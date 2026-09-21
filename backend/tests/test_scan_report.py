"""Feature-scan results: numbered findings, plain-language wording, a warning when the scan says yes to
nearly everything, and evidence that highlights only the flagged ground."""
import json
import os
import sys

import cv2
import numpy as np
from pypdf import PdfReader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.grid_scan import SCAN_YES_THRESHOLD  # noqa: E402
from agent_manager.scan_report import (  # noqa: E402
    build_scan_summary, confidence_label, find_findings, over_reporting_note, render_scan_evidence, scan_answer_text,
    _pin_tile,
)
from test_endpoint_wiring import client  # noqa: E402,F401

# The real mining-adapter grid of a mixed scene (two pits inside a town), from an earlier run
MIXED = np.array([[0.84, 0.41, 0.15, 0.12], [0.91, 0.25, 0.88, 0.27], [0.91, 0.59, 0.97, 0.35], [0.82, 0.62, 0.88, 0.85]])
SINGLE = np.array([[0.95, 0.9, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])


def test_findings_are_numbered_groups_with_position_and_confidence():
    found = find_findings(MIXED, SCAN_YES_THRESHOLD)
    assert [f["id"] for f in found] == [1, 2]
    left, right = found
    assert left["box"] == [0.0, 0.0, 1.0, 0.25] and len(left["tiles"]) == 4
    assert right["box"] == [0.25, 0.5, 1.0, 1.0]
    assert left["label"] == "High confidence" and right["label"] == "High confidence"   # best tiles 0.91 / 0.97
    assert 80 <= left["confidence"] <= 95 and left["peak"] == 91


def test_labels_follow_the_best_tile():
    assert confidence_label(0.95) == "High confidence" and confidence_label(0.8) == "Likely" and confidence_label(0.6) == "Possible"


def test_headline_is_plain_language_and_counts_areas():
    s = build_scan_summary("mining", SINGLE, SCAN_YES_THRESHOLD)
    assert s["level"] == "high" and "surface mining activity" in s["headline"].lower() and "1 area" in s["headline"]
    assert "2 areas" not in s["headline"] and s["coverage_pct"] == 12
    assert "adapter" not in scan_answer_text(s).lower() and "logit" not in scan_answer_text(s).lower()


def test_no_confident_area_says_so_and_mentions_faint_ones():
    faint = np.full((4, 4), 0.1)
    faint[1, 1] = faint[2, 2] = 0.6
    s = build_scan_summary("deforestation", faint, SCAN_YES_THRESHOLD)
    assert s["level"] == "none" and s["findings"] == [] and s["faint_areas"] == 2
    assert "No confident forest clearing found" in s["headline"] and "2 faint areas" in s["headline"]
    assert build_scan_summary("agriculture", np.zeros((4, 4)) + 0.1, SCAN_YES_THRESHOLD)["headline"] == "No confident cultivated land found."


def test_yes_almost_everywhere_is_flagged_as_possible_over_reporting():
    assert over_reporting_note(np.full((4, 4), 0.95)).startswith("The scan answered yes for 16 of 16")
    assert "over-reporting" in build_scan_summary("mining", MIXED, SCAN_YES_THRESHOLD)["note"]      # 10 of 16 tiles above 0.5
    assert over_reporting_note(SINGLE) == ""
    assert build_scan_summary("mining", SINGLE, SCAN_YES_THRESHOLD)["note"] == ""


def test_evidence_tints_flagged_ground_and_leaves_the_rest_alone():
    img = np.full((200, 300, 3), 120, np.uint8)
    out = render_scan_evidence(img, SINGLE, build_scan_summary("mining", SINGLE, SCAN_YES_THRESHOLD))
    assert out.shape == img.shape and out.dtype == np.uint8
    changed = (out != img).any(axis=2)
    assert changed[:50, :150].mean() > 0.9                    # the flagged top-left tiles are marked
    assert not changed[120:, 150:].any()                      # the far, clearly negative corner is untouched
    assert 0.05 < changed.mean() < 0.4                        # NOT the whole image


def test_a_mixed_scene_is_no_longer_highlighted_everywhere():
    img = np.full((240, 400, 3), 100, np.uint8)
    out = render_scan_evidence(img, MIXED, build_scan_summary("mining", MIXED, SCAN_YES_THRESHOLD))
    changed = (out != img).any(axis=2)
    assert not changed[:30, 320:].any()                       # the town corner (P 0.12) stays clean
    assert changed.mean() < 0.75


def test_numbered_pins_are_drawn_per_finding():
    img = np.full((200, 300, 3), 60, np.uint8)
    s = build_scan_summary("mining", MIXED, SCAN_YES_THRESHOLD)
    out = render_scan_evidence(img, MIXED, s)
    radius = max(11, int(300 / 40.0))
    for finding in s["findings"]:
        px, py = _pin_tile(finding, 4, 4)
        x, y = int(px * 300), int(py * 200)
        assert out[y, x + radius + 1].min() > 190                  # the pin's white ring (anti-aliased)
        assert tuple(out[y, x - radius + 3]) != tuple(img[y, x])   # its filled disc
    quiet = np.full((4, 4), 0.1)
    assert render_scan_evidence(img, quiet, build_scan_summary("mining", quiet, 0.8)).tolist() == img.tolist()


# ------------------------------------------------------------------ through /analyze

def _scan(client, grid, name="a.png"):
    ok, buf = cv2.imencode(".png", np.full((160, 240, 3), 90, np.uint8))
    client.agent.scan_grid = grid
    return client.post("/analyze", data={"query": "Extract features.", "adapter": "mining", "modality_a": "optical", "chat_history": "[]"},
                       files=[("images", (name, buf.tobytes(), "image/png"))])


def test_scan_response_carries_findings_and_plain_answer(client):
    r = _scan(client, SINGLE)
    assert r.status_code == 200, r.text
    body = r.json()
    scan = body["scan"]
    assert scan["adapter"] == "mining" and scan["level"] == "high" and len(scan["findings"]) == 1
    assert scan["findings"][0]["where"] == "upper-left" and scan["threshold"] == SCAN_YES_THRESHOLD
    assert body["answer"] == scan["headline"] and "potential region" not in body["answer"]
    assert client.agent.calls[-1] == {"scan": "mining", "prompt": "Extract features."}


def test_scan_evidence_and_data_json_and_pdf_show_the_summary(client):
    r = _scan(client, SINGLE)
    body = r.json()
    session = body["agent_execution_trace"]["pipeline_id"]
    evidence = cv2.imread(os.path.join("reports", session, "evidence.png"))
    assert evidence.shape[:2] == (160, 240) and (evidence != 90).any()
    record = json.load(open(os.path.join("reports", session, "data.json")))
    assert record["scan"]["headline"] == body["scan"]["headline"]
    text = " ".join(p.extract_text() for p in PdfReader(os.path.join("reports", session, "report.pdf")).pages)
    assert "RESULT AT A GLANCE" in text and "surface mining activity" in text.lower().replace("\n", " ")

    rebuilt = client.post("/report", data={"session_id": session, "chat_history": "[]", "map_link": "https://www.openstreetmap.org/#map=15/30.7/76.8"})
    assert rebuilt.status_code == 200 and rebuilt.json()["report_download_url"]
    text = " ".join(p.extract_text() for p in PdfReader(os.path.join("reports", session, "report_chat.pdf")).pages)
    assert "openstreetmap.org" in text
    bad = client.post("/report", data={"session_id": session, "chat_history": "[]", "map_link": "javascript:alert(1)"})
    assert bad.status_code == 200
    text = " ".join(p.extract_text() for p in PdfReader(os.path.join("reports", session, "report_chat.pdf")).pages)
    assert "javascript" not in text


def test_general_questions_have_no_scan_block(client):
    ok, buf = cv2.imencode(".png", np.full((64, 64, 3), 90, np.uint8))
    r = client.post("/analyze", data={"query": "what is this?", "adapter": "general", "modality_a": "optical", "chat_history": "[]"},
                    files=[("images", ("a.png", buf.tobytes(), "image/png"))])
    assert r.status_code == 200 and r.json()["scan"] is None
