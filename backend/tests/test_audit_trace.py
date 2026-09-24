"""The audit trace and PDF must describe what actually ran, and nothing else."""
import io
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.grid_scan import SCAN_YES_THRESHOLD  # noqa: E402
from agent_manager.schemas import ExecutionTrace, TaskType  # noqa: E402
from test_endpoint_wiring import client, png_bytes, post, speckled, tif_bytes  # noqa: E402,F401

RNG = np.random.default_rng(9)
GRAY = lambda: png_bytes(np.full((64, 64, 3), 90, np.uint8))  # noqa: E731


def one_image(client, **form):
    return post(client, [("images", ("a.png", GRAY(), "image/png"))], **form)


def trace_of(r):
    assert r.status_code == 200, r.text
    return r.json()["agent_execution_trace"]


# ------------------------------------------------------------------ schema

def test_unused_schema_pieces_are_gone_and_routing_is_labelled_rule_based():
    assert "GROUNDING" not in TaskType.__members__
    import agent_manager.schemas as s
    assert not hasattr(s, "ClassificationResult")
    t = ExecutionTrace(pipeline_id="x", task=TaskType.SINGLE_IMAGE_VQA, routing_reason="r", nodes_traversed=["a"])
    assert t.routing == "rule-based" and t.validation_status == "not performed" and "confidence" not in t.model_dump()


# ------------------------------------------------------------------ single image

def test_general_single_image_trace_is_honest(client):
    client.agent.base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    tr = trace_of(one_image(client))
    assert tr["task"] == "SINGLE_IMAGE_VQA" and tr["routing"] == "rule-based"
    assert tr["routing_reason"] == "1 image uploaded"
    assert tr["nodes_traversed"] == ["RuleBasedRouter", "InputPreprocessor", "SingleImageSpecialist"]
    assert "VerifierNode" not in str(tr) and "BigEarthNet" not in str(tr)
    tel = tr["telemetry"]
    assert tel["active_adapter"] is None and "adapters disabled" in tel["model_used"]
    assert tel["inference"].startswith("free-text generation") and tel["max_new_tokens"] >= 512
    # A real, computed generation signal - explicitly NOT claimed to be a correctness probability
    assert tel["confidence"] == 0.87
    assert "NOT a calibrated correctness score" in tel["confidence_method"]


def test_confidence_is_just_omitted_when_it_could_not_be_computed(client):
    # A real deployment can fail to compute this (see agent_controller.py's own try/except around it);
    # the trace must stay honest rather than inventing a placeholder number.
    client.agent.query_confidence = None
    tel = trace_of(one_image(client))["telemetry"]
    assert "confidence" not in tel and "confidence_method" not in tel


def test_a_scan_never_claims_the_generation_confidence_field(client):
    # Scans have their own, different confidence signal (per-finding, from the P(yes) grid - see
    # scan_report.py); the generation-token-probability field only applies to the free-text general path.
    tel = trace_of(one_image(client, adapter="mining"))["telemetry"]
    assert "confidence" not in tel


def test_grid_scan_trace_names_the_adapter_and_its_scoring(client):
    client.agent.base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    client.agent.adapter_info = {"mining": {"base_model": "Qwen/Qwen2-VL-2B-Instruct", "lora_rank": 16, "lora_alpha": 32}}
    tel = trace_of(one_image(client, adapter="mining"))["telemetry"]
    assert "LoRA adapter 'mining'" in tel["model_used"] and tel["active_adapter"] == "mining"
    assert tel["adapter_details"]["lora_rank"] == 16
    assert tel["adapter_training_data"] == "not recorded in the adapter files"
    assert tel["grid_yes_threshold"] == SCAN_YES_THRESHOLD and "logit" in tel["inference"]


def test_chat_turns_given_to_model_is_recorded(client):
    hist = [{"role": "user", "content": "hi"}, {"role": "ai", "content": "hello"}, {"role": "user", "content": "what is this?"}]
    tel = trace_of(one_image(client, query="what is this?", chat_history=json.dumps(hist)))["telemetry"]
    assert tel["chat_turns_given_to_model"] == 2                # current question is not counted twice


def test_unknown_adapter_is_rejected_before_any_work(client):
    r = one_image(client, adapter="../../etc/passwd")
    assert r.status_code == 400 and "Invalid adapter" in r.json()["detail"]
    assert client.agent.calls == []


# ------------------------------------------------------------------ change detection / fusion

def test_change_detection_trace_on_plain_images_names_orb_and_warns_of_nothing(client):
    img = png_bytes(RNG.integers(0, 255, (64, 64, 3)).astype(np.uint8))
    tr = trace_of(post(client, [("images", ("a.png", img, "image/png")), ("images", ("b.png", img, "image/png"))]))
    assert tr["task"] == "CHANGE_DETECTION" and "same sensor type" in tr["routing_reason"]
    assert tr["nodes_traversed"] == ["RuleBasedRouter", "InputPreprocessor", "OrbAlignment", "TemporalOrdering",
                                     "Hybrid-CDVQA-Engine", "Qwen2-VL-Bridge"]
    assert tr["validation_status"] == "image payload checks passed"
    assert "adapters disabled" in tr["telemetry"]["model_used"]


def test_change_detection_geotiff_uses_spatial_alignment_and_surfaces_loader_warnings(client):
    a = tif_bytes(client.tmp, "a.tif", RNG.uniform(500, 2500, (3, 96, 96)).astype(np.float32))
    tr = trace_of(post(client, [("images", ("a.tif", a, "image/tiff")), ("images", ("b.tif", a, "image/tiff"))]))
    assert "SpatialAlignment" in tr["nodes_traversed"] and "OrbAlignment" not in tr["nodes_traversed"]
    assert any("assumed the first three bands" in w for w in tr["warnings"])   # undeclared band order is reported


def test_temporal_warnings_reach_the_trace(client):
    from datetime import datetime
    a = tif_bytes(client.tmp, "a.tif", RNG.uniform(500, 2500, (3, 96, 96)).astype(np.float32))
    same = "S2A_MSIL2A_20240110T051121_N0510_R062_T43PGQ_20240110T091234.tif"
    tr = trace_of(post(client, [("images", (same, a, "image/tiff")), ("images", (same, a, "image/tiff"))]))
    assert any("same capture date" in w for w in tr["warnings"])


def test_fusion_trace(client):
    opt = png_bytes(RNG.integers(0, 255, (64, 64, 3)).astype(np.uint8))
    sar = png_bytes(np.dstack([np.clip(speckled(100.0, (64, 64)), 0, 255).astype(np.uint8)] * 3))
    tr = trace_of(post(client, [("images", ("o.png", opt, "image/png")), ("images", ("s.png", sar, "image/png"))],
                       modality_a="optical", modality_b="sar"))
    assert tr["task"] == "OPTICAL_SAR_FUSION" and "different sensor types" in tr["routing_reason"]
    assert tr["nodes_traversed"] == ["RuleBasedRouter", "InputPreprocessor", "OpticalSARFusion", "Qwen2-VL-Bridge"]
    assert set(tr) == {"pipeline_id", "task", "routing", "routing_reason", "nodes_traversed", "telemetry",
                       "warnings", "validation_status", "execution_status"}


def test_saved_record_matches_the_response(client):
    r = one_image(client)
    tr = trace_of(r)
    with open(os.path.join(client.tmp, "reports", tr["pipeline_id"], "data.json")) as f:
        assert json.load(f)["agent_execution_trace"] == tr


# ------------------------------------------------------------------ PDF

pypdf = pytest.importorskip("pypdf")


def pdf_pages(trace, history=None):
    from pdf_report_generator import generate_pdf_report
    buf = generate_pdf_report(query="q", answer="a", agent_execution_trace=trace, image_source=None,
                              chat_history=history or [])
    return [p.extract_text() for p in pypdf.PdfReader(io.BytesIO(buf.getvalue())).pages]


BASE = {"pipeline_id": "abc", "task": "CHANGE_DETECTION", "routing": "rule-based", "routing_reason": "2 images",
        "nodes_traversed": ["RuleBasedRouter", "Hybrid-CDVQA-Engine"], "telemetry": {"model_used": "M"},
        "warnings": [], "validation_status": "passed", "execution_status": "completed"}


def test_pdf_status_reflects_the_run_and_never_claims_verified():
    text = " ".join(pdf_pages(BASE))
    assert "COMPLETED" in text and "VERIFIED" not in text.upper()
    warned = " ".join(pdf_pages({**BASE, "warnings": ["band order guessed"]}))
    assert "COMPLETED WITH WARNINGS" in warned and "band order guessed" in warned
    assert "NOT REPORTED" in " ".join(pdf_pages({"pipeline_id": "x"}))


def test_pdf_pagination_is_real():
    one = pdf_pages(BASE)
    assert len(one) == 1 and "Page 1 of 1" in one[0]

    hist = [{"role": "user" if i % 2 == 0 else "ai", "content": f"message {i} " + "word " * 60} for i in range(40)]
    pages = pdf_pages(BASE, hist)
    assert len(pages) > 1
    for i, text in enumerate(pages, start=1):
        assert f"Page {i} of {len(pages)}" in text


def test_pdf_trace_is_not_truncated():
    big = {**BASE, "telemetry": {"model_used": "M", "notes": [f"note-{i:03d}" for i in range(150)], "last": "ENDMARKER"}}
    text = " ".join(pdf_pages(big))
    assert "ENDMARKER" in text and "note-149" in text            # old code cut the dump at 1200 characters


def test_pdf_shows_task_routing_model_and_temporal_summary():
    tr = {**BASE, "telemetry": {"model_used": "Qwen base", "temporal_order": {"summary": "Order confirmed by dates"}}}
    text = re.sub(r"\s+", " ", " ".join(pdf_pages(tr)))
    assert "CHANGE_DETECTION" in text and "rule-based" in text and "Qwen base" in text
    assert "Order confirmed by dates" in text
