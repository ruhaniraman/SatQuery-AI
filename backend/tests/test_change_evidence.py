"""Change-detection evidence: measured before->after shifts and per-image / per-region captions."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.scene_stats import compute_scene_stats, format_pair_stats  # noqa: E402
from change_detection.cdvqa_engine import (  # noqa: E402
    MAX_REGION_NOTES,
    ChangeDetectionEngine,
    build_change_prompt,
    gather_change_evidence,
)
from test_endpoint_wiring import client, png_bytes, post  # noqa: E402,F401


def solid(rgb, size=64):
    return np.full((size, size, 3), rgb, np.uint8)


def textured(seed, h=400, w=400):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(60, 190, (h // 8, w // 8, 3)).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ measured before -> after

def test_pair_stats_report_direction_and_stability():
    grey, green = compute_scene_stats(solid((120, 120, 120))), compute_scene_stats(solid((30, 160, 40)))
    up = format_pair_stats(grey, green)
    assert "green-dominant pixels up (0% -> 100%)" in up
    assert "green-dominant pixels down (100% -> 0%)" in format_pair_stats(green, grey)
    same = format_pair_stats(grey, grey)
    assert "green-dominant pixels about the same" in same and "up" not in same and "down" not in same


def test_pair_stats_skip_colour_for_sar_and_missing_sides():
    sar = compute_scene_stats(solid((30, 160, 40)), modality="sar")
    assert "green" not in format_pair_stats(sar, sar)
    assert format_pair_stats(None, sar) == "" and format_pair_stats(sar, None) == ""


def test_valid_mask_keeps_alignment_borders_out_of_the_proportions():
    img = solid((120, 120, 120))
    img[:, :32] = 0                                       # black border left by warping
    valid = np.ones((64, 64), bool)
    valid[:, :32] = False
    assert compute_scene_stats(img)["dark_pct"] == 50
    assert compute_scene_stats(img, valid_mask=valid)["dark_pct"] == 0
    assert compute_scene_stats(img, valid_mask=np.zeros((64, 64), bool)) is None


# ------------------------------------------------------------------ captions

REGIONS = [(10, 10, 80, 80, 0.04), (200, 200, 100, 100, 0.06), (300, 20, 60, 60, 0.02)]


def recorder(reply="a caption"):
    calls = []

    def fn(prompt, image):
        calls.append((prompt, image))
        return reply
    return fn, calls


def test_without_notes_fn_only_measured_numbers_are_returned():
    block, facts = gather_change_evidence(solid((120, 120, 120), 400), solid((30, 160, 40), 400), REGIONS)
    assert block.startswith("Measured before -> after") and "BEFORE image" not in block
    assert facts and facts in block


def test_each_image_and_the_first_regions_are_captioned_separately():
    fn, calls = recorder()
    a, b = textured(1), textured(2)
    block, _ = gather_change_evidence(a, b, REGIONS, fn)
    assert len(calls) == 2 + 2 * MAX_REGION_NOTES          # 2 whole images + before/after for each of 2 regions
    assert calls[0][1].shape == a.shape and calls[1][1].shape == b.shape
    crop = calls[2][1]
    assert crop.shape[0] < a.shape[0] and crop.flags["C_CONTIGUOUS"]
    assert "BEFORE image" in block and "AFTER image" in block
    assert "Flagged region, upper-left" in block and "may contain mistakes" in block


def test_a_failing_caption_call_is_not_fatal_and_keeps_the_measured_numbers():
    def boom(prompt, image):
        raise RuntimeError("gpu out of memory")
    block, facts = gather_change_evidence(textured(1), textured(2), REGIONS, boom)
    assert block == f"Measured before -> after (pixel counts, not materials): {facts}"


def test_prompt_puts_evidence_before_the_users_question():
    text = build_change_prompt([], (300, 300), "any new roads?", evidence="EVIDENCE-BLOCK")
    assert text.index("EVIDENCE-BLOCK") < text.index("Also answer: any new roads?")
    assert "CHANGES:" in text and "ASSESSMENT:" in text and "seasonal" in text


def test_detect_captions_first_then_asks_about_the_stitched_pair_last():
    a = textured(1)
    b = a.copy()
    b[150:300, 200:350] = 0                                # one genuine change
    seen = []
    res = ChangeDetectionEngine().detect(a, b, lambda p, i: seen.append((p, i)) or "ok", notes_fn=lambda p, i: "note")
    assert res.major_regions_detected == 1 and res.measured_facts
    # the notes_fn above is separate from vlm_fn, so only the final call reaches vlm_fn
    assert len(seen) == 1 and seen[0][1].shape[1] == 2 * a.shape[1]
    assert "BEFORE image, first look" in seen[0][0] and "Flagged region" in seen[0][0]


# ------------------------------------------------------------------ endpoint wiring

def test_endpoint_runs_captions_before_the_final_call_and_records_it_in_the_trace(client, monkeypatch):
    monkeypatch.setenv("DESCRIBE_FIRST", "1")
    img = png_bytes(np.full((64, 64, 3), 90, np.uint8))
    r = post(client, [("images", ("a.png", img, "image/png")), ("images", ("b.png", img, "image/png"))])
    assert r.status_code == 200, r.text
    assert len(client.vlm_calls) == 3                      # 2 image captions (no flagged regions) + the final call
    assert client.vlm_calls[-1][1].shape[1] == 128         # last call is the stitched pair
    assert "BEFORE image, first look" in client.vlm_calls[-1][0]
    tel = r.json()["agent_execution_trace"]["telemetry"]
    assert tel["describe_first"] is True and "pixels" in tel["measured_before_after"]


def test_endpoint_with_describe_first_off_makes_a_single_call(client):
    img = png_bytes(np.full((64, 64, 3), 90, np.uint8))
    r = post(client, [("images", ("a.png", img, "image/png")), ("images", ("b.png", img, "image/png"))])
    assert r.status_code == 200, r.text
    assert len(client.vlm_calls) == 1
    assert r.json()["agent_execution_trace"]["telemetry"]["describe_first"] is False
