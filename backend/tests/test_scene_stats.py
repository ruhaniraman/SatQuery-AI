"""scene_stats: measured proportions handed to the model, and their wiring into the general path."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.scene_stats import compute_scene_stats, format_region_shift, format_scene_stats  # noqa: E402


def solid(rgb, size=64):
    return np.full((size, size, 3), rgb, np.uint8)


def test_green_scene_is_mostly_green_dominant_and_smooth():
    stats = compute_scene_stats(solid((30, 160, 40)))
    assert stats["green_dominant_pct"] == 100
    assert stats["dark_pct"] == 0 and stats["bright_pct"] == 0
    assert stats["edge_pct"] == 0


def test_grey_and_dark_scenes_are_not_green():
    assert compute_scene_stats(solid((120, 120, 120)))["green_dominant_pct"] == 0
    dark = compute_scene_stats(solid((10, 10, 10)))
    assert dark["dark_pct"] == 100


def test_half_bright_half_dark_and_edges():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, 32:] = 255
    stats = compute_scene_stats(img)
    assert stats["dark_pct"] == 50 and stats["bright_pct"] == 50
    assert stats["edge_pct"] > 0            # the vertical seam


def test_sar_has_no_colour_measure():
    stats = compute_scene_stats(solid((30, 160, 40)), modality="sar")
    assert "green_dominant_pct" not in stats
    assert "green" not in format_scene_stats(stats)


def test_large_images_are_measured_on_a_downsample_but_report_true_size():
    stats = compute_scene_stats(np.zeros((3000, 2000, 3), np.uint8))
    assert (stats["width_px"], stats["height_px"]) == (2000, 3000)


def test_unusable_input_returns_none_and_empty_text():
    for bad in (None, np.zeros((4, 4), np.uint8), np.zeros((0, 4, 3), np.uint8)):
        assert compute_scene_stats(bad) is None
    assert format_scene_stats(None) == ""


def test_format_says_pixels_not_materials():
    text = format_scene_stats(compute_scene_stats(solid((30, 160, 40))))
    assert "green-dominant" in text and "vegetation" not in text and "water" not in text
    assert text.endswith(".")


# ------------------------------------------------------------------ wiring

def test_general_path_passes_measured_facts_to_the_engine_and_trace(client):
    from test_audit_trace import one_image, trace_of

    resp = one_image(client, query="what is this?")
    tel = trace_of(resp)["telemetry"]
    assert "pixels" in tel["measured_scene_facts"]
    assert "pixels" in client.agent.calls[-1]["scene_facts"]


def test_scans_do_not_compute_scene_facts(client):
    from test_audit_trace import one_image, trace_of

    resp = one_image(client, query="scan", adapter="mining")
    assert "measured_scene_facts" not in trace_of(resp)["telemetry"]
    assert client.agent.calls[-1] == {"scan": "mining", "prompt": "scan"}   # a scan never goes through query(), so no scene facts


from test_endpoint_wiring import client  # noqa: E402,F401


# ------------------------------------------------------------------ one flagged area, before -> after

def test_region_shift_lists_only_what_moved():
    forest, bare = compute_scene_stats(solid((30, 160, 40))), compute_scene_stats(solid((200, 170, 160)))
    text = format_region_shift(forest, bare)
    assert text.startswith("green-dominant pixels 100% -> 0%")
    assert "average brightness" in text and "-> 0%" in text
    assert format_region_shift(forest, forest) == "averages over this area barely moved; the change is patchy or local"


def test_region_shift_handles_sar_and_missing_sides():
    dark, light = compute_scene_stats(solid((40, 40, 40)), "sar"), compute_scene_stats(solid((180, 180, 180)), "sar")
    text = format_region_shift(dark, light)
    assert "green" not in text and "average brightness 40 -> 180 (0-255)" in text
    assert format_region_shift(None, light) == "" and format_region_shift(dark, None) == ""


def test_region_shift_ignores_small_brightness_drift():
    a, b = compute_scene_stats(solid((100, 100, 100))), compute_scene_stats(solid((110, 110, 110)))
    assert format_region_shift(a, b) == "averages over this area barely moved; the change is patchy or local"


def test_region_shift_reports_a_change_of_tone_when_brightness_and_green_did_not_move():
    warm, cool = compute_scene_stats(solid((150, 120, 100))), compute_scene_stats(solid((100, 120, 150)))
    assert format_region_shift(warm, cool) == "average colour (R,G,B) 150,120,100 -> 100,120,150"
    near = compute_scene_stats(solid((150, 120, 100)))
    assert format_region_shift(warm, near) == "averages over this area barely moved; the change is patchy or local"
