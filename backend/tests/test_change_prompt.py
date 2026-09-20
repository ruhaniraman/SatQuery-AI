import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from change_detection.cdvqa_engine import ChangeDetectionEngine, build_change_prompt, describe_location  # noqa: E402


def textured(seed, h=400, w=400):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(60, 190, (h // 8, w // 8, 3)).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_CUBIC)


def test_location_words():
    assert describe_location(10, 10, 300, 300) == "upper-left"
    assert describe_location(290, 290, 300, 300) == "lower-right"
    assert describe_location(150, 150, 300, 300) == "centre"
    assert describe_location(150, 10, 300, 300) == "upper-centre"
    assert describe_location(10, 150, 300, 300) == "left"
    assert describe_location(299, 150, 300, 300) == "right"


def test_prompt_lists_regions_or_says_none():
    with_regions = build_change_prompt([(10, 10, 50, 50, 0.04)], (300, 300), "any new roads?")
    assert "1 region(s)" in with_regions and "upper-left" in with_regions and "any new roads?" in with_regions
    none = build_change_prompt([], (300, 300))
    assert "no region of substantial change" in none and "region(s)" not in none
    assert "backscatter" in build_change_prompt([], (300, 300), modality="sar")
    assert "backscatter" not in build_change_prompt([], (300, 300))


def test_vlm_is_told_exactly_what_the_boxes_show():
    a = textured(1)
    b = a.copy()
    b[150:300, 200:350] = 0                                    # one genuine change
    seen = []
    res = ChangeDetectionEngine().detect(a, b, lambda prompt, img: seen.append((prompt, img)) or "ok")
    assert res.major_regions_detected == 1
    prompt, stitched = seen[0]
    assert "flagged 1 region(s)" in prompt
    assert stitched.shape == (400, 800, 3)


def test_vlm_is_told_when_nothing_was_found():
    a = textured(2)
    seen = []
    res = ChangeDetectionEngine().detect(a, a.copy(), lambda prompt, img: seen.append(prompt) or "ok")
    assert res.major_regions_detected == 0 and "no region of substantial change" in seen[0]
