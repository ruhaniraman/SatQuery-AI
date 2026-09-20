"""Warped/filled borders must not be reported as change; payload guard; ORB validity mask."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from change_detection.cdvqa_engine import ChangeDetectionEngine  # noqa: E402
from geospatial_preprocessing.payload_validation import validate_downstream_payload  # noqa: E402

RNG = np.random.default_rng(2)


def textured(h=400, w=400):
    base = RNG.integers(60, 190, (h // 8, w // 8, 3)).astype(np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def run(a, b, valid=None):
    return ChangeDetectionEngine().detect(a, b, lambda p, i: "x", valid_mask=valid)


def test_black_border_is_false_change_without_mask_and_ignored_with_mask():
    a = textured()
    b = a.copy()
    b[:, :100] = 0                                    # what warping leaves outside B's footprint
    valid = np.ones(a.shape[:2], bool)
    valid[:, :100] = False

    assert run(a, b).major_regions_detected >= 1      # the bug
    assert run(a, b, valid).major_regions_detected == 0


def test_real_change_inside_valid_area_is_still_found():
    a = textured()
    b = a.copy()
    b[150:300, 200:350] = 0                            # genuine change (dark: bright would be masked as cloud)
    b[:, :100] = 0
    valid = np.ones(a.shape[:2], bool)
    valid[:, :100] = False
    assert run(a, b, valid).major_regions_detected >= 1


def test_payload_guard():
    ok = np.zeros((8, 8, 3), np.uint8)
    assert validate_downstream_payload(ok, ok)
    assert not validate_downstream_payload(ok.astype(np.uint16))
    assert not validate_downstream_payload(ok[:, :, 0])
    assert not validate_downstream_payload(ok, np.zeros((4, 4, 3), np.uint8))


def test_orb_alignment_returns_mask_for_warped_border():
    from geospatial_preprocessing.standard_alignment import align_standard_images

    a = textured(300, 300)
    M = np.float32([[1, 0, 30], [0, 1, 0]])           # B is A shifted 30px right
    b = cv2.warpAffine(a, M, (300, 300))
    aligned, valid = align_standard_images(a, b)
    assert valid.shape == (300, 300) and valid.dtype == bool
    assert valid.mean() < 1.0 or np.array_equal(aligned, cv2.resize(b, (300, 300)))
