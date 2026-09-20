"""Optical-only cloud mask, size-scaled thresholds, and ORB transform validation."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from change_detection.cdvqa_engine import ChangeDetectionEngine  # noqa: E402
from geospatial_preprocessing.standard_alignment import _transform_is_plausible, align_standard_images  # noqa: E402


def textured(seed, h, w, cell=8):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 190, (h // cell, w // cell, 3)).astype(np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def regions(a, b, **kw):
    return ChangeDetectionEngine().detect(a, b, lambda p, i: "x", **kw).major_regions_detected


# ------------------------------------------------------------------ cloud mask is optical-only

def test_bright_change_is_found_in_sar_but_masked_as_cloud_in_optical():
    a = textured(1, 400, 400)
    b = a.copy()
    b[150:300, 200:350] = 255                       # bright new structure
    assert regions(a, b, modality="optical") == 0   # a bright blob in optical is treated as cloud
    assert regions(a, b, modality="sar") >= 1       # in SAR it is a real scatterer


# ------------------------------------------------------------------ size scaling

def test_small_image_change_is_detectable():
    """A 90x90 block (8100 px) can never pass the old absolute 15000 px area threshold."""
    a = textured(2, 256, 256)
    b = a.copy()
    b[80:170, 80:170] = 0
    assert regions(a, b) == 1


def test_same_scene_at_two_sizes_gives_same_answer():
    a = textured(3, 512, 512, cell=16)
    b = a.copy()
    b[150:350, 150:350] = 0
    big_a, big_b = (cv2.resize(x, (1024, 1024), interpolation=cv2.INTER_NEAREST) for x in (a, b))
    assert regions(a, b) == regions(big_a, big_b) == 1


def test_no_change_no_regions_at_large_size():
    a = textured(4, 1024, 1024, cell=16)
    assert regions(a, a.copy()) == 0


# ------------------------------------------------------------------ ORB validation

def test_transform_plausibility_rules():
    f = _transform_is_plausible
    ok_inl = np.ones((30, 1), np.uint8)
    ident = np.float32([[1, 0, 5], [0, 1, -3]])
    assert f(ident, ok_inl, (400, 400))[0]
    assert not f(None, None, (400, 400))[0]
    assert "inliers" in f(ident, np.ones((5, 1), np.uint8), (400, 400))[1]
    assert "scale" in f(np.float32([[2.0, 0, 0], [0, 2.0, 0]]), ok_inl, (400, 400))[1]
    th = np.radians(40)
    rot = np.float32([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0]])
    assert "rotation" in f(rot, ok_inl, (400, 400))[1]
    assert "translation" in f(np.float32([[1, 0, 300], [0, 1, 0]]), ok_inl, (400, 400))[1]


@pytest.mark.parametrize("seed", range(5))
def test_unrelated_images_fall_back_to_plain_resize(seed):
    align = align_standard_images
    a = textured(100 + seed, 300, 300, cell=4)
    b = textured(200 + seed, 300, 300, cell=4)
    out, valid = align(a, b)
    np.testing.assert_array_equal(out, b)            # untouched: no bogus warp applied
    assert valid.all()


def test_genuine_small_shift_is_still_corrected():
    align = align_standard_images
    rng = np.random.default_rng(7)
    a = cv2.GaussianBlur(rng.integers(0, 255, (400, 400, 3)).astype(np.uint8), (0, 0), 1.5)
    b = cv2.warpAffine(a, np.float32([[1, 0, 12], [0, 1, 6]]), (400, 400))
    out, valid = align(a, b)
    inner = (slice(40, 360), slice(40, 360))
    err_before = np.abs(a.astype(int) - b.astype(int))[inner].mean()
    err_after = np.abs(a.astype(int) - out.astype(int))[inner].mean()
    assert err_after < 0.2 * err_before
    assert not valid.all()                           # the warp introduced a border
