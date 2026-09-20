import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion.sar_optical_fusion import (  # noqa: E402
    FUSION_METHOD,
    create_fusion_composite,
    execute_optical_sar_fusion,
    fusion_system_prompt,
)


def optical(h=64, w=64, rgb=(40, 160, 60)):
    return np.full((h, w, 3), rgb, np.uint8)


def gradient_sar(h=64, w=64):
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def test_hue_and_saturation_come_from_optical_brightness_from_sar():
    opt = optical()
    out = create_fusion_composite(opt, gradient_sar())
    hsv_o = cv2.cvtColor(opt, cv2.COLOR_RGB2HSV).astype(int)
    hsv_f = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(int)
    bright = hsv_f[:, :, 2] > 90                            # hue is only well defined away from black
    assert bright.any()
    assert np.abs(hsv_f[:, :, 0][bright] - hsv_o[:, :, 0][bright]).max() <= 3
    v = hsv_f[:, :, 2].astype(float)
    assert np.corrcoef(v.ravel(), gradient_sar().astype(float).ravel())[0, 1] > 0.99


def test_sar_gap_keeps_optical_brightness_and_does_not_skew_normalisation():
    opt = optical(rgb=(200, 120, 50))
    sar = gradient_sar()
    sar[:, :16] = 0                                         # nodata strip
    valid = np.ones(sar.shape, bool)
    valid[:, :16] = False
    with_mask = create_fusion_composite(opt, sar, valid)
    without = create_fusion_composite(opt, sar)

    v_opt = cv2.cvtColor(opt, cv2.COLOR_RGB2HSV)[:, :, 2]
    v_masked = cv2.cvtColor(with_mask, cv2.COLOR_RGB2HSV)[:, :, 2]
    assert np.abs(v_masked[:, :16].astype(int) - v_opt[:, :16].astype(int)).max() <= 2   # gap keeps optical V
    v_unmasked = cv2.cvtColor(without, cv2.COLOR_RGB2HSV)[:, :, 2]
    assert v_unmasked[:, :16].mean() < 20                   # without a mask the gap is painted black


def test_mask_is_resized_with_sar_and_all_invalid_raises():
    opt = optical(64, 64)
    sar = gradient_sar(32, 32)
    create_fusion_composite(opt, sar, np.ones((32, 32), bool))
    with pytest.raises(ValueError):
        create_fusion_composite(opt, sar, np.zeros((32, 32), bool))


def test_non_uint8_optical_and_all_zero_do_not_crash():
    create_fusion_composite(np.zeros((16, 16, 3), np.float32), gradient_sar(16, 16))
    create_fusion_composite(np.random.rand(16, 16, 3).astype(np.float32), gradient_sar(16, 16))


def test_prompt_does_not_promise_colours_and_describes_the_method():
    p = fusion_system_prompt()
    assert "Cyan" not in p and "NEVER mention" not in p and "Internal Key" not in p
    assert "backscatter" in p and "hue" in p
    assert "HSV" in FUSION_METHOD and "False-Color" not in FUSION_METHOD


def test_execute_returns_system_and_user_prompt_separately():
    res = execute_optical_sar_fusion(optical(), gradient_sar(), "how much water?")
    assert res["generated_prompt"] == "how much water?"
    assert res["system_prompt"] == fusion_system_prompt() and res["method"] == FUSION_METHOD
    assert res["composite_image"].shape == (64, 64, 3)
