"""Tests for the SAR despeckling and GeoTIFF loading/normalisation modules (synthetic data only)."""
import os
import sys

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geospatial_preprocessing.geotiff_loader import (  # noqa: E402
    load_and_standardize_image,
    load_and_standardize_pair,
)
from geospatial_preprocessing.sar_preprocessor import (  # noqa: E402
    despeckle_band,
    lee_filter,
    preprocess_sar,
)

RNG = np.random.default_rng(0)
LOOKS = 4  # speckle: gamma-distributed multiplier with mean 1, Cu^2 = 1/LOOKS


def speckled(clean: np.ndarray, looks: int = LOOKS) -> np.ndarray:
    return (clean * RNG.gamma(looks, 1.0 / looks, clean.shape)).astype(np.float32)


def write_tif(path, data, *, nodata=None, colorinterp=None, descriptions=None, crs="EPSG:32643"):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
        dtype=data.dtype, crs=crs, transform=from_origin(500000, 1500000, 10, 10), nodata=nodata,
    ) as dst:
        dst.write(data)
        if colorinterp:
            dst.colorinterp = colorinterp
        if descriptions:
            for i, d in enumerate(descriptions, start=1):
                dst.set_band_description(i, d)
    return str(path)


# ----------------------------------------------------------------------------- lee filter

def test_lee_recovers_constant_region_and_keeps_edge():
    clean = np.full((200, 200), 1.0, np.float32)
    clean[:, 100:] = 5.0                                   # a sharp radiometric edge
    noisy = speckled(clean)
    out = lee_filter(noisy, window_size=7, enl=LOOKS)

    flat = (slice(20, 80), slice(20, 80))
    assert np.std(out[flat]) < 0.5 * np.std(noisy[flat])   # speckle strongly reduced
    assert abs(np.mean(out[flat]) - 1.0) < 0.05            # and the true level is recovered
    step = np.mean(out[:, 110:130]) - np.mean(out[:, 70:90])
    assert 3.6 < step < 4.4                                # edge preserved (true step is 4)


def test_lee_estimated_noise_level_works_without_enl():
    noisy = speckled(np.full((256, 256), 2.0, np.float32))
    out = lee_filter(noisy)
    assert np.std(out) < 0.6 * np.std(noisy)


# ----------------------------------------------------------------------------- units

def test_linear_and_db_inputs_agree_and_db_not_converted_twice():
    linear = speckled(np.full((128, 128), 0.1, np.float32))
    db = 10 * np.log10(linear)
    valid = np.ones(linear.shape, bool)

    out_lin, u_lin = despeckle_band(linear, valid, integer_source=False)
    out_db, u_db = despeckle_band(db, valid, integer_source=False)

    assert (u_lin, u_db) == ("linear", "db")
    np.testing.assert_allclose(out_lin, out_db, atol=0.05)
    assert abs(np.median(out_lin) - (-10.0)) < 1.0         # 0.1 linear == -10 dB, not -20 or +10


def test_db_data_with_a_positive_scatterer_is_still_treated_as_db():
    db = np.full((64, 64), -12.0, np.float32)
    db[10, 10] = 5.0                                       # one bright target; max() > 0
    _, units = despeckle_band(db, np.ones(db.shape, bool), integer_source=False)
    assert units == "db"


def test_integer_source_is_display_and_not_db_converted():
    dn = RNG.integers(50, 200, (64, 64)).astype(np.float32)
    out, units = despeckle_band(dn, np.ones(dn.shape, bool), integer_source=True)
    assert units == "display" and 40 < out.mean() < 210


def test_preprocess_sar_handles_2d_and_3d():
    a = speckled(np.full((64, 64), 1.0, np.float32))
    assert preprocess_sar(a).shape == (64, 64)
    assert preprocess_sar(np.dstack([a, a])).shape == (64, 64, 2)


# ----------------------------------------------------------------------------- loader

def test_nan_and_nodata_are_masked_not_all_black(tmp_path):
    data = RNG.uniform(100, 3000, (3, 64, 64)).astype(np.float32)
    data[:, :, :16] = -9999                                # nodata strip
    data[:, 5, 20:30] = np.nan
    p = write_tif(tmp_path / "a.tif", data, nodata=-9999)
    img, meta = load_and_standardize_image(p)

    valid = meta["valid_mask"]
    assert not valid[:, :16].any() and not valid[5, 20:30].any()
    assert valid[:, 16:].sum() > 0.9 * 64 * 48
    assert img.dtype == np.uint8 and img.shape == (64, 64, 3)
    assert (img[~valid] == 0).all()
    assert img[valid].max() > 200                          # stretched, not collapsed to black


def test_all_nodata_raises(tmp_path):
    p = write_tif(tmp_path / "n.tif", np.full((3, 8, 8), -9999, np.float32), nodata=-9999)
    with pytest.raises(ValueError):
        load_and_standardize_image(p)


def test_band_order_from_colorinterp(tmp_path):
    data = np.zeros((3, 32, 32), np.uint16)
    data[0], data[1], data[2] = 3000, 1000, 100            # file stores B, G, R order
    data += RNG.integers(0, 50, data.shape).astype(np.uint16)
    p = write_tif(tmp_path / "bgr.tif", data,
                  colorinterp=[ColorInterp.blue, ColorInterp.green, ColorInterp.red])
    img, meta = load_and_standardize_image(p)
    assert meta["warnings"] == []
    # so RGB output channel 2 (blue) must come from file band 1 (the bright one)
    assert img[:, :, 2].mean() > img[:, :, 0].mean()


def test_band_order_from_descriptions_and_fallback_warning(tmp_path):
    data = RNG.integers(0, 4000, (3, 32, 32)).astype(np.uint16)
    p = write_tif(tmp_path / "d.tif", data, descriptions=["B02", "B03", "B04"])
    img, meta = load_and_standardize_image(p)
    assert meta["warnings"] == []

    p2 = write_tif(tmp_path / "u.tif", data)
    _, meta2 = load_and_standardize_image(p2)
    assert meta2["warnings"] and "assumed" in meta2["warnings"][0]


def test_metadata_labels_resolution_unit(tmp_path):
    p = write_tif(tmp_path / "m.tif", RNG.integers(0, 255, (3, 16, 16)).astype(np.uint8))
    _, meta = load_and_standardize_image(p)
    assert meta["crs"] == "EPSG:32643" and meta["spatial_resolution"] == 10
    assert meta["resolution_unit"] in ("metre", "metres", "meter", "m")


def test_sar_tif_is_despeckled_and_reports_units(tmp_path):
    p = write_tif(tmp_path / "s.tif", speckled(np.full((96, 96), 0.2, np.float32)))
    img, meta = load_and_standardize_image(p, modality="sar")
    assert meta["sar_input_units"] == "linear"
    assert img.shape == (96, 96, 3)
    raw_img, _ = load_and_standardize_image(p, modality="optical")
    assert img[:, :, 0].std() < raw_img[:, :, 0].std()


# ----------------------------------------------------------------------------- shared stretch

def test_shared_stretch_removes_false_change_on_unchanged_pixels(tmp_path):
    """Same scene twice; B has a genuine bright change in a small patch. Independent per-image
    percentiles shift because of that patch and make every UNCHANGED pixel differ; a shared
    stretch keeps unchanged pixels identical and leaves the real change visible."""
    base = RNG.uniform(500, 2500, (3, 128, 128)).astype(np.float32)
    after = base.copy()
    after[:, 10:40, 10:40] = 9000                          # real change (>2% of pixels)
    pa = write_tif(tmp_path / "a.tif", base)
    pb = write_tif(tmp_path / "b.tif", after)
    unchanged = np.ones((128, 128), bool)
    unchanged[10:40, 10:40] = False

    ia, _ = load_and_standardize_image(pa)
    ib, _ = load_and_standardize_image(pb)
    indep = np.abs(ia.astype(int) - ib.astype(int))[unchanged].mean()

    sa, ma, sb, mb = load_and_standardize_pair(pa, pb)
    shared = np.abs(sa.astype(int) - sb.astype(int)).max(axis=2)
    assert indep > 5                                       # the bug is real
    assert shared[unchanged].max() == 0                    # and the fix removes it
    assert shared[~unchanged].mean() > 50                  # true change still stands out
    assert ma["normalization"] == "shared"


def test_pair_of_standard_images_passes_through_unchanged(tmp_path):
    a = RNG.integers(0, 255, (32, 32, 3)).astype(np.uint8)
    pa, pb = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    cv2.imwrite(pa, a)
    cv2.imwrite(pb, a)
    ia, _, ib, _ = load_and_standardize_pair(pa, pb)
    np.testing.assert_array_equal(ia, cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    np.testing.assert_array_equal(ia, ib)
