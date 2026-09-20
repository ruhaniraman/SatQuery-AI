"""Real capture dates, before/after resolution, and that nothing fabricates a date."""
import io
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.schemas import ImageInput  # noqa: E402
from geospatial_preprocessing.acquisition_date import (  # noqa: E402
    AcquisitionDate,
    date_from_filename,
    extract_acquisition_date,
    parse_date_string,
    resolve_temporal_order,
)
from test_endpoint_wiring import client, post  # noqa: E402,F401  (fixture + helper)

RNG = np.random.default_rng(5)
S2_OLD = "S2A_MSIL2A_20190110T051121_N0213_R062_T43PGQ_20190110T091234.tif"
S2_NEW = "S2B_MSIL2A_20240110T051121_N0510_R062_T43PGQ_20240110T091234.tif"


def write_tif(path, value, tags=None, x0=500000):
    data = np.full((3, 96, 96), value, np.float32) + RNG.uniform(0, 50, (3, 96, 96)).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=96, width=96, count=3, dtype="float32",
                       crs="EPSG:32643", transform=from_origin(x0, 1500000, 10, 10)) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
    return str(path)


def AD(day, trusted=True, has_time=False):
    from datetime import datetime
    return AcquisitionDate(datetime(2020, 1, day), has_time, "test", trusted)


# ------------------------------------------------------------------ parsing / filenames

@pytest.mark.parametrize("text,expected", [
    ("2024-01-10T05:11:21Z", ("2024-01-10 05:11:21", True)),
    ("2024:01:10 05:11:21", ("2024-01-10 05:11:21", True)),
    ("20240110", ("2024-01-10 00:00:00", False)),
    ("2024-01-10", ("2024-01-10 00:00:00", False)),
])
def test_parse_valid_dates(text, expected):
    when, has_time = parse_date_string(text)
    assert (when.strftime("%Y-%m-%d %H:%M:%S"), has_time) == expected


@pytest.mark.parametrize("text", ["", "no date here", "2024-13-45", "1850-01-01", "2999-01-01", "abcdefgh"])
def test_parse_rejects_implausible(text):
    assert parse_date_string(text) is None


@pytest.mark.parametrize("name,iso", [
    (S2_NEW, "2024-01-10 05:11:21 UTC"),
    ("S1A_IW_GRDH_1SDV_20240110T125034_20240110T125059_052000_064A2B_1A2B.tif", "2024-01-10 12:50:34 UTC"),
    ("S1A_IW_SLC__1SDV_20230505T010203_20230505T010230_048000_05C000_AAAA.tif", "2023-05-05 01:02:03 UTC"),
    ("LC08_L1TP_144051_20240110_20240120_02_T1.tif", "2024-01-10"),
    ("T43PGQ_20240110T051121_B04_10m.tif", "2024-01-10 05:11:21 UTC"),
])
def test_recognised_product_names_give_trusted_dates(name, iso):
    d = date_from_filename(name)
    assert d is not None and d.trusted and d.iso == iso


@pytest.mark.parametrize("name", [
    "photo_20240110.png", "site_2024_final.tif", "before.tif", "IMG_20240110_123456.jpg",
    "S2A_MSIL2A_notadate_N0510.tif", "S2A_MSIL2A_20241345T051121_N0510.tif", "myS2A_MSIL2A_20240110T051121_x.tif",
    "LC08_L1TP_XXXX_20240110.tif", None,
])
def test_loose_or_lookalike_names_are_not_dates(name):
    assert date_from_filename(name) is None


# ------------------------------------------------------------------ raster tags

def test_acquisition_tag_is_trusted_and_beats_filename(tmp_path):
    p = write_tif(tmp_path / "a.tif", 100, {"ACQUISITIONDATETIME": "2021-06-01T10:00:00Z"})
    d = extract_acquisition_date(p, S2_NEW)
    assert d.trusted and d.iso.startswith("2021-06-01") and "ACQUISITIONDATETIME" in d.source


def test_tiff_datetime_is_untrusted_and_filename_beats_it(tmp_path):
    p = write_tif(tmp_path / "a.tif", 100, {"TIFFTAG_DATETIME": "2025:03:03 12:00:00"})
    d = extract_acquisition_date(p, "scene.tif")
    assert d is not None and not d.trusted and d.iso.startswith("2025-03-03")
    d2 = extract_acquisition_date(p, S2_OLD)
    assert d2.trusted and d2.iso.startswith("2019-01-10")


def test_no_metadata_gives_none_not_a_placeholder(tmp_path):
    p = write_tif(tmp_path / "a.tif", 100)
    assert extract_acquisition_date(p, "before.tif") is None
    assert extract_acquisition_date(str(tmp_path / "missing.tif"), "x.tif") is None
    assert ImageInput(path="x").date is None


# ------------------------------------------------------------------ order resolution

def test_both_trusted_correct_order_is_confirmed():
    o = resolve_temporal_order(AD(1), AD(9))
    assert (o.swap, o.basis) == (False, "metadata") and o.prompt_dates == ("2020-01-01", "2020-01-09")


def test_both_trusted_reversed_order_is_swapped():
    o = resolve_temporal_order(AD(9), AD(1))
    assert o.swap and o.basis == "metadata" and o.prompt_dates == ("2020-01-01", "2020-01-09")
    assert "swapped" in o.summary


def test_same_date_keeps_upload_order_and_states_no_dates():
    o = resolve_temporal_order(AD(5), AD(5))
    assert not o.swap and o.basis == "as-uploaded" and o.prompt_dates is None and o.warnings


@pytest.mark.parametrize("a,b", [(None, None), (AD(1), None), (None, AD(1))])
def test_missing_dates_fall_back_to_as_uploaded(a, b):
    o = resolve_temporal_order(a, b)
    assert not o.swap and o.basis == "as-uploaded" and o.prompt_dates is None
    assert "as uploaded" in o.summary and "no date was assumed" in o.summary


def test_untrusted_dates_never_reorder_but_warn():
    o = resolve_temporal_order(AD(9, trusted=False), AD(1, trusted=False))
    assert not o.swap and o.basis == "as-uploaded" and o.prompt_dates is None
    assert any("not reliable" in w for w in o.warnings)


def test_one_trusted_one_untrusted_is_not_enough_to_reorder():
    o = resolve_temporal_order(AD(9), AD(1, trusted=False))
    assert not o.swap and o.basis == "as-uploaded"


# ------------------------------------------------------------------ endpoint behaviour

def upload(client, name_a, val_a, name_b, val_b, tags_a=None, tags_b=None):
    pa = write_tif(client.tmp / "ua.tif", val_a, tags_a)
    pb = write_tif(client.tmp / "ub.tif", val_b, tags_b)
    return post(client, [("images", (name_a, open(pa, "rb").read(), "image/tiff")),
                         ("images", (name_b, open(pb, "rb").read(), "image/tiff"))])


def test_reversed_uploads_are_swapped_using_filename_dates(client):
    # first upload is the NEWER scene and is bright; the older one is dark
    r = upload(client, S2_NEW, 2000, S2_OLD, 200)
    assert r.status_code == 200, r.text
    order = r.json()["agent_execution_trace"]["telemetry"]["temporal_order"]
    assert order["basis"] == "metadata" and order["reordered"] is True
    assert order["before"]["date"].startswith("2019-01-10") and order["after"]["date"].startswith("2024-01-10")
    prompt, stitched, _ = client.vlm_calls[0]
    assert "2019-01-10" in prompt and "2024-01-10" in prompt and "unknown" not in prompt
    left, right = stitched[:, :96], stitched[:, 96:]
    assert left.mean() < right.mean()                       # dark (older) image ended up on the left


def test_correctly_ordered_uploads_are_left_alone(client):
    r = upload(client, S2_OLD, 200, S2_NEW, 2000)
    order = r.json()["agent_execution_trace"]["telemetry"]["temporal_order"]
    assert order["basis"] == "metadata" and order["reordered"] is False
    stitched = client.vlm_calls[0][1]
    assert stitched[:, :96].mean() < stitched[:, 96:].mean()


def test_acquisition_tags_are_used_for_geotiffs_without_product_names(client):
    r = upload(client, "a.tif", 2000, "b.tif", 200,
               {"ACQUISITIONDATETIME": "2023-08-01T00:00:00Z"}, {"ACQUISITIONDATETIME": "2022-08-01T00:00:00Z"})
    order = r.json()["agent_execution_trace"]["telemetry"]["temporal_order"]
    assert order["reordered"] is True and order["before"]["source"].startswith("raster tag")


def test_unknown_dates_are_never_invented(client):
    r = upload(client, "before_2024.tif", 200, "after_2019.tif", 2000)   # misleading digits in names
    assert r.status_code == 200, r.text
    body = r.json()
    order = body["agent_execution_trace"]["telemetry"]["temporal_order"]
    assert order["basis"] == "as-uploaded" and order["reordered"] is False
    assert order["before"] is None and order["after"] is None
    prompt = client.vlm_calls[0][0]
    assert "capture dates are unknown" in prompt and "BEFORE" in prompt
    assert not any(d in prompt + str(body) for d in ("2019-01-10", "2024-01-10", "2024-02-10"))


def test_pdf_shows_temporal_summary_without_mutating_trace():
    from pdf_report_generator import generate_pdf_report
    trace = {"pipeline_id": "x", "telemetry": {"temporal_order": {"summary": "Order <b>test</b> & more"}, "k": 1}}
    import copy
    before = copy.deepcopy(trace)
    buf = generate_pdf_report(query="q", answer="a", agent_execution_trace=trace, image_source=None, chat_history=[])
    assert isinstance(buf, io.BytesIO) and buf.getbuffer().nbytes > 500
    assert trace == before
