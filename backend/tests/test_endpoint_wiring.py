"""Drives the real main_api.py through FastAPI's TestClient with the model stubbed out.

No GPU / weights here: agent_manager.agent_controller and torch are replaced before main_api is
imported, and the VLM bridge is monkeypatched to record what it was given.
"""
import contextlib
import os
import sys
import threading
import types

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

RNG = np.random.default_rng(1)


class FakeAgent:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []

    def query(self, **kw):
        self.calls.append(kw)
        return "A plain answer."

    # P(yes) grid a feature scan gets back; a test can replace it
    scan_grid = np.array([[0.9, 0.2, 0.1, 0.1], [0.9, 0.2, 0.1, 0.1], [0.3, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])

    def scan_scores(self, img_array, adapter, prompt=None):
        self.calls.append({"scan": adapter, "prompt": prompt})
        return self.scan_grid


@pytest.fixture()
def client(tmp_path, monkeypatch):
    fake_agent = FakeAgent()
    fake_mod = types.ModuleType("agent_manager.agent_controller")
    fake_mod.get_agent = lambda: fake_agent
    fake_mod.load_agent = lambda: fake_agent
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = contextlib.contextmanager(lambda: (yield))

    for name in ("main_api", "agent_manager.agent_controller", "torch"):
        sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "agent_manager.agent_controller", fake_mod)
    monkeypatch.chdir(tmp_path)                            # reports/ and temp_* land here
    monkeypatch.setenv("DESCRIBE_FIRST", "0")              # tests that want the extra caption calls turn it on
    monkeypatch.setenv("CHANGE_EVIDENCE_REPORT", "0")      # ... and the evidence-built comparison likewise

    import main_api
    real_vlm = main_api.shared_vram_caller           # kept so a test can exercise the real bridge
    seen = []

    def fake_vlm(prompt, img_a, img_b=None, img_c=None, system=None):
        seen.append((prompt, img_a, system))
        return "VLM says things changed."

    monkeypatch.setattr(main_api, "shared_vram_caller", fake_vlm)

    from fastapi.testclient import TestClient
    c = TestClient(main_api.app)
    c.agent, c.vlm_calls, c.tmp, c.real_vlm = fake_agent, seen, tmp_path, real_vlm
    yield c
    sys.modules.pop("main_api", None)


def tif_bytes(tmp_path, name, data, x0=500000, nodata=None):
    p = str(tmp_path / name)
    data = data[None] if data.ndim == 2 else data
    with rasterio.open(p, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=data.dtype, crs="EPSG:32643",
                       transform=from_origin(x0, 1500000, 10, 10), nodata=nodata) as dst:
        dst.write(data)
    return open(p, "rb").read()


def png_bytes(arr):
    return cv2.imencode(".png", arr)[1].tobytes()


def speckled(v, shape=(96, 96)):
    return (v * RNG.gamma(4, 0.25, shape)).astype(np.float32)


def post(client, files, **form):
    form.setdefault("query", "what changed?")
    return client.post("/analyze", data=form, files=files)


def leftovers(client):
    return [f for f in os.listdir(client.tmp) if f.startswith("temp_")]


def test_single_image_passes_modality_to_agent(client):
    r = post(client, [("images", ("a.png", png_bytes(np.full((64, 64, 3), 90, np.uint8)), "image/png"))],
             modality_a="sar")
    assert r.status_code == 200, r.text
    assert client.agent.calls[0]["modality"] == "sar"


def test_change_detection_geotiff_sar_pair(client):
    a = tif_bytes(client.tmp, "a.tif", speckled(0.2))
    b = tif_bytes(client.tmp, "b.tif", speckled(0.2))
    r = post(client, [("images", ("a.tif", a, "image/tiff")), ("images", ("b.tif", b, "image/tiff"))],
             modality_a="sar", modality_b="sar")
    assert r.status_code == 200, r.text
    assert r.json()["agent_execution_trace"]["telemetry"]["major_regions_detected"] == 0
    stitched = client.vlm_calls[0][1]
    assert stitched.dtype == np.uint8 and stitched.shape == (96, 192, 3)
    assert leftovers(client) == []


def test_change_detection_geotiff_optical_with_nodata_strip(client):
    base = RNG.uniform(500, 2500, (3, 96, 96)).astype(np.float32)
    a = tif_bytes(client.tmp, "a.tif", base, nodata=-9999)
    b = tif_bytes(client.tmp, "b.tif", base.copy(), nodata=-9999)
    r = post(client, [("images", ("a.tif", a, "image/tiff")), ("images", ("b.tif", b, "image/tiff"))])
    assert r.status_code == 200, r.text
    assert leftovers(client) == []


def test_change_detection_standard_sar_pair_is_despeckled(client):
    noisy = np.clip(speckled(100.0, (96, 96)), 0, 255).astype(np.uint8)
    img = np.dstack([noisy] * 3)
    r = post(client, [("images", ("a.png", png_bytes(img), "image/png")),
                      ("images", ("b.png", png_bytes(img), "image/png"))],
             modality_a="sar", modality_b="sar")
    assert r.status_code == 200, r.text
    left = client.vlm_calls[0][1][:, :96]
    assert left[:, :, 0].std() < noisy.std()              # Lee filter ran in the loader


def test_change_detection_geotiff_non_overlap_still_400(client):
    a = tif_bytes(client.tmp, "a.tif", speckled(0.2))
    b = tif_bytes(client.tmp, "b.tif", speckled(0.2), x0=900000)
    r = post(client, [("images", ("a.tif", a, "image/tiff")), ("images", ("b.tif", b, "image/tiff"))],
             modality_a="sar", modality_b="sar")
    assert r.status_code == 400, r.text
    assert leftovers(client) == []


def test_fusion_geotiff_optical_plus_sar(client):
    opt = tif_bytes(client.tmp, "o.tif", RNG.uniform(500, 2500, (3, 96, 96)).astype(np.float32))
    sar = tif_bytes(client.tmp, "s.tif", speckled(0.2))
    r = post(client, [("images", ("o.tif", opt, "image/tiff")), ("images", ("s.tif", sar, "image/tiff"))],
             modality_a="optical", modality_b="sar")
    assert r.status_code == 200, r.text
    assert client.vlm_calls[0][1].dtype == np.uint8
    assert leftovers(client) == []


def test_fusion_standard_images_sar_first(client):
    opt = png_bytes(RNG.integers(0, 255, (64, 64, 3)).astype(np.uint8))
    sar = png_bytes(np.dstack([np.clip(speckled(100.0, (64, 64)), 0, 255).astype(np.uint8)] * 3))
    r = post(client, [("images", ("s.png", sar, "image/png")), ("images", ("o.png", opt, "image/png"))],
             modality_a="sar", modality_b="optical")
    assert r.status_code == 200, r.text
    assert client.vlm_calls[0][1].shape == (64, 64, 3)


def test_change_detection_geotiff_shifted_grid_has_no_false_change(client):
    """B covers the same ground shifted 100px, so aligning it leaves a nodata strip on A's left.
    Nothing actually changed, so that strip must not be boxed as a change."""
    xs = np.arange(500)
    field = (1500 - 400 * np.sin(xs / 37.0))[None, :] + 300 * np.sin(np.arange(400) / 23.0)[:, None]
    field = np.stack([field, field * 0.9, field * 1.1]).astype(np.float32)
    a = tif_bytes(client.tmp, "a.tif", field[:, :, 0:400])
    b = tif_bytes(client.tmp, "b.tif", field[:, :, 100:500], x0=500000 + 1000)
    r = post(client, [("images", ("a.tif", a, "image/tiff")), ("images", ("b.tif", b, "image/tiff"))])
    assert r.status_code == 200, r.text
    assert r.json()["agent_execution_trace"]["telemetry"]["major_regions_detected"] == 0


def test_pdf_failure_reports_error_and_null_url(client, monkeypatch):
    import main_api

    def boom(**kw):
        raise RuntimeError("no pdf for you")
    monkeypatch.setattr(main_api, "generate_pdf_report", boom)
    r = post(client, [("images", ("a.png", png_bytes(np.full((64, 64, 3), 90, np.uint8)), "image/png"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report_download_url"] is None and "RuntimeError" in body["report_error"]


def test_pdf_success_has_no_error(client):
    r = post(client, [("images", ("a.png", png_bytes(np.full((64, 64, 3), 90, np.uint8)), "image/png"))])
    assert r.json()["report_download_url"] and r.json()["report_error"] is None


def test_chat_history_reaches_the_agent(client):
    import json
    hist = [{"role": "user", "content": "what is here?"}, {"role": "ai", "content": "A river."},
            {"role": "user", "content": "and north?"}]
    r = post(client, [("images", ("a.png", png_bytes(np.full((64, 64, 3), 90, np.uint8)), "image/png"))],
             query="and north?", chat_history=json.dumps(hist))
    assert r.status_code == 200, r.text
    assert client.agent.calls[0]["chat_history"][:2] == hist[:2]


def test_fusion_trace_and_system_prompt_are_honest(client):
    from fusion.sar_optical_fusion import FUSION_METHOD
    opt = tif_bytes(client.tmp, "o.tif", RNG.uniform(500, 2500, (3, 96, 96)).astype(np.float32))
    sar = tif_bytes(client.tmp, "s.tif", speckled(0.2))
    r = post(client, [("images", ("o.tif", opt, "image/tiff")), ("images", ("s.tif", sar, "image/tiff"))],
             modality_a="optical", modality_b="sar")
    assert r.status_code == 200, r.text
    trace = r.json()["agent_execution_trace"]
    assert trace["telemetry"]["fusion_mode"] == FUSION_METHOD
    assert "False-Color" not in str(trace)
    assert trace["telemetry"]["sar_despeckled"] and trace["telemetry"]["sar_input_units"] == "linear"
    assert "SpatialAlignment" in trace["nodes_traversed"]
    prompt, _, system = client.vlm_calls[0]
    assert prompt == "what changed?" and "backscatter" in system and "[SYSTEM]" not in prompt


def test_fusion_of_plain_images_does_not_claim_spatial_alignment(client):
    opt = png_bytes(RNG.integers(0, 255, (64, 64, 3)).astype(np.uint8))
    sar = png_bytes(np.dstack([np.clip(speckled(100.0, (64, 64)), 0, 255).astype(np.uint8)] * 3))
    r = post(client, [("images", ("o.png", opt, "image/png")), ("images", ("s.png", sar, "image/png"))],
             modality_a="optical", modality_b="sar")
    assert r.status_code == 200, r.text
    assert "SpatialAlignment" not in r.json()["agent_execution_trace"]["nodes_traversed"]


def test_fusion_with_partial_sar_footprint_keeps_optical_brightness_in_gap(client):
    """SAR covers only part of the optical scene, so aligning leaves a nodata strip."""
    xs = np.arange(500)
    field = (1500 - 400 * np.sin(xs / 37.0))[None, :] + 0 * np.arange(400)[:, None]
    opt = tif_bytes(client.tmp, "o.tif", np.stack([field[:, 0:400]] * 3).astype(np.float32))
    sar = tif_bytes(client.tmp, "s.tif", speckled(0.2, (400, 400)), x0=500000 + 1000)
    r = post(client, [("images", ("o.tif", opt, "image/tiff")), ("images", ("s.tif", sar, "image/tiff"))],
             modality_a="optical", modality_b="sar")
    assert r.status_code == 200, r.text
    composite = client.vlm_calls[0][1]
    strip = composite[:, :90]                               # left 100px of optical has no SAR
    assert strip.mean() > 20                                # not painted black
