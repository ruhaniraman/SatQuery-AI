"""Upload hardening, model-unavailable behaviour, and report retention."""
import io
import os
import sys
import time
import uuid

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from report_retention import purge_old_reports, retention_hours  # noqa: E402
from upload_validation import (  # noqa: E402
    UploadTooLarge,
    detect_image_kind,
    max_request_bytes,
    max_upload_bytes,
    read_limited,
)
from test_endpoint_wiring import client, png_bytes, post, tif_bytes  # noqa: E402,F401

RNG = np.random.default_rng(11)


def png():
    return png_bytes(np.full((64, 64, 3), 90, np.uint8))


def leftovers(client):
    return [f for f in os.listdir(client.tmp) if f.startswith("temp_")]


# ------------------------------------------------------------------ sniffing / limits (pure)

def encode(fmt):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt,kind", [("PNG", "png"), ("JPEG", "jpeg"), ("TIFF", "tiff"), ("WEBP", "webp")])
def test_real_files_are_recognised(fmt, kind):
    assert detect_image_kind(encode(fmt)) == kind


def test_geotiff_bytes_are_recognised(tmp_path):
    assert detect_image_kind(tif_bytes(tmp_path, "g.tif", np.zeros((3, 8, 8), np.float32))) == "tiff"


@pytest.mark.parametrize("data", [b"", b"hello world", b"%PDF-1.7 ...", b"<html></html>", b"RIFF\x00\x00\x00\x00WAVE", b"\x89PNG"])
def test_non_images_are_rejected(data):
    assert detect_image_kind(data) is None


def test_read_limited_boundaries():
    assert read_limited(io.BytesIO(b"x" * 100), 100) == b"x" * 100        # exactly at the limit is fine
    with pytest.raises(UploadTooLarge):
        read_limited(io.BytesIO(b"x" * 101), 100)


def test_limits_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    assert max_upload_bytes() == 5 * 1024 * 1024
    assert max_request_bytes() > 2 * max_upload_bytes()
    monkeypatch.setenv("MAX_UPLOAD_MB", "garbage")
    assert max_upload_bytes() == 100 * 1024 * 1024


# ------------------------------------------------------------------ endpoint hardening

def test_bytes_that_are_not_an_image_are_rejected_whatever_the_client_claims(client):
    r = post(client, [("images", ("a.png", b"this is not an image at all", "image/png"))])
    assert r.status_code == 415 and "not a valid" in r.json()["detail"]
    assert client.agent.calls == [] and leftovers(client) == []


def test_missing_filename_is_a_clean_client_error_not_a_crash(client):
    r = post(client, [("images", ("", png(), "image/png"))])
    assert 400 <= r.status_code < 500


def test_wrong_declared_type_or_extension_does_not_matter_if_bytes_are_valid(client):
    r = post(client, [("images", ("holiday.jpg", png(), "image/jpeg"))])          # a PNG called .jpg
    assert r.status_code == 200, r.text


def test_geotiff_with_a_misleading_name_is_still_read_as_geotiff(client):
    a = tif_bytes(client.tmp, "a.tif", RNG.uniform(500, 2500, (3, 96, 96)).astype(np.float32))
    r = post(client, [("images", ("first.png", a, "image/png")), ("images", ("second.jpg", a, "image/jpeg"))])
    assert r.status_code == 200, r.text
    assert "SpatialAlignment" in r.json()["agent_execution_trace"]["nodes_traversed"]


def test_mixed_geotiff_and_png_pair_is_rejected_up_front(client):
    a = tif_bytes(client.tmp, "a.tif", RNG.uniform(500, 2500, (3, 96, 96)).astype(np.float32))
    r = post(client, [("images", ("a.tif", a, "image/tiff")), ("images", ("b.png", png(), "image/png"))])
    assert r.status_code == 400 and "Cannot combine" in r.json()["detail"]
    assert leftovers(client) == [] and client.vlm_calls == []


def test_oversized_single_file_is_413(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")                                  # ~1 KB per file
    big = png_bytes(RNG.integers(0, 255, (64, 64, 3)).astype(np.uint8))           # incompressible, > 1 KB
    assert len(big) > 1100
    r = post(client, [("images", ("a.png", big, "image/png"))])
    assert r.status_code == 413 and "upload limit" in r.json()["detail"]
    assert leftovers(client) == []


def test_oversized_request_is_rejected_by_content_length_before_reading(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    r = client.post("/analyze", data={"query": "q"}, files=[("images", ("a.png", b"\x89PNG" + b"0" * 2_200_000, "image/png"))])
    assert r.status_code == 413 and r.json()["detail"] == "Request body too large."


def test_normal_sized_upload_still_works_with_default_limits(client):
    assert post(client, [("images", ("a.png", png(), "image/png"))]).status_code == 200


# ------------------------------------------------------------------ model availability

def test_analyze_answers_503_with_the_reason_when_the_model_is_missing(client, monkeypatch):
    import main_api
    monkeypatch.setattr(main_api, "get_agent", lambda: None)
    main_api.app.state.model_error = "FileNotFoundError: adapters/mining"
    r = post(client, [("images", ("a.png", png(), "image/png"))])
    assert r.status_code == 503 and "adapters/mining" in r.json()["detail"]
    assert leftovers(client) == []


def test_app_starts_and_reports_503_when_the_model_fails_to_load(client, monkeypatch):
    import main_api
    from fastapi.testclient import TestClient

    def boom():
        raise RuntimeError("no adapters found")

    monkeypatch.setattr(main_api, "get_agent", lambda: None)
    monkeypatch.setattr(main_api, "load_agent", boom)
    with TestClient(main_api.app) as c:                       # runs the real lifespan; must NOT raise
        r = c.post("/analyze", data={"query": "q"}, files=[("images", ("a.png", png(), "image/png"))])
        assert r.status_code == 503 and "RuntimeError: no adapters found" in r.json()["detail"]
        assert c.get("/docs").status_code == 200              # the rest of the API is still up


def test_lifespan_loads_the_model_when_absent(client, monkeypatch):
    import main_api
    from fastapi.testclient import TestClient
    loaded = []
    monkeypatch.setattr(main_api, "get_agent", lambda: None if not loaded else client.agent)
    monkeypatch.setattr(main_api, "load_agent", lambda: loaded.append(1))
    with TestClient(main_api.app):
        pass
    assert loaded == [1] and main_api.app.state.model_error is None


# ------------------------------------------------------------------ report retention

def make_session(root, name=None, age_hours=0.0, files=("evidence.png", "report.pdf")):
    name = name or str(uuid.uuid4())
    d = os.path.join(root, name)
    os.makedirs(d)
    for f in files:
        with open(os.path.join(d, f), "wb") as fh:
            fh.write(b"x")
    t = time.time() - age_hours * 3600
    for f in files:
        os.utime(os.path.join(d, f), (t, t))
    os.utime(d, (t, t))
    return name


def test_old_sessions_are_deleted_and_recent_ones_kept(tmp_path):
    old = make_session(tmp_path, age_hours=100)
    fresh = make_session(tmp_path, age_hours=1)
    assert purge_old_reports(str(tmp_path), 48) == [old]
    assert os.listdir(tmp_path) == [fresh]


def test_a_session_with_a_recently_written_file_is_kept(tmp_path):
    name = make_session(tmp_path, age_hours=100)
    fresh_file = os.path.join(tmp_path, name, "report.pdf")
    os.utime(fresh_file, None)                                # touched now
    assert purge_old_reports(str(tmp_path), 48) == []


def test_only_session_shaped_folders_are_ever_touched(tmp_path):
    keep = [make_session(tmp_path, name="important-notes", age_hours=1000),
            make_session(tmp_path, name="123", age_hours=1000)]
    with open(tmp_path / "stray.txt", "w") as f:
        f.write("stay")
    os.utime(tmp_path / "stray.txt", (0, 0))
    assert purge_old_reports(str(tmp_path), 1) == []
    assert sorted(os.listdir(tmp_path)) == sorted(keep + ["stray.txt"])


def test_retention_is_off_unless_configured(tmp_path, monkeypatch):
    make_session(tmp_path, age_hours=10_000)
    monkeypatch.delenv("REPORT_RETENTION_HOURS", raising=False)
    assert retention_hours() == 0
    assert purge_old_reports(str(tmp_path), retention_hours()) == [] and len(os.listdir(tmp_path)) == 1
    monkeypatch.setenv("REPORT_RETENTION_HOURS", "not a number")
    assert retention_hours() == 0
    monkeypatch.setenv("REPORT_RETENTION_HOURS", "24")
    assert retention_hours() == 24


def test_missing_root_is_harmless(tmp_path):
    assert purge_old_reports(str(tmp_path / "nope"), 1) == []


# ------------------------------------------------------------------ chat/PDF consistency

def test_answer_text_is_identical_in_response_history_and_saved_record(client):
    import json
    client.agent.query = lambda **kw: "**Water** covers *most* of the scene."
    r = post(client, [("images", ("a.png", png(), "image/png"))], query="q",
             chat_history=json.dumps([{"role": "user", "content": "q"}]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Water covers most of the scene."
    tr = body["agent_execution_trace"]
    with open(os.path.join(client.tmp, "reports", tr["pipeline_id"], "data.json")) as f:
        saved = json.load(f)
    assert saved["answer"] == body["answer"]
    assert saved["chat_history"][-1] == {"role": "ai", "content": body["answer"]}      # what the PDF prints
    assert "*" not in json.dumps(saved)


def test_pdf_labels_system_entries_as_system_not_user():
    pypdf = pytest.importorskip("pypdf")
    from pdf_report_generator import generate_pdf_report
    buf = generate_pdf_report(
        query="q", answer="a", agent_execution_trace={"pipeline_id": "x"}, image_source=None,
        chat_history=[{"role": "system", "content": "Initiating mining scan..."},
                      {"role": "ai", "content": "Scan complete."}, {"role": "user", "content": "thanks"}])
    text = " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(buf.getvalue())).pages)
    assert "SYSTEM: Initiating mining scan" in text and "USER: Initiating" not in text
    assert "USER: thanks" in text
