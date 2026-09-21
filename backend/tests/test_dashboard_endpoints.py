"""/health, /preview (TIFF -> browser-safe PNG) and /report (PDF rebuilt with the full conversation)."""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_endpoint_wiring import client, png_bytes, post, speckled, tif_bytes  # noqa: E402,F401

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def decode(png):
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def preview(client, name, content, mime, **form):
    return client.post("/preview", files={"image": (name, content, mime)}, data=form)


# ------------------------------------------------------------------ /health

def test_health_says_whether_the_model_is_loaded(client, monkeypatch):
    import main_api
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_error": None}
    monkeypatch.setattr(main_api, "get_agent", lambda: None)
    main_api.app.state.model_error = "RuntimeError: no weights"
    body = client.get("/health").json()
    assert body["model_loaded"] is False and body["model_error"] == "RuntimeError: no weights"


# ------------------------------------------------------------------ /preview

def test_a_geotiff_becomes_a_png_the_browser_can_draw(client):
    data = np.random.default_rng(3).uniform(500, 2500, (3, 96, 128)).astype(np.float32)
    r = preview(client, "scene.tif", tif_bytes(client.tmp, "scene.tif", data), "image/tiff")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content.startswith(PNG_MAGIC) and r.headers["cache-control"] == "no-store"
    assert decode(r.content).shape == (96, 128, 3)                        # small scenes are not resized
    assert os.listdir(client.tmp) == [f for f in os.listdir(client.tmp) if not f.startswith("temp_")]  # no temp leak


def test_a_large_scene_is_scaled_down_to_the_preview_limit(client):
    data = np.random.default_rng(4).integers(0, 255, (1, 1200, 3000), dtype=np.uint8)
    r = preview(client, "big.tif", tif_bytes(client.tmp, "big.tif", data), "image/tiff")
    assert r.status_code == 200
    height, width = decode(r.content).shape[:2]
    assert width == 2048 and height == round(1200 * 2048 / 3000)


def test_sar_previews_are_despeckled_like_an_analysis_run(client):
    tiff = tif_bytes(client.tmp, "sar.tif", speckled(0.2, (128, 128)))
    optical = decode(preview(client, "sar.tif", tiff, "image/tiff", modality="optical").content)
    sar = decode(preview(client, "sar.tif", tiff, "image/tiff", modality="sar").content)
    assert sar.std() < optical.std()                                      # the Lee filter ran for SAR


def test_ordinary_images_are_returned_as_png_too(client):
    jpg = cv2.imencode(".jpg", np.full((40, 50, 3), 120, np.uint8))[1].tobytes()
    r = preview(client, "a.jpg", jpg, "image/jpeg")
    assert r.status_code == 200 and decode(r.content).shape == (40, 50, 3)


def test_preview_rejects_bad_input(client, monkeypatch):
    assert preview(client, "x.tif", b"not an image at all", "image/tiff").status_code == 415
    assert preview(client, "a.png", png_bytes(np.zeros((8, 8, 3), np.uint8)), "image/png", modality="radar").status_code == 400
    truncated = tif_bytes(client.tmp, "t.tif", np.zeros((1, 32, 32), np.uint8))[:60]     # TIFF header, no image data
    assert preview(client, "t.tif", truncated, "image/tiff").status_code == 400
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")                            # about 1 KB
    noisy = png_bytes(np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8))   # noise does not compress
    assert len(noisy) > 1100 and preview(client, "a.png", noisy, "image/png").status_code == 413


# ------------------------------------------------------------------ /report

def run_id(response):
    return re.search(r"/reports/([0-9a-f-]{36})/", response.json()["visual_evidence_url"]).group(1)


def one_run(client):
    r = post(client, [("images", ("a.png", png_bytes(np.full((64, 64, 3), 90, np.uint8)), "image/png"))], query="what is here?")
    assert r.status_code == 200, r.text
    return run_id(r)


def pdf_text(client, url):
    from io import BytesIO
    from pypdf import PdfReader
    path = os.path.join(client.tmp, url.lstrip("/"))
    return "\n".join(page.extract_text() for page in PdfReader(BytesIO(open(path, "rb").read())).pages)


def test_a_rebuilt_report_carries_the_full_current_conversation(client):
    session = one_run(client)
    history = [{"role": "user", "content": "first question"}, {"role": "ai", "content": "first answer"},
               {"role": "user", "content": "a LATER question"}, {"role": "ai", "content": "a LATER answer"}]
    r = client.post("/report", data={"session_id": session, "chat_history": json.dumps(history)})
    assert r.status_code == 200 and r.json()["report_error"] is None
    assert r.json()["report_download_url"] == f"/reports/{session}/report_chat.pdf"
    text = pdf_text(client, r.json()["report_download_url"])
    assert "a LATER question" in text and "a LATER answer" in text
    assert "Built on demand" in text and session in text.replace("\n", "")     # says whose evidence/trace this is


def test_without_a_history_the_runs_own_conversation_is_used(client):
    session = one_run(client)
    r = client.post("/report", data={"session_id": session})
    assert r.status_code == 200 and "A plain answer." in pdf_text(client, r.json()["report_download_url"])


def test_report_rejects_bad_ids_unknown_runs_and_broken_history(client):
    session = one_run(client)
    for bad in ("../etc/passwd", "not-a-uuid", "", session.upper(), session + "x", "0" * 8 + "-0000-1000-8000-" + "0" * 12):
        assert client.post("/report", data={"session_id": bad}).status_code in (400, 404, 422), bad
    assert client.post("/report", data={"session_id": "12345678-1234-4234-8234-123456789012"}).status_code == 404
    assert client.post("/report", data={"session_id": session, "chat_history": "{oops"}).status_code == 400
    assert client.post("/report", data={"session_id": session, "chat_history": json.dumps({"not": "a list"})}).status_code == 200
