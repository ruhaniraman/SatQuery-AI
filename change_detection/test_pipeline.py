"""
Formal pytest suite for the SATQUERY AI end-to-end pipeline.
Run with: pytest tests/ -v
"""
import io
import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


def make_image_bytes(fill_val, shape=(200, 200, 3), draw_box=False):
    img = np.full(shape, fill_val, dtype=np.uint8)
    if draw_box:
        cv2.rectangle(img, (60, 60), (140, 140), (200, 30, 30), -1)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return io.BytesIO(buf.tobytes())


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_single_image_vqa():
    resp = client.post(
        "/analyze",
        data={"query": "What is visible in this image?"},
        files={"image1": ("a.png", make_image_bytes(120), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_execution_trace"]["task_selected"] == "SINGLE_IMAGE_VQA"
    assert "answer" in body and len(body["answer"]) > 0


def test_grounding():
    resp = client.post(
        "/analyze",
        data={"query": "Locate the red structure"},
        files={"image1": ("a.png", make_image_bytes(120, draw_box=True), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["agent_execution_trace"]["task_selected"] == "GROUNDING"


def test_change_detection_two_images():
    resp = client.post(
        "/analyze",
        data={"query": "What changed between these two dates?"},
        files={
            "image1": ("a.png", make_image_bytes(120), "image/png"),
            "image2": ("b.png", make_image_bytes(120, draw_box=True), "image/png"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_execution_trace"]["task_selected"] == "CHANGE_DETECTION"
    assert 0.0 <= body["confidence_score"] <= 1.0


def test_change_detection_missing_second_image_returns_422():
    resp = client.post(
        "/analyze",
        data={"query": "What changed between these two dates?"},
        files={"image1": ("a.png", make_image_bytes(120), "image/png")},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["required_images"] == 2
    assert detail["images_provided"] == 1


def test_optical_sar_fusion():
    resp = client.post(
        "/analyze",
        data={"query": "Where are the man-made structures based on radar structure?"},
        files={
            "image1": ("optical.png", make_image_bytes(100), "image/png"),
            "image2": ("sar.png", make_image_bytes(180), "image/png"),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["agent_execution_trace"]["task_selected"] == "OPTICAL_SAR_FUSION"


def test_pdf_report_is_downloadable():
    resp = client.post(
        "/analyze",
        data={"query": "What is in this image?"},
        files={"image1": ("a.png", make_image_bytes(120), "image/png")},
    )
    report_url = resp.json()["visual_evidence_url"]
    pdf_resp = client.get(report_url)
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert len(pdf_resp.content) > 500
