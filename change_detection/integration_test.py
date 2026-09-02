"""
End-to-end smoke test: spins up main_api.py in-process (TestClient) and
fires requests through the full Stage 1->4 pipeline using synthetic images,
covering SINGLE_IMAGE_VQA, GROUNDING, and CHANGE_DETECTION.
"""
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)

def make_image_bytes(fill_val, shape=(200, 200, 3), draw_box=False):
    img = np.full(shape, fill_val, dtype=np.uint8)
    if draw_box:
        cv2.rectangle(img, (60, 60), (140, 140), (200, 30, 30), -1)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return io.BytesIO(buf.tobytes())

print("=== TEST 1: SINGLE_IMAGE_VQA ===")
resp = client.post(
    "/analyze",
    data={"query": "What is visible in this image?"},
    files={"image1": ("a.png", make_image_bytes(120), "image/png")},
)
print(resp.status_code, resp.json())

print("\n=== TEST 2: GROUNDING ===")
resp = client.post(
    "/analyze",
    data={"query": "Locate the red structure"},
    files={"image1": ("a.png", make_image_bytes(120, draw_box=True), "image/png")},
)
print(resp.status_code, resp.json())

print("\n=== TEST 3: CHANGE_DETECTION (2 images) ===")
resp = client.post(
    "/analyze",
    data={"query": "What changed between these two dates?"},
    files={
        "image1": ("a.png", make_image_bytes(120), "image/png"),
        "image2": ("b.png", make_image_bytes(120, draw_box=True), "image/png"),
    },
)
print(resp.status_code, resp.json())

print("\n=== TEST 4: CHANGE_DETECTION missing 2nd image (should 422) ===")
resp = client.post(
    "/analyze",
    data={"query": "What changed between these two dates?"},
    files={"image1": ("a.png", make_image_bytes(120), "image/png")},
)
print(resp.status_code, resp.json())

print("\n=== TEST 5: Fetch generated PDF report ===")
resp = client.post(
    "/analyze",
    data={"query": "What is in this image?"},
    files={"image1": ("a.png", make_image_bytes(120), "image/png")},
)
report_url = resp.json()["visual_evidence_url"]
pdf_resp = client.get(report_url)
print("PDF fetch status:", pdf_resp.status_code, "bytes:", len(pdf_resp.content))
