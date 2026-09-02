"""
main_api.py
============
Member 6 (part 1): FastAPI backend wiring Stages 1-4 together.

    Stage 1 (Member 4): GeoImageLoader           -> normalized 8-bit arrays
    Stage 2 (Member 1): AgentController           -> task routing + trace
    Stage 3 (Member 2/3/5): specialist engines     -> answer + visual evidence
    Stage 4 (Member 6): this file + pdf_report_generator -> API response + PDF

POST /analyze accepts up to 2 images + a text query, runs the full pipeline,
and returns a structured JSON response plus a downloadable PDF audit report.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent_controller import AgentController, TaskIntent
from llm_classifier import LLMIntentClassifier
from cdvqa_engine import ChangeDetectionEngine
from single_image_vqa_engine import VQAGroundingEngine
from optical_sar_fusion_model import FusionEngine
from pdf_report_generator import generate_pdf_report

app = FastAPI(title="SATQUERY AI Backend", version="0.1.0")

OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared engine instances (loaded once, reused across requests).
# LLMIntentClassifier auto-upgrades routing to a real Claude call when
# ANTHROPIC_API_KEY is set; otherwise it transparently falls back to the
# same keyword classifier agent_controller.py ships with, so this never
# blocks anyone without a key.
_classifier = LLMIntentClassifier()
agent = AgentController(classifier_fn=_classifier.classify)
vqa_engine = VQAGroundingEngine()
change_engine = ChangeDetectionEngine()
fusion_engine = FusionEngine()


class AnalyzeResponse(BaseModel):
    answer: str
    confidence_score: float
    agent_execution_trace: dict
    visual_evidence_url: str


def _read_upload_to_array(upload_bytes: bytes) -> np.ndarray:
    """Standardizes an uploaded file's bytes to an 8-bit RGB numpy array."""
    arr = np.frombuffer(upload_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    query: str = Form(...),
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
):
    num_images = 1 if image2 is None else 2

    # --- Stage 2: route the query ---
    trace = agent.route(query=query, num_images_provided=num_images)

    if trace.validation_error is not None:
        raise HTTPException(status_code=422, detail=trace.validation_error.model_dump())

    # --- Standardize inputs (Stage 1 contract: 8-bit RGB arrays) ---
    img1_bytes = await image1.read()
    array1 = _read_upload_to_array(img1_bytes)
    array2 = None
    if image2 is not None:
        img2_bytes = await image2.read()
        array2 = _read_upload_to_array(img2_bytes)

    # --- Stage 3: dispatch to the specialist engine ---
    evidence_image = array1
    confidence_score = 0.75  # placeholder; wire to real model confidence when available

    if trace.task_selected == TaskIntent.SINGLE_IMAGE_VQA:
        answer = vqa_engine.answer_question(array1, query)
        evidence_image = array1

    elif trace.task_selected == TaskIntent.GROUNDING:
        result = vqa_engine.ground_object(array1, query)
        answer = f"Located {len(result.boxes)} region(s) matching '{query}'."
        evidence_image = result.image_with_boxes
        confidence_score = max((b.confidence for b in result.boxes), default=0.0)

    elif trace.task_selected == TaskIntent.CHANGE_DETECTION:
        cd_result = change_engine.detect(array1, array2, date_a="Date 1", date_b="Date 2")
        answer = cd_result.explanation
        evidence_image = cd_result.overlay_image
        confidence_score = cd_result.ssim_score

    elif trace.task_selected == TaskIntent.OPTICAL_SAR_FUSION:
        sar_gray = cv2.cvtColor(array2, cv2.COLOR_RGB2GRAY)
        fusion_result = fusion_engine.fuse(array1, sar_gray, user_query=query)
        answer = fusion_result.context_prompt
        evidence_image = fusion_result.composite_image

    else:
        raise HTTPException(status_code=500, detail="Unrecognized task intent.")

    # --- Stage 4: generate the PDF audit report ---
    report_id = str(uuid.uuid4())[:8]
    pdf_path = os.path.join(OUTPUT_DIR, f"report_{report_id}.pdf")
    generate_pdf_report(
        output_path=pdf_path,
        query=query,
        answer=answer,
        execution_trace=trace.model_dump(),
        evidence_image=evidence_image,
        confidence_score=confidence_score,
    )

    return AnalyzeResponse(
        answer=answer,
        confidence_score=confidence_score,
        agent_execution_trace=trace.model_dump(),
        visual_evidence_url=f"/reports/{report_id}",
    )


@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    pdf_path = os.path.join(OUTPUT_DIR, f"report_{report_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"satquery_report_{report_id}.pdf")


@app.get("/health")
async def health():
    return {"status": "ok"}