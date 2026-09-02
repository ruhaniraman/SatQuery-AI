import io
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from pdf_report_generator import generate_pdf_report

app = FastAPI(
    title="Agentic AI Backend & Reporting Service",
    version="1.0.0",
    description="FastAPI service for multi-modal agentic analysis and PDF report generation",
)

# In-memory session cache for demonstration (in production, use Redis or S3)
AUDIT_STORE: Dict[str, Dict[str, Any]] = {}


# --- Schema Definitions ---
class AnalysisResponse(BaseModel):
    answer: str = Field(..., description="Synthesized analytical response from the AI agents")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence rating between 0 and 1")
    agent_execution_trace: Dict[str, Any] = Field(..., description="Telemetry and reasoning steps of individual sub-agents")
    visual_evidence_url: str = Field(..., description="Endpoint or URL to retrieve the visual inspection artifact")
    report_download_url: Optional[str] = Field(None, description="Direct URL to download the single-page audit PDF")


# --- Analysis Endpoint ---
@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit query and up to 2 images for agentic analysis",
)
async def analyze(
    query: str = Form(..., description="Analytical query/prompt for the agent system"),
    images: List[UploadFile] = File(default=[], description="Up to 2 images (e.g. satellite, drone, or document scans)"),
):
    # Validation: Maximum of 2 image files
    if len(images) > 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum 2 images permitted. Received {len(images)}.",
        )

    # Validate image MIME types if files are provided
    processed_images_bytes: List[bytes] = []
    for img in images:
        if img.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format: {img.content_type}. Use JPEG, PNG, or WebP.",
            )
        content = await img.read()
        processed_images_bytes.append(content)

    # --- Simulated Agent Execution Pipeline ---
    session_id = str(uuid.uuid4())
    
    agent_trace = {
        "pipeline_id": session_id,
        "nodes_traversed": ["InputPreprocessor", "VisionExtractorNode", "ReasoningEngine", "VerifierNode"],
        "telemetry": {
            "preprocessor_ms": 42.1,
            "vision_model": "yolo-v11-aerial-finetuned",
            "detected_targets": 3,
            "bounding_boxes": [
                {"box": [120, 85, 310, 240], "class": "excavator", "confidence": 0.94},
                {"box": [400, 150, 520, 310], "class": "haul_truck", "confidence": 0.91},
                {"box": [60, 310, 180, 420], "class": "stockpile", "confidence": 0.88},
            ],
            "reasoning_latency_ms": 312.4,
            "verifier_passed": True,
        },
    }

    synthesized_answer = (
        f"Analysis complete for query: '{query}'. Identified 3 industrial activity targets across "
        f"{len(images)} provided inspection frame(s). Bounding coordinates confirmed high-density extraction activity "
        f"with strict boundary compliance verified by the sub-agent audit chain."
    )
    confidence = 0.93
    visual_evidence_url = f"/reports/{session_id}/evidence.png"
    report_url = f"/reports/{session_id}/download"

    # Persist session data to allow instant PDF download
    primary_image_bytes = processed_images_bytes[0] if processed_images_bytes else None
    AUDIT_STORE[session_id] = {
        "query": query,
        "answer": synthesized_answer,
        "confidence_score": confidence,
        "agent_execution_trace": agent_trace,
        "image_bytes": primary_image_bytes,
    }

    return AnalysisResponse(
        answer=synthesized_answer,
        confidence_score=confidence,
        agent_execution_trace=agent_trace,
        visual_evidence_url=visual_evidence_url,
        report_download_url=report_url,
    )


# --- PDF Report Download Endpoint ---
@app.get(
    "/reports/{session_id}/download",
    summary="Download the 1-page PDF audit report for a given analysis session",
)
async def download_report(session_id: str):
    record = AUDIT_STORE.get(session_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report session expired or does not exist.",
        )

    # Convert stored image bytes to an in-memory buffer if present
    img_buffer = io.BytesIO(record["image_bytes"]) if record["image_bytes"] else None

    # Generate the single-page PDF
    pdf_buffer = generate_pdf_report(
        query=record["query"],
        answer=record["answer"],
        confidence_score=record["confidence_score"],
        agent_execution_trace=record["agent_execution_trace"],
        image_source=img_buffer,
    )

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=audit_report_{session_id[:8]}.pdf"},
    )