import io
import os
import uuid
import shutil
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from pdf_report_generator import generate_pdf_report
# Import your newly created local AI engine
from single_image.single_image_vqa_engine import SingleImageSpecialist

app = FastAPI(
    title="SatQuery AI Backend & Reporting Service",
    version="1.0.0",
    description="FastAPI service for multi-modal agentic analysis and PDF report generation",
)

# In-memory session cache for demonstration
AUDIT_STORE: Dict[str, Dict[str, Any]] = {}

# --- Initialize the AI Engine ---
# This loads into your 6GB VRAM on startup so it doesn't have to reload for every request
print("Booting up the Single-Image Specialist...")
engine = SingleImageSpecialist(adapter_path="./models")


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
    if len(images) > 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum 2 images permitted. Received {len(images)}.",
        )
    
    if len(images) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 image must be provided for analysis.",
        )

    processed_images_bytes: List[bytes] = []
    for img in images:
        if img.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format: {img.content_type}. Use JPEG, PNG, or WebP.",
            )
        content = await img.read()
        processed_images_bytes.append(content)

    session_id = str(uuid.uuid4())
    temp_image_path = f"temp_{session_id}.jpg"
    
    try:
        # 1. Save the primary image temporarily to disk for the engine to read
        with open(temp_image_path, "wb") as f:
            f.write(processed_images_bytes[0])
            
        # 2. RUN REAL INFERENCE: Pass the image and query to your fine-tuned model
        ai_answer = engine.analyze_image(temp_image_path, query)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")
        
    finally:
        # 3. Clean up the temporary file immediately
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    # --- Simulated Agent Execution Pipeline ---
    # We will replace this with real orchestrator logic later
    agent_trace = {
        "pipeline_id": session_id,
        "nodes_traversed": ["InputPreprocessor", "SingleImageSpecialist", "VerifierNode"],
        "telemetry": {
            "model_used": "Qwen2-VL-2B-BigEarthNet-LoRA",
            "verifier_passed": True,
        },
    }

    confidence = 0.93 # Placeholder until we implement confidence scoring
    visual_evidence_url = f"/reports/{session_id}/evidence.png"
    report_url = f"/reports/{session_id}/download"

    # Persist session data to allow instant PDF download
    AUDIT_STORE[session_id] = {
        "query": query,
        "answer": ai_answer, # Using the real AI output here!
        "confidence_score": confidence,
        "agent_execution_trace": agent_trace,
        "image_bytes": processed_images_bytes[0],
    }

    return AnalysisResponse(
        answer=ai_answer,
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

    img_buffer = io.BytesIO(record["image_bytes"]) if record["image_bytes"] else None

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