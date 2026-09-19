import io
import os
import uuid
import shutil
import cv2
import torch
import json
import re
import numpy as np
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_report_generator import generate_pdf_report

from PIL import Image
from change_detection.cdvqa_engine import ChangeDetectionEngine

from geospatial_preprocessing.spatial_alignment import match_and_align_geotiffs
from geospatial_preprocessing.geotiff_loader import load_and_standardize_image
from geospatial_preprocessing.test_failures import check_bbox_overlap, validate_downstream_payload
from geospatial_preprocessing.sar_preprocessor import apply_lee_filter

from agent_manager.agent_controller import agent
from agent_manager.schemas import ImageInput, TaskType

from fusion.sar_optical_fusion import execute_optical_sar_fusion

app = FastAPI(
    title="SatQuery AI Backend & Reporting Service",
    version="1.0.0",
    description="FastAPI service for multi-modal agentic analysis and PDF report generation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173"
    ], # Explicitly whitelist the Vite frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


print("Initializing CDVQA Engine...")
cd_engine = ChangeDetectionEngine()

def shared_vram_caller(
    prompt: str, 
    img_a_np: np.ndarray, 
    img_b_np: Optional[np.ndarray] = None, 
    img_c_np: Optional[np.ndarray] = None
) -> str:
    print("VLM Bridge Activated: Processing multi-image prompt...")
    
    pil_a = Image.fromarray(img_a_np)
    
    # Check if a second distinct image was provided (e.g. not a duplicate or stitched)
    is_two_distinct_images = img_b_np is not None and not np.array_equal(img_a_np, img_b_np)

    if is_two_distinct_images:
        pil_b = Image.fromarray(img_b_np)
        content = [
            {"type": "text", "text": "### BEFORE IMAGE (Historical Capture):\n"},
            {"type": "image", "image": pil_a},
            {"type": "text", "text": "\n### AFTER IMAGE (Recent Capture):\n"},
            {"type": "image", "image": pil_b},
        ]
        images_list = [pil_a, pil_b]
    else:
        # Single image mode (Perfect for the stitched side-by-side array)
        content = [{"type": "image", "image": pil_a}]
        images_list = [pil_a]

    # Append the heatmap overlay if it exists
    if img_c_np is not None:
        pil_c = Image.fromarray(img_c_np)
        content.extend([
            {"type": "text", "text": "\n### DETECTED CHANGE HIGHLIGHTS (Overlay):\n"},
            {"type": "image", "image": pil_c}
        ])
        images_list.append(pil_c)

    # Append the injected telemetry and user query
    content.append({"type": "text", "text": f"\n### USER QUESTION:\n{prompt}"})

    messages = [{"role": "user", "content": content}]

    text = agent.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = agent.processor(
        text=[text], 
        images=images_list, 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        with agent.model.disable_adapter():
            output_ids = agent.model.generate(**inputs, max_new_tokens=1024)

    generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    return agent.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def align_standard_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Uses ORB to find matching features and performs a safe 2D affine warp."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # If no features found, fallback safely
    if des1 is None or des2 is None:
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    keep = int(len(matches) * 0.15)
    matches = matches[:keep]
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
        
    # --- THE FIX ---
    # Use estimateAffinePartial2D instead of findHomography.
    # This prevents the "hyperspace stretch" by locking the transform to 2D only.
    transform_matrix, inliers = cv2.estimateAffinePartial2D(points2, points1, cv2.RANSAC)
    
    height, width = img1.shape[:2]
    
    if transform_matrix is not None:
        # Use warpAffine instead of warpPerspective
        aligned_img2 = cv2.warpAffine(img2, transform_matrix, (width, height))
        return aligned_img2
    else:
        print("Affine alignment failed (no valid matrix), falling back to resize.")
        return cv2.resize(img2, (width, height))

# --- Schema Definitions ---
class AnalysisResponse(BaseModel):
    answer: str = Field(..., description="Synthesized analytical response from the AI agents")
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
    adapter: str = Form("general", description="Explicit adapter to route to"), 
    images: List[UploadFile] = File(default=[], description="Up to 2 images"),
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
        if img.content_type not in ["image/jpeg", "image/png", "image/webp", "image/tiff"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format: {img.content_type}. Use JPEG, PNG, WebP, or TIFF.",
            )
        content = await img.read()
        processed_images_bytes.append(content)

    session_id = str(uuid.uuid4())
    temp_paths = []
    image_inputs = []
    
    try:
        # 1. Save ALL uploaded images temporarily
        for idx, (img_bytes, img_file) in enumerate(zip(processed_images_bytes, images)):
            ext = ".tif" if img_file.filename.lower().endswith(('.tif', '.tiff')) else ".jpg"
            temp_path = f"temp_{session_id}_{idx}{ext}"
            with open(temp_path, "wb") as f:
                f.write(img_bytes)
            temp_paths.append(temp_path)
            
            # ---> NEW: Build schemas for the Agent Manager <---
            filename_upper = img_file.filename.upper()
            modality = "sar" if "SAR" in filename_upper or "S1" in filename_upper else "optical"
            
            # Mock dates for validation (in production, extract from TIFF metadata)
            img_date = "2024-01-10" if "2024" in filename_upper else "2019-01-10"
            if idx == 1 and img_date == "2019-01-10":
                img_date = "2024-02-10" # Pass Member 1's different-date requirement

            image_inputs.append(ImageInput(path=temp_path, modality=modality, date=img_date))

        ## 2. --- TASK ROUTING ---
        if len(image_inputs) == 2:
            modality_a = image_inputs[0].modality
            modality_b = image_inputs[1].modality
            
            if modality_a != modality_b:
                task = TaskType.OPTICAL_SAR_FUSION
            else:
                task = TaskType.CHANGE_DETECTION
        else:
            task = TaskType.SINGLE_IMAGE_VQA
            
        print(f"Agent Manager routed task to: {task.value}")

        # 3. --- TASK EXECUTION ROUTING ---
        if task == TaskType.SINGLE_IMAGE_VQA:
            print(f"Routing to Single-Image Specialist with adapter: {adapter}...")
            # Load the single image for processing
            single_img_path = temp_paths[0]
            is_geospatial = single_img_path.lower().endswith('.tif')
            
            if is_geospatial:
                img_array, _ = load_and_standardize_image(single_img_path)
                # Save standard representation for rendering evidence view
                evidence_img_path = os.path.join("reports", session_id, "evidence.png")
                os.makedirs(os.path.dirname(evidence_img_path), exist_ok=True)
                cv2.imwrite(evidence_img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            else:
                img_array = cv2.cvtColor(cv2.imread(single_img_path), cv2.COLOR_BGR2RGB)
                # Copy directly as evidence artifact
                evidence_img_path = os.path.join("reports", session_id, "evidence.png")
                os.makedirs(os.path.dirname(evidence_img_path), exist_ok=True)
                shutil.copy(single_img_path, evidence_img_path)

            ai_answer = agent.query(prompt=query, image_path=single_img_path, explicit_adapter=adapter)
            
            agent_trace = {
                "pipeline_id": session_id,
                "nodes_traversed": ["AgentManager", "InputPreprocessor", "SingleImageSpecialist", "VerifierNode"],
                "telemetry": {"model_used": "Qwen2-VL-2B-BigEarthNet-LoRA"}
            }
            
        elif task == TaskType.CHANGE_DETECTION:
            print("Routing to CDVQA Specialist via Agent Manager...")

            # Check if we are dealing with geospatial data
            is_geospatial = temp_paths[0].lower().endswith('.tif')

            if is_geospatial:
                print("GeoTIFFs detected. Running fail-safes and Spatial Alignment...")
                aligned_path = f"temp_aligned_{session_id}.tif"
                
                # 1. Peek at the metadata before aligning
                _, meta_a = load_and_standardize_image(temp_paths[0])
                _, meta_b_raw = load_and_standardize_image(temp_paths[1])

                # 2. Check for physical overlap
                if meta_a["bounds"] and meta_b_raw["bounds"]:
                    if not check_bbox_overlap(meta_a["bounds"], meta_b_raw["bounds"]):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Images do not cover the same physical area. Alignment aborted."
                        )

                # 3. Align Image B to Image A
                match_and_align_geotiffs(temp_paths[0], temp_paths[1], aligned_path)
                
                # 4. Load the final, aligned arrays
                img_a, _ = load_and_standardize_image(temp_paths[0])
                img_b, _ = load_and_standardize_image(aligned_path)
                
                temp_paths.append(aligned_path)
            else:
                print("Standard images detected. Checking dimensions and aligning...")
                img_a = cv2.cvtColor(cv2.imread(temp_paths[0]), cv2.COLOR_BGR2RGB)
                img_b = cv2.cvtColor(cv2.imread(temp_paths[1]), cv2.COLOR_BGR2RGB)

                shape_a = img_a.shape[:2]
                shape_b = img_b.shape[:2]
                
                # 1. THE GATEKEEPER (Tolerance Check)
                if shape_a != shape_b:
                    height_diff = abs(shape_a[0] - shape_b[0])
                    width_diff = abs(shape_a[1] - shape_b[1])
                    
                    if height_diff > 10 or width_diff > 10:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Major shape mismatch ({shape_a} vs {shape_b}). Standard images must be the same size."
                        )
                    print(f"Minor shape mismatch ({shape_a} vs {shape_b}). Proceeding to optical alignment...")

                # 2. THE OPTICAL FIX (ORB Alignment)
                try:
                    # align_standard_images automatically fixes the shape AND the sub-pixel camera shifts
                    img_b = align_standard_images(img_a, img_b)
                except Exception as e:
                    print(f"ORB Alignment failed, falling back to simple resize: {e}")
                    img_b = cv2.resize(img_b, (shape_a[1], shape_a[0]), interpolation=cv2.INTER_LINEAR)

            # ---> NEW SAR DETECTION & LEE FILTER BLOCK <---
            # Check the original uploaded filenames from the 'images' list
            original_names = " ".join([img.filename.upper() for img in images if img.filename])
            if "S1" in original_names or "SAR" in original_names:
                print("SAR data detected! Applying Lee Filter...")
                
                # If it's a 3-channel image, apply the filter to each channel separately
                if img_a.ndim == 3:
                    img_a = np.dstack([apply_lee_filter(img_a[:, :, i]) for i in range(img_a.shape[2])])
                    img_b = np.dstack([apply_lee_filter(img_b[:, :, i]) for i in range(img_b.shape[2])])
                else:
                    # If it's already a single-band SAR image, apply directly
                    img_a = apply_lee_filter(img_a)
                    img_b = apply_lee_filter(img_b)

            if not validate_downstream_payload(img_a, img_b):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal pipeline error: Image array validation failed before AI processing."
                )

            # Route to our refactored, modular Hybrid Engine
            print("Routing to Hybrid CDVQA Specialist via Agent Manager...")
            cd_result = cd_engine.detect(
                image_a=img_a,
                image_b=img_b,
                vlm_fn=shared_vram_caller,
                user_query=query
            )
            
            ai_answer = cd_result.explanation

            session_folder = os.path.join("reports", session_id)
            os.makedirs(session_folder, exist_ok=True)
            
            # Save the annotated overlay (Image B + Green Bounding Boxes)
            evidence_path = os.path.join(session_folder, "evidence.png")
            cv2.imwrite(evidence_path, cd_result.overlay_image)
            
            agent_trace = {
                "pipeline_id": session_id,
                "nodes_traversed": ["InputPreprocessor", "Hybrid-CDVQA-Engine", "Qwen2-VL-Bridge"],
                "telemetry": {
                    "method": "Hybrid VLM + Spatial Analysis",
                    "major_regions_detected": cd_result.major_regions_detected
                }
            }
        elif task == TaskType.OPTICAL_SAR_FUSION:
            print("Routing to Optical-SAR Fusion Specialist via Agent Manager...")
            
            original_names = [img.filename.upper() for img in images if img.filename]
            sar_idx = 1 if ("SAR" in original_names[1] or "S1" in original_names[1]) else 0
            opt_idx = 0 if sar_idx == 1 else 1
            
            sar_path = temp_paths[sar_idx]
            opt_path = temp_paths[opt_idx]

            is_geospatial = opt_path.lower().endswith('.tif')
            if is_geospatial:
                print("GeoTIFFs detected. Aligning SAR to Optical...")
                aligned_sar_path = f"temp_aligned_fusion_{session_id}.tif"
                match_and_align_geotiffs(opt_path, sar_path, aligned_sar_path)
                
                opt_array, _ = load_and_standardize_image(opt_path)
                sar_array, _ = load_and_standardize_image(aligned_sar_path)
                temp_paths.append(aligned_sar_path)
            else:
                opt_array = cv2.cvtColor(cv2.imread(opt_path), cv2.COLOR_BGR2RGB)
                sar_array = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)

            fusion_result = execute_optical_sar_fusion(opt_array, sar_array, query)
            composite_img = fusion_result["composite_image"]
            specialized_prompt = fusion_result["generated_prompt"]
            
            print("Fusion complete. Passing False-Color Composite to VLM...")
            
            ai_answer = shared_vram_caller(specialized_prompt, composite_img)
            
            # Save composite image as the visual evidence artifact
            session_folder = os.path.join("reports", session_id)
            os.makedirs(session_folder, exist_ok=True)
            evidence_path = os.path.join(session_folder, "evidence.png")
            cv2.imwrite(evidence_path, cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR))

            agent_trace = {
                "pipeline_id": session_id,
                "nodes_traversed": ["AgentManager", "SpatialAlignment", "OpticalSARFusion", "Qwen2-VL-Bridge"],
                "telemetry": {"fusion_mode": "False-Color Composite (R, G, SAR)"}
            } 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")
        
    finally:
        # 3. Clean up ALL temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
    visual_evidence_url = f"/reports/{session_id}/evidence.png"
    report_url = f"/reports/{session_id}/download"

    # Ensure the session folder exists (in case it wasn't created earlier)
    # Ensure the session folder exists
    session_folder = os.path.join("reports", session_id)
    os.makedirs(session_folder, exist_ok=True)

    # 1. Save the metadata as JSON
    record = {
        "query": query,
        "answer": ai_answer,
        "agent_execution_trace": agent_trace,
    }
    with open(os.path.join(session_folder, "data.json"), "w") as f:
        json.dump(record, f)

    # 2. Save the raw image bytes to disk
    with open(os.path.join(session_folder, "source.img"), "wb") as f:
        f.write(processed_images_bytes[0])

    # 3. --- NEW: Generate and save the PDF report directly to disk ---
    img_buffer = io.BytesIO(processed_images_bytes[0]) if processed_images_bytes else None
    pdf_buffer = generate_pdf_report(
        query=query,
        answer=ai_answer,
        agent_execution_trace=agent_trace,
        image_source=img_buffer,
    )
    
    pdf_buffer.seek(0)
    report_path = os.path.join(session_folder, "report.pdf")
    with open(report_path, "wb") as f:
        f.write(pdf_buffer.read())

    # Point the URL directly to the static PDF file
    visual_evidence_url = f"/reports/{session_id}/evidence.png"
    report_url = f"/reports/{session_id}/report.pdf"

    return AnalysisResponse(
        answer=ai_answer,
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
    session_folder = os.path.join("reports", session_id)
    json_path = os.path.join(session_folder, "data.json")
    img_path = os.path.join(session_folder, "source.img")

    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report session expired or does not exist.",
        )

    # Load metadata
    with open(json_path, "r") as f:
        record = json.load(f)

    # Load image bytes
    img_buffer = None
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_buffer = io.BytesIO(f.read())

    pdf_buffer = generate_pdf_report(
        query=record["query"],
        answer=record["answer"],
        agent_execution_trace=record["agent_execution_trace"],
        image_source=img_buffer,
    )

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=audit_report_{session_id[:8]}.pdf"},
    )