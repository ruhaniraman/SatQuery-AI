import asyncio
import io
import os
import re
import uuid
import shutil
import cv2
import torch
import json
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_report_generator import generate_pdf_report

from PIL import Image
from change_detection.cdvqa_engine import ChangeDetectionEngine

from geospatial_preprocessing.spatial_alignment import match_and_align_geotiffs, AlignmentError
from geospatial_preprocessing.geotiff_loader import load_and_standardize_image, load_and_standardize_pair
from geospatial_preprocessing.payload_validation import validate_downstream_payload
from geospatial_preprocessing.standard_alignment import align_with_status
from geospatial_preprocessing.acquisition_date import extract_acquisition_date, resolve_temporal_order

from agent_manager.agent_controller import get_agent, load_agent
from report_retention import purge_old_reports, retention_hours
from upload_validation import UploadTooLarge, detect_image_kind, max_request_bytes, max_upload_bytes, read_limited
from agent_manager.schemas import ExecutionTrace, ImageInput, TaskType
from agent_manager.prompts import DEFAULT_MAX_NEW_TOKENS, describe_first_enabled, history_for_model
from agent_manager.scene_stats import compute_scene_stats, format_scene_stats
from agent_manager.scan_compare import summarize_scan_change
from change_detection.change_report import compare_report_enabled, lora_compare_enabled
from agent_manager.grid_scan import DEFAULT_CONTEXT, SCAN_YES_THRESHOLD
from agent_manager.scan_report import build_scan_summary, render_scan_evidence, scan_answer_text

from fusion.sar_optical_fusion import execute_optical_sar_fusion

async def _retention_loop():
    while True:
        deleted = await asyncio.to_thread(purge_old_reports, "reports", retention_hours())
        if deleted:
            print(f"Report retention: removed {len(deleted)} old session folder(s).")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model here, not at import time: a missing adapter or failed download must not stop
    # the API from starting. /analyze answers 503 with the reason until the model is available.
    _app.state.model_error = None
    if get_agent() is None:
        try:
            await asyncio.to_thread(load_agent)
        except Exception as e:
            _app.state.model_error = f"{type(e).__name__}: {e}"
            print(f"MODEL FAILED TO LOAD: {_app.state.model_error}")
    retention_task = asyncio.create_task(_retention_loop()) if retention_hours() > 0 else None
    yield
    if retention_task:
        retention_task.cancel()


app = FastAPI(
    lifespan=lifespan,
    title="SatQuery AI Backend & Reporting Service",
    version="1.0.0",
    description="FastAPI service for multi-modal agentic analysis and PDF report generation",
)

@app.middleware("http")
async def limit_request_size(request, call_next):
    """Reject oversized uploads from the Content-Length header before the body is read."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_request_bytes():
        return JSONResponse(status_code=413, content={"detail": "Request body too large."})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


# Sensor type of each uploaded image; chosen by the user in the UI.
VALID_MODALITIES = {"optical", "sar"}
VALID_ADAPTERS = {"general", "mining", "deforestation", "agriculture"}

# Longest side of a browser preview. Browsers cannot draw TIFF/GeoTIFF, and a raw scene can be thousands of pixels wide.
PREVIEW_MAX_SIDE = 2048
# Session folders are named by uuid4 only; anything else is rejected before it can touch the filesystem.
_SESSION_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def strip_markdown_asterisks(text):
    """The UI shows plain text, so emphasis markers are removed ONCE, here, before the answer is
    returned, stored in the chat history, or printed in the PDF (previously only the browser stripped
    them, so the chat and the PDF showed different text for the same answer)."""
    return text.replace("*", "") if isinstance(text, str) else text


def _agent():
    """The loaded model engine, or a 503 that says why it isn't available."""
    engine = get_agent()
    if engine is None:
        reason = getattr(app.state, "model_error", None)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI model is not available" + (f": {reason}" if reason else " yet (still loading)."),
        )
    return engine


def _model_telemetry(adapter):
    """Which weights really produce the output. adapter=None means the base model with LoRA adapters
    switched off (which is how the general, change-detection and fusion paths all run)."""
    engine = get_agent()
    base = getattr(engine, "base_model_id", "unknown")
    if adapter is None:
        return {"model_used": f"{base} (base weights, LoRA adapters disabled)", "active_adapter": None}
    return {
        "model_used": f"{base} + LoRA adapter '{adapter}'",
        "active_adapter": adapter,
        "adapter_details": getattr(engine, "adapter_info", {}).get(adapter, {}),
        "adapter_training_data": "not recorded in the adapter files",
    }


def _make_trace(session_id, task, routing_reason, nodes, telemetry, warnings=(), validation_status="not performed"):
    return ExecutionTrace(
        pipeline_id=session_id, task=task, routing_reason=routing_reason,
        nodes_traversed=nodes, telemetry=telemetry, warnings=list(warnings),
        validation_status=validation_status,
    )

print("Initializing CDVQA Engine...")
cd_engine = ChangeDetectionEngine()

def shared_vram_caller(prompt: str, image_np: np.ndarray, system: Optional[str] = None) -> str:
    """Free-text VLM call used by change detection (stitched before|after image) and fusion
    (composite). Runs on the base weights with LoRA adapters disabled."""
    print("VLM Bridge Activated: Processing prompt...")
    engine = _agent()

    pil = Image.fromarray(image_np)
    content = [
        {"type": "image", "image": pil},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = engine.processor(
        text=[text],
        images=[pil],
        padding=True,
        return_tensors="pt",
    ).to(engine.model.device)

    # disable_adapter() flips global model state, so hold the same lock engine.query() uses.
    with engine.lock:
        with torch.no_grad():
            with engine.model.disable_adapter():
                output_ids = engine.model.generate(**inputs, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)

    generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    return engine.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# --- Schema Definitions ---
class AnalysisResponse(BaseModel):
    answer: str = Field(..., description="Synthesized analytical response from the AI agents")
    agent_execution_trace: ExecutionTrace = Field(..., description="What actually ran for this request: task, routing rule, stages, model, warnings")
    visual_evidence_url: str = Field(..., description="Endpoint or URL to retrieve the visual inspection artifact")
    report_download_url: Optional[str] = Field(None, description="Direct URL to download the single-page audit PDF")
    report_error: Optional[str] = Field(None, description="Why the PDF could not be produced, when report_download_url is null")
    scan: Optional[dict] = Field(None, description="Feature scans only: headline, confidence level, numbered findings (box, position, score) and the 4x4 score grid")


# --- Analysis Endpoint ---
# Deliberately a plain `def`, not `async def`: the body is blocking GPU/OpenCV work, and FastAPI
# runs sync endpoints in a threadpool so the event loop (static files, other requests) stays free.
@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit query and up to 2 images for agentic analysis",
)
def analyze(
    query: str = Form(..., description="Analytical query/prompt for the agent system"),
    adapter: str = Form("general", description="Explicit adapter to route to"),
    modality_a: str = Form("optical", description="Sensor type of image 1: optical | sar"),
    modality_b: str = Form("optical", description="Sensor type of image 2: optical | sar"), 
    chat_history: Optional[str] = Form(None, description="Stringified JSON of the conversation"), 
    images: List[UploadFile] = File(default=[], description="Up to 2 images"),
):
    engine = _agent()  # 503 before doing any work if the model is not available

    parsed_history = []
    if chat_history:
        try:
            parsed_history = json.loads(chat_history)
        except Exception:
            pass

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

    modality_a, modality_b = modality_a.lower(), modality_b.lower()
    for m in (modality_a, modality_b):
        if m not in VALID_MODALITIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid modality '{m}'. Choose one of: optical, sar.",
            )

    if adapter not in VALID_ADAPTERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid adapter '{adapter}'. Choose one of: {', '.join(sorted(VALID_ADAPTERS))}.",
        )

    processed_images_bytes: List[bytes] = []
    kinds: List[str] = []
    for img in images:
        if not img.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Every uploaded image needs a filename.")
        if img.content_type not in ["image/jpeg", "image/png", "image/webp", "image/tiff"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format: {img.content_type}. Use JPEG, PNG, WebP, or TIFF.",
            )
        try:
            content = read_limited(img.file, max_upload_bytes())
        except UploadTooLarge as e:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
        # The declared content type and the filename are the client's word; the bytes are the truth.
        kind = detect_image_kind(content)
        if kind is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"'{img.filename}' is not a valid JPEG, PNG, WebP, or TIFF image.",
            )
        kinds.append(kind)
        processed_images_bytes.append(content)

    # GeoTIFF pairs are aligned by georeferencing and ordinary images by feature matching; the two
    # can't be mixed, so say so up front instead of failing somewhere inside rasterio.
    if len(kinds) == 2 and (kinds[0] == "tiff") != (kinds[1] == "tiff"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot combine a TIFF/GeoTIFF with a PNG/JPEG/WebP image. Upload two files of the same kind.",
        )

    session_id = str(uuid.uuid4())
    temp_paths = []
    image_inputs = []
    acquisitions = []
    temporal_order = None
    routing_reason = ""
    loader_warnings: List[str] = []
    
    try:
        # 1. Save ALL uploaded images temporarily
        for idx, (img_bytes, img_file) in enumerate(zip(processed_images_bytes, images)):
            ext = ".tif" if kinds[idx] == "tiff" else ".jpg"   # from the bytes, not the client's filename
            temp_path = f"temp_{session_id}_{idx}{ext}"
            with open(temp_path, "wb") as f:
                f.write(img_bytes)
            temp_paths.append(temp_path)
            
            # Build schemas for the Agent Manager
            # Chosen by the user in the UI (sniffing filenames for "SAR"/"S1" was unreliable)
            modality = modality_a if idx == 0 else modality_b
            
            # Real capture date from raster tags / recognised product filenames, else None
            acq = extract_acquisition_date(temp_path, img_file.filename)
            acquisitions.append(acq)
            image_inputs.append(ImageInput(
                path=temp_path, modality=modality,
                date=acq.iso if acq else None, date_source=acq.source if acq else None,
            ))

        scan_summary = None   # set by a feature scan; goes into the response, data.json and the PDF

        ## 2. --- TASK ROUTING ---
        if len(image_inputs) == 2:
            modality_a = image_inputs[0].modality
            modality_b = image_inputs[1].modality
            
            if modality_a != modality_b:
                task = TaskType.OPTICAL_SAR_FUSION
                routing_reason = f"2 images of different sensor types ({modality_a} + {modality_b})"
            else:
                task = TaskType.CHANGE_DETECTION
                routing_reason = f"2 images of the same sensor type ({modality_a} + {modality_b})"
                # Before/after: from real dates when both are trustworthy, else as uploaded
                temporal_order = resolve_temporal_order(acquisitions[0], acquisitions[1])
                print(f"Temporal order: {temporal_order.summary}")
                if temporal_order.swap:
                    temp_paths[0], temp_paths[1] = temp_paths[1], temp_paths[0]
                    image_inputs[0], image_inputs[1] = image_inputs[1], image_inputs[0]
        else:
            task = TaskType.SINGLE_IMAGE_VQA
            routing_reason = "1 image uploaded"
            
        print(f"Agent Manager routed task to: {task.value}")

        # 3. --- TASK EXECUTION ROUTING ---
        if task == TaskType.SINGLE_IMAGE_VQA:
            print(f"Routing to Single-Image Specialist with adapter: {adapter}...")
            # Load the single image for processing
            single_img_path = temp_paths[0]
            is_geospatial = single_img_path.lower().endswith('.tif')
            
            if is_geospatial:
                img_array, _ = load_and_standardize_image(single_img_path, modality_a)
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

            scene_stats = compute_scene_stats(img_array, modality_a) if adapter == "general" else None
            scene_facts = format_scene_stats(scene_stats)
            if adapter == "general":
                ai_answer = engine.query(prompt=query, image_path=single_img_path, explicit_adapter=adapter, modality=modality_a,
                                        chat_history=parsed_history, scene_facts=scene_facts)
            else:
                # Feature scan: score the 16 tiles, then say what was found and draw it (tint + outlines + pins)
                probs = engine.scan_scores(img_array, adapter, prompt=query)
                print("Grid P(yes):" + chr(10) + np.array2string(np.asarray(probs), precision=2))
                scan_summary = build_scan_summary(adapter, probs, SCAN_YES_THRESHOLD)
                ai_answer = scan_answer_text(scan_summary)
                annotated_img = render_scan_evidence(img_array, probs, scan_summary)
                cv2.imwrite(evidence_img_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

            is_grid_scan = adapter != "general"
            telemetry = {
                **_model_telemetry(adapter if is_grid_scan else None),
                "sensor_type": modality_a,
            }
            if is_grid_scan:
                telemetry["inference"] = "4x4 tile yes/no scan, scored from first-token logits"
                telemetry["grid_yes_threshold"] = SCAN_YES_THRESHOLD
                telemetry["grid_tile_context"] = DEFAULT_CONTEXT
            else:
                telemetry["inference"] = (
                    "free-text generation after a describe-first pass (whole image + 4 quadrants)"
                    if describe_first_enabled() else "free-text generation"
                )
                telemetry["measured_scene_facts"] = scene_facts or "unavailable"
                telemetry["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
                telemetry["chat_turns_given_to_model"] = len(history_for_model(parsed_history, query))
            agent_trace = _make_trace(
                session_id, task, routing_reason,
                ["RuleBasedRouter", "InputPreprocessor", "SingleImageSpecialist"],
                telemetry,
            )
            print(f"[{session_id}] Executed {telemetry['model_used']}")
            
        elif task == TaskType.CHANGE_DETECTION:
            print("Routing to CDVQA Specialist via Agent Manager...")

            alignment_warnings: List[str] = []

            # Check if we are dealing with geospatial data
            is_geospatial = temp_paths[0].lower().endswith('.tif')

            if is_geospatial:
                print("GeoTIFFs detected. Aligning Image B onto Image A's grid...")
                aligned_path = f"temp_aligned_{session_id}.tif"
                # Register for cleanup BEFORE the file exists, so a failure below can't leak it
                temp_paths.append(aligned_path)

                # Measures footprint coverage in A's CRS first; raises AlignmentError (-> HTTP 400)
                # if the two images barely overlap.
                match_and_align_geotiffs(temp_paths[0], temp_paths[1], aligned_path)

                # One shared intensity stretch for both dates, so identical ground doesn't get
                # different pixel values in A and B (that would show up as false change).
                # SAR rasters are despeckled inside the loader, in the linear domain.
                img_a, meta_a, img_b, meta_b = load_and_standardize_pair(
                    temp_paths[0], aligned_path, modality=image_inputs[0].modality
                )
                loader_warnings = meta_a.get("warnings", []) + meta_b.get("warnings", [])
                # Only pixels real in BOTH images can show change; B's warped-in nodata border is excluded.
                valid_mask = meta_a["valid_mask"] & meta_b["valid_mask"]
            else:
                print("Standard images detected. Checking dimensions and aligning...")
                # SAR previews are despeckled inside the loader
                img_a, _, img_b, _ = load_and_standardize_pair(
                    temp_paths[0], temp_paths[1], modality=image_inputs[0].modality
                )
                valid_mask = None

                shape_a = img_a.shape[:2]
                shape_b = img_b.shape[:2]
                
                # 1. Tolerance Check
                if shape_a != shape_b:
                    height_diff = abs(shape_a[0] - shape_b[0])
                    width_diff = abs(shape_a[1] - shape_b[1])
                    
                    if height_diff > 10 or width_diff > 10:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Major shape mismatch ({shape_a} vs {shape_b}). Standard images must be the same size."
                        )
                    print(f"Minor shape mismatch ({shape_a} vs {shape_b}). Proceeding to optical alignment...")

                # 2. ORB Alignment
                try:
                    # align_standard_images automatically fixes the shape AND the sub-pixel camera shifts
                    img_b, valid_mask, align_note = align_with_status(img_a, img_b)
                    if align_note:
                        alignment_warnings.append(
                            f"Image B could not be registered to Image A ({align_note}); it was only resized to fit, so "
                            "pixel-level comparison is unreliable (different zoom, view or content is the usual cause)."
                        )
                except Exception as e:
                    print(f"ORB Alignment failed, falling back to simple resize: {e}")
                    img_b = cv2.resize(img_b, (shape_a[1], shape_a[0]), interpolation=cv2.INTER_LINEAR)
                    valid_mask = None

            if not validate_downstream_payload(img_a, img_b):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal pipeline error: Image array validation failed before AI processing."
                )

            # Located evidence from the trained LoRAs: the same yes/no tile scan on both dates.
            # Optical only (the adapters' training data is not recorded); a failure just omits it.
            compose = compare_report_enabled()
            scan_summaries = []
            scanner = getattr(_agent(), "scan_scores", None) if compose and lora_compare_enabled() else None
            if scanner and image_inputs[0].modality == "optical":
                for scan_adapter in ("mining", "deforestation", "agriculture"):
                    try:
                        scan_summaries.append(summarize_scan_change(scan_adapter, scanner(img_a, scan_adapter), scanner(img_b, scan_adapter)))
                    except Exception as err:
                        print(f"LoRA scan comparison skipped for {scan_adapter}: {err}")

            # Route to our refactored, modular Hybrid Engine
            print("Routing to Hybrid CDVQA Specialist via Agent Manager...")
            cd_result = cd_engine.detect(
                image_a=img_a,
                image_b=img_b,
                vlm_fn=shared_vram_caller,
                user_query=query,
                capture_dates=temporal_order.prompt_dates,
                valid_mask=valid_mask,
                modality=image_inputs[0].modality,
                notes_fn=shared_vram_caller if describe_first_enabled() else None,
                compose_report=compose,
                scan_summaries=scan_summaries,
                extra_warnings=alignment_warnings,
            )
            
            ai_answer = cd_result.explanation

            session_folder = os.path.join("reports", session_id)
            os.makedirs(session_folder, exist_ok=True)
            
            # Save the annotated overlay (Image B + Green Bounding Boxes)
            evidence_path = os.path.join(session_folder, "evidence.png")
            cv2.imwrite(evidence_path, cd_result.overlay_image)
            
            agent_trace = _make_trace(
                session_id, task, routing_reason,
                ["RuleBasedRouter", "InputPreprocessor", "SpatialAlignment" if is_geospatial else "OrbAlignment",
                 "TemporalOrdering", "Hybrid-CDVQA-Engine"]
                + (["EvidenceReport"] if compose else [])
                + ["Qwen2-VL-Bridge"],
                {
                    **_model_telemetry(None),
                    "method": "Hybrid VLM + Spatial Analysis",
                    "sensor_type": image_inputs[0].modality,
                    "major_regions_detected": cd_result.major_regions_detected,
                    "describe_first": describe_first_enabled(),
                    "answer_source": (
                        "assembled by code from measurements, flagged regions, LoRA scan scores and single-image "
                        "model captions (the model did not write the conclusion)"
                        if compose else "written by the model from the stitched before|after image"
                    ),
                    "subtle_regions_detected": cd_result.subtle_regions_detected,
                    "lora_scans_compared": [x["name"] for x in scan_summaries],
                    **({"comparison_details": cd_result.details} if cd_result.details else {}),
                    "measured_before_after": cd_result.measured_facts or "unavailable",
                    "temporal_order": temporal_order.to_dict(),
                },
                warnings=loader_warnings + temporal_order.warnings + cd_result.warnings,
                validation_status="image payload checks passed",
            )
        elif task == TaskType.OPTICAL_SAR_FUSION:
            print("Routing to Optical-SAR Fusion Specialist via Agent Manager...")
            
            sar_idx = 0 if image_inputs[0].modality == "sar" else 1
            opt_idx = 1 - sar_idx
            
            sar_path = temp_paths[sar_idx]
            opt_path = temp_paths[opt_idx]

            is_geospatial = opt_path.lower().endswith('.tif')
            if is_geospatial:
                print("GeoTIFFs detected. Aligning SAR to Optical...")
                aligned_sar_path = f"temp_aligned_fusion_{session_id}.tif"
                temp_paths.append(aligned_sar_path)  # cleanup even if alignment fails
                match_and_align_geotiffs(opt_path, sar_path, aligned_sar_path)
                
                # Different sensors measure different quantities, so each gets its own stretch.
                # The SAR side is despeckled by the loader.
                opt_array, opt_meta = load_and_standardize_image(opt_path, "optical")
                sar_array, sar_meta = load_and_standardize_image(aligned_sar_path, "sar")
            else:
                opt_array, opt_meta = load_and_standardize_image(opt_path, "optical")
                sar_array, sar_meta = load_and_standardize_image(sar_path, "sar")

            fusion_result = execute_optical_sar_fusion(
                opt_array, sar_array, query, sar_valid=sar_meta["valid_mask"]
            )
            composite_img = fusion_result["composite_image"]

            print(f"Fusion complete ({fusion_result['method']}). Passing composite to VLM...")

            ai_answer = shared_vram_caller(
                fusion_result["generated_prompt"], composite_img, system=fusion_result["system_prompt"]
            )
            
            # Save composite image as the visual evidence artifact
            session_folder = os.path.join("reports", session_id)
            os.makedirs(session_folder, exist_ok=True)
            evidence_path = os.path.join(session_folder, "evidence.png")
            cv2.imwrite(evidence_path, cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR))

            agent_trace = _make_trace(
                session_id, task, routing_reason,
                ["RuleBasedRouter", "InputPreprocessor"] + (["SpatialAlignment"] if is_geospatial else [])
                + ["OpticalSARFusion", "Qwen2-VL-Bridge"],
                {
                    **_model_telemetry(None),
                    "fusion_mode": fusion_result["method"],
                    "sar_despeckled": True,
                    "sar_input_units": sar_meta.get("sar_input_units"),
                },
                warnings=opt_meta.get("warnings", []) + sar_meta.get("warnings", []),
            )
    except HTTPException:
        # Deliberate 4xx errors raised above must reach the client unchanged, not become 500s
        raise
    except AlignmentError as e:
        # Bad input data (not georeferenced / images don't overlap) is the client's problem
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")
        
    finally:
        # 3. Clean up ALL temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
    ai_answer = strip_markdown_asterisks(ai_answer)
    visual_evidence_url = f"/reports/{session_id}/evidence.png"

    session_folder = os.path.join("reports", session_id)
    os.makedirs(session_folder, exist_ok=True)

    # Append the final answer so the PDF shows the full conversation
    parsed_history.append({"role": "ai", "content": ai_answer})

    record = {
        "query": query,
        "answer": ai_answer,
        "agent_execution_trace": agent_trace.model_dump(mode="json"),
        "chat_history": parsed_history,
        "scan": scan_summary,
    }
    with open(os.path.join(session_folder, "data.json"), "w") as f:
        json.dump(record, f)

    # Use the evidence image in the PDF when there is one, else fall back to the first upload
    evidence_path = os.path.join(session_folder, "evidence.png")
    if os.path.exists(evidence_path):
        with open(evidence_path, "rb") as f:
            img_buffer = io.BytesIO(f.read())
    else:
        img_buffer = io.BytesIO(processed_images_bytes[0]) if processed_images_bytes else None

    # The PDF is a convenience artifact: if building it fails, the analysis result (which took
    # the GPU time) must still be returned. The frontend simply shows no download link.
    report_url = None
    report_error = None
    try:
        pdf_buffer = generate_pdf_report(
            query=record["query"],
            answer=record["answer"],
            agent_execution_trace=record["agent_execution_trace"],
            image_source=img_buffer,
            chat_history=record.get("chat_history", []),
            scan=scan_summary,
        )
        pdf_buffer.seek(0)
        with open(os.path.join(session_folder, "report.pdf"), "wb") as f:
            f.write(pdf_buffer.read())
        # Served by the static /reports mount
        report_url = f"/reports/{session_id}/report.pdf"
    except Exception as e:
        print(f"[{session_id}] PDF report generation failed (analysis result still returned): {e}")
        report_error = f"PDF generation failed: {type(e).__name__}"

    return AnalysisResponse(
        answer=ai_answer,
        agent_execution_trace=agent_trace,
        visual_evidence_url=visual_evidence_url,
        report_download_url=report_url,
        report_error=report_error,
        scan=scan_summary,
    )

# --- Small helper endpoints used by the dashboard ---

@app.get("/health", summary="Is the backend up, and is the model loaded?")
def health():
    loaded = get_agent() is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_error": None if loaded else getattr(app.state, "model_error", None),
    }


@app.post("/preview", summary="Browser-safe PNG of an uploaded image (browsers cannot draw TIFF/GeoTIFF)")
def preview(
    modality: str = Form("optical", description="Sensor type: optical | sar (SAR is despeckled like an analysis run)"),
    image: UploadFile = File(..., description="The image to preview"),
):
    """Runs the file through the same loader the analysis uses, so the preview is what the model will
    see (percentile stretch, SAR despeckle), scaled down to at most PREVIEW_MAX_SIDE pixels."""
    modality = modality.lower()
    if modality not in VALID_MODALITIES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid modality '{modality}'. Choose one of: optical, sar.")
    if not image.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The image needs a filename.")
    try:
        content = read_limited(image.file, max_upload_bytes())
    except UploadTooLarge as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    kind = detect_image_kind(content)
    if kind is None:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"'{image.filename}' is not a valid JPEG, PNG, WebP, or TIFF image.")

    if kind == "tiff":
        temp_path = f"temp_preview_{uuid.uuid4()}.tif"
        try:
            with open(temp_path, "wb") as f:
                f.write(content)
            array, _ = load_and_standardize_image(temp_path, modality)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read '{image.filename}' as a raster image: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not decode '{image.filename}'.")
        array = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    height, width = array.shape[:2]
    scale = PREVIEW_MAX_SIDE / max(height, width)
    if scale < 1:
        array = cv2.resize(array, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the preview.")
    return Response(content=png.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/report", summary="Rebuild the PDF of a finished run with the full current conversation")
def rebuild_report(
    session_id: str = Form(..., description="The run (its evidence image and trace) the report is built on"),
    chat_history: Optional[str] = Form(None, description="Stringified JSON of the whole conversation so far"),
    map_link: Optional[str] = Form(None, description="Link to the analysed view on a map (live-map captures), shown in scan reports"),
):
    """Every /analyze call writes its own PDF, but a chat has many runs and later plain questions carry
    only the raw image as evidence. This lets the UI pick which run's evidence the report shows while
    still printing the complete conversation."""
    if not _SESSION_ID.match(session_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id.")
    folder = os.path.join("reports", session_id)
    data_path = os.path.join(folder, "data.json")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such analysis run.")
    with open(data_path) as f:
        record = json.load(f)

    history = record.get("chat_history", [])
    if chat_history:
        try:
            parsed = json.loads(chat_history)
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_history is not valid JSON.")
        if isinstance(parsed, list):
            history = parsed

    trace = json.loads(json.dumps(record["agent_execution_trace"]))
    trace.setdefault("telemetry", {})["report_note"] = (
        f"Built on demand: the conversation shown is current; the evidence image and this trace belong to run {session_id}."
    )
    evidence_path = os.path.join(folder, "evidence.png")
    image_source = None
    if os.path.exists(evidence_path):
        with open(evidence_path, "rb") as f:
            image_source = io.BytesIO(f.read())

    try:
        pdf = generate_pdf_report(
            query=record["query"], answer=record["answer"], agent_execution_trace=trace,
            image_source=image_source, chat_history=history,
            scan=record.get("scan"), map_link=map_link if (map_link or "").startswith(("https://", "http://")) else None,
        )
        pdf.seek(0)
        with open(os.path.join(folder, "report_chat.pdf"), "wb") as f:
            f.write(pdf.read())
    except Exception as e:
        print(f"[{session_id}] PDF rebuild failed: {e}")
        return {"report_download_url": None, "report_error": f"PDF generation failed: {type(e).__name__}"}
    return {"report_download_url": f"/reports/{session_id}/report_chat.pdf", "report_error": None}
