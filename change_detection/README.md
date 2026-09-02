# SATQUERY AI — 72-Hour Prototype

Agentic vision-language backend for remote-sensing imagery (GeoTIFF/SAR/Optical),
driven by natural-language queries.

## Status (end of Day 2)

All Stage 1→4 modules are built, wired, and passing an end-to-end pytest suite
(`tests/test_pipeline.py`, 7/7 passing) using mock/fallback data. Two upgrades
landed this round:

- **Member 1's classifier is now LLM-backed** (`llm_classifier.py`), using a
  structured Claude tool-call so output is always a valid `TaskIntent` enum —
  no free-text parsing. Falls back to the original keyword classifier if
  `ANTHROPIC_API_KEY` isn't set, so nobody is blocked without a key.
- **Repo is now actually runnable**: `requirements.txt`, `.env.example`, and a
  formal `tests/` suite replace the old ad-hoc smoke-test scripts.

## Architecture

```
Stage 1 (Member 4)  geotiff_loader.py, sar_preprocessor.py
                      -> normalized 8-bit RGB numpy arrays + metadata
Stage 2 (Member 1)  agent_controller.py + llm_classifier.py
                      -> task routing + auditable execution trace
Stage 3 (specialists, one of):
  Member 2  single_image_vqa_engine.py   -> Florence-2 grounding, Qwen2-VL VQA
  Member 3  cdvqa_engine.py              -> SSIM change detection + mask
  Member 5  optical_sar_fusion_model.py  -> false-color optical+SAR composite
Stage 4 (Member 6)  main_api.py + pdf_report_generator.py
                      -> FastAPI /analyze endpoint + downloadable PDF audit
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (optional but recommended)
```

## Run the API

```bash
uvicorn main_api:app --reload
# POST http://localhost:8000/analyze
#   form fields: query (str), image1 (file), image2 (file, optional)
```

## Run tests

```bash
pytest tests/ -v
```

## What's still mocked (next steps)

- **Member 2**: real Florence-2/Qwen2-VL inference code is written
  (`single_image_vqa_engine.py`) but needs the actual weights downloaded
  (~5–10GB) and a GPU to run at usable speed. Currently falls back to a
  deterministic mock so the rest of the pipeline isn't blocked.
- **Confidence scores**: `main_api.py` uses placeholder/derived values
  (SSIM score for change detection, 0.0 for mock grounding). Real per-task
  confidence calibration is still open.
- **Alignment/co-registration**: Member 4's loader does not yet
  georeference-align two images to each other — it assumes inputs to
  change detection / fusion are already pixel-aligned.
