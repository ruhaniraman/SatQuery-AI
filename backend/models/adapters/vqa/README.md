---
base_model: Qwen/Qwen2-VL-2B-Instruct
library_name: peft
tags:
- lora
- remote-sensing
- vqa
- rsvqa
---

# SatQuery-AI remote-sensing VQA adapter (`vqa`)

A LoRA adapter for Qwen2-VL-2B-Instruct that answers short remote-sensing questions (yes/no, counts, rural or urban)
about optical satellite images. SatQuery-AI uses it for closed questions in chat; an open question never uses it.

## Training data
- **RSVQA-LR, train split only**: 57,223 questions on 572 Sentinel-2 RGB tiles (10 m), from zenodo.org/records/6344334.
- The train, val and test splits share no images (checked).
- Each prompt is the evaluation harness's short-answer prompt (`backend/evaluation/answer_space.py`). The loss covers
  the answers only.

## Training
- QLoRA: 4-bit NF4 base with bf16 compute. Rank 16, alpha 32, dropout 0.05. Language model only (q, k, v, o, gate,
  up and down projections), 18.5M trainable parameters.
- One epoch: 1,001 optimizer steps. Learning rate 1e-4 with cosine decay and 30 warm-up steps, gradient accumulation 4.
  Each sequence packs 16 questions about one image.
- About 4.5 h on an RTX 3050 6 GB laptop GPU. The full record is in `training_info.json` and `training_log.jsonl`.
- Code: `backend/training/train_vqa_lora.py` and `train_in_chunks.py`.

## Results (RSVQA-LR test, 1,000 stratified questions, seed 0, `backend/run_eval.py`)
- **Paper protocol** (presence, comparison, rural/urban):
  - Base 58.4%, **adapted 89.9%** (95% CI 87.5 to 91.9).
  - Answer-prior floor 68.9%.
  - For reference: GeoChat 90.7%, RSGPT 92.3%.
- **Per question type, adapted:** presence 91.5%, comparison 88.5%, rural/urban 10 of 10.
- **Counting is not solved:** 24.7%, at the floor.
- **Calibration** (expected calibration error, ECE): base 0.245, adapted 0.139.

## Limits
- Trained on 10 m Sentinel-2 RGB only. It has not been checked on sub-metre imagery, SAR or Cartosat-2S, and the app
  never uses it for SAR.
- RSVQA answers come from OpenStreetMap, so they inherit its gaps.
- Counts are unreliable.
