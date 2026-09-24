#!/usr/bin/env python
"""QLoRA fine-tuning of Qwen2-VL-2B for remote-sensing VQA (SIH26167 requirement: a remote-sensing-adapted VL
component). Run from backend/ on the GPU machine:

    python -m training.train_vqa_lora --root data/benchmarks/RSVQA_LR --out models/adapters/vqa

Data: the RSVQA-LR TRAIN split only (images disjoint from val/test; checked). Prompts are the evaluation
harness's, so the result is scored with `run_eval.py ... --adapter vqa`. Pick settings on the VAL split, and
score the test split once at the end. Writes the adapter + training_info.json (data, counts, hyperparameters,
loss curve, git commit) into --out, so the model card can say exactly what the adapter was trained on.

Memory (6 GB RTX 3050): base in 4-bit NF4 with bf16 compute, LoRA on the language model only (same modules,
rank and alpha as the existing adapters), gradient checkpointing, one pack per step with gradient accumulation.
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone

import torch
from PIL import Image
import psutil
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from agent_manager.agent_controller import MAX_PIXELS, MIN_PIXELS
from evaluation.answer_space import build_answer_spaces
from evaluation.benchmarks import load_benchmark
from evaluation.runner import _git_commit
from geospatial_preprocessing.geotiff_loader import load_and_standardize_image, load_and_standardize_pair
from training.vqa_data import assistant_label_mask, build_packs, pack_messages

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OPTIMIZER_FILE = "optimizer.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="rsvqa-lr")
    p.add_argument("--root", required=True)
    p.add_argument("--image-dir")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True, help="Adapter folder to write (e.g. models/adapters/vqa)")
    p.add_argument("--epochs", type=float, default=1.0, help="Fractions allowed (0.5 = half the packs)")
    p.add_argument("--per-pack", type=int, default=8, help="Questions per image per training sequence")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=30, help="Warm-up optimizer steps")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=100, help="Save a checkpoint every N optimizer steps")
    p.add_argument("--resume-from", help="A checkpoint-<step> folder written by an earlier run with the SAME settings: "
                                         "loads its LoRA weights and continues from that step (the optimizer restarts, "
                                         "with a short re-warm-up)")
    p.add_argument("--rewarm", type=int, default=10,
                   help="Re-warm-up steps after --resume-from, only when the checkpoint has no saved optimizer state")
    p.add_argument("--stop-at", type=int, default=0,
                   help="Save checkpoint-<N> and exit after optimizer step N; unlike --max-steps the schedule is unchanged "
                        "(training/train_in_chunks.py uses this to restart the process every few hundred steps)")
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps (0 = no limit; for speed tests)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def checkpoint_step(path: str) -> int:
    """The optimizer step a checkpoint-<step> folder was saved at."""
    name = os.path.basename(os.path.normpath(path))
    if not name.startswith("checkpoint-") or not name[len("checkpoint-"):].isdigit():
        raise ValueError(f"{path}: expected a folder named checkpoint-<step>")
    return int(name[len("checkpoint-"):])


def load_images(paths):
    if len(paths) == 2:
        a, _, b, _ = load_and_standardize_pair(paths[0], paths[1], "optical")
        return [Image.fromarray(a), Image.fromarray(b)]
    img, _ = load_and_standardize_image(paths[0], "optical")
    return [Image.fromarray(img)]


def encode(processor, pack, header_ids, im_start, im_end, image_cache):
    key = tuple(pack["image_paths"])
    if key not in image_cache:
        image_cache.clear()                     # packs are shuffled; keep at most one image set in memory
        image_cache[key] = load_images(pack["image_paths"])
    text = processor.apply_chat_template(pack_messages(pack), tokenize=False)
    batch = processor(text=[text], images=image_cache[key], return_tensors="pt")
    ids = batch["input_ids"][0].tolist()
    mask = assistant_label_mask(ids, im_start, im_end, header_ids)
    labels = torch.tensor([t if m else -100 for t, m in zip(ids, mask)]).unsqueeze(0)
    batch["labels"] = labels
    return batch


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = load_benchmark(args.benchmark, args.root, args.split, args.image_dir)
    spaces = build_answer_spaces((i.qtype, i.answer) for i in items)
    packs = build_packs(items, spaces, per_pack=args.per_pack, seed=args.seed)
    n_packs = int(len(packs) * args.epochs) if args.epochs <= 1 else len(packs) * math.ceil(args.epochs)
    schedule = (packs * math.ceil(max(args.epochs, 1)))[:n_packs]
    total_steps = max(1, n_packs // args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"{len(items)} questions from {args.benchmark}/{args.split} -> {len(packs)} packs; "
          f"training on {min(n_packs, total_steps * args.grad_accum)} packs = {total_steps} optimizer steps")

    processor = AutoProcessor.from_pretrained(BASE_MODEL, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    tok = processor.tokenizer
    im_start, im_end = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    header_ids = tok.encode("assistant\n", add_special_tokens=False)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    start_step = 0
    if args.resume_from:
        start_step = checkpoint_step(args.resume_from)
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        print(f"Resuming from {args.resume_from} at optimizer step {start_step}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=TARGET_MODULES,
            bias="none", task_type="CAUSAL_LM",
        ))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    optimizer_restored = False
    if args.resume_from and os.path.exists(os.path.join(args.resume_from, OPTIMIZER_FILE)):
        optimizer.load_state_dict(torch.load(os.path.join(args.resume_from, OPTIMIZER_FILE), map_location=model.device))
        optimizer_restored = True
        print("Optimizer state restored: continuing exactly, no re-warm-up")

    def lr_at(step):
        if start_step and not optimizer_restored and step < start_step + args.rewarm:
            # The optimizer's moment estimates were not saved, so ramp up again to the scheduled value.
            return scheduled_lr(step) * (step - start_step + 1) / args.rewarm
        return scheduled_lr(step)

    def scheduled_lr(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "training_log.jsonl")
    process = psutil.Process()
    info = {
        "base_model": BASE_MODEL, "quantization": "4-bit NF4, double quant, bf16 compute",
        "trained_on": {"benchmark": args.benchmark, "split": args.split, "root": os.path.abspath(args.root),
                       "questions": len(items), "images": len({tuple(i.image_paths) for i in items}),
                       "packs": len(packs), "packs_used": min(n_packs, total_steps * args.grad_accum)},
        "prompt_format": "evaluation.answer_space.benchmark_prompt (the harness's short-answer prompt)",
        "hyperparameters": {k: getattr(args, k) for k in ("epochs", "per_pack", "grad_accum", "lr", "warmup", "rank",
                                                          "alpha", "dropout", "seed")},
        "target_modules": TARGET_MODULES, "git_commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    info_path = os.path.join(args.out, "training_info.json")
    if start_step and os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = {**json.load(f), "git_commit": info["git_commit"]}
    if start_step:
        info.setdefault("resumed", []).append({"from": os.path.abspath(args.resume_from), "at_step": start_step,
                                               "utc": datetime.now(timezone.utc).isoformat()})

    def save_checkpoint(step):
        folder = os.path.join(args.out, f"checkpoint-{step}")
        model.save_pretrained(folder)
        torch.save(optimizer.state_dict(), os.path.join(folder, OPTIMIZER_FILE))
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({**info, "last_checkpoint": step}, f, indent=2)

    model.train()
    cache, step, running, t0 = {}, start_step, [], time.perf_counter()
    optimizer.zero_grad()
    first = start_step * args.grad_accum          # packs already trained on (same seed + settings = same order)
    with open(log_path, "a" if start_step else "w", encoding="utf-8") as log:
        for k, pack in enumerate(schedule[first: total_steps * args.grad_accum], first + 1):
            batch = encode(processor, pack, header_ids, im_start, im_end, cache).to(model.device)
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
            if k % args.grad_accum == 0:
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                mean_loss = sum(running) / len(running)
                running = []
                elapsed = time.perf_counter() - t0
                ram_gb = process.memory_info().rss / 2**30
                log.write(json.dumps({"step": step, "loss": round(mean_loss, 4), "lr": lr_at(step - 1),
                                      "elapsed_s": round(elapsed, 1), "ram_gb": round(ram_gb, 2)}) + "\n")
                log.flush()
                if step % 10 == 0 or step == total_steps:
                    eta = elapsed / (step - start_step) * (total_steps - step) / 60
                    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                    print(f"step {step}/{total_steps}  loss {mean_loss:.4f}  lr {lr_at(step - 1):.2e}  "
                          f"peak {mem:.2f} GiB GPU, {ram_gb:.1f} GiB RAM  ~{eta:.0f} min left", flush=True)
                if step < total_steps and (step % args.save_every == 0 or step == args.stop_at):
                    save_checkpoint(step)
                if step == args.stop_at and step < total_steps:
                    print(f"Stopped at step {step} (--stop-at); resume with --resume-from {args.out}/checkpoint-{step}")
                    return

    model.save_pretrained(args.out)
    info.update(finished_utc=datetime.now(timezone.utc).isoformat(), optimizer_steps=step,
                train_minutes=round((time.perf_counter() - t0) / 60, 1))
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Saved adapter to {args.out}")


if __name__ == "__main__":
    main()
