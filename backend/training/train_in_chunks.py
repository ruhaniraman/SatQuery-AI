#!/usr/bin/env python
"""Run train_vqa_lora in fresh processes of --chunk optimizer steps each, resuming from the newest checkpoint every
time. Why: the trainer's resident memory grows by ~0.03 GB per step on this Windows laptop (3.6 -> 7.3 GB over 110
steps, cause not found), which starved the machine and got a run killed at step 549. A new process every 100 steps
keeps it bounded; checkpoints carry the optimizer state, so the chunks continue the same run exactly.

    python -m training.train_in_chunks --chunk 100 -- --root data/benchmarks/RSVQA_LR --out models/adapters/vqa \\
        --per-pack 16 --grad-accum 4 --epochs 1 --lr 1e-4

Everything after `--` goes to train_vqa_lora unchanged (use the SAME settings as the run being continued).
"""
import argparse
import json
import os
import re
import subprocess
import sys


def latest_checkpoint(out_dir: str):
    """(step, path) of the highest checkpoint-<step> folder that holds adapter weights, or (0, None)."""
    best = (0, None)
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            path = os.path.join(out_dir, name)
            if m and os.path.exists(os.path.join(path, "adapter_model.safetensors")) and int(m.group(1)) > best[0]:
                best = (int(m.group(1)), path)
    return best


def finished(out_dir: str) -> bool:
    try:
        with open(os.path.join(out_dir, "training_info.json"), encoding="utf-8") as f:
            return "finished_utc" in json.load(f)
    except (OSError, ValueError):
        return False


def value_of(flag: str, argv):
    return argv[argv.index(flag) + 1] if flag in argv else None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, train_args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--max-chunks", type=int, default=100)
    args = p.parse_args(ours)
    out_dir = value_of("--out", train_args)
    if not out_dir:
        sys.exit("Pass the trainer's --out after `--`.")

    for _ in range(args.max_chunks):
        if finished(out_dir):
            print(f"Training finished: adapter in {out_dir}")
            return 0
        step, path = latest_checkpoint(out_dir)
        cmd = [sys.executable, "-u", "-m", "training.train_vqa_lora", *train_args, "--stop-at", str(step + args.chunk)]
        if path:
            cmd += ["--resume-from", path]
        print(f"=== chunk: steps {step} -> {step + args.chunk} ===", flush=True)
        code = subprocess.call(cmd)
        if code != 0:
            print(f"Trainer exited with code {code}; stopping. Re-run this command to continue from the last checkpoint.")
            return code
    print("Reached --max-chunks without finishing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
