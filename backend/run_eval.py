#!/usr/bin/env python
"""Score SatQuery-AI on the public benchmarks SIH26167 names. Run from backend/ on the machine with the GPU:

    python run_eval.py rsvqa-lr --root D:/benchmarks/RSVQA_LR --limit 500
    python run_eval.py rsvqa-hr --root D:/benchmarks/RSVQA_HR --image-dir D:/benchmarks/RSVQA_HR/Data --limit 500
    python run_eval.py cdvqa --root D:/benchmarks/CDVQA --image-dir D:/benchmarks/SECOND/test --limit 500
    python run_eval.py vrsbench-vqa --root D:/benchmarks/VRSBench --limit 500
    python run_eval.py rsvqa-lr --root ... --limit 500 --predictor prior      # the answer-prior floor, no GPU

Results go to eval_results/<benchmark>-<split>-<predictor>-n<limit>-s<seed>/ (config.json, predictions.jsonl,
summary.json, summary.md). Re-running the same command resumes. See evaluation/benchmarks.py for the expected
file layouts and where each dataset is downloaded from.
"""
import argparse
import os
import sys

from evaluation.answer_space import build_answer_spaces
from evaluation.benchmarks import BENCHMARKS, load_benchmark, stratified_sample
from evaluation.runner import ModelPredictor, run_benchmark

METRIC_NOTES = {
    "vrsbench-vqa": "strict string match after normalization; the official VRSBench VQA score is GPT-judged "
                    "(synonyms count), so this is a lower bound and NOT directly comparable with published numbers.",
    "cdvqa": "exact match against CDVQA's answer vocabulary (the before/after pair is given to the model as two images).",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=BENCHMARKS)
    p.add_argument("--root", required=True, help="Folder holding the benchmark's annotation files")
    p.add_argument("--image-dir", help="Image folder, if not the default inside --root (CDVQA: the SECOND folder with im1/ and im2/)")
    p.add_argument("--split", help="Split name (default: test for RSVQA, Test for CDVQA)")
    p.add_argument("--limit", type=int, default=500, help="Stratified sample size; 0 = the whole split (default 500)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--predictor", choices=["model", "prior"], default="model",
                   help="model = the real engine; prior = always the most common answer per question type (no GPU)")
    p.add_argument("--adapter", help="Answer with this loaded LoRA instead of the base weights (for base-vs-adapted runs)")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--out", help="Output folder (default: eval_results/<run name>)")
    return p.parse_args(argv)


def run_name(args) -> str:
    who = args.predictor if not args.adapter else f"{args.predictor}-{args.adapter}"
    return f"{args.benchmark}-{args.split or 'default'}-{who}-n{args.limit or 'all'}-s{args.seed}"


def main(argv=None):
    args = parse_args(argv)
    try:
        all_items = load_benchmark(args.benchmark, args.root, args.split, args.image_dir)
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"Could not load {args.benchmark} from {args.root}: {e}")
    if not all_items:
        sys.exit(f"No active questions found for {args.benchmark} in {args.root}.")
    spaces = build_answer_spaces((i.qtype, i.answer) for i in all_items)
    items = stratified_sample(all_items, args.limit, args.seed)
    print(f"{args.benchmark}: {len(all_items)} questions in the split, evaluating {len(items)}.")

    if args.predictor == "prior":
        # Most common answer per type over the WHOLE split: a floor that never looks at the image.
        from collections import Counter
        top = {t: Counter(i.answer for i in all_items if i.qtype == t).most_common(1)[0][0] for t in spaces}
        predictor = lambda item, prompt: (top[item.qtype], None)  # noqa: E731
    else:
        from agent_manager.agent_controller import load_agent
        predictor = ModelPredictor(load_agent(), adapter=args.adapter, max_new_tokens=args.max_new_tokens)

    out_dir = args.out or os.path.join("eval_results", run_name(args))
    config = {
        "benchmark": args.benchmark, "split": args.split, "limit": args.limit, "seed": args.seed,
        "predictor": args.predictor, "adapter": args.adapter, "root": os.path.abspath(args.root),
        "image_dir": args.image_dir, "max_new_tokens": args.max_new_tokens,
        "model": "Qwen/Qwen2-VL-2B-Instruct (4-bit)" + (f" + LoRA {args.adapter}" if args.adapter else ", adapters disabled")
                 if args.predictor == "model" else None,
        "metric_note": METRIC_NOTES.get(args.benchmark),
    }
    summary = run_benchmark(items, predictor, out_dir, config, spaces=spaces)
    with open(os.path.join(out_dir, "summary.md"), encoding="utf-8") as f:
        print("\n" + f.read())
    print(f"Saved to {out_dir}")
    return summary


if __name__ == "__main__":
    main()
