"""Benchmark runner: asks every item, scores it, and writes <out_dir>/predictions.jsonl (one line per item, flushed
as it goes, so an interrupted run resumes where it stopped), summary.json and summary.md. The model is only
touched through a `predictor(item, prompt) -> (reply, confidence)` callable, so the tests drive this with a
fake and `run_eval.py` passes the real engine (see ModelPredictor)."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from evaluation.answer_space import AnswerSpace, benchmark_prompt, build_answer_spaces
from evaluation.benchmarks import EvalItem
from evaluation.metrics import majority_baseline, summarize

Predictor = Callable[[EvalItem, str], Tuple[str, Optional[float]]]

# GeoChat and the RSVQA papers leave out count and area questions when they report RSVQA accuracy (the answers
# are open numbers), so that figure is reported separately to be comparable with them.
PAPER_EXCLUDED_TYPES = {"rsvqa-lr": {"count", "area"}, "rsvqa-hr": {"count", "area"}}


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")
    except Exception:
        return None


def _read_existing(path: str) -> Dict[str, dict]:
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done[record["item_id"]] = record
    return done


def run_benchmark(
    items: List[EvalItem],
    predictor: Predictor,
    out_dir: str,
    config: dict,
    spaces: Optional[Dict[str, AnswerSpace]] = None,
    progress_every: int = 25,
    log: Callable[[str], None] = print,
) -> dict:
    """`spaces` should be built from the WHOLE split (build_answer_spaces over all loaded items), not only the
    sampled items, so the answer options shown never depend on which items were drawn. `config` must hold
    benchmark, split, limit, seed and predictor; a resumed run must use the same config."""
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "config.json")
    predictions_path = os.path.join(out_dir, "predictions.jsonl")
    identity = {k: config.get(k) for k in ("benchmark", "split", "limit", "seed", "predictor", "adapter")}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            previous = json.load(f)
        old_identity = {k: previous.get(k) for k in identity}
        if old_identity != identity:
            raise ValueError(f"{out_dir} holds a different run ({old_identity}); use another --out or delete it.")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({**config, "git_commit": _git_commit(), "started_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)

    spaces = spaces or build_answer_spaces((i.qtype, i.answer) for i in items)
    done = _read_existing(predictions_path)
    todo = [i for i in items if i.item_id not in done]
    if done:
        log(f"Resuming: {len(done)} already answered, {len(todo)} to go.")

    started = time.perf_counter()
    with open(predictions_path, "a", encoding="utf-8") as out:
        for n, item in enumerate(todo, 1):
            space = spaces[item.qtype]
            prompt = benchmark_prompt(item.question, space, pair=len(item.image_paths) == 2)
            record = {"item_id": item.item_id, "qtype": item.qtype, "question": item.question, "answer": item.answer}
            t0 = time.perf_counter()
            try:
                missing = [p for p in item.image_paths if not os.path.exists(p)]
                if missing:
                    raise FileNotFoundError(f"image not found: {missing[0]}")
                reply, confidence = predictor(item, prompt)
                record.update(reply=reply, predicted=space.canonical(reply), confidence=confidence,
                              correct=space.is_correct(reply, item.answer))
            except Exception as e:
                record.update(reply=None, predicted=None, confidence=None, correct=False, error=f"{type(e).__name__}: {e}")
            record["latency_s"] = round(time.perf_counter() - t0, 3)
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[item.item_id] = record
            if n % progress_every == 0 or n == len(todo):
                rate = (time.perf_counter() - started) / n
                acc = sum(1 for r in done.values() if r["correct"]) / len(done)
                log(f"  {len(done)}/{len(items)}  running accuracy {acc:.1%}  ~{rate:.2f} s/item  "
                    f"~{rate * (len(todo) - n) / 60:.0f} min left")

    wanted = {i.item_id for i in items}
    records = [r for r in done.values() if r["item_id"] in wanted]
    excluded = PAPER_EXCLUDED_TYPES.get(config.get("benchmark"), set())
    summary = {
        "config": identity,
        "model": summarize(records),
        "answer_prior_baseline": majority_baseline(records),
    }
    if excluded:
        comparable = [r for r in records if r["qtype"] not in excluded]
        summary["paper_protocol"] = {
            "excludes": sorted(excluded),
            "model": summarize(comparable),
            "answer_prior_baseline": majority_baseline(comparable),
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(format_summary_markdown(summary, config))
    return summary


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def format_summary_markdown(summary: dict, config: dict) -> str:
    model, base = summary["model"], summary["answer_prior_baseline"]
    o = model["overall"]
    lines = [
        f"# {config.get('benchmark')} ({config.get('split') or 'default split'}): {config.get('predictor')}"
        + (f" + {config['adapter']}" if config.get("adapter") else ""),
        "",
        f"- Items: {o['n']} (limit {config.get('limit') or 'all'}, seed {config.get('seed')}), failed to run: {model['errors']}",
        f"- **Overall accuracy: {_pct(o['accuracy'])}** (95% CI {_pct(o['ci95'][0])} to {_pct(o['ci95'][1])})",
        f"- Average over question types: {_pct(model['average_over_types'])}",
        f"- Answer-prior floor (most common answer per type): {_pct(base['overall']['accuracy'])}",
    ]
    if "paper_protocol" in summary:
        pp = summary["paper_protocol"]
        lines.append(f"- Paper protocol (without {', '.join(pp['excludes'])}): "
                     f"{_pct(pp['model']['overall']['accuracy'])} (floor {_pct(pp['answer_prior_baseline']['overall']['accuracy'])})")
    cal = model.get("calibration")
    if cal:
        lines.append(f"- Confidence calibration: ECE {cal['ece']:.3f}, Brier {cal['brier']:.3f}, "
                     f"mean confidence {cal['mean_confidence']:.2f} (n={cal['n']})")
    if model.get("latency_s"):
        lines.append(f"- Latency: mean {model['latency_s']['mean']:.2f} s, median {model['latency_s']['median']:.2f} s")
    if config.get("metric_note"):
        lines.append(f"- Note: {config['metric_note']}")
    lines += ["", "| Question type | n | Accuracy | 95% CI | Floor |", "|---|---|---|---|---|"]
    for qtype, row in model["per_type"].items():
        floor = base["per_type"].get(qtype, {}).get("accuracy", 0.0)
        lines.append(f"| {qtype} | {row['n']} | {_pct(row['accuracy'])} | {_pct(row['ci95'][0])}-{_pct(row['ci95'][1])} | {_pct(floor)} |")
    return "\n".join(lines) + "\n"


class ModelPredictor:
    """The real engine behind the runner. Images go through the SAME loader the product uses (percentile
    stretch; a pair shares one stretch), so the score is the system's, not a differently-fed model's."""

    def __init__(self, engine, adapter: Optional[str] = None, modality: str = "optical", max_new_tokens: int = 16):
        self.engine, self.adapter, self.modality, self.max_new_tokens = engine, adapter, modality, max_new_tokens

    def __call__(self, item: EvalItem, prompt: str):
        from PIL import Image
        from geospatial_preprocessing.geotiff_loader import load_and_standardize_image, load_and_standardize_pair

        if len(item.image_paths) == 2:
            img_a, _, img_b, _ = load_and_standardize_pair(item.image_paths[0], item.image_paths[1], self.modality)
            images = [Image.fromarray(img_a), Image.fromarray(img_b)]
        else:
            img, _ = load_and_standardize_image(item.image_paths[0], self.modality)
            images = [Image.fromarray(img)]
        return self.engine.short_answer(images, prompt, adapter=self.adapter, max_new_tokens=self.max_new_tokens)
