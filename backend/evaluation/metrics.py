"""Benchmark summary statistics (pure): overall and per-type accuracy with Wilson 95% intervals, the average
over question types (AA, the number papers report next to overall accuracy), calibration of the confidence
score, and latency. `records` are the dicts the runner writes, one per item: at least qtype, correct (bool),
and optionally confidence (0-1 or None), latency_s, error."""
import math
from collections import defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

Z95 = 1.959964


def wilson_interval(correct: int, total: int, z: float = Z95):
    """95% Wilson score interval for a proportion; (0, 0) for an empty set. Better than the normal
    approximation at small n or near 0% / 100%, which is exactly where small benchmark runs land."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _accuracy(rows: List[dict]) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r["correct"])
    lo, hi = wilson_interval(k, n)
    return {"n": n, "correct": k, "accuracy": (k / n) if n else 0.0, "ci95": [lo, hi]}


def calibration(rows: Iterable[dict], bins: int = 10) -> Optional[dict]:
    """Expected / maximum calibration error and Brier score of `confidence` against correctness, over the
    rows that have a confidence. None when no row has one."""
    scored = [(float(r["confidence"]), 1.0 if r["correct"] else 0.0) for r in rows if r.get("confidence") is not None]
    if not scored:
        return None
    buckets = defaultdict(list)
    for conf, hit in scored:
        buckets[min(int(conf * bins), bins - 1)].append((conf, hit))
    ece, mce = 0.0, 0.0
    for members in buckets.values():
        gap = abs(mean(c for c, _ in members) - mean(h for _, h in members))
        ece += gap * len(members) / len(scored)
        mce = max(mce, gap)
    return {
        "n": len(scored),
        "ece": ece,
        "mce": mce,
        "brier": mean((c - h) ** 2 for c, h in scored),
        "mean_confidence": mean(c for c, _ in scored),
        "note": "confidence = mean generation-token probability; ECE/MCE over 10 equal-width bins",
    }


def summarize(records: List[dict], exclude_types_from_aa: Iterable[str] = ()) -> Dict:
    """Items that failed to run (error set) count as wrong in accuracy AND are counted separately, so a
    broken run can never look like a good one by silently dropping its failures."""
    records = list(records)
    by_type = defaultdict(list)
    for r in records:
        by_type[r["qtype"]].append(r)
    per_type = {t: _accuracy(rows) for t, rows in sorted(by_type.items())}
    excluded = set(exclude_types_from_aa)
    aa_types = [t for t in per_type if t not in excluded]
    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None and not r.get("error")]
    return {
        "overall": _accuracy(records),
        "average_over_types": mean(per_type[t]["accuracy"] for t in aa_types) if aa_types else 0.0,
        "average_over_types_excludes": sorted(excluded & set(per_type)),
        "per_type": per_type,
        "errors": sum(1 for r in records if r.get("error")),
        "calibration": calibration(r for r in records if not r.get("error")),
        "latency_s": {"mean": mean(latencies), "median": median(latencies), "max": max(latencies)} if latencies else None,
    }


def majority_baseline(records: List[dict], exclude_types_from_aa: Iterable[str] = ()) -> Dict:
    """The answer-prior floor: always give the most common answer of each question type (computed on these
    same items, so it is an optimistic floor). A model that does not beat this has learned nothing from
    the image. Needs `answer` on each record."""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["qtype"]].append(r["answer"])
    rows = []
    for qtype, answers in by_type.items():
        top = max(set(answers), key=answers.count)
        rows.extend({"qtype": qtype, "correct": a == top} for a in answers)
    return summarize(rows, exclude_types_from_aa)
