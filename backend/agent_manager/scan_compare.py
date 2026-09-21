"""Compare a LoRA tile scan run on two dates. Pure numpy/Python (no model), so it is unit-testable.

The scan gives P(yes) per grid cell for a fixed question ("is there a mining pit?"). Comparing the two
dates' grids says WHERE the answer flipped, which is located evidence the base model cannot produce
by reading a stitched image. The scores come from adapters whose training data is not recorded and
whose accuracy is not validated, and every line built here says so via the caller's header.
"""
from typing import Dict, List

import numpy as np

from agent_manager.grid_scan import DEFAULT_YES_THRESHOLD
from change_detection.locations import describe_location

# A cell only counts as newly positive / no longer positive if its score also moved by at least this
# much, so a cell hovering around the threshold does not flip on noise.
MIN_SHIFT = 0.4

# When at least this share of tiles is positive on either date the scan is 'saturated' (it says yes almost
# everywhere), so a flip of a couple of tiles is within noise and the line says so.
SATURATED_SHARE = 0.5

LABELS = {
    "mining": "mining / extraction pits",
    "deforestation": "cleared or logged land",
    "agriculture": "cultivated fields",
}


def _cells(mask: np.ndarray) -> List[str]:
    rows, cols = mask.shape
    # A coarse word (3x3 areas) alone is ambiguous on a 4x4 grid: two different cells can share it.
    return [f"{describe_location(c + 0.5, r + 0.5, cols, rows)} (row {r + 1}, col {c + 1})"
            for r, c in zip(*np.nonzero(mask))]


def _coords(mask: np.ndarray) -> List[tuple]:
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]


def summarize_scan_change(name: str, before, after, threshold: float = DEFAULT_YES_THRESHOLD,
                          min_shift: float = MIN_SHIFT) -> Dict[str, object]:
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.shape != after.shape or before.ndim != 2 or before.size == 0:
        raise ValueError("scan grids must be 2-D and the same shape")
    was, now = before >= threshold, after >= threshold
    gained = now & ~was & ((after - before) >= min_shift)
    lost = was & ~now & ((before - after) >= min_shift)
    return {
        "name": name,
        "total": int(before.size),
        "positive_before": int(was.sum()),
        "positive_after": int(now.sum()),
        "grid": tuple(int(n) for n in before.shape),
        "gained": _cells(gained),
        "lost": _cells(lost),
        "gained_cells": _coords(gained),
        "lost_cells": _coords(lost),
        "crossed_up": int((now & ~was).sum()),
        "crossed_down": int((was & ~now).sum()),
        "saturated": bool(max(was.sum(), now.sum()) >= SATURATED_SHARE * before.size),
        "mean_before": round(float(before.mean()), 2),
        "mean_after": round(float(after.mean()), 2),
    }


def _join(places: List[str]) -> str:
    return ", ".join(places) if places else "none"


def format_scan_change(summary: Dict[str, object]) -> str:
    """One line per scan. Rows count from the top, columns from the left.

    Tiles that crossed the 50% line but moved by less than MIN_SHIFT are counted, not listed, and the
    line says so: otherwise "5 tiles positive after" next to two listed cells looks like a mistake."""
    label = LABELS.get(str(summary["name"]), str(summary["name"]))
    text = (
        f"{label}: {summary['positive_before']} of {summary['total']} tiles positive before, "
        f"{summary['positive_after']} after. Clearly newly positive (score up by {MIN_SHIFT:.1f} or more): "
        f"{_join(summary['gained'])}. Clearly dropped: {_join(summary['lost'])}."
    )
    quiet_up = int(summary["crossed_up"]) - len(summary["gained"])
    quiet_down = int(summary["crossed_down"]) - len(summary["lost"])
    if quiet_up > 0:
        text += f" {quiet_up} more tile(s) crossed to positive by a smaller margin."
    if quiet_down > 0:
        text += f" {quiet_down} more tile(s) crossed to negative by a smaller margin."
    if summary["saturated"]:
        text += " Already positive on many tiles, so this scan separates the dates poorly."
    return text
