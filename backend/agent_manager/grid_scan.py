"""Pure helpers for the tile yes/no scan (numpy/scipy only, no model)."""
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import label

# Fraction of a tile's size added as extra image context on every side when the tile is shown to
# the model. The verdict still belongs to the un-padded cell, but a feature that crosses a cell
# boundary is now visible whole from both neighbouring cells.
DEFAULT_CONTEXT = 0.25

# Probability of "yes" (vs "no") at or above which a tile counts as positive. A guess to tune.
DEFAULT_YES_THRESHOLD = 0.5


def tile_windows(width: int, height: int, rows: int, cols: int,
                 context: float = DEFAULT_CONTEXT) -> List[Tuple[int, int, Tuple[int, int, int, int]]]:
    """(row, col, (left, top, right, bottom)) for every cell, crop box padded by `context` and
    clamped to the image. Cells tile the whole image, including the remainder pixels."""
    out = []
    for r in range(rows):
        for c in range(cols):
            left, right = c * width // cols, (c + 1) * width // cols
            top, bottom = r * height // rows, (r + 1) * height // rows
            pad_x = int(round((right - left) * context))
            pad_y = int(round((bottom - top) * context))
            out.append((r, c, (max(0, left - pad_x), max(0, top - pad_y),
                               min(width, right + pad_x), min(height, bottom + pad_y))))
    return out


def yes_probability(last_logits: np.ndarray, yes_ids: Sequence[int], no_ids: Sequence[int]) -> np.ndarray:
    """P(yes) per row from next-token logits, comparing only the yes-like vs no-like tokens
    (e.g. "yes"/"Yes" vs "no"/"No"), so probability mass on unrelated tokens is ignored.

    last_logits: (batch, vocab). Returns (batch,) floats in [0, 1]."""
    logits = np.asarray(last_logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]

    def logsumexp(ids):
        sel = logits[:, list(ids)]
        m = sel.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(sel - m).sum(axis=1, keepdims=True)))[:, 0]

    ly, ln = logsumexp(yes_ids), logsumexp(no_ids)
    return 1.0 / (1.0 + np.exp(ln - ly))


def merge_positive_tiles(positive: np.ndarray) -> List[List[float]]:
    """Merge edge-adjacent positive cells into one region each.

    positive: (rows, cols) bool. Returns boxes [ymin, xmin, ymax, xmax] normalised to 0..1 (the
    bounding box of each connected group, so an L-shaped group is covered by its bounding box)."""
    positive = np.asarray(positive, dtype=bool)
    rows, cols = positive.shape
    labels, n = label(positive)          # default structure = 4-connectivity (no diagonals)
    boxes = []
    for k in range(1, n + 1):
        rr, cc = np.where(labels == k)
        boxes.append([
            round(float(rr.min()) / rows, 3), round(float(cc.min()) / cols, 3),
            round(float(rr.max() + 1) / rows, 3), round(float(cc.max() + 1) / cols, 3),
        ])
    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes
