"""Cheap, deterministic measurements of an image (or a before/after pair), handed to the model as
labelled facts.

Pure numpy (no torch/cv2), so it is unit-testable. The numbers describe PIXELS, not materials:
"green-dominant" is not "vegetation" and "dark" is not "water". The prompt tells the model so.
The point is to give a 2B model a few things it is bad at estimating itself (proportions, how
busy the scene is, whether one date is greener than the other) so its answer can be specific
without inventing figures.
"""
from typing import Dict, Optional

import numpy as np

# 8-bit thresholds. Starting guesses, not tuned values (see CLAUDE.md).
GREEN_MARGIN = 20        # 2G - R - B above this counts as green-dominant
DARK_MAX = 50            # gray at or below this counts as dark
BRIGHT_MIN = 200         # gray at or above this counts as bright
EDGE_MIN = 25            # gradient magnitude above this counts as an edge pixel
MAX_SIDE = 512           # measured on a downsample; proportions do not need full resolution
# Two images of one place should have a similar level of detail. Edge share differing by more than this
# factor (and by at least EDGE_MIN_GAP points) usually means different zoom / resolution / sharpness.
EDGE_RATIO_WARN = 2.0
EDGE_MIN_GAP = 5
REGION_COLOUR_MIN = 12       # an average colour channel must move this much (when brightness did not) to be mentioned
REGION_BRIGHTNESS_MIN = 15   # average brightness (0-255) must move this much to be mentioned for one area
CHANGE_MIN_POINTS = 3    # a before/after difference smaller than this many percentage points is "about the same"


def _downsample(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    step = max(1, max(h, w) // MAX_SIDE)
    return arr[::step, ::step]


def compute_scene_stats(rgb: np.ndarray, modality: str = "optical",
                        valid_mask: Optional[np.ndarray] = None) -> Optional[Dict[str, object]]:
    """Proportions and texture of a uint8 (H, W, 3) image, or None if the input is unusable.

    valid_mask: (H, W) bool, True where the pixel is real data. Alignment leaves black borders that
    would otherwise read as a large 'dark' share on one date only."""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
        return None
    height, width = rgb.shape[:2]
    small = _downsample(rgb[:, :, :3]).astype(np.float32)
    if valid_mask is not None and valid_mask.shape == (height, width):
        valid = _downsample(valid_mask).astype(bool)
    else:
        valid = np.ones(small.shape[:2], dtype=bool)
    if not valid.any():
        return None

    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(gray)
    edges = np.hypot(gx, gy) > EDGE_MIN

    def pct(mask: np.ndarray) -> int:
        return int(round(100.0 * float(mask[valid].mean())))

    stats: Dict[str, object] = {
        "width_px": int(width),
        "height_px": int(height),
        "modality": modality,
        "dark_pct": pct(gray <= DARK_MAX),
        "bright_pct": pct(gray >= BRIGHT_MIN),
        "mean_brightness": int(round(float(gray[valid].mean()))),
        "edge_pct": pct(edges),
        "mean_rgb": tuple(int(round(float(channel[valid].mean()))) for channel in (r, g, b)),
    }
    if modality != "sar":
        # SAR brightness is backscatter, so a colour-based measure would be meaningless there
        stats["green_dominant_pct"] = pct((2 * g - r - b) > GREEN_MARGIN)
    return stats


def _texture_word(edge_pct: int) -> str:
    if edge_pct < 3:
        return "very smooth (few edges)"
    if edge_pct < 10:
        return "mostly smooth"
    if edge_pct < 25:
        return "moderately detailed"
    return "highly detailed / busy"


def format_scene_stats(stats: Optional[Dict[str, object]]) -> str:
    """One compact block for the prompt; empty string when there is nothing to report."""
    if not stats:
        return ""
    parts = [f"image {stats['width_px']}x{stats['height_px']} px"]
    if "green_dominant_pct" in stats:
        parts.append(f"{stats['green_dominant_pct']}% of pixels green-dominant")
    parts.append(f"{stats['dark_pct']}% dark pixels")
    parts.append(f"{stats['bright_pct']}% bright pixels")
    parts.append(f"texture {_texture_word(int(stats['edge_pct']))}")
    return "; ".join(parts) + "."


def _shift(name: str, before: int, after: int) -> str:
    delta = after - before
    if abs(delta) < CHANGE_MIN_POINTS:
        return f"{name} about the same ({before}% -> {after}%)"
    word = "up" if delta > 0 else "down"
    return f"{name} {word} ({before}% -> {after}%)"


def format_pair_stats(before: Optional[Dict[str, object]], after: Optional[Dict[str, object]]) -> str:
    """Before -> after shifts of the same measures, or "" when either side could not be measured."""
    if not before or not after:
        return ""
    parts = []
    if "green_dominant_pct" in before and "green_dominant_pct" in after:
        parts.append(_shift("green-dominant pixels", int(before["green_dominant_pct"]), int(after["green_dominant_pct"])))
    parts.append(_shift("dark pixels", int(before["dark_pct"]), int(after["dark_pct"])))
    parts.append(_shift("bright pixels", int(before["bright_pct"]), int(after["bright_pct"])))
    parts.append(_shift("edge/detail pixels", int(before["edge_pct"]), int(after["edge_pct"])))
    return "; ".join(parts) + "."


def comparability_warning(before: Optional[Dict[str, object]], after: Optional[Dict[str, object]]) -> str:
    """A sentence when the pair is probably not comparable pixel to pixel, else "".

    Without this a zoom or resolution mismatch reads as "no change": the detector removes the overall
    shift and looks for local differences, and nothing lines up well enough to find any."""
    if not before or not after:
        return ""
    first, second = int(before["edge_pct"]), int(after["edge_pct"])
    low, high = sorted((first, second))
    if high - low < EDGE_MIN_GAP or high < EDGE_RATIO_WARN * max(low, 1):
        return ""
    finer = "second" if second > first else "first"
    return (
        f"The two images differ strongly in level of detail ({first}% vs {second}% edge pixels; the {finer} is much "
        "sharper or closer). That usually means different zoom, resolution or sharpness, so they cannot be compared "
        "pixel to pixel: 'no change' may be false and any located change may be misaligned. Capture both at the "
        "same zoom and position."
    )


def format_region_shift(before: Optional[Dict[str, object]], after: Optional[Dict[str, object]]) -> str:
    """Short before -> after measurement of ONE flagged area, e.g.
    "green-dominant pixels 71% -> 22%; average brightness 62 -> 138".

    Only the measures that actually moved are listed. This is counted from the pixels, so unlike a model
    caption it cannot invent things; it says nothing about materials. "" when either side is unmeasured."""
    if not before or not after:
        return ""
    parts = []
    if "green_dominant_pct" in before and "green_dominant_pct" in after:
        b, a = int(before["green_dominant_pct"]), int(after["green_dominant_pct"])
        if abs(a - b) >= CHANGE_MIN_POINTS:
            parts.append(f"green-dominant pixels {b}% -> {a}%")
    b, a = int(before["bright_pct"]), int(after["bright_pct"])
    if abs(a - b) >= CHANGE_MIN_POINTS:
        parts.append(f"bright pixels {b}% -> {a}%")
    b, a = int(before["mean_brightness"]), int(after["mean_brightness"])
    if abs(a - b) >= REGION_BRIGHTNESS_MIN:
        parts.append(f"average brightness {b} -> {a} (0-255)")
    elif "mean_rgb" in before and "mean_rgb" in after and (
            max(abs(x - y) for x, y in zip(before["mean_rgb"], after["mean_rgb"])) >= REGION_COLOUR_MIN):
        # a change of tone (e.g. paler or redder ground) with no change in overall brightness or green share
        parts.append("average colour (R,G,B) {} -> {}".format(",".join(map(str, before["mean_rgb"])),
                                                              ",".join(map(str, after["mean_rgb"]))))
    return "; ".join(parts) if parts else "averages over this area barely moved; the change is patchy or local"
