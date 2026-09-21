"""Turns a feature scan's 4x4 P(yes) grid into something a person can read (numpy/scipy/OpenCV, no model).

The scan itself is unchanged: each tile gets a probability from the adapter's yes/no question. This module
decides which tiles are CONFIDENT (>= threshold), groups touching ones into numbered findings, words the
result (headline, confidence, a warning when the scan says yes almost everywhere) and draws the evidence:
a soft heat tint that follows the probabilities, a smooth outline round each finding and a numbered pin.
Faint tiles (P 0.5 to the threshold) are only tinted, never outlined or counted as findings.
"""
from typing import Dict, List, Sequence

import cv2
import numpy as np
from scipy.ndimage import label

from change_detection.locations import describe_location

FAINT_FLOOR = 0.5            # below this a tile gets no tint at all
HIGH_CONFIDENCE = 0.9        # best tile of a finding at or above this reads "High confidence"
LIKELY = 0.75                # ... at or above this "Likely", else "Possible"
OVER_REPORTING_SHARE = 0.5   # this share of tiles at or above FAINT_FLOOR means the scan says yes to nearly everything

# What each scan looks for, in words a person would use, and its highlight colour (matches the UI tones)
SCAN_SUBJECTS = {
    "mining": "surface mining activity",
    "deforestation": "forest clearing",
    "agriculture": "cultivated land",
}
SCAN_COLOURS = {                # RGB
    "mining": (255, 150, 30),
    "deforestation": (240, 70, 90),
    "agriculture": (50, 200, 120),
}
DEFAULT_COLOUR = (255, 150, 30)


def confidence_label(peak: float) -> str:
    if peak >= HIGH_CONFIDENCE:
        return "High confidence"
    return "Likely" if peak >= LIKELY else "Possible"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def find_findings(probs: np.ndarray, threshold: float) -> List[Dict]:
    """One finding per group of edge-adjacent confident tiles, top-to-bottom / left-to-right."""
    probs = np.asarray(probs, dtype=float)
    rows, cols = probs.shape
    groups, count = label(probs >= threshold)            # 4-connectivity
    found = []
    for k in range(1, count + 1):
        rr, cc = np.where(groups == k)
        tile_p = probs[rr, cc]
        box = [float(rr.min()) / rows, float(cc.min()) / cols, float(rr.max() + 1) / rows, float(cc.max() + 1) / cols]
        cx, cy = (float(cc.mean()) + 0.5) / cols, (float(rr.mean()) + 0.5) / rows
        found.append({
            "box": [round(v, 3) for v in box],
            "tiles": [[int(r), int(c)] for r, c in zip(rr, cc)],
            "where": describe_location(cx, cy, 1.0, 1.0),
            "coverage_pct": round(100.0 * len(rr) / probs.size),
            "confidence": round(100.0 * float(tile_p.mean())),
            "peak": round(100.0 * float(tile_p.max())),
            "label": confidence_label(float(tile_p.max())),
        })
    found.sort(key=lambda f: (f["box"][0], f["box"][1]))
    for i, f in enumerate(found, 1):
        f["id"] = i
    return found


def over_reporting_note(probs: np.ndarray) -> str:
    """A warning when the scan answered yes for most of the view (a single feature rarely fills it)."""
    probs = np.asarray(probs, dtype=float)
    yes = int((probs >= FAINT_FLOOR).sum())
    if yes / probs.size < OVER_REPORTING_SHARE:
        return ""
    return (f"The scan answered yes for {yes} of {probs.size} areas, which is more than one feature usually covers, "
            "so it may be over-reporting. Treat the highlights as a starting point and check them by eye.")


def build_scan_summary(adapter: str, probs: np.ndarray, threshold: float) -> Dict:
    """Everything the UI and the PDF show about a scan, as plain JSON."""
    probs = np.asarray(probs, dtype=float)
    subject = SCAN_SUBJECTS.get(adapter, "the target feature")
    findings = find_findings(probs, threshold)
    faint = int(((probs >= FAINT_FLOOR) & (probs < threshold)).sum())
    note = over_reporting_note(probs)
    covered = sum(len(f["tiles"]) for f in findings)
    coverage = round(100.0 * covered / probs.size)

    if findings:
        top = max(f["peak"] for f in findings) / 100.0
        level = "high" if top >= HIGH_CONFIDENCE else "likely" if top >= LIKELY else "possible"
        where = _plural(len(findings), "area")
        headline = {
            "high": f"{subject[0].upper()}{subject[1:]} found in {where} (high confidence)",
            "likely": f"Likely {subject} in {where}",
            "possible": f"Possible {subject} in {where}",
        }[level]
        headline += f", covering about {coverage}% of the view."
    else:
        level = "none"
        headline = f"No confident {subject} found."
        if faint:
            headline += f" {_plural(faint, 'faint area')} scored above chance and {'is' if faint == 1 else 'are'} tinted lightly."

    return {
        "adapter": adapter,
        "subject": subject,
        "level": level,                                   # none | possible | likely | high
        "headline": headline,
        "note": note,
        "coverage_pct": coverage,
        "faint_areas": faint,
        "threshold": threshold,
        "findings": findings,
        "grid": [[round(float(v), 2) for v in row] for row in probs],
    }


def scan_answer_text(summary: Dict) -> str:
    """The chat / PDF sentence: the headline plus the warning, if any."""
    return f"{summary['headline']} {summary['note']}".strip()


def _pin_tile(finding: Dict, rows: int, cols: int):
    """The tile of a finding nearest its centre, so a pin never lands outside an L-shaped group."""
    tiles = np.array(finding["tiles"], dtype=float)
    mean = tiles.mean(axis=0)
    r, c = tiles[np.argmin(((tiles - mean) ** 2).sum(axis=1))]
    return (c + 0.5) / cols, (r + 0.5) / rows


def render_scan_evidence(img_rgb: np.ndarray, probs: np.ndarray, summary: Dict) -> np.ndarray:
    """The image with a probability-shaped tint, a rounded outline per finding and numbered pins."""
    img = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    probs = np.asarray(probs, dtype=np.float32)
    rows, cols = probs.shape
    colour = np.array(SCAN_COLOURS.get(summary["adapter"], DEFAULT_COLOUR), dtype=np.float32)
    side = float(max(h, w))

    # Tint: smooth field from the 4x4 grid; 0 at P=0.5 up to 40% at P=1, so faint tiles stay faint
    field = cv2.GaussianBlur(cv2.resize(probs, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), side / 60.0)
    alpha = (np.clip((field - FAINT_FLOOR) / (1.0 - FAINT_FLOOR), 0.0, 1.0) * 0.40)[..., None]
    out = img.astype(np.float32) * (1.0 - alpha) + colour * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    line = max(2, int(round(side / 400.0)))
    for finding in summary["findings"]:
        mask = np.zeros((rows, cols), np.float32)
        for r, c in finding["tiles"]:
            mask[r, c] = 1.0
        smooth = cv2.GaussianBlur(cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST), (0, 0), side / 45.0)
        contours, _ = cv2.findContours((smooth > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), line + 2, cv2.LINE_AA)   # halo keeps it readable on any ground
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in colour), line, cv2.LINE_AA)

        px, py = _pin_tile(finding, rows, cols)
        centre = (int(px * w), int(py * h))
        radius = max(11, int(side / 40.0))
        cv2.circle(out, centre, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, centre, radius, tuple(int(v * 0.8) for v in colour), -1, cv2.LINE_AA)
        text = str(finding["id"])
        scale = radius / 20.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, line - 1))
        cv2.putText(out, text, (centre[0] - tw // 2, centre[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, line - 1), cv2.LINE_AA)
    return out


def summary_lines(summary: Dict) -> Sequence[str]:
    """Plain lines for the PDF's first page: one per finding."""
    lines = []
    for f in summary["findings"]:
        lines.append(f"{f['id']}. {f['label']} - {f['where']}, {_plural(len(f['tiles']), 'tile')} "
                     f"(about {f['coverage_pct']}% of the view), average score {f['confidence']}%")
    return lines
