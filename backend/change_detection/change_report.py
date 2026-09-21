"""The comparison answer, assembled by code from measurements and single-image looks.

Why: a 2B vision-language model cannot reliably compare two images (it invents "urbanization" or
parrots its own captions). It does describe ONE image well, so the model is only asked to look at
single-image crops of flagged regions; every claim about change comes from a measurement (pixel
shifts, flagged regions, LoRA scan scores), and the text says so. Pure Python, no model imports.

Two outputs, on purpose:
  * compose_change_answer   - what the user reads: warnings, changed areas, before / after.
  * compose_change_details  - everything else (measured shifts, scan scores, agreement between them,
                              how the answer was built). It goes into the trace and the PDF report.
"""
import os
import re
from typing import Dict, List, Optional, Sequence

from agent_manager.scan_compare import format_scan_change
from change_detection.locations import describe_location

MAX_CAPTION_CHARS = 500

# A scan tile counts as backing up a flagged region when at least this share of the TILE lies inside
# the region's box (a region merely touching a tile's corner does not).
TILE_OVERLAP_MIN = 0.15

# A full stop / ! / ? followed by whitespace or the end. Whether it really ends a sentence is checked in _clip.
_STOP = re.compile(r"[.!?](?=\s|$)")

# Habits of the small model that make captions worse to read, cleaned up in clean_caption:
_BOILERPLATE = re.compile(
    r"^\s*The (?:image|crop) shows a satellite (?:view|image) of [^.:]*[.:]\s*(?:Visible features include:?\s*)?", re.I)
_ITEM_NUMBER = re.compile(r"(?:(?<=\s)|^)\d+\.\s+")
_ITEM_LABEL = re.compile(r"(?:(?<=\s)|^)\d+\.\s+[A-Z][^:.]{1,40}:\s*")
_NEGATION = re.compile(
    r"\b(?:there (?:is|are) no|no visible|not visible|nothing visible|no (?:[\w-]+ ){0,3}(?:is|are) (?:visible|present))\b",
    re.I)

_GENERIC_QUESTION = re.compile(r"chang|differ|compar|what.*(happen|new)|^\s*$", re.IGNORECASE)


def compare_report_enabled() -> bool:
    """Env CHANGE_EVIDENCE_REPORT (default on): build the comparison answer from evidence instead of
    asking the model to write it. "0" restores the old stitched-image answer."""
    return os.environ.get("CHANGE_EVIDENCE_REPORT", "1").strip().lower() not in {"0", "false", "no", "off"}


def lora_compare_enabled() -> bool:
    """Env LORA_COMPARE (default on): run the three LoRA tile scans on both dates and compare the scores."""
    return os.environ.get("LORA_COMPARE", "1").strip().lower() not in {"0", "false", "no", "off"}


def is_generic_question(query: str) -> bool:
    return bool(_GENERIC_QUESTION.search(query or ""))


def _last_sentence_end(head: str) -> int:
    """Index just past the last real sentence end in `head`, or -1. A number before the full stop is
    a list marker ("4."), not a sentence end: cutting there left captions ending in a stray "4."."""
    best = -1
    for stop in _STOP.finditer(head):
        words = head[: stop.start()].split()
        if words and words[-1].isdigit():
            continue
        best = stop.end()
    return best


def clean_caption(text: str) -> str:
    """Make a model caption readable without shortening what it actually says.

    Removes the stock opener ("The image shows a satellite view of a landscape ..."), turns the numbered
    "1. Label: sentence" habit into plain sentences, and drops sentences that only report an absence
    ("There is no visible water body"): those were the model's filler and often wrong. If that leaves
    nothing, the original text is returned rather than an empty caption."""
    original = " ".join((text or "").split())
    t = _BOILERPLATE.sub("", original)
    if _ITEM_NUMBER.search(t):
        if ":" in t:
            t = _ITEM_NUMBER.sub("", _ITEM_LABEL.sub("", t))
        else:                                   # "1. Bare ground 2. Buildings 3. Roads"
            items = [i.strip(" ,;.") for i in _ITEM_NUMBER.split(t) if i.strip(" ,;.")]
            t = "Noted: " + ", ".join(items) + "." if items else ""
    kept = [sentence for sentence in re.split(r"(?<=[.!?])\s+", t) if sentence and not _NEGATION.search(sentence)]
    return " ".join(kept).strip() or original


def _clip(text: str, limit: int) -> str:
    """Shorten at a sentence end when one fits (never mid-word), else at a word boundary."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = _last_sentence_end(head)
    if end >= limit // 3:
        return head[:end]
    return head.rsplit(" ", 1)[0].rstrip(",;:") + "…"


def region_items(major: Sequence, subtle: Sequence, shape, notes: Optional[Sequence] = None,
                 measured: Optional[Sequence] = None) -> List[Dict]:
    """Flatten major then subtle regions (x, y, w, h, frac) into dicts with a location word and the
    pixel box. `notes` (before/after captions) and `measured` (before -> after pixel counts) line up with
    that order; entries may be None / "" when a region has none."""
    h, w = shape
    items = []
    for kind, group in (("major", major), ("possible", subtle)):
        for x, y, rw, rh, frac in group:
            items.append({"kind": kind, "where": describe_location(x + rw / 2, y + rh / 2, w, h), "frac": frac,
                          "box": (x, y, rw, rh), "before": None, "after": None, "measured": "", "scans": []})
    for item, note in zip(items, notes or []):
        if note:
            item["before"], item["after"] = note
    for item, text in zip(items, measured or []):
        item["measured"] = text or ""
    return items


def match_scans_to_regions(items: List[Dict], shape, summaries: Sequence[Dict]) -> List[Dict]:
    """Record on each region which LoRA scans clearly turned positive on a tile that overlaps it.
    Two independent signals pointing at the same place is the strongest evidence this report has."""
    h, w = shape
    for item in items:
        x, y, rw, rh = item["box"]
        names = []
        for summary in summaries:
            rows, cols = summary["grid"]
            for r, c in summary["gained_cells"]:
                left, right, top, bottom = c * w / cols, (c + 1) * w / cols, r * h / rows, (r + 1) * h / rows
                inter = max(0.0, min(x + rw, right) - max(x, left)) * max(0.0, min(y + rh, bottom) - max(y, top))
                if inter >= TILE_OVERLAP_MIN * (right - left) * (bottom - top):
                    names.append(summary["name"])
                    break
        item["scans"] = names
    return items


def _scan_names(names: Sequence[str]) -> str:
    if len(names) == 1:
        return f"the {names[0]} scan"
    return "the " + ", ".join(names[:-1]) + f" and {names[-1]} scans"


def _label(item: Dict) -> str:
    """Major changes stay major. A subtler area is only 'possible' until a specialist scan independently
    flags the same place, then it is 'likely'."""
    if item["kind"] == "major":
        return "Major change"
    return "Likely change" if item.get("scans") else "Possible smaller change"


def compose_change_answer(items: Sequence[Dict], warnings: Sequence[str] = (), user_query: str = "") -> str:
    """What the user reads: warnings, changed areas, and each area's before / after. Nothing else."""
    lines = [f"WARNING: {w}" for w in warnings]
    if lines:
        lines.append("")
    lines.append("Changed areas")
    if not items:
        lines.append(
            "- No changed area was flagged, but the warning above means this is NOT evidence that nothing changed."
            if warnings else "- No changed area was found."
        )
    for item in items:
        head = f"- {_label(item)}, {item['where']} (~{item['frac']:.0%} of the scene)."
        if item.get("scans"):
            head += f" Also flagged by {_scan_names(item['scans'])}."
        lines.append(head)
        if item.get("before") or item.get("after"):
            lines.append(f"    Before: {_clip(clean_caption(item.get('before') or 'not described'), MAX_CAPTION_CHARS)}")
            lines.append(f"    After: {_clip(clean_caption(item.get('after') or 'not described'), MAX_CAPTION_CHARS)}")
        if item.get("measured"):
            caveat = " This may reflect the two images differing in source, not the ground." if warnings else ""
            lines.append(f"    Measured: {item['measured']}.{caveat}")
    if user_query and not is_generic_question(user_query):
        lines += ["", f"Your question ({_clip(user_query, 120)}) is not answered separately in comparison mode. "
                      "Ask about one image in the Assistant panel for follow-up questions."]
    return "\n".join(lines)


def compose_change_details(
    items: Sequence[Dict],
    measured_facts: str = "",
    scan_summaries: Sequence[Dict] = (),
    capture_dates: Optional[tuple] = None,
    modality: str = "optical",
    warnings: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """The supporting data, as {section title: lines}. Kept out of the answer so it stays readable;
    the PDF report prints it and it is stored in the trace."""
    details: Dict[str, List[str]] = {
        "How this was produced": [
            "Assembled by code from pixel measurements, flagged regions, LoRA scan scores and single-image model "
            "captions of the flagged regions. The model did not write the conclusion, so nothing in the answer is "
            "guessed about change."
        ],
        "Image order": [
            f"Before: captured {capture_dates[0]}. After: captured {capture_dates[1]}." if capture_dates else
            "First image treated as BEFORE, second as AFTER (as uploaded). Capture dates are unknown, so no time "
            "span is stated."
        ],
    }
    if measured_facts:
        lines = [f"Pixel counts, not materials: {measured_facts}"]
        if warnings:
            lines.append("Because of the warning(s), these shifts may reflect the two images differing in source, "
                         "zoom, haze or exposure rather than a change on the ground.")
        details["Measured overall shift"] = lines
    if items:
        lines = [f"{_label(i)}, {i['where']} "
                 f"(~{i['frac']:.0%} of the scene)" + (f": {i['measured']} (pixel counts, not materials)" if i.get('measured') else "")
                 for i in items]
        if any(i["kind"] == "possible" for i in items):
            lines.append("'Possible' and 'Likely' areas are subtler colour differences after removing the overall shift; "
                         "'Likely' means a specialist scan also flagged the same place. They can also be lighting or season.")
        details["Flagged regions"] = lines
    if scan_summaries:
        details["Specialist scans on both dates"] = [
            "LoRA adapter scores over a 4x4 grid. WEAK evidence: accuracy is not validated, and a scan that is "
            "already positive on many tiles separates the dates poorly.",
            *[format_scan_change(s) for s in scan_summaries],
        ]
        if items:                       # with nothing flagged there is nothing for a scan to agree with
            backed = [f"{i['where']}: {_scan_names(i['scans'])} also turned positive on a tile overlapping this area."
                      for i in items if i.get("scans")]
            details["Agreement between scans and flagged regions"] = backed or [
                "No scan flip overlaps a flagged region, so nothing independently backs the flagged areas."
            ]
    if modality == "sar":
        details["SAR note"] = ["Brightness is backscatter, not colour; the colour-based measures are omitted."]
    return details
