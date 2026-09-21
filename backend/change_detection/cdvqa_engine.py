import cv2
import numpy as np
from dataclasses import dataclass, field

from agent_manager.prompts import REPLY_WORD_LIMIT, describe_region_prompt, describe_scene_prompt
from agent_manager.scene_stats import comparability_warning, compute_scene_stats, format_pair_stats, format_region_shift
from change_detection.locations import describe_location  # noqa: F401  (re-exported)

# The kernel sizes and minimum region area below were hand-tuned as absolute pixel counts, so they
# only suited one tile size. They are now expressed for this reference side length and scaled to
# the real image. NOTE: 512 is a guess at the size they were tuned on, not a validated value.
REFERENCE_SIDE = 512
MIN_REGION_AREA_AT_REFERENCE = 15000

# Describe-first evidence: at most this many flagged regions get a before and an after caption (each
# is one extra model call), cropped with this fraction of padding on every side.
MAX_REGION_NOTES = 3
REGION_PAD = 0.15

# Subtler-change pass: colour difference (Lab) AFTER removing the overall shift between the dates
# (season / exposure), so only local differences remain. Starting guesses, not tuned values.
SUBTLE_MIN_DELTA = 14.0
SUBTLE_AREA_FACTOR = 0.4        # of the major-change minimum area
MAX_SUBTLE_REGIONS = 4
# Two-level threshold (hysteresis): pixels above the strict level are seeds, and a region grows into the
# connected pixels above SUBTLE_LOW_FACTOR of it, so a box covers the whole changed patch and not just its
# strongest core. Growth that would multiply a region's area by more than SUBTLE_MAX_GROWTH is treated as a
# leak into ordinary variation and refused (the region keeps only its seeds). Chosen from two real pairs:
# legitimate growth measured 1.3-2.1x at 0.7, leaks appeared from 0.5 (x5.9) and 0.4. Guesses, not tuned.
SUBTLE_LOW_FACTOR = 0.7
SUBTLE_MAX_GROWTH = 3.0



def _odd_kernel(base: int, scale: float) -> tuple:
    k = max(3, int(round(base * scale)))
    k += 1 - k % 2
    return (k, k)

def build_change_prompt(regions, shape, user_query: str = "", modality: str = "optical",
                        capture_dates: tuple = None, evidence: str = "") -> str:
    """regions: list of (x, y, w, h, area_fraction) from the pixel-difference analysis.
    capture_dates: (before, after) strings, ONLY when both are real dates read from the files.
    evidence: optional block of measured before->after shifts and per-image / per-region notes.

    Kept short on purpose (measured with the Qwen tokenizer): the image tokens dominate the cost, and
    a bounded reply saves more than prompt words. The pixel-analysis result is included so the text
    cannot contradict the boxes drawn on the evidence image."""
    h, w = shape
    if capture_dates:
        order = f"Side-by-side satellite images of one place: left captured {capture_dates[0]}, right captured {capture_dates[1]}."
    else:
        order = (
            "Side-by-side satellite images of one place: left = BEFORE, right = AFTER (order as supplied; "
            "capture dates are unknown, so do not state or guess dates or time gaps)."
        )
    # Order matters for a 2B model: it tends to imitate whatever it read last. So the evidence (captions,
    # measured shifts) comes right after the framing, and the instruction to COMPARE comes after it.
    instruction = (
        f"Now compare the two halves. Reply in two parts, under {REPLY_WORD_LIMIT} words total. Do not "
        "describe one image on its own: every point must contrast before with after.\n"
        "CHANGES: each visible difference (land cover, buildings, water, vegetation, roads, excavation), "
        "with where it is and what it was before -> after.\n"
        "ASSESSMENT: one or two sentences on what changed overall. Report only differences supported by "
        "the images, the measured numbers or the region analysis; never list something present in both "
        "halves as a change. A general shift in greenness or brightness may be seasonal or lighting, so "
        "say that rather than assuming construction or clearing."
    )
    if modality == "sar":
        instruction += " These are SAR images: brightness is backscatter, not colour; fine grain is speckle, not change."
    if regions:
        listed = "; ".join(
            f"{describe_location(x + rw / 2, y + rh / 2, w, h)} (~{frac:.0%} of the scene)"
            for x, y, rw, rh, frac in regions
        )
        instruction += (
            f" Pixel analysis flagged {len(regions)} region(s) of change, at the same position in each half: "
            f"{listed}. Check each and say what changed there."
        )
    else:
        instruction += (
            " Pixel analysis found no region of substantial change: report only clearly visible "
            "differences, otherwise say there is no major change."
        )
    parts = [order]
    if evidence:
        parts.append(evidence)
    parts.append(instruction)
    if user_query:
        parts.append(f"Also answer: {user_query}")
    return "\n".join(parts)


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def gather_change_evidence(image_a, image_b, regions, notes_fn=None, modality="optical", valid_mask=None):
    """Returns (evidence_block, measured_facts).

    measured_facts: before -> after shifts of pixel proportions (no model involved).
    notes_fn(prompt, image) -> str, when given, captions each image and the first flagged regions
    SEPARATELY (a 2B model reads one image far better than a stitched pair). Captions are handed over
    as "may contain mistakes"; a failing caption call is skipped, never fatal."""
    facts = format_pair_stats(
        compute_scene_stats(image_a, modality, valid_mask), compute_scene_stats(image_b, modality, valid_mask)
    )
    lines = []
    if facts:
        lines.append(f"Measured before -> after (pixel counts, not materials): {facts}")
    if notes_fn is not None:
        h, w = image_a.shape[:2]
        try:
            prompt = describe_scene_prompt(modality)
            lines.append(f"BEFORE image, first look (may contain mistakes): {_one_line(notes_fn(prompt, image_a))}")
            lines.append(f"AFTER image, first look (may contain mistakes): {_one_line(notes_fn(prompt, image_b))}")
            for x, y, rw, rh, _ in regions[:MAX_REGION_NOTES]:
                px, py = int(rw * REGION_PAD), int(rh * REGION_PAD)
                y0, y1, x0, x1 = max(0, y - py), min(h, y + rh + py), max(0, x - px), min(w, x + rw + px)
                where = describe_location(x + rw / 2, y + rh / 2, w, h)
                ask = describe_region_prompt(where, modality)
                before = _one_line(notes_fn(ask, np.ascontiguousarray(image_a[y0:y1, x0:x1])))
                after = _one_line(notes_fn(ask, np.ascontiguousarray(image_b[y0:y1, x0:x1])))
                lines.append(f"Flagged region, {where}: before: {before} | after: {after}")
        except Exception as err:   # notes are an aid; the analysis must still complete
            print(f"Describe-first notes skipped: {err}")
            lines = [l for l in lines if l.startswith("Measured")]
    return "\n".join(lines), facts


def find_subtle_regions(image_a, image_b, safe_mask, blocked_mask, blur_k, scale, exclude=()):
    """Local colour differences too small for the strict pass, as (x, y, w, h, area_fraction).

    The median Lab shift between the dates is removed first (a greener or brighter season moves the
    whole scene, which is not a local change). safe_mask: bool (H, W) of trustworthy pixels or None.
    blocked_mask: uint8 (H, W), 255 where clouds are (optical) or None. exclude: boxes already
    reported as major changes; a subtle region whose centre lies in one is dropped."""
    h, w = image_a.shape[:2]
    lab_a = cv2.cvtColor(cv2.GaussianBlur(image_a, blur_k, 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(cv2.GaussianBlur(image_b, blur_k, 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    valid = safe_mask if safe_mask is not None else np.ones((h, w), dtype=bool)
    if not valid.any():
        return []
    diff = lab_b - lab_a
    shift = np.array([np.median(diff[..., c][valid]) for c in range(3)], dtype=np.float32)
    delta = np.linalg.norm(diff - shift, axis=2)

    spread = delta[valid]
    med = float(np.median(spread))
    mad = float(np.median(np.abs(spread - med))) * 1.4826
    usable = valid if blocked_mask is None else valid & (blocked_mask != 255)
    strict = max(SUBTLE_MIN_DELTA, med + 4 * mad)
    seeds = (delta > strict) & usable
    loose = (delta > strict * SUBTLE_LOW_FACTOR) & usable
    count, labels = cv2.connectedComponents(loose.astype(np.uint8))
    seed_count = np.bincount(labels[seeds], minlength=count)
    size = np.bincount(labels.ravel(), minlength=count)
    grow = (seed_count > 0) & (size <= SUBTLE_MAX_GROWTH * seed_count)
    grow[0] = False                                       # label 0 is the background
    binary = (grow[labels] | seeds).astype(np.uint8) * 255

    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _odd_kernel(11, scale))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _odd_kernel(35, scale))
    closed = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel, iterations=2),
                              cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_REGION_AREA_AT_REFERENCE * scale * scale * SUBTLE_AREA_FACTOR
    found = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        x, y, rw, rh = cv2.boundingRect(contour)
        cx, cy = x + rw / 2, y + rh / 2
        inside_major = any(ex <= cx <= ex + ew and ey <= cy <= ey + eh for ex, ey, ew, eh in exclude)
        if area > min_area and (rw * rh) / (w * h) < 0.75 and not inside_major:
            found.append((x, y, rw, rh, area / (w * h)))
        if len(found) == MAX_SUBTLE_REGIONS:
            break
    return found


def _safe_note(fn, *args, **kwargs) -> str:
    try:
        return _one_line(fn(*args, **kwargs))
    except Exception as err:   # a caption is an aid; the report must still be produced
        print(f"Caption skipped: {err}")
        return ""


def gather_report_notes(image_a, image_b, regions, notes_fn, modality="optical"):
    """Single-image captions of the flagged regions, for the evidence report.

    Each call sees ONE crop (before, then after), the situation the model handles well. Whole-image
    captions are deliberately not requested: on real scenes they were generic, near-identical for
    both dates and sometimes wrong ("urban area" for a mining scene), so they only misled.
    Returns a list lined up with `regions`: the first MAX_REGION_NOTES get a (before, after) pair,
    the rest (and any whose calls failed) get None."""
    if notes_fn is None:
        return [None] * len(regions)
    h, w = image_a.shape[:2]
    region_notes = []
    for index, (x, y, rw, rh, _) in enumerate(regions):
        if index >= MAX_REGION_NOTES:
            region_notes.append(None)
            continue
        px, py = int(rw * REGION_PAD), int(rh * REGION_PAD)
        y0, y1, x0, x1 = max(0, y - py), min(h, y + rh + py), max(0, x - px), min(w, x + rw + px)
        ask = describe_region_prompt(describe_location(x + rw / 2, y + rh / 2, w, h), modality)
        b_text = _safe_note(notes_fn, ask, np.ascontiguousarray(image_a[y0:y1, x0:x1]))
        a_text = _safe_note(notes_fn, ask, np.ascontiguousarray(image_b[y0:y1, x0:x1]))
        region_notes.append((b_text, a_text) if (b_text or a_text) else None)
    return region_notes


@dataclass
class ChangeDetectionResult:
    explanation: str
    overlay_image: np.ndarray
    major_regions_detected: int
    measured_facts: str = ""
    subtle_regions_detected: int = 0
    warnings: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

class ChangeDetectionEngine:
    def detect(self, image_a: np.ndarray, image_b: np.ndarray, vlm_fn, user_query: str = "",
               valid_mask: np.ndarray = None, modality: str = "optical",
               capture_dates: tuple = None, notes_fn=None, compose_report: bool = False,
               scan_summaries=(), extra_warnings=()) -> ChangeDetectionResult:
        """valid_mask: (H, W) bool, True where BOTH images hold real data. Filled borders left by
        alignment (nodata / warped-in black) are otherwise seen as a sharp edge and boxed as change."""

        # Calibrated OpenCV Spatial Differencing + Cloud Masking]
        gray1 = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
        h1, w1 = image_a.shape[:2]
        scale = float(np.sqrt(h1 * w1)) / REFERENCE_SIDE
        # Blur suppresses per-pixel noise (sensor noise, SAR speckle), which does NOT shrink with the
        # image, so it only grows with size. Region area and morphology below are object-sized and
        # scale both ways.
        blur_k = _odd_kernel(21, max(scale, 1.0))
        blur1 = cv2.GaussianBlur(gray1, blur_k, 0)
        blur2 = cv2.GaussianBlur(gray2, blur_k, 0)
        
        diff = cv2.absdiff(blur1, blur2)

        safe = None
        if valid_mask is not None:
            # The blur smears the black border ~half a kernel into real data, so shrink the
            # valid region by that much before masking.
            erode = cv2.getStructuringElement(cv2.MORPH_RECT, (blur_k[0] + 2, blur_k[1] + 2))
            safe = cv2.erode(valid_mask.astype(np.uint8), erode) > 0
            diff[~safe] = 0

        # Cloud masking is optical-only: in SAR, bright pixels are strong scatterers (buildings,
        # metal), which are exactly the things that change, not clouds.
        combined_clouds = None
        if modality != "sar":
            _, cloud_mask1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY)
            _, cloud_mask2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY)
            combined_clouds = cv2.bitwise_or(cloud_mask1, cloud_mask2)
            cloud_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _odd_kernel(21, scale))
            # Drop bright specks first: scattered pale ground (bare soil, roofs) is not cloud, and dilating
            # it used to blank most of a scene (1.7% bright pixels grew to a 63% mask on a mining scene).
            # Real cloud patches are bigger than the kernel and survive the opening.
            combined_clouds = cv2.morphologyEx(combined_clouds, cv2.MORPH_OPEN, cloud_kernel)
            combined_clouds = cv2.dilate(combined_clouds, cloud_kernel, iterations=2)

            # Zero out difference map where clouds exist
            diff[combined_clouds == 255] = 0

        # Strict Threshold & Noise Filtering 
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _odd_kernel(11, scale))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _odd_kernel(35, scale))
        
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare the BGR evidence artifact
        evidence_img = cv2.cvtColor(image_b, cv2.COLOR_RGB2BGR)
        box_count = 0
        regions = []
        min_area = MIN_REGION_AREA_AT_REFERENCE * scale * scale
        total_area = w1 * h1

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                coverage_ratio = (w * h) / total_area
                
                if area > min_area and coverage_ratio < 0.75:
                    cv2.rectangle(evidence_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
                    
                    # Clamp the text coordinates so they never render off-screen
                    text_x = max(x, 10)
                    text_y = max(y - 10, 25)
                    
                    cv2.putText(
                        evidence_img, "Major Change", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    box_count += 1
                    regions.append((x, y, w, h, area / total_area))

        # Is the pair even comparable? A zoom / resolution mismatch otherwise reads as "no change".
        stats_a = compute_scene_stats(image_a, modality, valid_mask)
        stats_b = compute_scene_stats(image_b, modality, valid_mask)
        warnings = [w for w in (*extra_warnings, comparability_warning(stats_a, stats_b)) if w]

        # Subtler differences (optical only: SAR speckle would flood this pass). Drawn in yellow.
        subtle = []
        if compose_report and modality != "sar":
            subtle = find_subtle_regions(image_a, image_b, safe, combined_clouds, blur_k, scale,
                                         exclude=[(x, y, rw, rh) for x, y, rw, rh, _ in regions])
            for x, y, rw, rh, _ in subtle:
                cv2.rectangle(evidence_img, (x, y), (x + rw, y + rh), (0, 255, 255), thickness=2)
                cv2.putText(evidence_img, "Possible change", (max(x, 10), max(y - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if compose_report:
            # The model only looks at single images; the conclusion is assembled from measurements.
            from change_detection.change_report import (
                compose_change_answer, compose_change_details, match_scans_to_regions, region_items)
            all_regions = list(regions) + list(subtle)
            region_notes = gather_report_notes(image_a, image_b, all_regions, notes_fn, modality)
            measured_facts = format_pair_stats(stats_a, stats_b)
            measured = [
                format_region_shift(*(compute_scene_stats(img[y:y + rh, x:x + rw], modality,
                                                          None if valid_mask is None else valid_mask[y:y + rh, x:x + rw])
                                      for img in (image_a, image_b)))
                for x, y, rw, rh, _ in all_regions
            ]
            items = match_scans_to_regions(
                region_items(regions, subtle, (h1, w1), region_notes, measured), (h1, w1), scan_summaries)
            explanation = compose_change_answer(items, warnings, user_query)
            details = compose_change_details(items, measured_facts, scan_summaries, capture_dates, modality, warnings)
        else:
            details = {}
            # VLM semantic analysis runs AFTER the pixel analysis so it can be told what the pixel
            # analysis found; otherwise the two are independent and can contradict each other
            # (text says "no major change" while boxes are drawn, or the reverse).
            stitched_image = np.concatenate((image_a, image_b), axis=1)
            evidence, measured_facts = gather_change_evidence(image_a, image_b, regions, notes_fn, modality, valid_mask)
            prompt = build_change_prompt(regions, (h1, w1), user_query, modality, capture_dates, evidence)
            # We pass ONLY the single stitched image to the VLM
            explanation = vlm_fn(prompt, stitched_image)

        return ChangeDetectionResult(
            explanation=explanation,
            overlay_image=evidence_img,
            major_regions_detected=box_count,
            measured_facts=measured_facts,
            subtle_regions_detected=len(subtle),
            warnings=warnings,
            details=details,
        )