import cv2
import numpy as np
from dataclasses import dataclass

from agent_manager.prompts import REPLY_WORD_LIMIT

# The kernel sizes and minimum region area below were hand-tuned as absolute pixel counts, so they
# only suited one tile size. They are now expressed for this reference side length and scaled to
# the real image. NOTE: 512 is a guess at the size they were tuned on, not a validated value.
REFERENCE_SIDE = 512
MIN_REGION_AREA_AT_REFERENCE = 15000


def _odd_kernel(base: int, scale: float) -> tuple:
    k = max(3, int(round(base * scale)))
    k += 1 - k % 2
    return (k, k)

def describe_location(cx: float, cy: float, width: int, height: int) -> str:
    """Plain-language position of a point in the scene, on a 3x3 grid."""
    col = min(2, int(3 * cx / width))
    row = min(2, int(3 * cy / height))
    if row == 1 and col == 1:
        return "centre"
    return f"{('upper', 'middle', 'lower')[row]}-{('left', 'centre', 'right')[col]}".replace("middle-", "")


def build_change_prompt(regions, shape, user_query: str = "", modality: str = "optical",
                        capture_dates: tuple = None) -> str:
    """regions: list of (x, y, w, h, area_fraction) from the pixel-difference analysis.
    capture_dates: (before, after) strings, ONLY when both are real dates read from the files.

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
    prompt = (
        f"{order} In one short paragraph (under {REPLY_WORD_LIMIT} words) describe only the visible "
        "differences (land cover, buildings, water, vegetation, roads, excavation)."
    )
    if modality == "sar":
        prompt += " These are SAR images: brightness is backscatter, not colour; fine grain is speckle, not change."
    if regions:
        listed = "; ".join(
            f"{describe_location(x + rw / 2, y + rh / 2, w, h)} (~{frac:.0%} of the scene)"
            for x, y, rw, rh, frac in regions
        )
        prompt += (
            f" Pixel analysis flagged {len(regions)} region(s) of change, at the same position in each half: "
            f"{listed}. Check each and say what changed there."
        )
    else:
        prompt += (
            " Pixel analysis found no region of substantial change: report only clearly visible "
            "differences, otherwise say there is no major change."
        )
    if user_query:
        prompt += f" Also answer: {user_query}"
    return prompt


@dataclass
class ChangeDetectionResult:
    explanation: str
    overlay_image: np.ndarray
    major_regions_detected: int

class ChangeDetectionEngine:
    def detect(self, image_a: np.ndarray, image_b: np.ndarray, vlm_fn, user_query: str = "",
               valid_mask: np.ndarray = None, modality: str = "optical",
               capture_dates: tuple = None) -> ChangeDetectionResult:
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

        if valid_mask is not None:
            # The blur smears the black border ~half a kernel into real data, so shrink the
            # valid region by that much before masking.
            erode = cv2.getStructuringElement(cv2.MORPH_RECT, (blur_k[0] + 2, blur_k[1] + 2))
            safe = cv2.erode(valid_mask.astype(np.uint8), erode) > 0
            diff[~safe] = 0

        # Cloud masking is optical-only: in SAR, bright pixels are strong scatterers (buildings,
        # metal), which are exactly the things that change, not clouds.
        if modality != "sar":
            _, cloud_mask1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY)
            _, cloud_mask2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY)
            combined_clouds = cv2.bitwise_or(cloud_mask1, cloud_mask2)
            cloud_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _odd_kernel(21, scale))
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

        # VLM semantic analysis runs AFTER the pixel analysis so it can be told what the pixel
        # analysis found; otherwise the two are independent and can contradict each other
        # (text says "no major change" while boxes are drawn, or the reverse).
        stitched_image = np.concatenate((image_a, image_b), axis=1)
        prompt = build_change_prompt(regions, (h1, w1), user_query, modality, capture_dates)
        # We pass ONLY the single stitched image to the VLM
        explanation = vlm_fn(prompt, stitched_image)

        return ChangeDetectionResult(
            explanation=explanation,
            overlay_image=evidence_img,
            major_regions_detected=box_count
        )