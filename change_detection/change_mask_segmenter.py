"""
change_mask_segmenter.py
========================
Member 3: Bi-Temporal Change Mask Segmenter (SATQUERY AI Backend)

Mission: Answers "Where did it change?"
Takes two co-registered 8-bit images (Date 1 -> Date 2), applies optional illumination/shadow
histogram matching, computes structural similarity (SSIM) differencing, uses adaptive
Otsu / threshold binarization + morphological closing + denoising, and outputs binary masks,
JET diff heatmaps, and annotated overlays.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms


class ChangeMaskSegmenter:
    """
    Deterministic OpenCV + SSIM Segmentation Engine.
    Handles image alignment, illumination/cloud-shadow histogram matching,
    SSIM differencing, adaptive Otsu thresholding, morphological denoising,
    JET heatmaps, and overlay drawing.
    """

    def __init__(
        self,
        ssim_win_size: int = 7,
        change_threshold: int = 30,
        min_blob_area: int = 25,
        use_histogram_matching: bool = True,
        use_otsu_thresholding: bool = True,
    ):
        self.ssim_win_size = ssim_win_size
        self.change_threshold = change_threshold
        self.min_blob_area = min_blob_area
        self.use_histogram_matching = use_histogram_matching
        self.use_otsu_thresholding = use_otsu_thresholding

    def segment(
        self, image_a: np.ndarray, image_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Segment changed pixels between image_a and image_b.

        Returns:
            binary_mask: HxW uint8 array (255 = changed, 0 = unchanged)
            diff_heatmap: HxWx3 RGB colorized JET diff map
            diff_map: HxW uint8 raw SSIM difference intensity
            ssim_score: float global SSIM (0 to 1)
            image_b_aligned: uint8 RGB image_b (aligned & histogram matched)
        """
        image_a, image_b = self._validate_and_align(image_a, image_b)

        # 1. Illumination / Cloud Shadow Normalization (Histogram Matching)
        if self.use_histogram_matching:
            image_b_proc = self._match_illumination(image_b, image_a)
        else:
            image_b_proc = image_b

        gray_a = self._to_gray(image_a)
        gray_b = self._to_gray(image_b_proc)

        # 2. SSIM Structural Difference
        ssim_score, diff_map = self._compute_ssim_diff(gray_a, gray_b)

        # 3. Adaptive / Otsu Thresholding
        binary_mask = self._threshold_mask(diff_map)

        # 4. Morphological Closing + Denoising
        binary_mask = self._denoise_mask(binary_mask)

        # 5. JET Colorized Heatmap
        heatmap = self._build_heatmap(diff_map)

        return binary_mask, heatmap, diff_map, ssim_score, image_b_proc

    def _match_illumination(self, source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
        """Normalizes brightness, sun angle, and cloud shadow shifts using histogram matching."""
        try:
            # Skip histogram matching for uniform synthetic test images with zero variance
            if np.std(source_img) < 1.0 or np.std(reference_img) < 1.0:
                return source_img

            if source_img.ndim == 3 and source_img.shape[2] >= 3:
                matched = match_histograms(source_img[:, :, :3], reference_img[:, :, :3], channel_axis=-1)
            else:
                matched = match_histograms(source_img, reference_img)
            return np.clip(matched, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[ChangeMaskSegmenter] Histogram matching skipped: {e}")
            return source_img

    def _validate_and_align(self, image_a: np.ndarray, image_b: np.ndarray):
        if image_a is None or image_b is None:
            raise ValueError("Both image_a and image_b are required for change segmentation.")
        if image_a.shape[:2] != image_b.shape[:2]:
            h_a, w_a = image_a.shape[:2]
            print(
                f"[ChangeMaskSegmenter] Warning: Images differ in shape ({image_a.shape} vs {image_b.shape}). "
                f"Auto-resizing image_b to match image_a ({w_a}x{h_a})."
            )
            image_b = cv2.resize(image_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
        if image_a.dtype != np.uint8:
            image_a = image_a.astype(np.uint8)
        if image_b.dtype != np.uint8:
            image_b = image_b.astype(np.uint8)
        return image_a, image_b

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] >= 3:
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        return img.squeeze()

    def _compute_ssim_diff(self, gray_a: np.ndarray, gray_b: np.ndarray):
        score, diff = ssim(gray_a, gray_b, win_size=self.ssim_win_size, full=True)
        diff_map = (1 - diff) * 127.5
        diff_map = np.clip(diff_map, 0, 255).astype(np.uint8)
        return float(score), diff_map

    def _threshold_mask(self, diff_map: np.ndarray) -> np.ndarray:
        """Applies Otsu's adaptive binarization or fixed cutoff thresholding."""
        if self.use_otsu_thresholding:
            try:
                otsu_val, mask = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if otsu_val >= 15:
                    return mask
            except Exception:
                pass
        _, mask = cv2.threshold(diff_map, self.change_threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _denoise_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(closed)
        for c in contours:
            if cv2.contourArea(c) >= self.min_blob_area:
                cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
        return clean_mask

    def _build_heatmap(self, diff_map: np.ndarray) -> np.ndarray:
        jet_bgr = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
        return cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)

    def build_overlay(
        self,
        base_image: np.ndarray,
        mask: np.ndarray,
        regions: Optional[List[dict]] = None,
    ) -> np.ndarray:
        """Highlight changed regions in red and overlay bounding boxes with labels."""
        if base_image.ndim == 2:
            overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = base_image[:, :, :3].copy()

        red_layer = np.zeros_like(overlay)
        red_layer[:, :] = (255, 0, 0)
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(
            overlay, 0.35, red_layer, 0.65, 0
        )[mask_bool]

        if regions:
            for r in regions[:5]:
                bbox = r.get("bbox") if isinstance(r, dict) else getattr(r, "bbox", None)
                r_id = r.get("region_id") if isinstance(r, dict) else getattr(r, "region_id", 0)
                area_ha = r.get("area_ha") if isinstance(r, dict) else getattr(r, "area_ha", None)
                area_px = r.get("area_px") if isinstance(r, dict) else getattr(r, "area_px", 0)
                category = r.get("category") if isinstance(r, dict) else getattr(r, "category", "")
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 230, 0), thickness=2)
                    tag = f" {category}" if category else ""
                    label = f"R#{r_id}{tag}: {area_ha:.2f}ha" if area_ha else f"R#{r_id}{tag}: {area_px}px"
                    cv2.putText(
                        overlay, label, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 0), 1, cv2.LINE_AA
                    )
        return overlay
