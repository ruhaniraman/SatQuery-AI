"""
cdvqa_engine.py
================
Member 3: Bi-Temporal & Multi-Temporal Change Engine (SATQUERY AI Backend)

Mission: Detect what changed between co-registered satellite images, track multi-date
series progression (3+ timestamps), compute hazard severity ratings, export GeoJSON GPS
polygons, and produce natural-language explanations via Qwen2-VL.

Modularized Architecture:
- `change_mask_segmenter.py` -> Segmenter (Histogram Matching, SSIM, Otsu Binarization, JET heatmap, overlays)
- `change_metrics.py` -> Metrics Calculator (% change, Severity, GeoJSON, IoU evaluation)
- `cdvqa_engine.py` -> Main VQA engine, coordinator & multi-date series analyzer
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from change_detection.change_mask_segmenter import ChangeMaskSegmenter
from change_detection.change_metrics import ChangeMetricsCalculator, ChangeRegion

# ---------------------------------------------------------------------------
# Optional: Qwen2-VL (same lazy-load + fallback pattern as Member 2)
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Top-level result containers                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ChangeDetectionResult:
    change_percentage: float
    binary_mask: np.ndarray      # HxW uint8 — 255=changed, 0=unchanged
    overlay_image: np.ndarray    # image_b with changed regions highlighted in red
    diff_heatmap: np.ndarray     # colorized JET heatmap of SSIM diff (HxWx3 RGB)
    ssim_score: float            # global SSIM (0-1; 1 = identical)
    confidence_score: float      # 1 - ssim_score, clamped; used by Member 6 API
    diff_map: np.ndarray         # raw per-pixel SSIM difference map (0-255 uint8)
    change_regions: List[ChangeRegion]
    overall_severity: str        # CRITICAL, HIGH, MODERATE, LOW
    prompt_sent_to_vlm: str
    explanation: str
    processing_time_ms: float
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "change_percentage": round(self.change_percentage, 2),
            "ssim_score": round(self.ssim_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "overall_severity": self.overall_severity,
            "explanation": self.explanation,
            "prompt_sent_to_vlm": self.prompt_sent_to_vlm,
            "num_change_regions": len(self.change_regions),
            "change_regions": [r.to_dict() for r in self.change_regions],
            "processing_time_ms": round(self.processing_time_ms, 2),
            "metadata": self.metadata,
            "visual_evidence_shape": {
                "binary_mask": list(self.binary_mask.shape),
                "overlay_image": list(self.overlay_image.shape),
                "diff_heatmap": list(self.diff_heatmap.shape),
            },
        }


@dataclass
class MultiDateSeriesResult:
    """Result for 3+ multi-date temporal image series."""
    num_timestamps: int
    timestamps: List[str]
    cumulative_change_percentage: float
    pairwise_results: List[ChangeDetectionResult]
    series_explanation: str
    processing_time_ms: float

    def to_json(self) -> dict:
        return {
            "num_timestamps": self.num_timestamps,
            "timestamps": self.timestamps,
            "cumulative_change_percentage": round(self.cumulative_change_percentage, 2),
            "series_explanation": self.series_explanation,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "pairwise_steps": [
                {
                    "step": f"{self.timestamps[i]} -> {self.timestamps[i+1]}",
                    "change_percentage": p.change_percentage,
                    "overall_severity": p.overall_severity,
                    "num_regions": len(p.change_regions),
                }
                for i, p in enumerate(self.pairwise_results)
            ],
        }


# --------------------------------------------------------------------------- #
# Qwen2-VL wrapper (real inference + mock fallback)                            #
# --------------------------------------------------------------------------- #

class _Qwen2VLWrapper:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Qwen2VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_id).to(self.device)

    def call(self, prompt: str, image_a: np.ndarray, image_b: np.ndarray) -> str:
        if not TRANSFORMERS_AVAILABLE:
            return self._mock(prompt)
        try:
            self._load()
            return self._infer(prompt, image_a, image_b)
        except Exception as exc:
            print(f"[ChangeDetectionEngine] Qwen2-VL fallback ({exc})")
            return self._mock(prompt)

    def _infer(self, prompt: str, image_a: np.ndarray, image_b: np.ndarray) -> str:
        from PIL import Image
        pil_a = Image.fromarray(image_a)
        pil_b = Image.fromarray(image_b)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=[text_prompt], images=[pil_a, pil_b], return_tensors="pt").to(self.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        generated = self._processor.batch_decode(output_ids, skip_special_tokens=True)
        return generated[0]

    @staticmethod
    def _mock(prompt: str) -> str:
        return (
            "[MOCK VLM RESPONSE] Structural analysis suggests new built-up "
            "surfaces and/or vegetation loss in the highlighted region between "
            "the two capture dates. Replace _mock_vlm_call with a real "
            "Qwen2-VL inference call to generate a grounded explanation. "
            f"(Prompt received: '{prompt[:120]}...')"
        )


# --------------------------------------------------------------------------- #
# Core engine                                                                  #
# --------------------------------------------------------------------------- #

class ChangeDetectionEngine:
    """
    Bi-temporal & Multi-temporal Change Detection Engine.
    Coordinates segmenter, metrics calculator, hazard alerts, GeoJSON exporter, and Qwen2-VL.
    """

    def __init__(
        self,
        ssim_win_size: int = 7,
        change_threshold: int = 30,
        min_blob_area: int = 25,
        use_histogram_matching: bool = True,
        use_otsu_thresholding: bool = True,
        qwen_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
    ):
        self.ssim_win_size = ssim_win_size
        self.change_threshold = change_threshold
        self.min_blob_area = min_blob_area
        self.segmenter = ChangeMaskSegmenter(
            ssim_win_size=ssim_win_size,
            change_threshold=change_threshold,
            min_blob_area=min_blob_area,
            use_histogram_matching=use_histogram_matching,
            use_otsu_thresholding=use_otsu_thresholding,
        )
        self.metrics = ChangeMetricsCalculator(min_blob_area=min_blob_area)
        self._vlm = _Qwen2VLWrapper(model_id=qwen_model_id, device=device)

    def detect(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        date_a: str = "Date 1",
        date_b: str = "Date 2",
        resolution_m_per_px: Optional[float] = None,
        top_left_lat_lon: Optional[Tuple[float, float]] = (37.7749, -122.4194),
        vlm_fn: Optional[Callable[[str, np.ndarray, np.ndarray], str]] = None,
        user_query: str = "",
    ) -> ChangeDetectionResult:
        """Run full bi-temporal change detection pipeline."""
        t0 = time.perf_counter()

        binary_mask, heatmap, diff_map, ssim_score, image_b_aligned = self.segmenter.segment(
            image_a, image_b
        )

        change_pct = self.metrics.compute_change_percentage(binary_mask)
        change_regions = self.metrics.extract_regions(
            binary_mask,
            image_a=image_a,
            image_b=image_b_aligned,
            resolution_m_per_px=resolution_m_per_px,
            top_left_lat_lon=top_left_lat_lon,
        )
        overlay = self.segmenter.build_overlay(image_b_aligned, binary_mask, change_regions)

        # Overall severity is the highest severity across regions
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
        overall_sev = max((r.severity for r in change_regions), key=lambda s: severity_order.get(s, 0), default="LOW")

        base_prompt = self._build_prompt(date_a, date_b, change_pct, change_regions, overall_sev)
        
        # DEBUG MODE: If a user query exists, ONLY use the user query. Ignore the math.
        if user_query:
            final_prompt = user_query
        else:
            final_prompt = base_prompt

        caller = vlm_fn or self._vlm.call
        explanation = caller(final_prompt, image_a, image_b_aligned)
        
        prompt = final_prompt

        elapsed_ms = (time.perf_counter() - t0) * 1000
        confidence = float(np.clip(1.0 - ssim_score, 0.0, 1.0))

        return ChangeDetectionResult(
            change_percentage=change_pct,
            binary_mask=binary_mask,
            overlay_image=overlay,
            diff_heatmap=heatmap,
            ssim_score=ssim_score,
            confidence_score=confidence,
            diff_map=diff_map,
            change_regions=change_regions,
            overall_severity=overall_sev,
            prompt_sent_to_vlm=prompt,
            explanation=explanation,
            processing_time_ms=elapsed_ms,
            metadata={
                "date_a": date_a,
                "date_b": date_b,
                "image_shape": list(image_a.shape),
                "ssim_win_size": self.ssim_win_size,
                "change_threshold": self.change_threshold,
                "min_blob_area": self.min_blob_area,
                "resolution_m_per_px": resolution_m_per_px,
                "top_left_lat_lon": list(top_left_lat_lon) if top_left_lat_lon else None,
                "histogram_matching": self.segmenter.use_histogram_matching,
                "otsu_thresholding": self.segmenter.use_otsu_thresholding,
            },
        )

    def detect_series(
        self,
        images: List[np.ndarray],
        timestamps: Optional[List[str]] = None,
        resolution_m_per_px: Optional[float] = None,
    ) -> MultiDateSeriesResult:
        """
        Multi-date series change tracker (3+ images).
        Computes pairwise change steps, cumulative change progression, and timeline explanation.
        """
        t0 = time.perf_counter()
        if len(images) < 2:
            raise ValueError("Multi-date series analysis requires at least 2 images.")

        if not timestamps or len(timestamps) != len(images):
            timestamps = [f"T{i+1}" for i in range(len(images))]

        pairwise_results = []
        cumulative_mask = np.zeros(images[0].shape[:2], dtype=np.uint8)

        for i in range(len(images) - 1):
            res = self.detect(
                images[i], images[i+1],
                date_a=timestamps[i], date_b=timestamps[i+1],
                resolution_m_per_px=resolution_m_per_px
            )
            pairwise_results.append(res)
            cumulative_mask = np.logical_or(cumulative_mask > 0, res.binary_mask > 0).astype(np.uint8) * 255

        cum_pct = self.metrics.compute_change_percentage(cumulative_mask)

        # Multi-date series summary explanation
        series_summary = (
            f"[MULTI-DATE SERIES ANALYSIS] Processed {len(images)} timestamps ({' -> '.join(timestamps)}). "
            f"Cumulative area change across the full series is {cum_pct:.2f}%. "
            f"Step changes: " + ", ".join([f"{timestamps[i]}->{timestamps[i+1]}: {p.change_percentage:.1f}% ({p.overall_severity})" for i, p in enumerate(pairwise_results)])
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return MultiDateSeriesResult(
            num_timestamps=len(images),
            timestamps=timestamps,
            cumulative_change_percentage=cum_pct,
            pairwise_results=pairwise_results,
            series_explanation=series_summary,
            processing_time_ms=elapsed_ms,
        )

    def export_geojson(
        self,
        result: ChangeDetectionResult,
        top_left_lat_lon: Optional[Tuple[float, float]] = (37.7749, -122.4194),
        resolution_m_per_px: float = 10.0,
    ) -> dict:
        """Generates a GeoJSON FeatureCollection from a ChangeDetectionResult."""
        return self.metrics.export_geojson(
            result.change_regions,
            top_left_lat_lon=top_left_lat_lon,
            resolution_m_per_px=resolution_m_per_px,
        )

    def _build_prompt(
        self,
        date_a: str,
        date_b: str,
        change_pct: float,
        regions: List[ChangeRegion],
        severity: str,
    ) -> str:
        region_info = ""
        if regions:
            largest = regions[0]
            area_str = f"{largest.area_ha:.2f} hectares" if largest.area_ha else f"{largest.area_px} pixels"
            cat_str = f"classified as '{largest.category}'"
            region_info = (
                f"There are {len(regions)} distinct changed region(s). "
                f"The primary region ({cat_str}) covers {area_str} "
                f"centred at ({largest.centroid_x}, {largest.centroid_y}). "
            )
        return (
            f"Image A is from {date_a}. Image B is from {date_b}. "
            f"{change_pct:.2f}% of the area has changed between the two captures (HAZARD SEVERITY: {severity}). "
            f"{region_info}"
            "Carefully compare the two images and describe exactly what physical changes occurred in the landscape. "
            "Focus strictly on the visual evidence, such as the appearance or disappearance of water bodies, vegetation, or infrastructure. "
            "Do not invent details."
        )

    def save_outputs(
        self,
        result: ChangeDetectionResult,
        out_dir: str = ".",
        prefix: str = "cd",
    ) -> dict:
        import os
        os.makedirs(out_dir, exist_ok=True)
        paths = {}

        mask_path = os.path.join(out_dir, f"{prefix}_binary_mask.png")
        cv2.imwrite(mask_path, result.binary_mask)
        paths["binary_mask"] = mask_path

        overlay_path = os.path.join(out_dir, f"{prefix}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(result.overlay_image, cv2.COLOR_RGB2BGR))
        paths["overlay"] = overlay_path

        heatmap_path = os.path.join(out_dir, f"{prefix}_heatmap.png")
        cv2.imwrite(heatmap_path, cv2.cvtColor(result.diff_heatmap, cv2.COLOR_RGB2BGR))
        paths["diff_heatmap"] = heatmap_path

        return paths


if __name__ == "__main__":
    import json

    img_a = np.full((256, 256, 3), 120, dtype=np.uint8)
    img_b = img_a.copy()
    cv2.rectangle(img_b, (80, 80), (170, 170), (220, 40, 40), thickness=-1)
    cv2.rectangle(img_b, (20, 200), (70, 240), (30, 90, 30), thickness=-1)

    engine = ChangeDetectionEngine(ssim_win_size=7, change_threshold=30, min_blob_area=25)
    result = engine.detect(img_a, img_b, date_a="2023-01-15", date_b="2024-06-02", resolution_m_per_px=10.0)

    print("=== Change Detection Result ===")
    print(f"Change %:        {result.change_percentage:.2f}%")
    print(f"Severity Alert:  {result.overall_severity}")
    print(f"SSIM score:      {result.ssim_score:.4f}")
    print(f"Regions found:   {len(result.change_regions)}")
    for r in result.change_regions:
        print(f"  Region #{r.region_id} [{r.category}] ({r.severity}): {r.area_ha:.4f} ha")

    geojson_data = engine.export_geojson(result)
    print(f"\nGeoJSON Features Exported: {len(geojson_data['features'])}")

    saved = engine.save_outputs(result, out_dir=".", prefix="mock")
    print(f"\nSaved visual outputs: {saved}")