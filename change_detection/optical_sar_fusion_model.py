"""
optical_sar_fusion_model.py
=============================
Member 5: Cross-Modal (Optical + SAR) Fusion (Stage 3 specialist)

Builds a false-color composite from co-registered Optical RGB + SAR
intensity arrays, and generates a context prompt telling the VLM how
to interpret each channel. No neural network training required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionResult:
    composite_image: np.ndarray  # HxWx3 uint8
    context_prompt: str

    def to_json(self) -> dict:
        return {
            "context_prompt": self.context_prompt,
            "composite_shape": list(self.composite_image.shape),
        }


class FusionEngine:
    """Deterministic optical+SAR false-color compositor."""

    def fuse(
        self,
        optical_rgb: np.ndarray,
        sar_intensity: np.ndarray,
        user_query: str = "",
    ) -> FusionResult:
        """
        Args:
            optical_rgb: HxWx3 uint8 array (co-registered optical image).
            sar_intensity: HxW uint8 array (despeckled SAR backscatter,
                e.g. output of sar_preprocessor.SARPreprocessor.despeckle).
            user_query: original user query, folded into the generated prompt
                for extra context.

        Returns:
            FusionResult(composite_image, context_prompt)
        """
        self._validate(optical_rgb, sar_intensity)

        composite = np.zeros_like(optical_rgb)
        composite[:, :, 0] = optical_rgb[:, :, 0]  # Red   <- Optical Red
        composite[:, :, 1] = optical_rgb[:, :, 1]  # Green <- Optical Green
        composite[:, :, 2] = sar_intensity          # Blue  <- SAR VV/VH backscatter

        prompt = self._build_prompt(user_query)
        return FusionResult(composite_image=composite, context_prompt=prompt)

    def _validate(self, optical_rgb: np.ndarray, sar_intensity: np.ndarray):
        if optical_rgb.shape[:2] != sar_intensity.shape[:2]:
            raise ValueError(
                f"Optical ({optical_rgb.shape[:2]}) and SAR ({sar_intensity.shape[:2]}) "
                "must be co-registered to the same H x W before fusion."
            )
        if optical_rgb.ndim != 3 or optical_rgb.shape[2] < 2:
            raise ValueError("optical_rgb must be an HxWx3 (or more) array.")

    def _build_prompt(self, user_query: str) -> str:
        base = (
            "This is a false-color composite image combining optical and radar data. "
            "The Red and Green channels show the visual (optical) spectrum of the scene. "
            "The Blue channel shows SAR (radar) backscatter intensity, which represents "
            "surface structural density and roughness -- bright blue areas indicate strong "
            "radar returns (e.g. buildings, metal structures, rough terrain), while dark "
            "blue areas indicate smooth surfaces (e.g. calm water, flat pavement) or radar shadow."
        )
        if user_query:
            base += f" User question about this composite: '{user_query}'."
        return base


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    optical = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    sar = np.random.randint(0, 255, (128, 128), dtype=np.uint8)

    engine = FusionEngine()
    result = engine.fuse(optical, sar, user_query="Where are the man-made structures?")

    print("=== Fusion Test ===")
    print(result.to_json())

    import cv2
    cv2.imwrite("mock_fusion_composite.png", cv2.cvtColor(result.composite_image, cv2.COLOR_RGB2BGR))
    print("Saved mock_fusion_composite.png")
