from typing import Optional

import numpy as np
import cv2

from agent_manager.prompts import REPLY_WORD_LIMIT

# What the composite really is; reported verbatim in the audit trace.
FUSION_METHOD = "HSV value-channel replacement (hue/saturation from optical, brightness from SAR)"


def create_fusion_composite(optical_array: np.ndarray, sar_array: np.ndarray,
                            sar_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Fuse by HSV value-channel replacement.

    Hue and saturation are kept from the optical image; the brightness (V) channel is replaced by
    the SAR backscatter. This is NOT true IHS/Brovey fusion, and it does not guarantee any
    particular colour for a given surface: colour is whatever the optical image had, so it only
    says something about the surface where the optical image did.

    sar_valid: optional (H, W) bool, False where the SAR raster has no data (e.g. outside its
    footprint after alignment). Those pixels are left out of the SAR normalisation and keep the
    optical image's own brightness instead of being painted black.
    """
    # 1. Resize SAR (and its validity mask) to match Optical exactly
    target_shape = (optical_array.shape[1], optical_array.shape[0])
    if sar_array.shape[:2] != target_shape[::-1]:
        sar_array = cv2.resize(sar_array, target_shape, interpolation=cv2.INTER_AREA)
        if sar_valid is not None:
            sar_valid = cv2.resize(sar_valid.astype(np.uint8), target_shape, interpolation=cv2.INTER_NEAREST) > 0

    # 2. Extract single channel SAR and normalize robustly
    # SAR data often has extreme bright spots, so percentile-clip (over valid pixels only) to 0-255
    sar_channel = sar_array if len(sar_array.shape) == 2 else sar_array[:, :, 0]
    valid = np.ones(sar_channel.shape, dtype=bool) if sar_valid is None else np.asarray(sar_valid, dtype=bool)
    if not valid.any():
        raise ValueError("SAR image has no valid pixels to fuse.")
    p2, p98 = np.percentile(sar_channel[valid], (2, 98))
    sar_normalized = np.clip((sar_channel.astype(np.float32) - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

    # 3. Convert Optical RGB to HSV
    if optical_array.dtype != np.uint8:
        peak = float(np.max(optical_array))
        optical_array = (optical_array / (peak if peak > 0 else 1.0) * 255).astype(np.uint8)

    hsv_image = cv2.cvtColor(optical_array, cv2.COLOR_RGB2HSV)

    # 4. Replace the 'V' (Value/Brightness) channel with the Normalized SAR where SAR has data
    hsv_image[:, :, 2] = np.where(valid, sar_normalized, hsv_image[:, :, 2])

    # 5. Convert back to RGB for the VLM to interpret naturally
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def fusion_system_prompt() -> str:
    """System-role instructions. Describes what the composite actually encodes, and does not
    promise specific colours for specific surfaces (the fusion maths cannot guarantee any)."""
    return (
        "You are a satellite-imagery analyst. This image fuses two sensors: the colour (hue and "
        "saturation) comes from an optical image, the brightness from radar (SAR) backscatter, not "
        "sunlight. Very dark areas usually mean low backscatter (calm water, smooth surfaces); very "
        "bright areas usually mean strong backscatter (buildings, rough ground). These are hints: say "
        "so when a feature is ambiguous. It is not a photograph.\n"
        f"Reply in two short parts, under {REPLY_WORD_LIMIT} words in total.\n"
        "OBSERVATIONS: the visible features relevant to the question; only those present.\n"
        "ASSESSMENT: the answer, based only on those observations. If the image cannot answer it, say "
        "so. Give amounts and proportions in words (\"about half\"), not numbers. No generic advice."
    )


def generate_fusion_prompt(user_query: str) -> str:
    """The user-turn text (the instructions live in the system role)."""
    return user_query


def execute_optical_sar_fusion(optical_array: np.ndarray, sar_array: np.ndarray, user_query: str,
                               sar_valid: Optional[np.ndarray] = None) -> dict:
    """
    Main interface function called by the Agent Controller.
    """
    composite = create_fusion_composite(optical_array, sar_array, sar_valid)

    return {
        "status": "success",
        "composite_image": composite,
        "system_prompt": fusion_system_prompt(),
        "generated_prompt": generate_fusion_prompt(user_query),
        "method": FUSION_METHOD,
    }
