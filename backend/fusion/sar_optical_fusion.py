import numpy as np
import cv2

def create_fusion_composite(optical_array: np.ndarray, sar_array: np.ndarray) -> np.ndarray:
    """
    Implements IHS (Intensity-Hue-Saturation) Fusion.
    Preserves the color (multispectral) information from the optical image, 
    but replaces the brightness/intensity with the SAR backscatter to highlight structures.
    """
    # 1. Resize SAR to match Optical exactly
    target_shape = (optical_array.shape[1], optical_array.shape[0])
    if sar_array.shape[:2] != target_shape[::-1]:
        sar_array = cv2.resize(sar_array, target_shape, interpolation=cv2.INTER_AREA)

    # 2. Extract single channel SAR and normalize robustly
    # SAR data often has extreme bright spots. We use percentile clipping to normalize it to 0-255
    sar_channel = sar_array if len(sar_array.shape) == 2 else sar_array[:, :, 0]
    p2, p98 = np.percentile(sar_channel, (2, 98))
    sar_normalized = np.clip((sar_channel - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

    # 3. Convert Optical RGB to HSV
    if optical_array.dtype != np.uint8:
        optical_array = (optical_array / np.max(optical_array) * 255).astype(np.uint8)
        
    hsv_image = cv2.cvtColor(optical_array, cv2.COLOR_RGB2HSV)

    # 4. Replace the 'V' (Value/Brightness) channel with the Normalized SAR
    hsv_image[:, :, 2] = sar_normalized

    # 5. Convert back to RGB for the VLM to interpret naturally
    fused_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return fused_rgb


def generate_fusion_prompt(user_query: str) -> str:
    """
    Generates a structured VLM prompt matching the successful single-image architecture.
    """
    instruction = (
        "[SYSTEM]: You are a precise geospatial analyst examining a fused satellite image. "
        "(Internal Key: Green=Vegetation, Cyan/White=Buildings, Dark=Water). You must NEVER mention these colors in your output.\n\n"
        "You must follow this exact structure:\n"
        "1. OBSERVATIONS: First, describe the visible density of buildings, the geographic spread of vegetation, and the layout of any water bodies in the image.\n"
        "2. ASSESSMENT: Answer the user's query strictly based on those physical observations. If asked for a ratio or proportion, compare their physical footprints using words instead of numerical math. Do not offer generic advice.\n\n"
        f"[USER QUERY]: {user_query}"
    )
    return instruction


def execute_optical_sar_fusion(optical_array: np.ndarray, sar_array: np.ndarray, user_query: str) -> dict:
    """
    Main interface function called by the Agent Controller.
    """
    composite = create_fusion_composite(optical_array, sar_array)
    prompt = generate_fusion_prompt(user_query)

    return {
        "status": "success",
        "composite_image": composite,
        "generated_prompt": prompt
    }