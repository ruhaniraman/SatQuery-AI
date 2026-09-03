import numpy as np
import cv2

def create_fusion_composite(optical_array: np.ndarray, sar_array: np.ndarray) -> np.ndarray:
    """
    Combines an Optical RGB array and a SAR intensity array into a 3-channel
    False-Color Composite (Red = Optical Red, Green = Optical Green, Blue = SAR).
    """
    # 1. Ensure SAR array matches the spatial dimensions of the Optical image
    target_shape = (optical_array.shape[1], optical_array.shape[0])
    if sar_array.shape[:2] != optical_array.shape[:2]:
        sar_array = cv2.resize(sar_array, target_shape, interpolation=cv2.INTER_AREA)

    # 2. Extract Red and Green channels from Optical RGB
    red_channel = optical_array[:, :, 0]
    green_channel = optical_array[:, :, 1]

    # 3. Extract single SAR intensity channel
    sar_channel = sar_array if len(sar_array.shape) == 2 else sar_array[:, :, 0]

    # 4. Stack into a 3-channel composite image: [Red, Green, SAR]
    composite_array = np.dstack((red_channel, green_channel, sar_channel))
    return composite_array


def generate_fusion_prompt(user_query: str) -> str:
    """
    Generates structured VLM prompt instructions to interpret the multi-modal composite.
    """
    instruction = (
        "You are analyzing a False-Color Composite satellite image combining Optical and SAR data.\n"
        "- RED and GREEN channels represent optical spectral reflections (vegetation, soil).\n"
        "- BLUE channel represents SAR radar backscatter (bright blue = hard/built structures, dark blue = water/flat surfaces).\n\n"
        f"Answer the query based on this multi-modal context: '{user_query}'"
    )
    return instruction


def execute_optical_sar_fusion(optical_array: np.ndarray, sar_array: np.ndarray, user_query: str) -> dict:
    """
    Main interface function called by the Agent Controller (Member 1).
    """
    composite = create_fusion_composite(optical_array, sar_array)
    prompt = generate_fusion_prompt(user_query)

    return {
        "status": "success",
        "composite_image": composite,
        "generated_prompt": prompt
    }