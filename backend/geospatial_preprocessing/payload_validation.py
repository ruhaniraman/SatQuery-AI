import numpy as np

def validate_downstream_payload(img_a: np.ndarray, img_b: np.ndarray = None) -> bool:
    if img_a.dtype != np.uint8:
        print(f"Validation Failed: Expected uint8, but got {img_a.dtype}")
        return False
    if len(img_a.shape) != 3 or img_a.shape[2] != 3:
        print(f"Validation Failed: Expected shape (H, W, 3), but got {img_a.shape}")
        return False
    if img_b is not None:
        if img_a.shape != img_b.shape:
            print(f"Validation Failed: Shape mismatch: {img_a.shape} vs {img_b.shape}")
            return False
    return True
