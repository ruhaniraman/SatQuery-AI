import numpy as np

def check_bbox_overlap(bbox1, bbox2) -> bool:
    """bbox format: [min_lon, min_lat, max_lon, max_lat]"""
    overlap_lon = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    overlap_lat = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    return (overlap_lon * overlap_lat) > 0

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

if __name__ == "__main__":
    print("--- 1. Testing Overlap Failure ---")
    bengaluru = [77.50, 12.90, 77.65, 13.05]
    delhi = [77.10, 28.50, 77.30, 28.70]
    print("Overlap Check:", check_bbox_overlap(bengaluru, delhi))

    print("\n--- 2. Testing Data Type Failure ---")
    bad_img = np.random.randint(0, 4096, (256, 256, 3), dtype=np.uint16)
    print("Result:", validate_downstream_payload(bad_img))

    print("\n--- 3. Testing Dimension Mismatch Failure ---")
    img1 = np.zeros((512, 512, 3), dtype=np.uint8)
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    print("Result:", validate_downstream_payload(img1, img2))