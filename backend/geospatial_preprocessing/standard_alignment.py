"""Alignment of ordinary (non-georeferenced) image pairs: ORB feature matching + a validated
affine warp, with a plain-resize fallback."""
import cv2
import numpy as np

# A before/after pair is already meant to be co-registered, so a good transform is close to the
# identity. Anything else usually means RANSAC latched onto a few chance matches (common between
# different seasons/sensors). These bounds are conservative guesses, not validated values.
MIN_ORB_INLIERS = 12
MAX_ORB_SCALE_DEVIATION = 0.15     # similarity scale must be within 1 +/- this
MAX_ORB_ROTATION_DEG = 10.0
MAX_ORB_SHIFT_FRACTION = 0.25      # translation as a fraction of the image side


def _transform_is_plausible(matrix, inliers, shape) -> tuple:
    """Returns (ok, reason). Rejects a missing matrix, too few RANSAC inliers, or an extreme
    scale / rotation / translation."""
    if matrix is None or inliers is None:
        return False, "no valid matrix"
    n_inliers = int(np.count_nonzero(inliers))
    if n_inliers < MIN_ORB_INLIERS:
        return False, f"only {n_inliers} inliers"
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(scale - 1.0) > MAX_ORB_SCALE_DEVIATION:
        return False, f"implausible scale {scale:.2f}"
    rotation = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    if rotation > MAX_ORB_ROTATION_DEG:
        return False, f"implausible rotation {rotation:.0f} deg"
    h, w = shape
    if abs(matrix[0, 2]) > MAX_ORB_SHIFT_FRACTION * w or abs(matrix[1, 2]) > MAX_ORB_SHIFT_FRACTION * h:
        return False, "implausible translation"
    return True, ""


def align_standard_images(img1: np.ndarray, img2: np.ndarray):
    """Uses ORB to find matching features and performs a safe 2D affine warp.

    Returns (aligned_img2, valid_mask): the mask is False where the warp pulled in pixels from
    outside img2 (the filled border), so callers don't mistake that edge for real change.
    """
    height, width = img1.shape[:2]
    all_valid = np.ones((height, width), dtype=bool)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # If no features found, fallback safely
    if des1 is None or des2 is None:
        return cv2.resize(img2, (width, height)), all_valid
        
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    keep = int(len(matches) * 0.15)
    matches = matches[:keep]
    if len(matches) < MIN_ORB_INLIERS:
        print(f"Only {len(matches)} candidate feature matches; falling back to resize.")
        return cv2.resize(img2, (width, height)), all_valid
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
        
    transform_matrix, inliers = cv2.estimateAffinePartial2D(points2, points1, cv2.RANSAC)
    
    ok, reason = _transform_is_plausible(transform_matrix, inliers, (height, width))
    if ok:
        # Use warpAffine instead of warpPerspective
        aligned_img2 = cv2.warpAffine(img2, transform_matrix, (width, height))
        src_ones = np.full(img2.shape[:2], 255, dtype=np.uint8)
        valid = cv2.warpAffine(src_ones, transform_matrix, (width, height)) == 255
        return aligned_img2, valid
    else:
        print(f"Affine alignment rejected ({reason}), falling back to resize.")
        return cv2.resize(img2, (width, height)), all_valid
