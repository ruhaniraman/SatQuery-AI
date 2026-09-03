import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
import cv2


def _normalize_per_band(img: np.ndarray) -> np.ndarray:
    """Percentile-normalize (2%-98%) each band independently, then cast to uint8.

    Doing this per-band (instead of on the flattened array) prevents one band
    with a much wider value range from washing out or blowing out the others.
    """
    img = img.astype(np.float32)
    p2 = np.percentile(img, 2, axis=(0, 1), keepdims=True)
    p98 = np.percentile(img, 98, axis=(0, 1), keepdims=True)
    clipped = np.clip(img, p2, p98)
    norm = (clipped - p2) / (p98 - p2 + 1e-8) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def load_and_standardize_image(file_path: str):
    # Support standard image formats
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image (bad path or corrupt file): {file_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, {"crs": None, "bounds": None, "resolution": None}

    # Support GeoTIFF
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        crs = str(src.crs) if src.crs else None
        res = src.res
        raw_data = src.read()  # Shape: (Bands, Height, Width)

        # Reorder to (Height, Width, Bands)
        img = reshape_as_image(raw_data)
        n_bands = img.shape[2]

        # Handle 1-band (grayscale), 2-band (e.g. SAR VV/VH), or 3+ bands
        if n_bands == 1:
            img = np.repeat(img, 3, axis=2)
        elif n_bands == 2:
            # No natural 3rd channel exists (e.g. VV/VH only) — synthesize one
            # as the per-pixel mean of the two bands, a common radar false-color
            # convention (R=VV, G=VH, B=mean(VV,VH)).
            third = np.mean(img[:, :, :2], axis=2, keepdims=True).astype(img.dtype)
            img = np.concatenate([img, third], axis=2)
        elif n_bands >= 3:
            img = img[:, :, :3]  # Take first 3 bands (RGB)

        norm_8bit = _normalize_per_band(img)

        metadata = {
            "crs": crs,
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "spatial_resolution_m": res[0]
        }
        return norm_8bit, metadata
