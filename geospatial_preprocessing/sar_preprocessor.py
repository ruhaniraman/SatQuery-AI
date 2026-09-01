import numpy as np
from scipy.ndimage import uniform_filter


def apply_lee_filter(img_array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Removes speckle noise while preserving hard structural edges.

    Expects a single 2D band (H, W). Call once per band for multi-band data.
    """
    if img_array.ndim != 2:
        raise ValueError(
            f"apply_lee_filter expects a single 2D band, got shape {img_array.shape}. "
            "Call it once per band for multi-band arrays."
        )

    img_float = img_array.astype(np.float32)
    local_mean = uniform_filter(img_float, (window_size, window_size))
    local_sqr_mean = uniform_filter(img_float**2, (window_size, window_size))
    local_variance = np.maximum(local_sqr_mean - local_mean**2, 0)

    overall_variance = np.var(img_float)
    # Weighting factor K
    k = local_variance / (local_variance + overall_variance + 1e-8)
    filtered = local_mean + k * (img_float - local_mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def _preprocess_single_band(band: np.ndarray) -> np.ndarray:
    # Apply dB scaling if data is raw linear power
    if np.issubdtype(band.dtype, np.floating) and np.max(band) > 0:
        band = 10 * np.log10(np.maximum(band, 1e-5))

    # Min-max scaling to 0-255
    min_val, max_val = np.percentile(band, (2, 98))
    band_8bit = np.clip((band - min_val) / (max_val - min_val + 1e-8) * 255.0, 0, 255).astype(np.uint8)

    # Filter speckle
    return apply_lee_filter(band_8bit)


def preprocess_sar(sar_raw_array: np.ndarray) -> np.ndarray:
    """Preprocesses SAR data.

    Accepts either:
      - a single band, shape (H, W)   -> e.g. VV alone
      - stacked bands, shape (H, W, N) -> e.g. VV/VH stacked as N=2

    Each band is dB-scaled, normalized, and speckle-filtered independently
    (a single global percentile/variance would let one band's brightness
    or noise level distort the other).
    """
    if sar_raw_array.ndim == 2:
        return _preprocess_single_band(sar_raw_array)

    if sar_raw_array.ndim == 3:
        bands = [_preprocess_single_band(sar_raw_array[:, :, i])
                  for i in range(sar_raw_array.shape[2])]
        return np.stack(bands, axis=-1)

    raise ValueError(f"Expected a 2D (H,W) or 3D (H,W,bands) array, got shape {sar_raw_array.shape}")
