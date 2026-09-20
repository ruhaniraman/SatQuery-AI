import numpy as np
from scipy.ndimage import uniform_filter

# Floor for linear backscatter before taking log10 (-60 dB); avoids log(0).
_MIN_LINEAR = 1e-6

# Bounds for the estimated noise coefficient of variation squared (Cu^2 = 1/ENL).
# 0.02 ~ 50 looks, 0.5 ~ 2 looks: anything outside is almost certainly a bad estimate.
_CU2_BOUNDS = (0.02, 0.5)


def _estimate_noise_cv2(ci2: np.ndarray) -> float:
    """Estimate the speckle noise coefficient of variation squared (Cu^2 = 1/ENL) from the image.

    In homogeneous areas the local variance-to-mean^2 ratio equals Cu^2, while in textured
    areas it is larger. The lower quartile of the local ratios is therefore a robust proxy
    for the homogeneous (noise-only) level. Pass `enl` to lee_filter() if the product's
    equivalent number of looks is known, which is more reliable than this estimate.
    """
    step_r = max(1, ci2.shape[0] // 512)
    step_c = max(1, ci2.shape[1] // 512)
    cu2 = float(np.percentile(ci2[::step_r, ::step_c], 25))
    return float(np.clip(cu2, *_CU2_BOUNDS))


def lee_filter(img: np.ndarray, window_size: int = 7, enl: float = None) -> np.ndarray:
    """Lee speckle filter using the multiplicative speckle model.

    Output = local_mean + W * (pixel - local_mean), with W = 1 - Cu^2 / Ci^2 clipped to [0, 1]:
      - Ci^2 is the local squared coefficient of variation (var / mean^2),
      - Cu^2 = 1/ENL is the speckle noise level.
    Where the local variation is no bigger than pure speckle (flat areas) W -> 0 and the pixel
    is replaced by the local mean; where it's much bigger (edges, targets) W -> 1 and the pixel
    is kept. This is what actually preserves edges. (The previous version used the whole-image
    variance as a stand-in for the noise variance, which is small almost everywhere in a scene
    with real texture, so it over-smoothed uniformly instead of adapting per pixel.)

    Should be applied to LINEAR intensity, where speckle is multiplicative.
    """
    x = np.asarray(img, dtype=np.float32)
    mean = uniform_filter(x, window_size, mode="reflect")
    sqr_mean = uniform_filter(x * x, window_size, mode="reflect")
    var = np.maximum(sqr_mean - mean * mean, 0.0)
    ci2 = var / np.maximum(mean * mean, 1e-12)
    cu2 = (1.0 / enl) if enl else _estimate_noise_cv2(ci2)
    weight = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-12), 0.0, 1.0)
    return (mean + weight * (x - mean)).astype(np.float32)


def apply_lee_filter(img_array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Despeckle a single 8-bit display-scaled band (H, W) and return uint8.

    Only for data that is ALREADY display-scaled (PNG/JPEG previews). For real SAR rasters,
    use preprocess_sar()/despeckle_band(), which filter in the linear domain before any
    display scaling.
    """
    if img_array.ndim != 2:
        raise ValueError(
            f"apply_lee_filter expects a single 2D band, got shape {img_array.shape}. "
            "Call it once per band for multi-band arrays."
        )
    return np.clip(lee_filter(img_array, window_size), 0, 255).astype(np.uint8)


def despeckle_band(band: np.ndarray, valid: np.ndarray, integer_source: bool):
    """Despeckle one raw SAR band. Returns (float32 array, input_units).

    input_units is one of:
      "linear"  - float, all valid values >= 0 (power/intensity; speckle is multiplicative)
      "db"      - float with some negative valid values (linear power can't be negative,
                  so a negative value means this is already in dB)
      "display" - integer DNs (already stretched for display; dB conversion isn't meaningful)

    Linear and dB inputs are filtered in the LINEAR domain (correct for multiplicative speckle)
    and returned in dB. Display data is filtered as-is. Invalid pixels (nodata/NaN) are filled
    with the valid median before filtering so they don't bleed into their neighbours; callers
    keep using the validity mask to exclude them afterwards.
    """
    band = np.asarray(band, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise ValueError("SAR band has no valid pixels.")

    work = np.where(valid, band, np.float32(np.median(band[valid])))

    if integer_source:
        return lee_filter(work), "display"

    units = "db" if float(band[valid].min()) < 0 else "linear"
    linear = np.power(10.0, work / 10.0, dtype=np.float32) if units == "db" else work
    linear = np.maximum(linear, _MIN_LINEAR)
    filtered = np.maximum(lee_filter(linear), _MIN_LINEAR)
    return (10.0 * np.log10(filtered)).astype(np.float32), units


def preprocess_sar(sar_raw_array: np.ndarray, valid_mask: np.ndarray = None,
                   integer_source: bool = None) -> np.ndarray:
    """Despeckle raw SAR and return a float32 array (dB for float sources, display units for
    integer sources). Intensity stretching to 8-bit is left to the caller, so a before/after
    pair can share one stretch (see geotiff_loader.load_and_standardize_pair).

    Accepts a single band (H, W) or stacked bands (H, W, N); each band is handled independently.
    """
    arr = np.asarray(sar_raw_array)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2D (H,W) or 3D (H,W,bands) array, got shape {sar_raw_array.shape}")

    if integer_source is None:
        integer_source = bool(np.issubdtype(arr.dtype, np.integer))
    valid = np.ones(arr.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)

    out = np.empty(arr.shape, dtype=np.float32)
    for i in range(arr.shape[2]):
        out[:, :, i], _ = despeckle_band(arr[:, :, i], valid, integer_source)
    return out[:, :, 0] if squeeze else out
