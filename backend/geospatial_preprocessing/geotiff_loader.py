from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.enums import ColorInterp

from geospatial_preprocessing.sar_preprocessor import apply_lee_filter, despeckle_band

_STANDARD_EXTS = (".png", ".jpg", ".jpeg")
_SAMPLE_LIMIT = 2_000_000          # max pixels per band used to estimate stretch percentiles
_LOW_PCT, _HIGH_PCT = 2, 98

# Band names that unambiguously identify a colour band (Sentinel-2 B02/B03/B04 are B/G/R).
_BAND_NAME_TO_COLOUR = {
    "red": "r", "b04": "r",
    "green": "g", "b03": "g",
    "blue": "b", "b02": "b",
}


@dataclass
class _Prepared:
    img: np.ndarray            # (H, W, 3): float32 raw values if needs_stretch, else final uint8
    valid: np.ndarray          # (H, W) bool
    meta: dict
    needs_stretch: bool


def _empty_meta() -> dict:
    return {
        "crs": None, "bounds": None,
        "spatial_resolution": None, "resolution_unit": None,
        "warnings": [], "sar_input_units": None,
    }


def _resolution_unit(crs) -> Optional[str]:
    if crs is None:
        return None
    if crs.is_geographic:
        return "degree"
    try:
        return crs.linear_units or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- band selection

def _optical_band_indexes(src) -> Tuple[List[int], List[str]]:
    """Choose which (1-based) bands are R, G, B, instead of blindly taking the first three.

    Order of evidence: declared colour interpretation, then unambiguous band names, then a
    fallback to bands 1-3 (with a warning, since e.g. a Sentinel-2 B02/B03/B04 stack is B,G,R).
    """
    n = src.count
    if n <= 2:
        return list(range(1, n + 1)), []

    found = {}
    for i, ci in enumerate(src.colorinterp, start=1):
        if ci == ColorInterp.red:
            found.setdefault("r", i)
        elif ci == ColorInterp.green:
            found.setdefault("g", i)
        elif ci == ColorInterp.blue:
            found.setdefault("b", i)
    if len(found) == 3:
        return [found["r"], found["g"], found["b"]], []

    found = {}
    for i, desc in enumerate(src.descriptions, start=1):
        colour = _BAND_NAME_TO_COLOUR.get((desc or "").strip().lower())
        if colour:
            found.setdefault(colour, i)
    if len(found) == 3:
        return [found["r"], found["g"], found["b"]], []

    return [1, 2, 3], ["Band order is not declared in the file; assumed the first three bands are R, G, B."]


def _sar_band_indexes(src) -> Tuple[List[int], List[str]]:
    return list(range(1, min(src.count, 3) + 1)), []


# --------------------------------------------------------------------------- reading

def _prepare_standard(path: str, modality: str) -> _Prepared:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image (bad path or corrupt file): {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if modality == "sar":
        # Already display-scaled 8-bit: despeckle directly. Grayscale radar previews usually
        # have identical channels, so filter once and copy.
        if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
            g = apply_lee_filter(img[:, :, 0])
            img = np.repeat(g[:, :, None], 3, axis=2)
        else:
            img = np.dstack([apply_lee_filter(img[:, :, c]) for c in range(3)])

    meta = _empty_meta()
    return _Prepared(img, np.ones(img.shape[:2], dtype=bool), meta, needs_stretch=False)


def _prepare_geotiff(path: str, modality: str) -> _Prepared:
    meta = _empty_meta()
    with rasterio.open(path) as src:
        indexes, warns = _sar_band_indexes(src) if modality == "sar" else _optical_band_indexes(src)
        meta["warnings"].extend(warns)

        data = src.read(indexes=indexes).astype(np.float32)          # (k, H, W): only the bands we need
        integer_source = all(np.issubdtype(np.dtype(src.dtypes[i - 1]), np.integer) for i in indexes)
        nodata = src.nodata

        alpha_idx = next((i for i, ci in enumerate(src.colorinterp, start=1) if ci == ColorInterp.alpha), None)
        alpha = (src.read(alpha_idx) > 0) if alpha_idx else None

        bounds = src.bounds
        meta.update({
            "crs": str(src.crs) if src.crs else None,
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "spatial_resolution": src.res[0],
            "resolution_unit": _resolution_unit(src.crs),
        })

    img = np.transpose(data, (1, 2, 0))                              # (H, W, k)

    # Validity: finite in every band, not (all bands == nodata), and inside the alpha mask.
    valid = np.isfinite(img).all(axis=2)
    if nodata is not None and not np.isnan(nodata):
        valid &= ~(img == nodata).all(axis=2)
    if alpha is not None:
        valid &= alpha
    if not valid.any():
        raise ValueError(f"No valid (non-nodata) pixels in {path}")
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if modality == "sar":
        units = None
        bands = []
        for c in range(img.shape[2]):
            band, units = despeckle_band(img[:, :, c], valid, integer_source)
            bands.append(band)
        img = np.stack(bands, axis=2)
        meta["sar_input_units"] = units

    # Bring to exactly 3 channels
    k = img.shape[2]
    if k == 1:
        img = np.repeat(img, 3, axis=2)
    elif k == 2:
        # No natural 3rd channel (e.g. VV/VH): synthesize the per-pixel mean of the two bands.
        img = np.concatenate([img, np.mean(img[:, :, :2], axis=2, keepdims=True)], axis=2)

    return _Prepared(img.astype(np.float32), valid, meta, needs_stretch=True)


def _prepare(path: str, modality: str) -> _Prepared:
    if path.lower().endswith(_STANDARD_EXTS):
        return _prepare_standard(path, modality)
    return _prepare_geotiff(path, modality)


# --------------------------------------------------------------------------- normalisation

def _sample(values: np.ndarray) -> np.ndarray:
    step = max(1, values.size // _SAMPLE_LIMIT)
    return values[::step]


def _shared_bounds(items: List[_Prepared]):
    """Per-band low/high percentiles over the VALID pixels of all given images together."""
    lo, hi = np.zeros(3, np.float32), np.zeros(3, np.float32)
    for c in range(3):
        samples = np.concatenate([_sample(p.img[:, :, c][p.valid]) for p in items])
        lo[c], hi[c] = np.percentile(samples, [_LOW_PCT, _HIGH_PCT])
        if hi[c] <= lo[c]:
            hi[c] = lo[c] + 1.0
    return lo, hi


def _stretch(img: np.ndarray, valid: np.ndarray, lo, hi) -> np.ndarray:
    out = np.empty(img.shape, dtype=np.uint8)
    for c in range(3):
        scaled = (img[:, :, c] - lo[c]) / (hi[c] - lo[c]) * 255.0
        out[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def _finalize(items: List[_Prepared]):
    """Stretch all raster inputs with ONE shared set of percentiles.

    Stretching each image on its own would map the same ground reflectance to different pixel
    values in the "before" and "after" images and create differences that aren't real. A shared
    stretch keeps the two on the same scale (the difference between them stays meaningful).
    """
    pending = [p for p in items if p.needs_stretch]
    if pending:
        lo, hi = _shared_bounds(pending)
        for p in pending:
            p.img = _stretch(p.img, p.valid, lo, hi)
            p.meta["normalization"] = "shared" if len(pending) > 1 else "single-image"
    return [(p.img, {**p.meta, "valid_mask": p.valid}) for p in items]


# --------------------------------------------------------------------------- public API

def load_and_standardize_image(file_path: str, modality: str = "optical"):
    """Load one image as uint8 RGB (H, W, 3) plus a metadata dict.

    metadata: crs, bounds [left, bottom, right, top], spatial_resolution + resolution_unit,
    warnings (list of str), sar_input_units, valid_mask ((H, W) bool: False for nodata/NaN).
    modality "sar" applies dB conversion + speckle filtering (rasters) or speckle filtering
    (already display-scaled PNG/JPEG).
    """
    ((img, meta),) = _finalize([_prepare(file_path, modality)])
    return img, meta


def load_and_standardize_pair(path_a: str, path_b: str, modality: str = "optical"):
    """Load a before/after pair with a SHARED intensity stretch. Returns (img_a, meta_a, img_b, meta_b)."""
    (img_a, meta_a), (img_b, meta_b) = _finalize([_prepare(path_a, modality), _prepare(path_b, modality)])
    return img_a, meta_a, img_b, meta_b
