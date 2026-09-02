"""
geotiff_loader.py
==================
Member 4: Geospatial Pipeline & Ingestion (Critical Path / Stage 1 bottleneck)

Reads .tif/.geotiff (via rasterio) OR standard .png/.jpg, extracts spatial
metadata when available, and normalizes everything to an 8-bit RGB numpy
array that Members 2, 3, and 5 can consume directly.

Integration contract:
    output image  -> HxWx3 uint8 numpy array (matches what cdvqa_engine.py /
                      single_image_vqa_engine.py / optical_sar_fusion_model.py expect)
    output meta   -> dict, JSON-serializable
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class GeoImageLoader:
    """Loads GeoTIFF or standard raster images and normalizes to 8-bit RGB."""

    TIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
    STANDARD_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def load(self, filepath: str) -> tuple[np.ndarray, dict]:
        """
        Args:
            filepath: path to a .tif/.geotiff or .png/.jpg file.

        Returns:
            (normalized_8bit_rgb_array, metadata_dict)
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext in self.TIFF_EXTENSIONS:
            return self._load_tiff(filepath)
        elif ext in self.STANDARD_EXTENSIONS:
            return self._load_standard(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Expected {self.TIFF_EXTENSIONS | self.STANDARD_EXTENSIONS}")

    # ------------------------------------------------------------------ #

    def _load_tiff(self, filepath: str) -> tuple[np.ndarray, dict]:
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio is not installed. Run: pip install rasterio")

        with rasterio.open(filepath) as src:
            bands = src.read()  # shape: (bands, H, W)
            metadata = {
                "bounding_box": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "resolution": {"x": src.res[0], "y": src.res[1]},
                "crs": str(src.crs) if src.crs else None,
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "dtype": str(src.dtypes[0]),
                "source_file": filepath,
            }

        normalized = self.normalize_to_8bit(bands)
        return normalized, metadata

    def _load_standard(self, filepath: str) -> tuple[np.ndarray, dict]:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {filepath}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        metadata = {
            "bounding_box": None,
            "resolution": None,
            "crs": None,
            "width": img_rgb.shape[1],
            "height": img_rgb.shape[0],
            "band_count": img_rgb.shape[2],
            "dtype": str(img_rgb.dtype),
            "source_file": filepath,
        }
        return img_rgb, metadata

    # ------------------------------------------------------------------ #

    def normalize_to_8bit(self, bands: np.ndarray) -> np.ndarray:
        """
        Converts a (bands, H, W) array of any dtype (uint16, float32, etc.)
        into a standard HxWx3 uint8 RGB array using per-band min-max stretch.
        """
        n_bands = bands.shape[0]

        # Pick RGB-ish bands: if >=3 bands, use first three; if fewer, replicate.
        if n_bands >= 3:
            rgb_bands = bands[:3]
        elif n_bands == 1:
            rgb_bands = np.repeat(bands, 3, axis=0)
        else:
            rgb_bands = np.concatenate([bands, bands[-1:]], axis=0)[:3]

        out = np.zeros((rgb_bands.shape[1], rgb_bands.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            band = rgb_bands[i].astype(np.float32)
            band_min, band_max = np.percentile(band, 1), np.percentile(band, 99)
            if band_max - band_min < 1e-6:
                stretched = np.zeros_like(band)
            else:
                stretched = np.clip((band - band_min) / (band_max - band_min), 0, 1) * 255
            out[:, :, i] = stretched.astype(np.uint8)

        return out


class SARPreprocessor:
    """
    Basic despeckling for SAR intensity data before 8-bit conversion.
    Implements a lightweight 3x3 Lee filter (fallback: median blur).
    """

    def __init__(self, method: str = "lee", kernel_size: int = 3):
        assert method in ("lee", "median")
        self.method = method
        self.kernel_size = kernel_size

    def despeckle(self, sar_array: np.ndarray) -> np.ndarray:
        """
        Args:
            sar_array: 2D float/uint array of raw SAR backscatter intensity.

        Returns:
            Despeckled, 8-bit single-channel array.
        """
        arr = sar_array.astype(np.float32)

        if self.method == "median":
            denoised = cv2.medianBlur(self._to_uint8(arr), self.kernel_size)
            return denoised

        return self._lee_filter(arr)

    def _lee_filter(self, arr: np.ndarray) -> np.ndarray:
        """Simple adaptive Lee filter using local mean/variance in a k x k window."""
        k = self.kernel_size
        mean = cv2.boxFilter(arr, ddepth=-1, ksize=(k, k))
        mean_sq = cv2.boxFilter(arr * arr, ddepth=-1, ksize=(k, k))
        variance = mean_sq - mean * mean
        variance = np.clip(variance, 1e-6, None)

        overall_variance = np.var(arr)
        weight = variance / (variance + overall_variance + 1e-6)

        filtered = mean + weight * (arr - mean)
        return self._to_uint8(filtered)

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        arr_min, arr_max = np.percentile(arr, 1), np.percentile(arr, 99)
        if arr_max - arr_min < 1e-6:
            return np.zeros_like(arr, dtype=np.uint8)
        stretched = np.clip((arr - arr_min) / (arr_max - arr_min), 0, 1) * 255
        return stretched.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Standalone smoke test (mock data, per Hour 0-12 plan)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # --- Build a synthetic 3-band 16-bit "GeoTIFF" to test normalization ---
    if RASTERIO_AVAILABLE:
        import rasterio
        from rasterio.transform import from_origin

        synthetic_bands = (np.random.rand(3, 128, 128) * 60000).astype(np.uint16)
        transform = from_origin(-105.0, 40.0, 0.001, 0.001)

        test_tif_path = "mock_test.tif"
        with rasterio.open(
            test_tif_path, "w", driver="GTiff", height=128, width=128,
            count=3, dtype="uint16", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(synthetic_bands)

        loader = GeoImageLoader()
        img, meta = loader.load(test_tif_path)
        print("=== GeoTIFF Load Test ===")
        print("Output shape:", img.shape, "dtype:", img.dtype)
        print("Metadata:", meta)

    # --- Test SAR despeckling ---
    sar_synthetic = (np.random.rayleigh(scale=50, size=(128, 128))).astype(np.float32)
    sar_pre = SARPreprocessor(method="lee")
    despeckled = sar_pre.despeckle(sar_synthetic)
    print("\n=== SAR Despeckle Test ===")
    print("Output shape:", despeckled.shape, "dtype:", despeckled.dtype)
    print("Min/Max:", despeckled.min(), despeckled.max())
