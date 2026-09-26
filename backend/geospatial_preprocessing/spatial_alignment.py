import os

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling

# Minimum share of Image A's grid that Image B must cover for alignment to be meaningful.
DEFAULT_MIN_COVERAGE = 0.5


class AlignmentError(ValueError):
    """Raised when two rasters can't be meaningfully aligned because of the input data
    (no georeference, footprints don't overlap). It's a client error, not a server bug."""


def _footprint_coverage(ref, src, max_dim: int = 512) -> float:
    """Fraction of `ref`'s pixel grid that `src`'s footprint covers, measured in ref's CRS.

    Works by reprojecting a mask of ones (not the real pixel data), so it is independent of
    what the pixel values are (a legitimately black SAR water pixel is still "covered").
    Both grids are coarsened to <= max_dim px per side, so it stays cheap on huge scenes.
    """
    def coarse(ds):
        step = max(1.0, max(ds.width, ds.height) / max_dim)
        w = max(1, round(ds.width / step))
        h = max(1, round(ds.height / step))
        return w, h, ds.transform @ Affine.scale(ds.width / w, ds.height / h)

    dst_w, dst_h, dst_transform = coarse(ref)
    src_w, src_h, src_transform = coarse(src)

    src_mask = np.ones((src_h, src_w), dtype=np.uint8)
    dst_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=src_mask,
        destination=dst_mask,
        src_transform=src_transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=ref.crs,
        src_nodata=None,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return float(np.count_nonzero(dst_mask)) / dst_mask.size


def match_and_align_geotiffs(src_path_a: str, src_path_b: str, output_path_b_aligned: str,
                             resampling: Resampling = Resampling.bilinear,
                             min_coverage: float = DEFAULT_MIN_COVERAGE) -> float:
    """Reprojects and resamples Image B to match the exact CRS, extent, and shape of Image A.

    Before doing any real work it measures how much of Image A's grid Image B actually covers
    (after reprojecting into A's CRS). If that is below `min_coverage` it raises AlignmentError
    and writes nothing. Pass min_coverage=0 to disable the check.

    Returns the measured coverage (0..1). Pixels of the output outside Image B's footprint are
    filled with B's nodata value (or 0 if none is set).
    """
    if not os.path.exists(src_path_a):
        raise FileNotFoundError(f"Reference image not found: {src_path_a}")
    if not os.path.exists(src_path_b):
        raise FileNotFoundError(f"Image to align not found: {src_path_b}")

    with rasterio.open(src_path_a) as ref:
        if ref.crs is None:
            raise AlignmentError("Image A has no coordinate reference system (not georeferenced).")

        with rasterio.open(src_path_b) as src:
            if src.crs is None:
                raise AlignmentError("Image B has no coordinate reference system (not georeferenced).")

            coverage = _footprint_coverage(ref, src)
            if coverage < min_coverage:
                raise AlignmentError(
                    f"Image B covers only {coverage:.0%} of Image A's area "
                    f"(minimum {min_coverage:.0%}). The images likely show different places."
                )

            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height

            src_nodata = src.nodata
            dst_nodata = src_nodata if src_nodata is not None else 0

            profile = src.profile.copy()
            profile.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': dst_nodata,
            })

            with rasterio.open(output_path_b_aligned, 'w', **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src_nodata,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        dst_nodata=dst_nodata,
                        resampling=resampling
                    )

    return coverage