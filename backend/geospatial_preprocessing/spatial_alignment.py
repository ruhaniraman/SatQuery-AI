import os
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np


def match_and_align_geotiffs(src_path_a: str, src_path_b: str, output_path_b_aligned: str,
                              resampling: Resampling = Resampling.bilinear):
    """Reprojects and resamples Image B to match the exact CRS, extent, and shape of Image A.

    Pixels in the output that fall outside Image B's original coverage are
    filled with B's nodata value (or 0 if none is set) — check for this
    before assuming an all-zero region means alignment failed.
    """
    if not os.path.exists(src_path_a):
        raise FileNotFoundError(f"Reference image not found: {src_path_a}")
    if not os.path.exists(src_path_b):
        raise FileNotFoundError(f"Image to align not found: {src_path_b}")

    with rasterio.open(src_path_a) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

        with rasterio.open(src_path_b) as src:
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

    # Sanity check: warn (don't fail) if the two images turn out not to overlap at all
    with rasterio.open(output_path_b_aligned) as aligned:
        arr = aligned.read()
        valid = arr != dst_nodata
        if not valid.any():
            print(
                f"Warning: aligned output '{output_path_b_aligned}' has no valid pixels — "
                "src_path_a and src_path_b likely do not spatially overlap."
            )

    return True
