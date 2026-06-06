"""
Raster Alignment Module

Warps source rasters to match target raster grids (transform, resolution, extent).
Uses GDAL/rasterio for high-quality reprojection and resampling.

Reference: agent.md §3.4, agent-2.md align_raster_to_target tool
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

logger = logging.getLogger(__name__)


# Resampling method mapping
RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
}


def align_raster_to_target(
    src_path: str,
    target_path: str,
    out_path: str,
    resampling: str = "cubic"
) -> str:
    """
    Warp source raster to match target raster's grid.

    Aligns source to target's:
    - CRS (coordinate reference system)
    - Transform (origin, pixel size)
    - Extent (bounding box)
    - Dimensions (width, height)

    Reference: agent-2.md align_raster_to_target tool

    Args:
        src_path: Path to source raster
        target_path: Path to target raster (grid to match)
        out_path: Output path for aligned raster
        resampling: Resampling method (nearest, bilinear, cubic, etc.)

    Returns:
        Output file path

    Raises:
        FileNotFoundError: If source or target doesn't exist
        RuntimeError: If alignment fails
    """
    src_path = Path(src_path)
    target_path = Path(target_path)
    out_path = Path(out_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Source raster not found: {src_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target raster not found: {target_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    resample_method = RESAMPLING_METHODS.get(resampling, Resampling.cubic)

    logger.info(f"Aligning {src_path.name} to {target_path.name} grid")

    try:
        # Read target properties
        with rasterio.open(target_path) as target:
            target_crs = target.crs
            target_transform = target.transform
            target_width = target.width
            target_height = target.height
            target_bounds = target.bounds

        logger.debug(
            f"Target grid: {target_width}x{target_height}, "
            f"CRS={target_crs}, bounds={target_bounds}"
        )

        # Read and reproject source
        with rasterio.open(src_path) as src:
            # Create output profile
            out_profile = src.profile.copy()
            out_profile.update({
                'crs': target_crs,
                'transform': target_transform,
                'width': target_width,
                'height': target_height,
                'compress': 'lzw'
            })

            # Allocate output array
            out_data = np.zeros(
                (src.count, target_height, target_width),
                dtype=src.dtypes[0]
            )

            # Reproject each band
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=out_data[band_idx - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resample_method
                )

            # Write aligned raster
            with rasterio.open(str(out_path), 'w', **out_profile) as dst:
                dst.write(out_data)

        logger.info(
            f"Aligned raster saved: {out_path} "
            f"({target_width}x{target_height}, {out_profile['count']} bands)"
        )

        return str(out_path)

    except Exception as e:
        error_msg = f"Raster alignment failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def reproject_raster(
    src_path: str,
    out_path: str,
    target_crs: str,
    resolution: Optional[float] = None,
    resampling: str = "bilinear"
) -> str:
    """
    Reproject raster to target CRS with optional resolution change.

    Reference: agent-2.md reproject_raster tool

    Args:
        src_path: Path to source raster
        out_path: Output path
        target_crs: Target CRS (e.g., "EPSG:26917")
        resolution: Optional target resolution in CRS units
        resampling: Resampling method

    Returns:
        Output file path

    Raises:
        FileNotFoundError: If source doesn't exist
        RuntimeError: If reprojection fails
    """
    src_path = Path(src_path)
    out_path = Path(out_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Source raster not found: {src_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    resample_method = RESAMPLING_METHODS.get(resampling, Resampling.bilinear)
    dst_crs = CRS.from_string(target_crs)

    logger.info(f"Reprojecting {src_path.name} to {target_crs}")

    try:
        with rasterio.open(src_path) as src:
            # Calculate transform and dimensions
            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=resolution
            )

            # Update profile
            out_profile = src.profile.copy()
            out_profile.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'compress': 'lzw'
            })

            # Allocate output
            out_data = np.zeros(
                (src.count, height, width),
                dtype=src.dtypes[0]
            )

            # Reproject each band
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=out_data[band_idx - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resample_method
                )

            # Write output
            with rasterio.open(str(out_path), 'w', **out_profile) as dst:
                dst.write(out_data)

        logger.info(f"Reprojected raster saved: {out_path} ({width}x{height})")

        return str(out_path)

    except Exception as e:
        error_msg = f"Raster reprojection failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_raster_info(raster_path: str) -> dict:
    """
    Get raster metadata for alignment verification.

    Args:
        raster_path: Path to raster file

    Returns:
        Dict with crs, transform, width, height, bounds, bands, dtype
    """
    with rasterio.open(raster_path) as src:
        return {
            "crs": str(src.crs),
            "transform": list(src.transform),
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "resolution": (src.transform.a, abs(src.transform.e))
        }


def verify_alignment(raster1_path: str, raster2_path: str) -> Tuple[bool, str]:
    """
    Verify two rasters are aligned to the same grid.

    Args:
        raster1_path: First raster path
        raster2_path: Second raster path

    Returns:
        Tuple of (is_aligned, message)
    """
    info1 = get_raster_info(raster1_path)
    info2 = get_raster_info(raster2_path)

    issues = []

    if info1["crs"] != info2["crs"]:
        issues.append(f"CRS mismatch: {info1['crs']} vs {info2['crs']}")

    if info1["width"] != info2["width"] or info1["height"] != info2["height"]:
        issues.append(
            f"Dimension mismatch: {info1['width']}x{info1['height']} "
            f"vs {info2['width']}x{info2['height']}"
        )

    if info1["transform"] != info2["transform"]:
        issues.append("Transform mismatch (different origin or resolution)")

    if issues:
        return False, "; ".join(issues)

    return True, "Rasters are aligned"
