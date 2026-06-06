"""
Point Extraction Module

Converts clipped fused raster to 3D points with filtering.
Extracts (x, y, z, R, G, B, NIR) point tables from 5-band rasters.

Reference: agent.md §4.2-4.3, agent-2.md raster_to_points tool
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
import pandas as pd

logger = logging.getLogger(__name__)


# Default thresholds per agent spec
DEFAULT_HEIGHT_MIN_M = 1.5  # agent.md §4.3
DEFAULT_NDVI_THRESHOLD = 0.3  # agent-2.md rules.vegetation_filter
DEFAULT_NIR_RED_RATIO = 1.3  # agent-2.md rules.vegetation_filter


def raster_to_points(
    raster_path: str,
    out_points_path: Optional[str] = None,
    height_band: int = 5,
    height_min: float = DEFAULT_HEIGHT_MIN_M,
    ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    Convert clipped fused raster to (x, y, z, R, G, B, NIR) point table.

    Applies height filtering and vegetation masking per agent specs.

    Reference: agent.md §4.2, agent-2.md raster_to_points tool

    Args:
        raster_path: Path to fused 5-band raster (R,G,B,NIR,Height)
        out_points_path: Optional path to save points as CSV/Parquet
        height_band: Band index for height (1-based, default 5)
        height_min: Minimum height to keep (default 1.5m)
        ndvi_threshold: NDVI threshold for vegetation mask (default 0.3)
        mask: Optional binary mask (True = valid pixels)

    Returns:
        Tuple of (points_array, metadata) where:
        - points_array: structured array with x, y, z, r, g, b, nir fields
        - metadata: dict with counts and filter stats

    Raises:
        FileNotFoundError: If raster doesn't exist
        ValueError: If raster doesn't have expected bands
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    logger.info(f"Extracting points from {raster_path.name}")

    with rasterio.open(raster_path) as src:
        if src.count < 5:
            raise ValueError(
                f"Expected 5-band raster (R,G,B,NIR,Height), got {src.count} bands"
            )

        # Read all bands
        data = src.read()  # (bands, height, width)
        transform = src.transform

        # Extract individual bands
        r_band = data[0].astype(np.float32)
        g_band = data[1].astype(np.float32)
        b_band = data[2].astype(np.float32)
        nir_band = data[3].astype(np.float32)
        height_band_data = data[height_band - 1].astype(np.float32)

    # Get pixel coordinates
    rows, cols = np.indices(height_band_data.shape)

    # Convert pixel indices to world coordinates using geotransform
    # x = transform.c + col * transform.a + row * transform.b
    # y = transform.f + col * transform.d + row * transform.e
    x_coords = transform.c + cols * transform.a + rows * transform.b
    y_coords = transform.f + cols * transform.d + rows * transform.e

    # Flatten arrays
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    z_flat = height_band_data.ravel()
    r_flat = r_band.ravel()
    g_flat = g_band.ravel()
    b_flat = b_band.ravel()
    nir_flat = nir_band.ravel()

    total_pixels = len(z_flat)

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.ravel()
    else:
        mask_flat = np.ones(total_pixels, dtype=bool)

    # Apply height filter (agent.md §4.3)
    height_mask = z_flat >= height_min
    valid_after_height = height_mask.sum()

    # Compute NDVI for vegetation filtering
    # NDVI = (NIR - Red) / (NIR + Red)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_flat - r_flat) / (nir_flat + r_flat + 1e-10)
        ndvi = np.nan_to_num(ndvi, nan=0.0)

    # Also compute NIR/Red ratio (agent-2.md alternative)
    with np.errstate(divide='ignore', invalid='ignore'):
        nir_red_ratio = nir_flat / (r_flat + 1e-10)
        nir_red_ratio = np.nan_to_num(nir_red_ratio, nan=0.0)

    # Vegetation mask: exclude high NDVI OR high NIR/Red ratio
    vegetation_mask = (ndvi > ndvi_threshold) | (nir_red_ratio > DEFAULT_NIR_RED_RATIO)
    non_vegetation_mask = ~vegetation_mask
    valid_after_vegetation = (height_mask & non_vegetation_mask).sum()

    # Combined valid mask
    valid_mask = mask_flat & height_mask & non_vegetation_mask

    # Handle nodata (assume 0 or very negative values)
    nodata_mask = (z_flat > -1000) & (z_flat != 0)
    valid_mask = valid_mask & nodata_mask

    final_count = valid_mask.sum()

    logger.info(
        f"Point extraction: {total_pixels:,} total → "
        f"{valid_after_height:,} (height>{height_min}m) → "
        f"{valid_after_vegetation:,} (non-vegetation) → "
        f"{final_count:,} final points"
    )

    # Extract valid points
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    r_valid = r_flat[valid_mask]
    g_valid = g_flat[valid_mask]
    b_valid = b_flat[valid_mask]
    nir_valid = nir_flat[valid_mask]

    # Create structured array
    dtype = np.dtype([
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float32),
        ('r', np.float32),
        ('g', np.float32),
        ('b', np.float32),
        ('nir', np.float32)
    ])

    points = np.zeros(final_count, dtype=dtype)
    points['x'] = x_valid
    points['y'] = y_valid
    points['z'] = z_valid
    points['r'] = r_valid
    points['g'] = g_valid
    points['b'] = b_valid
    points['nir'] = nir_valid

    # Metadata for QC
    vegetation_pct = vegetation_mask.sum() / total_pixels * 100
    metadata = {
        "total_pixels": total_pixels,
        "valid_after_height_filter": valid_after_height,
        "valid_after_vegetation_filter": valid_after_vegetation,
        "final_point_count": final_count,
        "height_min_threshold": height_min,
        "ndvi_threshold": ndvi_threshold,
        "vegetation_percentage": vegetation_pct,
        "retention_rate": final_count / total_pixels * 100 if total_pixels > 0 else 0
    }

    # Save to file if requested
    if out_points_path:
        out_points_path = Path(out_points_path)
        out_points_path.parent.mkdir(parents=True, exist_ok=True)

        if out_points_path.suffix.lower() == '.parquet':
            df = pd.DataFrame(points)
            df.to_parquet(str(out_points_path), index=False)
        else:
            # Default to CSV
            df = pd.DataFrame(points)
            df.to_csv(str(out_points_path), index=False)

        logger.info(f"Points saved to: {out_points_path}")

    return points, metadata


def filter_points_by_height(
    points: np.ndarray,
    height_min: float = DEFAULT_HEIGHT_MIN_M,
    height_max: Optional[float] = None
) -> np.ndarray:
    """
    Filter points by height range.

    Reference: agent.md §4.3

    Args:
        points: Structured array with 'z' field
        height_min: Minimum height (default 1.5m)
        height_max: Optional maximum height

    Returns:
        Filtered points array
    """
    mask = points['z'] >= height_min

    if height_max is not None:
        mask = mask & (points['z'] <= height_max)

    filtered = points[mask]

    logger.debug(
        f"Height filter: {len(points):,} → {len(filtered):,} points "
        f"(range: {height_min}-{height_max if height_max else 'inf'}m)"
    )

    return filtered


def filter_points_by_ndvi(
    points: np.ndarray,
    ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD
) -> np.ndarray:
    """
    Filter out vegetation points based on NDVI.

    Reference: agent-2.md rules.vegetation_filter

    Args:
        points: Structured array with 'r', 'nir' fields
        ndvi_threshold: NDVI threshold (default 0.3)

    Returns:
        Filtered points array (vegetation removed)
    """
    # Compute NDVI
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (points['nir'] - points['r']) / (points['nir'] + points['r'] + 1e-10)
        ndvi = np.nan_to_num(ndvi, nan=0.0)

    # Keep non-vegetation points
    mask = ndvi <= ndvi_threshold
    filtered = points[mask]

    logger.debug(
        f"NDVI filter (threshold={ndvi_threshold}): "
        f"{len(points):,} → {len(filtered):,} points"
    )

    return filtered


def compute_vegetation_percentage(
    points: np.ndarray,
    ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD
) -> float:
    """
    Compute percentage of points classified as vegetation.

    Reference: agent-2.md io_schemas.metrics_json.qc.vegetation_pct

    Args:
        points: Structured array with 'r', 'nir' fields
        ndvi_threshold: NDVI threshold

    Returns:
        Percentage of vegetation points (0-100)
    """
    if len(points) == 0:
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (points['nir'] - points['r']) / (points['nir'] + points['r'] + 1e-10)
        ndvi = np.nan_to_num(ndvi, nan=0.0)

    vegetation_count = (ndvi > ndvi_threshold).sum()

    return vegetation_count / len(points) * 100


def points_to_xyz_array(points: np.ndarray) -> np.ndarray:
    """
    Extract XYZ coordinates as simple (N, 3) array.

    Useful for plane fitting and convex hull operations.

    Args:
        points: Structured array with x, y, z fields

    Returns:
        (N, 3) array of XYZ coordinates
    """
    return np.column_stack([points['x'], points['y'], points['z']])


def points_to_xy_array(points: np.ndarray) -> np.ndarray:
    """
    Extract XY coordinates as simple (N, 2) array.

    Useful for 2D operations like convex hull.

    Args:
        points: Structured array with x, y fields

    Returns:
        (N, 2) array of XY coordinates
    """
    return np.column_stack([points['x'], points['y']])
