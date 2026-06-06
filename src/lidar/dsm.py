"""
DSM/DTM/nDSM Generation Module

Creates Digital Surface Models (DSM), Digital Terrain Models (DTM), and
normalized DSM (nDSM = DSM - DTM) from LiDAR point clouds.

Reference: agent.md §3, agent-2.md lidar_to_models tool
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from src.config import DSM_RESOLUTION_M

logger = logging.getLogger(__name__)


# Ground classification code per ASPRS LAS standard
CLASSIFICATION_GROUND = 2


def build_dtm_from_points(
    points: np.ndarray,
    resolution: float = None,
    smoothing_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Digital Terrain Model (DTM) from ground-classified LiDAR points.

    Filters points to ground classification (class 2) and interpolates
    to create a bare-earth elevation surface.

    Reference: agent.md §3.1

    Args:
        points: Structured numpy array with X, Y, Z, Classification fields
        resolution: Grid cell size in meters (default from config)
        smoothing_sigma: Gaussian smoothing sigma (default 1.0)

    Returns:
        Tuple of (grid_x, grid_y, grid_z_dtm):
        - grid_x, grid_y: 2D meshgrid coordinates
        - grid_z_dtm: 2D ground elevation grid

    Raises:
        ValueError: If insufficient ground points found
    """
    if points is None or len(points) == 0:
        raise ValueError("Cannot build DTM from empty point array")

    if resolution is None:
        resolution = DSM_RESOLUTION_M

    # Filter to ground-classified points
    if 'Classification' in points.dtype.names:
        ground_mask = points['Classification'] == CLASSIFICATION_GROUND
        ground_points = points[ground_mask]

        if len(ground_points) < 10:
            logger.warning(
                f"Only {len(ground_points)} ground points (class 2). "
                "Falling back to lowest 10% of points for DTM."
            )
            # Fallback: use lowest 10% of points as pseudo-ground
            z_threshold = np.percentile(points['Z'], 10)
            ground_points = points[points['Z'] <= z_threshold]
    else:
        # No classification - use lowest 10% as pseudo-ground
        logger.warning("No Classification field. Using lowest 10% of points for DTM.")
        z_threshold = np.percentile(points['Z'], 10)
        ground_points = points[points['Z'] <= z_threshold]

    if len(ground_points) < 10:
        raise ValueError(f"Insufficient ground points for DTM: {len(ground_points)}")

    logger.info(f"Building DTM at {resolution}m resolution from {len(ground_points):,} ground points")

    x = ground_points['X']
    y = ground_points['Y']
    z = ground_points['Z']

    # Use full point extent for grid (not just ground points)
    x_min, x_max = points['X'].min(), points['X'].max()
    y_min, y_max = points['Y'].min(), points['Y'].max()

    grid_x_1d = np.arange(x_min, x_max, resolution)
    grid_y_1d = np.arange(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    # Interpolate ground elevation
    grid_z = griddata(
        (x, y), z,
        (grid_x, grid_y),
        method='linear',
        fill_value=np.nan
    )

    # Fill NaN with nearest neighbor
    if np.any(np.isnan(grid_z)):
        grid_z_nearest = griddata(
            (x, y), z,
            (grid_x, grid_y),
            method='nearest'
        )
        grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

    # Smooth to reduce noise
    grid_z_dtm = gaussian_filter(grid_z, sigma=smoothing_sigma)

    logger.info(
        f"DTM created: {grid_z_dtm.shape[1]}x{grid_z_dtm.shape[0]} grid, "
        f"elevation range {grid_z_dtm.min():.2f}-{grid_z_dtm.max():.2f}m"
    )

    return grid_x, grid_y, grid_z_dtm


def build_dsm_from_points(
    points: np.ndarray,
    resolution: float = None,
    smoothing_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Digital Surface Model (DSM) from LiDAR point cloud.

    Uses all returns (first return surface) to capture buildings, trees,
    and other above-ground features.

    Reference: agent.md §3.2

    Args:
        points: Structured numpy array with X, Y, Z fields
        resolution: Grid cell size in meters (default from config)
        smoothing_sigma: Gaussian smoothing sigma (default 1.0)

    Returns:
        Tuple of (grid_x, grid_y, grid_z_dsm):
        - grid_x, grid_y: 2D meshgrid coordinates
        - grid_z_dsm: 2D surface elevation grid

    Raises:
        ValueError: If points array is empty
    """
    if points is None or len(points) == 0:
        raise ValueError("Cannot build DSM from empty point array")

    if resolution is None:
        resolution = DSM_RESOLUTION_M

    logger.info(f"Building DSM at {resolution}m resolution from {len(points):,} points")

    x = points['X']
    y = points['Y']
    z = points['Z']

    # Create regular grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    grid_x_1d = np.arange(x_min, x_max, resolution)
    grid_y_1d = np.arange(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    # Interpolate elevation (linear for smooth surface)
    grid_z = griddata(
        (x, y), z,
        (grid_x, grid_y),
        method='linear',
        fill_value=z.min()
    )

    # Fill gaps with maximum filter (captures highest returns)
    grid_z = maximum_filter(grid_z, size=3)

    # Smooth to reduce noise
    grid_z_dsm = gaussian_filter(grid_z, sigma=smoothing_sigma)

    logger.info(
        f"DSM created: {grid_z_dsm.shape[1]}x{grid_z_dsm.shape[0]} grid, "
        f"elevation range {grid_z_dsm.min():.2f}-{grid_z_dsm.max():.2f}m"
    )

    return grid_x, grid_y, grid_z_dsm


def compute_ndsm(
    dsm: np.ndarray,
    dtm: np.ndarray
) -> np.ndarray:
    """
    Compute normalized DSM (above-ground height).

    nDSM = DSM - DTM

    Reference: agent.md §3.3

    Args:
        dsm: Digital Surface Model array
        dtm: Digital Terrain Model array (must match DSM shape)

    Returns:
        nDSM array (height above ground)

    Raises:
        ValueError: If DSM and DTM shapes don't match
    """
    if dsm.shape != dtm.shape:
        raise ValueError(
            f"DSM shape {dsm.shape} doesn't match DTM shape {dtm.shape}. "
            "Ensure both rasters are aligned to the same grid."
        )

    ndsm = dsm - dtm

    # Clip negative values (can occur from interpolation artifacts)
    ndsm = np.maximum(ndsm, 0.0)

    logger.info(
        f"nDSM computed: height range {ndsm.min():.2f}-{ndsm.max():.2f}m"
    )

    return ndsm


def save_raster_geotiff(
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_path: str,
    crs: str = "EPSG:4326",
    nodata: float = -9999.0
) -> str:
    """
    Save elevation grid as GeoTIFF with proper geotransform.

    Reference: agent-2.md lidar_to_models output

    Args:
        grid_z: 2D elevation array
        grid_x: 2D meshgrid X coordinates
        grid_y: 2D meshgrid Y coordinates
        output_path: Output file path
        crs: Coordinate reference system (default EPSG:4326)
        nodata: NoData value (default -9999.0)

    Returns:
        Output file path

    Raises:
        RuntimeError: If file write fails
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract bounds from grid
    x_min = grid_x.min()
    x_max = grid_x.max()
    y_min = grid_y.min()
    y_max = grid_y.max()

    height, width = grid_z.shape

    # Create geotransform from bounds
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Handle NaN values
    grid_z_out = np.where(np.isnan(grid_z), nodata, grid_z)

    try:
        with rasterio.open(
            str(output_path),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=grid_z_out.dtype,
            crs=CRS.from_string(crs),
            transform=transform,
            nodata=nodata,
            compress='lzw'
        ) as dst:
            # Note: rasterio expects (bands, height, width), origin at top-left
            # Flip Y axis since our grid has origin at bottom-left
            dst.write(np.flipud(grid_z_out), 1)

        logger.info(f"Saved raster: {output_path} ({width}x{height}, CRS={crs})")
        return str(output_path)

    except Exception as e:
        error_msg = f"Failed to save GeoTIFF: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def generate_height_models(
    points: np.ndarray,
    output_dir: str,
    building_id: str,
    resolution: float = None,
    crs: str = "EPSG:4326"
) -> Dict[str, Any]:
    """
    Generate DSM, DTM, and nDSM from LiDAR points and save as GeoTIFFs.

    Complete workflow that produces all height models from a point cloud.

    Reference: agent-2.md lidar_to_models tool

    Args:
        points: Structured numpy array with X, Y, Z, Classification fields
        output_dir: Directory for output files
        building_id: Identifier for file naming
        resolution: Grid resolution in meters (default from config)
        crs: Coordinate reference system for output

    Returns:
        Dict with paths and metadata:
        {
            "dsm_path": str,
            "dtm_path": str,
            "ndsm_path": str,
            "grid_x": np.ndarray,
            "grid_y": np.ndarray,
            "dsm": np.ndarray,
            "dtm": np.ndarray,
            "ndsm": np.ndarray,
            "resolution": float,
            "crs": str,
            "bounds": tuple  # (x_min, y_min, x_max, y_max)
        }

    Raises:
        ValueError: If points array is invalid
        RuntimeError: If processing fails
    """
    if resolution is None:
        resolution = DSM_RESOLUTION_M

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating height models for building {building_id}")

    # Build DSM (all returns)
    grid_x, grid_y, dsm = build_dsm_from_points(points, resolution)

    # Build DTM (ground only)
    _, _, dtm = build_dtm_from_points(points, resolution)

    # Compute nDSM
    ndsm = compute_ndsm(dsm, dtm)

    # Define output paths
    dsm_path = output_dir / f"{building_id}_dsm.tif"
    dtm_path = output_dir / f"{building_id}_dtm.tif"
    ndsm_path = output_dir / f"{building_id}_ndsm.tif"

    # Save GeoTIFFs
    save_raster_geotiff(dsm, grid_x, grid_y, dsm_path, crs=crs)
    save_raster_geotiff(dtm, grid_x, grid_y, dtm_path, crs=crs)
    save_raster_geotiff(ndsm, grid_x, grid_y, ndsm_path, crs=crs)

    # Compute bounds
    bounds = (grid_x.min(), grid_y.min(), grid_x.max(), grid_y.max())

    result = {
        "dsm_path": str(dsm_path),
        "dtm_path": str(dtm_path),
        "ndsm_path": str(ndsm_path),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "dsm": dsm,
        "dtm": dtm,
        "ndsm": ndsm,
        "resolution": resolution,
        "crs": crs,
        "bounds": bounds
    }

    logger.info(
        f"Height models complete: DSM/DTM/nDSM saved to {output_dir}"
    )

    return result


def get_ndsm_stats(ndsm: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for nDSM array.

    Reference: agent-2.md io_schemas.metrics_json.qc.ndsm_stats

    Args:
        ndsm: Normalized DSM array

    Returns:
        Dict with min, max, mean, std values
    """
    # Filter out nodata/nan values
    valid_mask = ~np.isnan(ndsm) & (ndsm > -9000)
    valid_ndsm = ndsm[valid_mask]

    if len(valid_ndsm) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    return {
        "min": float(valid_ndsm.min()),
        "max": float(valid_ndsm.max()),
        "mean": float(valid_ndsm.mean()),
        "std": float(valid_ndsm.std())
    }
