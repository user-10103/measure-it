"""
EPT LiDAR Client Module

Fetches and processes LiDAR point clouds via PDAL EPT (Entwine Point Tiles).
Single responsibility: transform raw point clouds into roof polygons.
"""

import json
import logging
import os
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pdal
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from sklearn.linear_model import RANSACRegressor

from src.config import (
    DSM_RESOLUTION_M,
    EPT_RESOLUTION_TARGET,
    LIDAR_BUFFER_M,
    POLYGON_SIMPLIFY_M,
    RANSAC_RESIDUAL_M,
)

logger = logging.getLogger(__name__)


def build_pdal_pipeline(ept_url: str, polygon_wkt_epsg4326: str) -> dict:
    """
    Construct PDAL pipeline configuration for EPT point cloud extraction.

    Single responsibility: create pipeline JSON with filtering and classification.

    Pipeline stages:
    1. Read EPT data for polygon bounds
    2. Remove statistical outliers
    3. Filter to building-relevant classifications (1-6)

    Args:
        ept_url: EPT endpoint URL (e.g., "https://.../ept.json")
        polygon_wkt_epsg4326: WKT polygon with CRS tag (e.g., "POLYGON(...)/EPSG:4326")

    Returns:
        Pipeline configuration dict ready for pdal.Pipeline()
    """
    pipeline = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": ept_url,
                "polygon": polygon_wkt_epsg4326,
                "resolution": EPT_RESOLUTION_TARGET
            },
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 12,
                "multiplier": 2.0
            },
            {
                "type": "filters.range",
                "limits": "Classification[1:6]"  # Unclassified through building
            }
        ]
    }

    logger.debug(f"Built PDAL pipeline with {len(pipeline['pipeline'])} stages")
    return pipeline


def fetch_lidar_points(ept_url: str, polygon_wkt_epsg4326: str) -> Optional[np.ndarray]:
    """
    Execute PDAL pipeline to fetch LiDAR points for a polygon area.

    Single responsibility: retrieve and return point cloud array.

    Args:
        ept_url: EPT endpoint URL
        polygon_wkt_epsg4326: WKT polygon with CRS tag

    Returns:
        Structured numpy array with fields: X, Y, Z, Classification, etc.
        or None if execution fails

    Raises:
        RuntimeError: If PDAL pipeline execution fails
    """
    # Configure AWS for requester-pays S3 access
    os.environ['AWS_REQUEST_PAYER'] = 'requester'

    logger.info(f"Fetching LiDAR points from EPT: {ept_url}")

    try:
        pipeline_config = build_pdal_pipeline(ept_url, polygon_wkt_epsg4326)
        pipeline = pdal.Pipeline(json.dumps(pipeline_config))

        n_points = pipeline.execute()

        if n_points == 0:
            logger.warning("EPT query returned 0 points - area may not have LiDAR coverage")
            return None

        points = pipeline.arrays[0]

        logger.info(
            f"Retrieved {n_points:,} points. "
            f"Z range: {points['Z'].min():.2f} to {points['Z'].max():.2f} meters"
        )

        return points

    except Exception as e:
        error_msg = f"PDAL pipeline execution failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def build_dsm_from_points(points: np.ndarray, resolution: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Digital Surface Model (DSM) from LiDAR point cloud.

    Single responsibility: convert sparse 3D points to gridded elevation raster.

    Processing steps:
    1. Grid interpolation (linear)
    2. Maximum filter to fill gaps
    3. Gaussian smoothing for noise reduction

    Args:
        points: Structured array with X, Y, Z fields
        resolution: Grid cell size in meters (defaults to config value)

    Returns:
        Tuple of (grid_x, grid_y, grid_z) where:
        - grid_x, grid_y: 2D meshgrid coordinates
        - grid_z: 2D elevation grid (smoothed DSM)

    Raises:
        ValueError: If points array is empty or missing required fields
    """
    if points is None or len(points) == 0:
        raise ValueError("Cannot build DSM from empty point array")

    if resolution is None:
        resolution = DSM_RESOLUTION_M

    logger.info(f"Building DSM at {resolution}m resolution from {len(points):,} points")

    try:
        x = points['X']
        y = points['Y']
        z = points['Z']

        # Create regular grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        grid_x_1d = np.arange(x_min, x_max, resolution)
        grid_y_1d = np.arange(y_min, y_max, resolution)
        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

        # Interpolate elevation
        grid_z = griddata(
            (x, y), z,
            (grid_x, grid_y),
            method='linear',
            fill_value=z.min()
        )

        # Fill gaps with maximum filter
        grid_z = maximum_filter(grid_z, size=3)

        # Smooth to reduce noise
        grid_z_smooth = gaussian_filter(grid_z, sigma=1.0)

        logger.info(
            f"DSM created: {grid_z_smooth.shape[1]}x{grid_z_smooth.shape[0]} grid, "
            f"elevation range {grid_z_smooth.min():.2f}-{grid_z_smooth.max():.2f}m"
        )

        return grid_x, grid_y, grid_z_smooth

    except Exception as e:
        error_msg = f"DSM generation failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def extract_roof_polygon(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    threshold_offset: float = 2.0,
    simplify_tolerance: float = None
) -> Optional[Polygon]:
    """
    Extract roof polygon from DSM using RANSAC plane fitting.

    Single responsibility: identify roof surface and trace its boundary.

    Processing steps:
    1. Threshold DSM to isolate roof region (max elevation - threshold_offset)
    2. RANSAC plane fitting to find dominant roof surface
    3. Convex hull boundary extraction
    4. Polygon simplification

    Args:
        grid_x: 2D meshgrid X coordinates
        grid_y: 2D meshgrid Y coordinates
        grid_z: 2D elevation grid (DSM)
        threshold_offset: Meters below max elevation to consider as roof (default: 2.0m)
        simplify_tolerance: Douglas-Peucker tolerance in meters (defaults to config value)

    Returns:
        Shapely Polygon in the DSM's native CRS (projected meters)
        or None if extraction fails

    Raises:
        ValueError: If insufficient roof points found
        RuntimeError: If RANSAC or convex hull fails
    """
    if simplify_tolerance is None:
        simplify_tolerance = POLYGON_SIMPLIFY_M

    logger.info(f"Extracting roof polygon (threshold offset: {threshold_offset}m)")

    try:
        # Segment roof region
        max_elevation = grid_z.max()
        roof_threshold = max_elevation - threshold_offset
        roof_mask = grid_z >= roof_threshold

        roof_x = grid_x[roof_mask]
        roof_y = grid_y[roof_mask]
        roof_z = grid_z[roof_mask]

        if len(roof_x) < 10:
            logger.error(f"Insufficient roof points: {len(roof_x)} (need at least 10)")
            raise ValueError("Not enough points above roof threshold")

        logger.debug(
            f"Roof segmentation: {len(roof_x):,} points above {roof_threshold:.2f}m "
            f"({len(roof_x)/roof_mask.size*100:.1f}% of grid)"
        )

        # Fit dominant plane with RANSAC
        X_roof = np.column_stack([roof_x, roof_y])
        y_roof = roof_z

        ransac = RANSACRegressor(
            residual_threshold=RANSAC_RESIDUAL_M,
            max_trials=1000,
            random_state=42
        )
        ransac.fit(X_roof, y_roof)

        inlier_mask = ransac.inlier_mask_
        n_inliers = inlier_mask.sum()

        if n_inliers < 10:
            logger.error(f"RANSAC found only {n_inliers} inliers (need at least 10)")
            raise ValueError("RANSAC plane fitting failed - insufficient inliers")

        logger.info(
            f"RANSAC plane fitting: {n_inliers:,} inliers "
            f"({n_inliers/len(X_roof)*100:.1f}% of roof points)"
        )

        # Extract boundary via convex hull
        boundary_x = roof_x[inlier_mask]
        boundary_y = roof_y[inlier_mask]

        points_2d = np.column_stack([boundary_x, boundary_y])
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        polygon = Polygon(hull_points)

        # Simplify to reduce vertex count
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

        logger.info(
            f"Roof polygon extracted: {len(polygon.exterior.coords)} vertices, "
            f"area {polygon.area:.2f} sq m"
        )

        return polygon

    except Exception as e:
        error_msg = f"Roof polygon extraction failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_ept_lidar_for_location(
    lat: float,
    lon: float,
    building_polygon_wgs84: Polygon,
    ept_url: str,
    buffer_meters: float = None
) -> Optional[dict]:
    """
    Complete workflow: fetch LiDAR and extract roof polygon for a location.

    Single responsibility: orchestrate full EPT → polygon pipeline.

    Args:
        lat: Latitude (WGS84)
        lon: Longitude (WGS84)
        building_polygon_wgs84: Building footprint in WGS84
        ept_url: EPT endpoint URL
        buffer_meters: Buffer around building in meters (defaults to config value)

    Returns:
        Dict with keys:
        {
            "points": np.ndarray,          # Raw point cloud
            "grid_x": np.ndarray,          # DSM X coordinates
            "grid_y": np.ndarray,          # DSM Y coordinates
            "grid_z": np.ndarray,          # DSM elevation grid
            "roof_polygon": Polygon,       # Extracted roof in projected CRS
            "dataset_name": str,           # EPT dataset identifier
        }
        or None if processing fails
    """
    if buffer_meters is None:
        buffer_meters = LIDAR_BUFFER_M

    logger.info(
        f"Processing EPT LiDAR for ({lat:.6f}, {lon:.6f}) "
        f"with {buffer_meters}m buffer"
    )

    try:
        # Create buffered polygon in WGS84
        from pyproj import CRS, Transformer

        # Create local AEQD projection for accurate meter buffering
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
        )
        fwd = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True).transform
        inv = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True).transform

        # Buffer in meters, convert back to WGS84
        building_m = shp_transform(fwd, building_polygon_wgs84)
        buffer_m = building_m.buffer(buffer_meters)
        buffer_wgs84 = shp_transform(inv, buffer_m)

        # Create CRS-tagged WKT for PDAL
        polygon_wkt_epsg4326 = buffer_wgs84.wkt + "/EPSG:4326"

        # Fetch points
        points = fetch_lidar_points(ept_url, polygon_wkt_epsg4326)

        if points is None or len(points) == 0:
            logger.warning("No LiDAR points retrieved")
            return None

        # Build DSM
        grid_x, grid_y, grid_z = build_dsm_from_points(points)

        # Extract roof
        roof_polygon = extract_roof_polygon(grid_x, grid_y, grid_z)

        dataset_name = ept_url.split("/")[-2] if "/" in ept_url else "unknown"

        result = {
            "points": points,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "roof_polygon": roof_polygon,
            "dataset_name": dataset_name
        }

        logger.info(f"EPT LiDAR processing complete for dataset: {dataset_name}")

        return result

    except Exception as e:
        logger.error(f"EPT LiDAR processing failed: {str(e)}")
        return None
