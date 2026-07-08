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


def utm_epsg_for(lon: float, lat: float) -> int:
    """NAD83 UTM zone EPSG for a CONUS lon/lat (e.g. Tampa -> 26917)."""
    zone = int((lon + 180.0) / 6.0) + 1
    return 26900 + zone


def build_pdal_pipeline(
    ept_url: str,
    polygon_wkt_epsg4326: str,
    out_srs: Optional[str] = None,
) -> dict:
    """
    Construct PDAL pipeline configuration for EPT point cloud extraction.

    Single responsibility: create pipeline JSON with filtering and classification.

    Pipeline stages:
    1. Read EPT data for polygon bounds
    2. Remove statistical outliers
    3. Filter to building-relevant classifications (1-6)
    4. Reproject to `out_srs` (when given) — several FL EPTs are published in
       EPSG:3857 Web Mercator, where XY is NOT true meters at FL latitudes
       (areas +28%, lengths +13%, slopes -12% at 28N). Reprojecting to UTM at
       the source keeps every downstream gradient/length/area in real meters.

    Args:
        ept_url: EPT endpoint URL (e.g., "https://.../ept.json")
        polygon_wkt_epsg4326: WKT polygon with CRS tag (e.g., "POLYGON(...)/EPSG:4326")
        out_srs: optional target CRS (e.g. "EPSG:26917")

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
    if out_srs:
        pipeline["pipeline"].append({
            "type": "filters.reprojection",
            "out_srs": out_srs,
        })

    logger.debug(f"Built PDAL pipeline with {len(pipeline['pipeline'])} stages")
    return pipeline


def fetch_lidar_points(
    ept_url: str,
    polygon_wkt_epsg4326: str,
    out_srs: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    Execute PDAL pipeline to fetch LiDAR points for a polygon area.

    Single responsibility: retrieve and return point cloud array and its CRS.

    Args:
        ept_url: EPT endpoint URL
        polygon_wkt_epsg4326: WKT polygon with CRS tag

    Returns:
        Tuple of (points, srswkt) where points is a structured numpy array with
        fields X, Y, Z, Classification etc., and srswkt is the WKT CRS string
        that PDAL reported for the output points.
        Returns (None, None) if the EPT tile has no coverage.

    Raises:
        RuntimeError: If PDAL pipeline execution fails
    """
    # Configure AWS for requester-pays S3 access
    os.environ['AWS_REQUEST_PAYER'] = 'requester'

    logger.info(f"Fetching LiDAR points from EPT: {ept_url}")

    try:
        import pdal  # lazy — not needed unless LiDAR path is active
        pipeline_config = build_pdal_pipeline(ept_url, polygon_wkt_epsg4326, out_srs=out_srs)
        pipeline = pdal.Pipeline(json.dumps(pipeline_config))

        n_points = pipeline.execute()

        if n_points == 0:
            logger.warning("EPT query returned 0 points - area may not have LiDAR coverage")
            return None, None

        points = pipeline.arrays[0]
        # Canonical point order: the threaded EPT reader returns points in a
        # nondeterministic order, and downstream consumers are order-sensitive
        # (Qhull triangulation in griddata, KDTree tie-breaks, RANSAC/KMeans
        # index sampling). Sorting here makes the whole pipeline reproducible
        # for identical EPT content.
        points = points[np.lexsort((points['Z'], points['Y'], points['X']))]
        # With an explicit reprojection stage the output CRS is out_srs by
        # construction; srswkt2 may still report the reader's native SRS.
        if out_srs:
            srswkt = out_srs
        else:
            # srswkt2 is the PDAL 3.x attribute name; fall back to srswkt for older builds.
            srswkt = (
                getattr(pipeline, "srswkt2", None)
                or getattr(pipeline, "srswkt", None)
                or ""
            )

        logger.info(
            f"Retrieved {n_points:,} points. "
            f"Z range: {points['Z'].min():.2f} to {points['Z'].max():.2f} meters"
        )

        return points, srswkt

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
        # Use a ground-relative threshold: everything more than threshold_offset metres
        # above the estimated ground plane is "building".  The old max-relative approach
        # (max_z - 2 m) captured only the roof peak — useless for the footprint.
        z_flat = grid_z.ravel()
        # 10th percentile of non-minimum cells ≈ ground elevation
        ground_z = float(np.percentile(z_flat[z_flat > z_flat.min() + 0.1], 10))
        roof_threshold = ground_z + threshold_offset
        roof_mask = grid_z >= roof_threshold

        roof_x = grid_x[roof_mask]
        roof_y = grid_y[roof_mask]

        if len(roof_x) < 10:
            logger.error(f"Insufficient roof points: {len(roof_x)} (need at least 10)")
            raise ValueError("Not enough points above roof threshold")

        logger.info(
            f"Footprint extraction: {len(roof_x):,} points above {roof_threshold:.2f}m "
            f"(ground≈{ground_z:.2f}m + {threshold_offset}m offset)"
        )

        # Convex hull of ALL above-ground points — no RANSAC here.
        # RANSAC is for plane fitting (step 8), not footprint extraction.
        points_2d = np.column_stack([roof_x, roof_y])
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

        # Fetch points — returns (array, srswkt) so we know the actual output CRS.
        # Reproject at the source to true-meter UTM: several FL EPTs publish in
        # EPSG:3857, which distorts every downstream area/length/slope at 28N.
        out_srs = f"EPSG:{utm_epsg_for(lon, lat)}"
        points, points_crs = fetch_lidar_points(ept_url, polygon_wkt_epsg4326, out_srs=out_srs)

        if points is None or len(points) == 0:
            logger.warning("No LiDAR points retrieved")
            return None

        logger.info(f"EPT output CRS: {points_crs[:80] if points_crs else '(unknown)'}")

        # Clip points to building footprint + 10m buffer.
        # The fetch used a 30m buffer to ensure coverage; now we discard trees
        # and adjacent structures outside the actual building boundary.
        if points_crs:
            try:
                from pyproj import CRS as _CRS
                _pts_crs = _CRS.from_user_input(points_crs)
                _to_pts = Transformer.from_crs("EPSG:4326", _pts_crs, always_xy=True).transform
                _bldg_native = shp_transform(_to_pts, building_polygon_wgs84)
                _bldg_clip = _bldg_native.buffer(10.0)
                import shapely as _shp_lib
                _mask = _shp_lib.contains_xy(
                    _bldg_clip,
                    points['X'].astype(float),
                    points['Y'].astype(float),
                )
                n_before = len(points)
                points = points[_mask]
                logger.info(
                    f"Clipped to building footprint (+10m): "
                    f"{len(points):,}/{n_before:,} points retained"
                )
            except Exception as _ce:
                logger.warning(f"Building footprint clip failed: {_ce} — using all points")

        if len(points) == 0:
            logger.warning("No points inside building footprint")
            return None

        # Build DSM
        grid_x, grid_y, grid_z = build_dsm_from_points(points)

        # Extract roof
        roof_polygon = extract_roof_polygon(grid_x, grid_y, grid_z)

        dataset_name = ept_url.split("/")[-2] if "/" in ept_url else "unknown"

        result = {
            "points": points,
            "points_crs": points_crs,
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
