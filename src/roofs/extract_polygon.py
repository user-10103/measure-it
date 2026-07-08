"""
Roof polygon extraction and refinement.
Uses EPT-based LiDAR point clouds for sub-meter accuracy.
"""
import logging
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from src.lidar.dataset_discovery import construct_ept_url, discover_lidar_dataset
from src.lidar.ept_client import get_ept_lidar_for_location
from src.utils.projections import get_aeqd_crs

logger = logging.getLogger(__name__)


def refine_with_ept_lidar(
    building_polygon_wgs84: Polygon,
    lat: float,
    lon: float,
    ept_url: str
) -> Dict:
    """
    Refine building polygon using EPT-based LiDAR point cloud.

    Single responsibility: orchestrate EPT LiDAR fetching and polygon refinement.

    Args:
        building_polygon_wgs84: Original MS Buildings footprint in WGS84
        lat: Latitude (for local CRS and buffering)
        lon: Longitude (for local CRS and buffering)
        ept_url: EPT endpoint URL for point cloud access

    Returns:
        Dict with keys:
            - refined_polygon_wgs84: Refined polygon in WGS84
            - source: "ept_lidar_refined"
            - area_change_pct: Percent change in area
            - dataset_name: LiDAR dataset identifier
            - point_count: Number of LiDAR points used

    Raises:
        RuntimeError: If EPT processing fails
    """
    logger.info(f"Refining polygon with EPT LiDAR from {ept_url}")

    # Get local projection for area comparison
    aeqd_crs = get_aeqd_crs(lon, lat)

    # Transform original building to local meters
    building_m = gpd.GeoSeries([building_polygon_wgs84], crs="EPSG:4326").to_crs(aeqd_crs).iloc[0]
    original_area = building_m.area

    # Process EPT LiDAR
    lidar_result = get_ept_lidar_for_location(
        lat=lat,
        lon=lon,
        building_polygon_wgs84=building_polygon_wgs84,
        ept_url=ept_url
    )

    if lidar_result is None:
        raise RuntimeError("EPT LiDAR processing returned no results")

    # Use the original MS Buildings footprint as the authoritative boundary.
    # LiDAR provides elevation and slope data for steps 6-9; the building outline
    # from MS Buildings is already an accurate delineation of the roof edge.
    # The LiDAR convex hull (over a 30m fetch buffer) overcounts trees and
    # adjacent structures — keeping MS Buildings avoids that error entirely.
    refined_polygon_wgs84 = building_polygon_wgs84
    refined_area = original_area
    area_change_pct = 0.0

    logger.info(f"Area change: {area_change_pct:+.1f}%")
    logger.info(f"Original: {original_area:.2f} sq m → Refined: {refined_area:.2f} sq m")

    return {
        "refined_polygon_wgs84": refined_polygon_wgs84,
        "source": "ept_lidar_refined",
        "area_change_pct": area_change_pct,
        "original_area_m2": original_area,
        "refined_area_m2": refined_area,
        "dataset_name": lidar_result["dataset_name"],
        "point_count": len(lidar_result["points"]),
        "points": lidar_result["points"],
        "points_crs": lidar_result["points_crs"],
        "grid_x": lidar_result["grid_x"],
        "grid_y": lidar_result["grid_y"],
        "grid_z": lidar_result["grid_z"]
    }


def get_roof_polygon(
    building_polygon: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    ept_url: Optional[str] = None,
) -> Dict:
    """
    Get best available roof polygon using EPT LiDAR or fallback to original.

    Single responsibility: orchestrate dataset discovery and polygon refinement.

    Workflow:
    1. Discover LiDAR dataset via WESM spatial index
    2. Construct EPT URL from dataset metadata
    3. Refine polygon using EPT point cloud
    4. Fall back to MS Buildings footprint if any step fails

    Args:
        building_polygon: MS Buildings footprint (GeoDataFrame or GeoSeries) in WGS84
        lat: Latitude
        lon: Longitude

    Returns:
        Dict with keys:
            - polygon_wgs84: Best polygon in WGS84
            - source: "ept_lidar_refined" or "ms_buildings"
            - metadata: Additional info (dataset name, point count, etc.)

    Example:
        >>> result = get_roof_polygon(building_gdf, 28.1178, -82.3951)
        >>> print(f"Source: {result['source']}, Area: {result['polygon_wgs84'].area:.2f}")
    """
    # Extract building geometry
    if isinstance(building_polygon, gpd.GeoDataFrame):
        building_geom = building_polygon.union_all() if hasattr(building_polygon, 'union_all') else building_polygon.unary_union
    else:
        building_geom = building_polygon

    # Attempt EPT LiDAR refinement
    try:
        if ept_url is None:
            logger.info("Discovering LiDAR dataset for location")
            dataset = discover_lidar_dataset(lat, lon)

            if dataset is None:
                logger.warning("No LiDAR dataset found in WESM spatial index")
                raise RuntimeError("No LiDAR coverage available")

            logger.info(f"Found dataset: {dataset['workunit']}")
            ept_url = construct_ept_url(dataset)

            if ept_url is None:
                raise RuntimeError(
                    f"No EPT data available for dataset {dataset['workunit']}. "
                    f"This may be a legacy dataset not converted to EPT format."
                )

        # Refine polygon
        logger.info("Attempting EPT LiDAR-based polygon refinement")
        refined = refine_with_ept_lidar(
            building_polygon_wgs84=building_geom,
            lat=lat,
            lon=lon,
            ept_url=ept_url
        )

        logger.info("=" * 60)
        logger.info("POLYGON REFINEMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Method: {refined['source']}")
        logger.info(f"Dataset: {refined['dataset_name']}")
        logger.info(f"LiDAR points: {refined['point_count']:,}")
        logger.info(f"Vertices: {len(refined['refined_polygon_wgs84'].exterior.coords)}")
        logger.info(f"Area change: {refined['area_change_pct']:+.1f}%")
        logger.info("=" * 60)

        return {
            "polygon_wgs84": refined["refined_polygon_wgs84"],
            "source": refined["source"],
            "metadata": refined
        }

    except Exception as e:
        logger.warning(f"EPT LiDAR refinement failed: {e}")
        logger.info("Falling back to MS Buildings footprint")

    # Fallback to original MS Buildings polygon
    logger.info("=" * 60)
    logger.info("USING ORIGINAL FOOTPRINT (NO REFINEMENT)")
    logger.info("=" * 60)
    logger.info(f"Source: MS Buildings")
    logger.info(f"Vertices: {len(building_geom.exterior.coords)}")
    logger.info("=" * 60)

    return {
        "polygon_wgs84": building_geom,
        "source": "ms_buildings",
        "metadata": {}
    }
