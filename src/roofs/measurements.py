"""
Roof measurement calculations.
Computes area, perimeter, and other metrics for roof polygons.
"""
import logging
from typing import Dict
import geopandas as gpd
from shapely.geometry import LineString
from src.utils.projections import get_aeqd_crs, calculate_geodesic_area

logger = logging.getLogger(__name__)


def calculate_plan_area(
    polygon_wgs84,
    lat: float,
    lon: float,
    method: str = "local"
) -> float:
    """
    Calculate plan area (2D footprint) of a roof polygon.

    Args:
        polygon_wgs84: Polygon in WGS84
        lat: Reference latitude for local projection
        lon: Reference longitude for local projection
        method: "local" (AEQD) or "geodesic" (equal-area projection)

    Returns:
        Area in square meters

    Example:
        >>> area = calculate_plan_area(polygon, 28.1178, -82.3951)
        >>> print(f"{area:.2f} sq m")
    """
    if method == "local":
        # Use local azimuthal equidistant projection
        aeqd_crs = get_aeqd_crs(lon, lat)
        gdf = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326")
        gdf_m = gdf.to_crs(aeqd_crs)
        area = gdf_m.area.iloc[0]

    elif method == "geodesic":
        # Use geodesic (equal-area) calculation
        gdf = gpd.GeoDataFrame(geometry=[polygon_wgs84], crs="EPSG:4326")
        area = calculate_geodesic_area(gdf).iloc[0]

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Plan area ({method}): {area:.2f} sq m")
    return area


def calculate_perimeter(
    polygon_wgs84,
    lat: float,
    lon: float
) -> float:
    """
    Calculate perimeter of a roof polygon.

    Args:
        polygon_wgs84: Polygon in WGS84
        lat: Reference latitude for local projection
        lon: Reference longitude for local projection

    Returns:
        Perimeter in meters

    Example:
        >>> perim = calculate_perimeter(polygon, 28.1178, -82.3951)
        >>> print(f"{perim:.2f} m")
    """
    # Use local projection for accurate distance
    aeqd_crs = get_aeqd_crs(lon, lat)
    gdf = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326")
    gdf_m = gdf.to_crs(aeqd_crs)
    perimeter = gdf_m.length.iloc[0]

    logger.info(f"Perimeter: {perimeter:.2f} m")
    return perimeter


def calculate_edge_lengths(
    polygon_wgs84,
    lat: float,
    lon: float
) -> Dict[int, float]:
    """
    Calculate length of each edge in a polygon.

    Args:
        polygon_wgs84: Polygon in WGS84
        lat: Reference latitude
        lon: Reference longitude

    Returns:
        Dict mapping edge index to length in meters

    Example:
        >>> edges = calculate_edge_lengths(polygon, 28.1178, -82.3951)
        >>> for i, length in edges.items():
        ...     print(f"Edge {i}: {length:.2f}m")
    """
    # Transform to local projection
    aeqd_crs = get_aeqd_crs(lon, lat)
    gdf = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326")
    polygon_m = gdf.to_crs(aeqd_crs).iloc[0]

    # Get coordinates
    coords = list(polygon_m.exterior.coords)

    # Calculate edge lengths
    edge_lengths = {}
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        edge = LineString([p1, p2])
        edge_lengths[i] = edge.length

    logger.info(f"Calculated {len(edge_lengths)} edge lengths")
    return edge_lengths


def calculate_bounding_box(
    polygon_wgs84,
    lat: float,
    lon: float
) -> Dict[str, float]:
    """
    Calculate oriented bounding box dimensions.

    Args:
        polygon_wgs84: Polygon in WGS84
        lat: Reference latitude
        lon: Reference longitude

    Returns:
        Dict with keys: width, height, area

    Example:
        >>> bbox = calculate_bounding_box(polygon, 28.1178, -82.3951)
        >>> print(f"{bbox['width']:.2f}m x {bbox['height']:.2f}m")
    """
    # Transform to local projection
    aeqd_crs = get_aeqd_crs(lon, lat)
    gdf = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326")
    polygon_m = gdf.to_crs(aeqd_crs).iloc[0]

    # Get minimum rotated rectangle
    min_rect = polygon_m.minimum_rotated_rectangle

    # Get dimensions
    coords = list(min_rect.exterior.coords)
    edge1 = LineString([coords[0], coords[1]]).length
    edge2 = LineString([coords[1], coords[2]]).length

    width = min(edge1, edge2)
    height = max(edge1, edge2)

    return {
        "width": width,
        "height": height,
        "area": min_rect.area
    }


def get_all_measurements(
    polygon_wgs84,
    lat: float,
    lon: float,
    source: str = "unknown"
) -> Dict:
    """
    Calculate all roof measurements.

    Args:
        polygon_wgs84: Polygon in WGS84
        lat: Reference latitude
        lon: Reference longitude
        source: Data source label

    Returns:
        Dict with all measurements

    Example:
        >>> metrics = get_all_measurements(polygon, 28.1178, -82.3951, "lidar")
    """
    logger.info("Calculating roof measurements...")

    # Calculate metrics
    plan_area = calculate_plan_area(polygon_wgs84, lat, lon, method="local")
    perimeter = calculate_perimeter(polygon_wgs84, lat, lon)
    edge_lengths = calculate_edge_lengths(polygon_wgs84, lat, lon)
    bbox = calculate_bounding_box(polygon_wgs84, lat, lon)

    # Get vertex count
    n_vertices = len(polygon_wgs84.exterior.coords) - 1  # Exclude repeated first point

    measurements = {
        "plan_area_m2": plan_area,
        "perimeter_m": perimeter,
        "n_vertices": n_vertices,
        "edge_lengths_m": edge_lengths,
        "bbox_width_m": bbox["width"],
        "bbox_height_m": bbox["height"],
        "bbox_area_m2": bbox["area"],
        "source": source,
        "coordinates_wgs84": list(polygon_wgs84.exterior.coords)
    }

    logger.info("=" * 60)
    logger.info("ROOF MEASUREMENTS")
    logger.info("=" * 60)
    logger.info(f"Plan area: {plan_area:.2f} sq m ({plan_area * 10.764:.2f} sq ft)")
    logger.info(f"Perimeter: {perimeter:.2f} m ({perimeter * 3.281:.2f} ft)")
    logger.info(f"Vertices: {n_vertices}")
    logger.info(f"Bounding box: {bbox['width']:.2f}m x {bbox['height']:.2f}m")
    logger.info(f"Source: {source}")
    logger.info("=" * 60)

    return measurements


def compare_measurements(
    measurements1: Dict,
    measurements2: Dict,
    label1: str = "Method 1",
    label2: str = "Method 2"
) -> Dict:
    """
    Compare measurements from two different methods.

    Args:
        measurements1: First measurement dict
        measurements2: Second measurement dict
        label1: Label for first method
        label2: Label for second method

    Returns:
        Dict with comparison results

    Example:
        >>> comp = compare_measurements(ms_metrics, lidar_metrics, "MS", "LiDAR")
        >>> print(f"Area diff: {comp['area_diff_pct']:.1f}%")
    """
    area_diff = measurements2["plan_area_m2"] - measurements1["plan_area_m2"]
    area_diff_pct = (area_diff / measurements1["plan_area_m2"]) * 100

    perim_diff = measurements2["perimeter_m"] - measurements1["perimeter_m"]
    perim_diff_pct = (perim_diff / measurements1["perimeter_m"]) * 100

    logger.info("=" * 60)
    logger.info(f"COMPARISON: {label1} vs {label2}")
    logger.info("=" * 60)
    logger.info(f"Area: {measurements1['plan_area_m2']:.2f} vs {measurements2['plan_area_m2']:.2f} sq m ({area_diff_pct:+.1f}%)")
    logger.info(f"Perimeter: {measurements1['perimeter_m']:.2f} vs {measurements2['perimeter_m']:.2f} m ({perim_diff_pct:+.1f}%)")
    logger.info("=" * 60)

    return {
        "area_diff_m2": area_diff,
        "area_diff_pct": area_diff_pct,
        "perim_diff_m": perim_diff,
        "perim_diff_pct": perim_diff_pct
    }
