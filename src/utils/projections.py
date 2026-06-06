"""
Coordinate reference system (CRS) transformation utilities.
Handles projections between WGS84, UTM, and custom equal-area projections.
"""
import logging
from typing import Tuple
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import CRS, Transformer
import geopandas as gpd

logger = logging.getLogger(__name__)


def get_utm_crs(lon: float, lat: float) -> CRS:
    """
    Determine the appropriate UTM zone CRS for a given location.

    Args:
        lon: Longitude in WGS84
        lat: Latitude in WGS84

    Returns:
        PyProj CRS object for the UTM zone

    Example:
        >>> crs = get_utm_crs(-82.395, 28.118)
        >>> print(crs)
        EPSG:32617  # UTM Zone 17N
    """
    # Calculate UTM zone
    zone = int((lon + 180) / 6) + 1

    # Determine hemisphere
    hemisphere = 'north' if lat >= 0 else 'south'

    # Create CRS
    crs = CRS.from_dict({
        'proj': 'utm',
        'zone': zone,
        'hemisphere': hemisphere,
        'datum': 'WGS84'
    })

    logger.debug(f"UTM Zone: {zone}{hemisphere[0].upper()}, EPSG: {crs.to_epsg()}")
    return crs


def get_aeqd_crs(lon: float, lat: float) -> CRS:
    """
    Create an Azimuthal Equidistant (AEQD) projection centered on a point.
    This projection preserves distances from the center point (ideal for buffers).

    Args:
        lon: Center longitude in WGS84
        lat: Center latitude in WGS84

    Returns:
        PyProj CRS object for AEQD projection

    Example:
        >>> crs = get_aeqd_crs(-82.395, 28.118)
    """
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 "
        f"+datum=WGS84 +units=m +no_defs"
    )
    return crs


def create_meter_buffer(
    lon: float,
    lat: float,
    radius_meters: float
) -> gpd.GeoDataFrame:
    """
    Create a circular buffer in meters around a WGS84 point.
    Uses AEQD projection to ensure accurate distance.

    Args:
        lon: Center longitude in WGS84
        lat: Center latitude in WGS84
        radius_meters: Buffer radius in meters

    Returns:
        GeoDataFrame with buffered polygon in WGS84

    Example:
        >>> buffer = create_meter_buffer(-82.395, 28.118, 150)
        >>> print(buffer.crs)
        EPSG:4326
    """
    # Create point in WGS84
    point = Point(lon, lat)

    # Get AEQD projection centered on point
    aeqd_crs = get_aeqd_crs(lon, lat)

    # Transform to AEQD
    transformer = Transformer.from_crs("EPSG:4326", aeqd_crs, always_xy=True)
    point_aeqd = transform(transformer.transform, point)

    # Create buffer
    buffered_aeqd = point_aeqd.buffer(radius_meters)

    # Transform back to WGS84
    transformer_back = Transformer.from_crs(aeqd_crs, "EPSG:4326", always_xy=True)
    buffered_wgs84 = transform(transformer_back.transform, buffered_aeqd)

    # Return as GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"geometry": [buffered_wgs84]},
        crs="EPSG:4326"
    )

    logger.debug(f"Created {radius_meters}m buffer around ({lon:.6f}, {lat:.6f})")
    return gdf


def transform_to_utm(gdf: gpd.GeoDataFrame, lon: float, lat: float) -> gpd.GeoDataFrame:
    """
    Transform a GeoDataFrame to the appropriate UTM projection.

    Args:
        gdf: GeoDataFrame in WGS84
        lon: Reference longitude for UTM zone selection
        lat: Reference latitude for UTM zone selection

    Returns:
        GeoDataFrame in UTM projection

    Example:
        >>> gdf_utm = transform_to_utm(buildings_wgs84, -82.395, 28.118)
    """
    utm_crs = get_utm_crs(lon, lat)
    gdf_utm = gdf.to_crs(utm_crs)
    logger.debug(f"Transformed to {utm_crs}")
    return gdf_utm


def transform_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Transform a GeoDataFrame to WGS84 (EPSG:4326).

    Args:
        gdf: GeoDataFrame in any CRS

    Returns:
        GeoDataFrame in WGS84

    Example:
        >>> gdf_wgs84 = transform_to_wgs84(buildings_utm)
    """
    if gdf.crs is None:
        logger.warning("GeoDataFrame has no CRS defined. Assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")

    if gdf.crs.to_epsg() == 4326:
        return gdf

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    logger.debug("Transformed to WGS84")
    return gdf_wgs84


def calculate_geodesic_area(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Calculate geodesic area in square meters.
    More accurate than planar area for large polygons.

    Args:
        gdf: GeoDataFrame in WGS84

    Returns:
        Series with areas in square meters

    Example:
        >>> areas = calculate_geodesic_area(buildings)
        >>> print(areas)
        0    245.3
        1    180.7
        dtype: float64
    """
    # Use equal area projection (e.g., Mollweide)
    gdf_ea = gdf.to_crs("ESRI:54009")  # World Mollweide
    return gdf_ea.geometry.area


def get_bounds_in_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:4326"
) -> Tuple[float, float, float, float]:
    """
    Get bounding box of a GeoDataFrame in a specific CRS.

    Args:
        gdf: Input GeoDataFrame
        target_crs: Target CRS (default: WGS84)

    Returns:
        Tuple of (minx, miny, maxx, maxy)

    Example:
        >>> bounds = get_bounds_in_crs(buildings, "EPSG:4326")
        >>> minx, miny, maxx, maxy = bounds
    """
    gdf_target = gdf.to_crs(target_crs)
    return gdf_target.total_bounds
