"""
Building footprint candidate selection and ranking.
Selects the most likely building from multiple candidates.
"""
import logging
from typing import Dict
import geopandas as gpd
from shapely.geometry import Point
from src.utils.projections import get_aeqd_crs, create_meter_buffer
from src.ingestion.ms_footprints import get_buildings_in_buffer
from src.config import DEFAULT_BUFFER_METERS

logger = logging.getLogger(__name__)


def rank_candidates(
    buildings: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    top_n: int = 3
) -> gpd.GeoDataFrame:
    """
    Rank building candidates by distance from pin.

    Args:
        buildings: GeoDataFrame of building footprints in WGS84
        lat: Pin latitude
        lon: Pin longitude
        top_n: Number of top candidates to return

    Returns:
        GeoDataFrame with top N candidates ranked by distance

    Example:
        >>> candidates = rank_candidates(buildings, 28.1178, -82.3951, top_n=3)
        >>> print(candidates[["dist_m"]])
    """
    if len(buildings) == 0:
        logger.warning("No buildings to rank")
        return buildings

    # Create point in local projection for accurate distance
    aeqd_crs = get_aeqd_crs(lon, lat)
    point_m = Point(0, 0)  # Center of AEQD is at origin

    # Transform buildings to local projection
    buildings_m = buildings.to_crs(aeqd_crs)

    # Calculate distance from pin
    buildings_m["dist_m"] = buildings_m.geometry.distance(point_m)

    # Sort by distance and take top N
    candidates = buildings_m.nsmallest(top_n, "dist_m")

    # Transform back to WGS84
    candidates = candidates.to_crs("EPSG:4326")

    logger.info(
        f"Ranked {len(buildings)} buildings, selected top {len(candidates)}"
    )
    logger.info(
        f"Closest building distance: {candidates['dist_m'].min():.2f}m"
    )

    return candidates


def select_building(
    lat: float,
    lon: float,
    buffer_meters: float = DEFAULT_BUFFER_METERS,
    auto_select: bool = True
) -> Dict:
    """
    Complete building selection workflow.

    Args:
        lat: Latitude
        lon: Longitude
        buffer_meters: Search radius in meters
        auto_select: Automatically select closest (True) or return candidates (False)

    Returns:
        Dict with keys:
            - selected: Selected building geometry (GeoSeries)
            - candidates: Top 3 candidates (GeoDataFrame)
            - dist_m: Distance to selected building
            - rank: Rank of selected (0=closest)

    Example:
        >>> result = select_building(28.1178, -82.3951)
        >>> print(f"Distance: {result['dist_m']:.2f}m")
    """
    logger.info(f"Selecting building at ({lat:.6f}, {lon:.6f})")
    logger.info(f"Search radius: {buffer_meters}m")

    # Create buffer
    buffer_gdf = create_meter_buffer(lon, lat, buffer_meters)
    logger.info("Created search buffer")

    # Get buildings in buffer
    buildings = get_buildings_in_buffer(buffer_gdf)

    if len(buildings) == 0:
        raise ValueError(
            f"No buildings found within {buffer_meters}m of location. "
            "Try increasing search radius or check coordinates."
        )

    logger.info(f"Found {len(buildings)} buildings in buffer")

    # Rank candidates
    candidates = rank_candidates(buildings, lat, lon, top_n=3)

    # Select best candidate
    if auto_select:
        selected_idx = candidates["dist_m"].idxmin()
        selected = candidates.loc[selected_idx]

        logger.info("=" * 60)
        logger.info("SELECTED BUILDING")
        logger.info("=" * 60)
        logger.info(f"Distance from pin: {selected['dist_m']:.2f}m")
        logger.info(f"Geometry type: {selected.geometry.geom_type}")
        logger.info(f"Bounds: {selected.geometry.bounds}")
        logger.info("=" * 60)

        return {
            "selected": selected,
            "candidates": candidates,
            "dist_m": selected["dist_m"],
            "rank": 0
        }
    else:
        logger.info(f"Returning {len(candidates)} candidates for manual selection")
        return {
            "selected": None,
            "candidates": candidates,
            "dist_m": None,
            "rank": None
        }


def export_candidates(
    candidates: gpd.GeoDataFrame,
    output_path: str,
    format: str = "geojson"
) -> None:
    """
    Export candidate buildings to file.

    Args:
        candidates: GeoDataFrame of candidates
        output_path: Output file path
        format: Output format ("geojson", "shapefile", "gpkg")

    Example:
        >>> export_candidates(candidates, "candidates.geojson")
    """
    driver_map = {
        "geojson": "GeoJSON",
        "shapefile": "ESRI Shapefile",
        "gpkg": "GPKG"
    }

    driver = driver_map.get(format.lower(), "GeoJSON")

    candidates.to_file(output_path, driver=driver)
    logger.info(f"Exported {len(candidates)} candidates to {output_path}")
