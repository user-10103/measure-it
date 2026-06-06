"""
Microsoft Buildings Footprint ingestion.
Downloads and filters building polygons from MS Global Buildings dataset.
"""
import logging
import math
import gzip
import json
from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape
from shapely.validation import make_valid
from src.config import MS_BUILDINGS_INDEX_URL, DATA_CACHE_DIR, QUADKEY_LEVEL
from src.utils.cache import cached_download, download_file
from src.ingestion.exceptions import (
    MSBuildingsError,
    MSBuildingsIndexError,
    MSBuildingsDownloadError,
    MSBuildingsParseError
)

logger = logging.getLogger(__name__)

# Constants for Web Mercator projection and quadkey operations
WGS84_MIN_LAT = -85.05112878  # Web Mercator latitude limit (south)
WGS84_MAX_LAT = 85.05112878   # Web Mercator latitude limit (north)
WGS84_MIN_LON = -180.0
WGS84_MAX_LON = 180.0
TILE_PADDING = 1  # Padding tiles to ensure complete coverage at boundaries
ASCII_DIGIT_OFFSET = 48  # ord('0') - for converting char to digit
MIN_ZOOM_LEVEL = 0
MAX_ZOOM_LEVEL = 23  # Maximum practical zoom level for quadkeys


def latlon_to_tile(lat: float, lon: float, z: int) -> Tuple[int, int]:
    """
    Convert lat/lon to Web Mercator tile coordinates.

    Uses the Web Mercator projection (EPSG:3857) which has latitude limits
    of approximately ±85.05° due to the Mercator projection's singularity at the poles.

    Args:
        lat: Latitude in WGS84 (-85.05 to 85.05)
        lon: Longitude in WGS84 (-180 to 180)
        z: Zoom level (0 to 23)

    Returns:
        Tuple of (tile_x, tile_y)

    Raises:
        ValueError: If coordinates or zoom level are out of valid range
    """
    if not (WGS84_MIN_LAT <= lat <= WGS84_MAX_LAT):
        raise ValueError(
            f"Latitude must be between {WGS84_MIN_LAT} and {WGS84_MAX_LAT}, got {lat}"
        )
    if not (WGS84_MIN_LON <= lon <= WGS84_MAX_LON):
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
    if not (MIN_ZOOM_LEVEL <= z <= MAX_ZOOM_LEVEL):
        raise ValueError(f"Zoom level must be between {MIN_ZOOM_LEVEL} and {MAX_ZOOM_LEVEL}, got {z}")

    num_tiles_at_zoom = 2 ** z
    x = int((lon + 180.0) / 360.0 * num_tiles_at_zoom)

    # Web Mercator Y coordinate using inverse Gudermannian function
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * num_tiles_at_zoom
    )
    return x, y


def tile_to_quadkey(x: int, y: int, z: int) -> str:
    """
    Convert tile coordinates to Bing Maps quadkey.

    Args:
        x: Tile X coordinate (must be non-negative)
        y: Tile Y coordinate (must be non-negative)
        z: Zoom level (0 to 23)

    Returns:
        Quadkey string

    Raises:
        ValueError: If coordinates are negative or zoom is invalid
    """
    if x < 0:
        raise ValueError(f"Tile X coordinate must be non-negative, got {x}")
    if y < 0:
        raise ValueError(f"Tile Y coordinate must be non-negative, got {y}")
    if not (MIN_ZOOM_LEVEL <= z <= MAX_ZOOM_LEVEL):
        raise ValueError(f"Zoom level must be between {MIN_ZOOM_LEVEL} and {MAX_ZOOM_LEVEL}, got {z}")

    quadkey = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


def quadkey_to_tile(qk: str) -> Tuple[int, int, int]:
    """
    Convert Bing Maps quadkey to tile coordinates.

    Args:
        qk: Quadkey string (must contain only digits 0-3)

    Returns:
        Tuple of (tile_x, tile_y, zoom_level)

    Raises:
        ValueError: If quadkey is empty or contains invalid characters
    """
    if not qk or not qk.strip():
        raise ValueError("Quadkey cannot be empty")

    # Validate quadkey contains only digits 0-3
    if not all(c in '0123' for c in qk):
        raise ValueError(
            f"Quadkey must contain only digits 0-3, got: {qk[:20]}{'...' if len(qk) > 20 else ''}"
        )

    x = y = 0
    z = len(qk)
    for i, ch in enumerate(qk):
        bit = z - i - 1
        mask = 1 << bit
        digit = ord(ch) - ASCII_DIGIT_OFFSET  # '0'..'3' -> 0..3
        if digit & 1:  # bit 0 -> x
            x |= mask
        if digit & 2:  # bit 1 -> y
            y |= mask
    return x, y, z


def tile_bounds(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    """
    Get geographic bounds of a Web Mercator tile.

    Args:
        x: Tile X coordinate (must be non-negative)
        y: Tile Y coordinate (must be non-negative)
        z: Zoom level (0 to 23)

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84

    Raises:
        ValueError: If coordinates are negative or zoom is invalid
    """
    if x < 0:
        raise ValueError(f"Tile X coordinate must be non-negative, got {x}")
    if y < 0:
        raise ValueError(f"Tile Y coordinate must be non-negative, got {y}")
    if not (MIN_ZOOM_LEVEL <= z <= MAX_ZOOM_LEVEL):
        raise ValueError(f"Zoom level must be between {MIN_ZOOM_LEVEL} and {MAX_ZOOM_LEVEL}, got {z}")

    num_tiles_at_zoom = 2 ** z
    lon_min = x / num_tiles_at_zoom * 360.0 - 180.0
    lon_max = (x + 1) / num_tiles_at_zoom * 360.0 - 180.0

    # Inverse Web Mercator projection for latitude
    lat_min_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / num_tiles_at_zoom)))
    lat_max_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / num_tiles_at_zoom)))
    lat_min = math.degrees(lat_min_rad)
    lat_max = math.degrees(lat_max_rad)

    return lon_min, lat_min, lon_max, lat_max


def download_index(force: bool = False) -> Path:
    """
    Download Microsoft Buildings dataset index.

    Args:
        force: Force re-download even if cached

    Returns:
        Path to downloaded index CSV

    Raises:
        MSBuildingsDownloadError: If download fails

    Example:
        >>> index_path = download_index()
    """
    cache_path = DATA_CACHE_DIR / "msbuildings" / "dataset-links.csv"

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        error_msg = f"Failed to create cache directory {cache_path.parent}: {str(e)}"
        logger.error(error_msg)
        raise MSBuildingsDownloadError(error_msg) from e

    if cache_path.exists() and not force:
        logger.info(f"Using cached index: {cache_path}")
        return cache_path

    logger.info(f"Downloading MS Buildings index from {MS_BUILDINGS_INDEX_URL}")
    try:
        download_file(MS_BUILDINGS_INDEX_URL, cache_path)
        logger.info(f"Downloaded index: {cache_path.stat().st_size / 1024:.1f} KB")
        return cache_path
    except Exception as e:
        error_msg = f"Failed to download MS Buildings index from {MS_BUILDINGS_INDEX_URL}: {str(e)}"
        logger.error(error_msg)
        # Clean up partial download
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception:
                pass
        raise MSBuildingsDownloadError(error_msg) from e


def load_index() -> pd.DataFrame:
    """
    Load and normalize MS Buildings dataset index.

    Returns:
        DataFrame with 'quadkey' and 'url' columns

    Example:
        >>> index_df = load_index()
        >>> print(index_df.head())
    """
    index_path = download_index()

    df = pd.read_csv(index_path, dtype={"QuadKey": "string"})
    df.columns = df.columns.str.strip().str.lower()

    # Find quadkey and URL columns with proper error handling
    quad_col = next((c for c in df.columns if "quad" in c and "key" in c), None)
    if not quad_col:
        raise ValueError(
            f"Index CSV missing quadkey column. Available columns: {list(df.columns)}"
        )

    url_col = next((c for c in df.columns if c == "url" or c.endswith("_url")), None)
    if not url_col:
        raise ValueError(
            f"Index CSV missing URL column. Available columns: {list(df.columns)}"
        )

    df = df[[quad_col, url_col]].rename(
        columns={quad_col: "quadkey", url_col: "url"}
    )
    df["quadkey"] = df["quadkey"].str.strip()

    logger.info(f"Loaded index: {len(df)} shards")
    return df


def find_covering_quadkeys(
    buffer_polygon: gpd.GeoDataFrame,
    level: int = QUADKEY_LEVEL
) -> set:
    """
    Find quadkeys that intersect with a buffer polygon.

    Args:
        buffer_polygon: GeoDataFrame with buffer geometry in WGS84
        level: Quadkey zoom level (default: 9, range: 0-23)

    Returns:
        Set of quadkey strings

    Raises:
        ValueError: If buffer_polygon is empty or level is invalid

    Example:
        >>> quadkeys = find_covering_quadkeys(buffer_gdf, level=9)
    """
    if buffer_polygon.empty:
        raise ValueError("Buffer polygon GeoDataFrame is empty")
    if not (MIN_ZOOM_LEVEL <= level <= MAX_ZOOM_LEVEL):
        raise ValueError(f"Level must be between {MIN_ZOOM_LEVEL} and {MAX_ZOOM_LEVEL}, got {level}")

    bounds = buffer_polygon.total_bounds  # (minx, miny, maxx, maxy)
    min_lon, min_lat, max_lon, max_lat = bounds

    # Get tile range
    x_min, y_max = latlon_to_tile(min_lat, min_lon, level)
    x_max, y_min = latlon_to_tile(max_lat, max_lon, level)

    # Add padding to ensure complete coverage at tile boundaries
    x_min -= TILE_PADDING
    y_min -= TILE_PADDING
    x_max += TILE_PADDING
    y_max += TILE_PADDING

    # Ensure non-negative coordinates
    x_min = max(0, x_min)
    y_min = max(0, y_min)

    # Find intersecting tiles
    buffer_geom = buffer_polygon.union_all() if hasattr(buffer_polygon, 'union_all') else buffer_polygon.unary_union
    selected_tiles = []

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            lon0, lat0, lon1, lat1 = tile_bounds(x, y, level)
            tile_poly = box(lon0, lat0, lon1, lat1)
            if tile_poly.intersects(buffer_geom):
                selected_tiles.append((x, y))

    quadkeys = {tile_to_quadkey(x, y, level) for x, y in selected_tiles}
    logger.info(f"Found {len(quadkeys)} covering quadkeys")
    return quadkeys


def download_shards(subset_links: pd.DataFrame) -> List[Path]:
    """
    Download Microsoft Buildings shard files with error recovery.

    Args:
        subset_links: DataFrame with 'quadkey' and 'url' columns

    Returns:
        List of paths to successfully downloaded shard files

    Raises:
        MSBuildingsDownloadError: If all downloads fail

    Example:
        >>> paths = download_shards(subset_df)
    """
    shards_dir = DATA_CACHE_DIR / "msbuildings" / "shards"

    try:
        shards_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        error_msg = f"Failed to create shards directory {shards_dir}: {str(e)}"
        logger.error(error_msg)
        raise MSBuildingsDownloadError(error_msg) from e

    urls = subset_links["url"].dropna().unique().tolist()
    local_paths = []
    failed_downloads = []

    for url in urls:
        fname = shards_dir / Path(url.split("/")[-1])

        if fname.exists() and fname.stat().st_size > 0:
            logger.debug(f"Using cached shard: {fname.name}")
            local_paths.append(fname)
            continue

        logger.info(f"Downloading shard: {fname.name}")
        try:
            download_file(url, fname, show_progress=True)
            local_paths.append(fname)
            logger.debug(f"Successfully downloaded: {fname.name}")
        except Exception as e:
            logger.warning(f"Failed to download {fname.name}: {str(e)[:100]}")
            failed_downloads.append((url, str(e)))
            # Continue with other downloads
            continue

    # Log summary
    if failed_downloads:
        logger.warning(
            f"Downloaded {len(local_paths)}/{len(urls)} shards successfully. "
            f"{len(failed_downloads)} failed."
        )
        for url, error in failed_downloads:
            logger.debug(f"Failed: {url} - {error[:50]}")

    if not local_paths:
        error_msg = f"All {len(urls)} shard downloads failed"
        logger.error(error_msg)
        raise MSBuildingsDownloadError(error_msg)

    return local_paths


def parse_shard_file(shard_path: Path, buffer_polygon: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Parse a shard file and extract buildings within buffer.

    Args:
        shard_path: Path to .csv.gz shard file
        buffer_polygon: Buffer polygon to filter buildings

    Returns:
        GeoDataFrame with buildings in WGS84

    Example:
        >>> buildings = parse_shard_file(shard_path, buffer_gdf)
    """
    buffer_geom = buffer_polygon.union_all() if hasattr(buffer_polygon, 'union_all') else buffer_polygon.unary_union
    geoms = []
    props = []

    # Track parsing statistics
    total_lines = 0
    skipped_json_errors = 0
    skipped_no_geometry = 0
    skipped_invalid_geometry = 0
    skipped_outside_buffer = 0

    with gzip.open(shard_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] != "{":
                continue

            total_lines += 1

            try:
                feat = json.loads(line)
            except json.JSONDecodeError as e:
                skipped_json_errors += 1
                logger.debug(f"Skipping invalid JSON in {shard_path.name}: {str(e)[:50]}")
                continue

            try:
                geom_dict = feat.get("geometry")
                if not geom_dict:
                    skipped_no_geometry += 1
                    continue

                geometry = shape(geom_dict)
                if not geometry.is_valid:
                    geometry = make_valid(geometry)

                if not geometry.is_valid:
                    skipped_invalid_geometry += 1
                    logger.debug(f"Skipping invalid geometry in {shard_path.name}")
                    continue

                if geometry.intersects(buffer_geom):
                    geoms.append(geometry)
                    props.append(feat.get("properties", {}))
                else:
                    skipped_outside_buffer += 1

            except Exception as e:
                # Catch geometry processing errors
                logger.warning(f"Error processing geometry in {shard_path.name}: {type(e).__name__}: {str(e)[:100]}")
                skipped_invalid_geometry += 1
                continue

    # Log parsing statistics
    total_valid = len(geoms)
    total_skipped = skipped_json_errors + skipped_no_geometry + skipped_invalid_geometry
    if total_lines > 0:
        success_rate = (total_valid / total_lines) * 100
        logger.info(
            f"Parsed {shard_path.name}: {total_valid} buildings ({success_rate:.1f}% of {total_lines} records)"
        )
        if total_skipped > 0:
            logger.debug(
                f"Skipped: {skipped_json_errors} JSON errors, {skipped_no_geometry} no geometry, "
                f"{skipped_invalid_geometry} invalid geometry, {skipped_outside_buffer} outside buffer"
            )

    if not geoms:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")

    # Deduplicate by geometry
    gdf["__wkb"] = gdf.geometry.apply(lambda g: g.wkb_hex)
    gdf = gdf.drop_duplicates("__wkb").drop(columns="__wkb")

    logger.info(f"After deduplication: {len(gdf)} unique buildings from {shard_path.name}")
    return gdf


def get_buildings_in_buffer(
    buffer_polygon: gpd.GeoDataFrame,
    level: int = QUADKEY_LEVEL
) -> gpd.GeoDataFrame:
    """
    Get all Microsoft Buildings footprints within a buffer.

    Args:
        buffer_polygon: Buffer geometry in WGS84
        level: Quadkey zoom level (default: 9, range: 0-23)

    Returns:
        GeoDataFrame with building footprints in WGS84

    Raises:
        ValueError: If buffer_polygon is None, empty, or level is invalid
        MSBuildingsError: If index loading or shard downloads fail

    Example:
        >>> buildings = get_buildings_in_buffer(buffer_gdf)
    """
    # Input validation
    if buffer_polygon is None:
        raise ValueError("buffer_polygon cannot be None")
    if buffer_polygon.empty:
        raise ValueError("buffer_polygon GeoDataFrame is empty")
    if not (MIN_ZOOM_LEVEL <= level <= MAX_ZOOM_LEVEL):
        raise ValueError(f"Level must be between {MIN_ZOOM_LEVEL} and {MAX_ZOOM_LEVEL}, got {level}")

    logger.info(f"Querying MS Buildings with quadkey level {level}")

    # Load index
    index_df = load_index()

    # Find covering quadkeys
    quadkeys = find_covering_quadkeys(buffer_polygon, level)

    # Build tile geometries for intersection test
    tiles_data = []
    for qk in quadkeys:
        x, y, z = quadkey_to_tile(qk)
        lon0, lat0, lon1, lat1 = tile_bounds(x, y, z)
        tiles_data.append({
            "quadkey": qk,
            "geometry": box(lon0, lat0, lon1, lat1)
        })

    tiles_gdf = gpd.GeoDataFrame(tiles_data, crs="EPSG:4326")

    # Filter index to intersecting tiles
    buffer_geom = buffer_polygon.union_all() if hasattr(buffer_polygon, 'union_all') else buffer_polygon.unary_union
    subset_tiles = tiles_gdf[tiles_gdf.intersects(buffer_geom)]
    subset_links = index_df[index_df["quadkey"].isin(subset_tiles["quadkey"])]

    logger.info(f"Downloading {len(subset_links)} shard(s)")

    # Download shards
    shard_paths = download_shards(subset_links)

    # Parse all shards and combine
    all_buildings = []
    for shard_path in shard_paths:
        buildings = parse_shard_file(shard_path, buffer_polygon)
        if len(buildings) > 0:
            all_buildings.append(buildings)

    if not all_buildings:
        logger.warning("No buildings found in buffer")
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    combined = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True), crs="EPSG:4326")
    logger.info(f"Total buildings in buffer: {len(combined)}")
    return combined
