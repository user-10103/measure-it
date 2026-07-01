"""
NAIP (National Agriculture Imagery Program) imagery ingestion.
Downloads and clips aerial imagery from AWS S3 Public Dataset.
"""
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path
import boto3
import rasterio
from rasterio.mask import mask
from rasterio.session import AWSSession
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from PIL import Image
from src.config import NAIP_S3_BUCKET, DATA_CACHE_DIR, NAIP_CLIP_PADDING_METERS, OUTPUT_DIR
from src.ingestion.exceptions import (
    NAIPNotFoundError,
    NAIPDownloadError,
    NAIPClipError,
    NAIPTileSearchError
)

logger = logging.getLogger(__name__)

# Constants
S3_LIST_MAX_KEYS = 1000
TILE_CHECK_LOG_INTERVAL = 10
DEFAULT_AWS_REGION = "us-west-2"
NAIP_RESOLUTION_ORDER = ["30cm", "60cm", "100cm"]
NAIP_PRODUCT_ORDER = ["rgbir_cog", "rgbir_mrf", "rgb_cog", "rgb_mrf"]


def latlon_to_naip_quadrangle(lat: float, lon: float) -> str:
    """
    Convert lat/lon to NAIP quadrangle name.
    NAIP uses SE corner reference point of 1-degree blocks.

    Args:
        lat: Latitude in WGS84 (-90 to 90)
        lon: Longitude in WGS84 (-180 to 180)

    Returns:
        Quadrangle name (e.g., "28082" for Tampa area)

    Raises:
        ValueError: If coordinates are out of valid range

    Example:
        >>> quad = latlon_to_naip_quadrangle(28.1178, -82.3951)
        >>> print(quad)
        28082
    """
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}")

    lat_se = int(lat)
    lon_se = abs(int(lon))
    return f"{lat_se:02d}{lon_se:03d}"


# ============================================================================
# Pure logic functions (no I/O, no side effects)
# ============================================================================

def _prioritize_years(available_years: list, preferred_year=None) -> list:
    """
    Sort years with preferred year(s) first, then newest to oldest.

    Args:
        available_years: List of year strings
        preferred_year: Year string, int, or list of year strings/ints to try first

    Returns:
        Sorted list of years
    """
    prioritized = []

    if preferred_year is not None:
        candidates = preferred_year if isinstance(preferred_year, (list, tuple)) else [preferred_year]
        for y in candidates:
            y = str(y)
            if y in available_years and y not in prioritized:
                prioritized.append(y)

    other_years = [y for y in available_years if y not in prioritized]
    prioritized.extend(sorted(other_years, reverse=True))

    return prioritized


def _build_candidate_paths(state: str, quadrangle: str, years: list) -> list:
    """
    Build list of candidate S3 paths to check for NAIP data.

    Args:
        state: State abbreviation
        quadrangle: Quadrangle ID
        years: Prioritized list of years to check

    Returns:
        List of tuples: (full_path, year, resolution, product)
    """
    res_order = ["30cm", "60cm", "100cm"]
    prod_order = ["rgbir_cog", "rgbir_mrf", "rgb_cog", "rgb_mrf"]

    candidates = []
    for year in years:
        for res in res_order:
            for prod in prod_order:
                path = f"{state}/{year}/{res}/{prod}/{quadrangle}/"
                candidates.append((path, year, res, prod))

    return candidates


def _point_in_tile_bounds(point_x: float, point_y: float, bounds: tuple) -> bool:
    """
    Check if a point is within tile bounds.

    Args:
        point_x: X coordinate in tile CRS
        point_y: Y coordinate in tile CRS
        bounds: Tile bounds (left, bottom, right, top)

    Returns:
        True if point is in bounds
    """
    left, bottom, right, top = bounds
    return left <= point_x <= right and bottom <= point_y <= top


def _create_buffered_clip_geometry(
    geometry,
    padding_meters: float,
    source_crs: str = "EPSG:4326"
):
    """
    Create buffered geometry for clipping in meters.

    Args:
        geometry: Shapely geometry in source CRS
        padding_meters: Buffer distance in meters
        source_crs: Source CRS (default WGS84)

    Returns:
        Buffered geometry in source CRS
    """
    from src.utils.projections import get_aeqd_crs

    centroid = geometry.centroid
    aeqd_crs = get_aeqd_crs(centroid.x, centroid.y)

    geom_m = gpd.GeoSeries([geometry], crs=source_crs).to_crs(aeqd_crs).iloc[0]
    buffered_m = geom_m.buffer(padding_meters)
    buffered_source = gpd.GeoSeries([buffered_m], crs=aeqd_crs).to_crs(source_crs).iloc[0]

    return buffered_source


# ============================================================================
# Data acquisition functions
# ============================================================================

def _list_s3_subdirectories(s3_client, bucket: str, prefix: str) -> list:
    """
    List subdirectories under an S3 prefix.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix to search

    Returns:
        List of subdirectory prefixes

    Raises:
        NAIPNotFoundError: If S3 listing fails
    """
    try:
        resp = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter="/",
            RequestPayer="requester",
            MaxKeys=S3_LIST_MAX_KEYS,
        )
        return [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    except Exception as e:
        error_msg = f"Failed to list S3 subdirectories at {prefix}: {str(e)}"
        logger.error(error_msg)
        raise NAIPNotFoundError(error_msg) from e


def _list_available_years(s3_client, bucket: str, state: str) -> list:
    """
    List available NAIP years for a state.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        state: State abbreviation

    Returns:
        List of year strings
    """
    year_dirs = _list_s3_subdirectories(s3_client, bucket, f"{state}/")
    return [p.split("/")[1] for p in year_dirs]


def _find_populated_naip_path(
    s3_client,
    bucket: str,
    state: str,
    quadrangle: str,
    years: list
) -> Optional[Dict[str, str]]:
    """
    Find first populated NAIP path from candidate years.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        state: State abbreviation
        quadrangle: Quadrangle ID
        years: Prioritized list of years to check

    Returns:
        Dict with year, res, prod, prefix or None
    """
    for year in years:
        res_dirs = _list_s3_subdirectories(s3_client, bucket, f"{state}/{year}/")
        available_res = [p.split("/")[2] for p in res_dirs]

        for res in [r for r in NAIP_RESOLUTION_ORDER if r in available_res]:
            prod_dirs = _list_s3_subdirectories(s3_client, bucket, f"{state}/{year}/{res}/")
            available_prod = [p.split("/")[3] for p in prod_dirs]

            for prod in [p for p in NAIP_PRODUCT_ORDER if p in available_prod]:
                path = f"{state}/{year}/{res}/{prod}/{quadrangle}/"

                # Check if path has any files
                probe = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=path,
                    MaxKeys=1,
                    RequestPayer="requester"
                )

                if probe.get("KeyCount", 0) > 0:
                    logger.info(f"Found NAIP data: {year}/{res}/{prod}")
                    return {
                        "year": year,
                        "res": res,
                        "prod": prod,
                        "prefix": path
                    }

    return None


def resolve_naip_prefix(
    state: str,
    quadrangle: str,
    preferred_year: Optional[str] = None,
    s3_client=None,
    bucket: str = NAIP_S3_BUCKET,
    aws_region: str = DEFAULT_AWS_REGION
) -> Optional[Dict[str, str]]:
    """
    Auto-resolve NAIP S3 prefix by finding available year/resolution/product.

    Args:
        state: State abbreviation (e.g., "fl")
        quadrangle: Quadrangle ID
        preferred_year: Preferred year (tries this first)
        s3_client: Optional boto3 S3 client (created if not provided)
        bucket: S3 bucket name
        aws_region: AWS region for client creation

    Returns:
        Dict with keys: year, res, prod, prefix; or None if not found

    Raises:
        ValueError: If state or quadrangle is empty

    Example:
        >>> info = resolve_naip_prefix("fl", "28082", "2023")
        >>> print(info["prefix"])
        fl/2023/30cm/rgbir_cog/28082/
    """
    if not state or not state.strip():
        raise ValueError("State abbreviation cannot be empty")
    if not quadrangle or not quadrangle.strip():
        raise ValueError("Quadrangle ID cannot be empty")

    s3 = s3_client if s3_client is not None else boto3.client("s3", region_name=aws_region)

    # Get available years
    available_years = _list_available_years(s3, bucket, state)
    if not available_years:
        return None

    # Prioritize years
    prioritized_years = _prioritize_years(available_years, preferred_year)

    # Find first populated path
    return _find_populated_naip_path(s3, bucket, state, quadrangle, prioritized_years)


def list_naip_tiles(
    prefix: str,
    s3_client=None,
    bucket: str = NAIP_S3_BUCKET,
    aws_region: str = DEFAULT_AWS_REGION
) -> list:
    """
    List all NAIP .tif tiles under a prefix.

    Args:
        prefix: S3 prefix path
        s3_client: Optional boto3 S3 client (created if not provided)
        bucket: S3 bucket name
        aws_region: AWS region for client creation

    Returns:
        List of S3 keys

    Raises:
        ValueError: If prefix is empty
        NAIPTileSearchError: If listing tiles fails

    Example:
        >>> tiles = list_naip_tiles("fl/2023/30cm/rgbir_cog/28082/")
    """
    if not prefix or not prefix.strip():
        raise ValueError("S3 prefix cannot be empty")

    s3 = s3_client if s3_client is not None else boto3.client("s3", region_name=aws_region)
    tiles = []

    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=prefix,
            RequestPayer="requester"
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".tif"):
                    tiles.append(key)
    except Exception as e:
        error_msg = f"Failed to list NAIP tiles at {prefix}: {str(e)}"
        logger.error(error_msg)
        raise NAIPTileSearchError(error_msg) from e

    logger.info(f"Found {len(tiles)} NAIP tiles in {prefix}")
    return tiles


# Persisted tile-bounds index: each NAIP COG's CRS+bounds, recorded whenever a
# tile header is read. Makes the per-quad S3 header sweep a one-time cost.
_TILE_BOUNDS_PATH = DATA_CACHE_DIR / "naip_tile_bounds.json"


def _load_tile_bounds_index() -> dict:
    try:
        import json as _json
        return _json.load(open(_TILE_BOUNDS_PATH))
    except Exception:
        return {}


def _index_tile_bounds(index: dict, tile_key: str, src) -> None:
    if tile_key not in index:
        b = src.bounds
        index[tile_key] = {"crs": str(src.crs),
                           "bounds": [b.left, b.bottom, b.right, b.top]}


def _save_tile_bounds_index(index: dict) -> None:
    try:
        import json as _json
        _TILE_BOUNDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _json.dump(index, open(_TILE_BOUNDS_PATH, "w"))
    except Exception as e:
        logger.debug(f"tile bounds index write failed: {e}")


def find_tile_containing_point(
    tiles: list,
    lat: float,
    lon: float,
    max_checks: Optional[int] = None,
    bucket: str = NAIP_S3_BUCKET,
    aws_session=None
) -> Optional[str]:
    """
    Find the NAIP tile that contains a specific point.
    Uses COG header reading to avoid full downloads.

    Args:
        tiles: List of S3 tile keys
        lat: Latitude of point
        lon: Longitude of point
        max_checks: Maximum tiles to check (None = check all)
        bucket: S3 bucket name
        aws_session: Optional AWSSession (created if not provided)

    Returns:
        S3 key of matching tile, or None

    Example:
        >>> tile = find_tile_containing_point(tiles, 28.1178, -82.3951)
    """
    building_wgs84 = Point(lon, lat)

    # Fast path 1: persisted bounds index — every tile bounds ever read
    # (remotely or locally) is recorded, so each quad's S3 header sweep
    # happens at most once. Critical with windowed S3 clipping, which no
    # longer leaves tiles on disk for the local-cache fast path to find.
    bounds_index = _load_tile_bounds_index()
    for tile_key in tiles:
        rec = bounds_index.get(tile_key)
        if not rec:
            continue
        try:
            tr = Transformer.from_crs("EPSG:4326", rec["crs"], always_xy=True)
            pt = shp_transform(tr.transform, building_wgs84)
            if box(*rec["bounds"]).contains(pt):
                logger.info(f"Found matching tile in bounds index: {Path(tile_key).name}")
                return tile_key
        except Exception:
            continue
    # If the index already covers every candidate tile and none matched,
    # the point genuinely has no tile in this prefix.
    if tiles and all(k in bounds_index for k in tiles):
        logger.warning(f"Bounds index covers all {len(tiles)} tiles; no tile contains point")
        return None

    # Fast path 2: a locally cached tile that contains the point.
    naip_cache = DATA_CACHE_DIR / "naip"
    if naip_cache.exists():
        by_name = {Path(k).name: k for k in tiles}
        for local in sorted(naip_cache.glob("*.tif")):
            key = by_name.get(local.name)
            if key is None:
                continue
            try:
                with rasterio.open(local) as src:
                    _index_tile_bounds(bounds_index, key, src)
                    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    pt = shp_transform(tr.transform, building_wgs84)
                    if box(*src.bounds).contains(pt):
                        logger.info(f"Found matching tile in local cache: {local.name}")
                        _save_tile_bounds_index(bounds_index)
                        return key
            except Exception:
                continue

    aws_session = aws_session if aws_session is not None else AWSSession(boto3.Session(), requester_pays=True)

    checks = max_checks or len(tiles)
    logger.info(f"Searching {checks} tiles for point ({lat:.4f}, {lon:.4f})")

    # Create transformer and transformed point outside loop (only needs to be done once)
    transformer = None
    building_transformed = None

    with rasterio.Env(aws_session):
        for i, tile_key in enumerate(tiles[:checks], 1):
            if i % TILE_CHECK_LOG_INTERVAL == 0:
                logger.debug(f"Checked {i}/{checks} tiles...")

            try:
                # Read tile metadata without full download
                with rasterio.open(f"/vsis3/{bucket}/{tile_key}") as src:
                    _index_tile_bounds(bounds_index, tile_key, src)
                    # Create transformer on first successful tile read
                    if transformer is None:
                        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                        building_transformed = shp_transform(transformer.transform, building_wgs84)

                    # Check if point is in tile bounds
                    tile_poly = box(*src.bounds)
                    if tile_poly.contains(building_transformed):
                        logger.info(f"Found matching tile: {Path(tile_key).name}")
                        _save_tile_bounds_index(bounds_index)
                        return tile_key

            except Exception as e:
                # Log individual tile errors at debug level
                logger.debug(f"Could not read tile {i}/{checks}: {str(e)[:100]}")
                continue

    _save_tile_bounds_index(bounds_index)
    # No tile found after checking all candidates
    logger.warning(f"No tile contains point ({lat:.4f}, {lon:.4f}) after checking {checks} tiles")
    return None


def download_naip_tile(
    tile_key: str,
    force: bool = False,
    s3_client=None,
    bucket: str = NAIP_S3_BUCKET,
    cache_dir: Path = DATA_CACHE_DIR,
    aws_region: str = DEFAULT_AWS_REGION
) -> Path:
    """
    Download a NAIP COG tile from S3.

    Args:
        tile_key: S3 key of tile
        force: Force re-download
        s3_client: Optional boto3 S3 client (created if not provided)
        bucket: S3 bucket name
        cache_dir: Directory for caching downloads
        aws_region: AWS region for client creation

    Returns:
        Path to downloaded file

    Example:
        >>> path = download_naip_tile("fl/2023/30cm/.../tile.tif")
    """
    naip_dir = cache_dir / "naip"
    naip_dir.mkdir(parents=True, exist_ok=True)

    local_path = naip_dir / Path(tile_key).name

    if local_path.exists() and not force:
        logger.info(f"Using cached NAIP tile: {local_path.name}")
        return local_path

    logger.info(f"Downloading NAIP tile: {Path(tile_key).name}")
    logger.info("This may take 30-60 seconds for large files...")

    s3 = s3_client if s3_client is not None else boto3.client("s3", region_name=aws_region)
    try:
        s3.download_file(
            bucket,
            tile_key,
            str(local_path),
            ExtraArgs={"RequestPayer": "requester"}
        )
        size_mb = local_path.stat().st_size / 1024 / 1024
        logger.info(f"Downloaded: {local_path.name} ({size_mb:.1f} MB)")
        return local_path

    except Exception as e:
        error_msg = f"Failed to download NAIP tile {tile_key}: {str(e)}"
        logger.error(error_msg)

        # Clean up partial download
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass

        raise NAIPDownloadError(error_msg) from e


def _clip_raster_to_geometry(src_path: Path, geometry, target_crs: str):
    """
    Clip raster file to geometry.

    Args:
        src_path: Path to source raster
        geometry: Shapely geometry in WGS84
        target_crs: Target CRS for geometry transformation

    Returns:
        Tuple of (clipped_array, transform, metadata_dict)

    Raises:
        NAIPClipError: If clipping operation fails
    """
    try:
        with rasterio.open(src_path) as src:
            # Transform geometry to raster CRS
            transformer = Transformer.from_crs(target_crs, src.crs, always_xy=True)
            geom_raster_crs = shp_transform(transformer.transform, geometry)

            # Check intersection
            raster_bounds = box(*src.bounds)
            if not raster_bounds.intersects(geom_raster_crs):
                raise NAIPClipError(
                    f"Geometry is outside raster bounds. "
                    f"Geometry: {geom_raster_crs.bounds}, Raster: {src.bounds}"
                )

            # Clip
            geom_mapping = [mapping(geom_raster_crs)]
            clipped_array, clipped_transform = mask(src, geom_mapping, crop=True, filled=True, nodata=0)

            # Build metadata
            metadata = src.meta.copy()
            metadata.update({
                "driver": "GTiff",
                "height": clipped_array.shape[1],
                "width": clipped_array.shape[2],
                "transform": clipped_transform
            })

            return clipped_array, clipped_transform, metadata

    except NAIPClipError:
        raise
    except Exception as e:
        error_msg = f"Failed to clip raster {src_path}: {str(e)}"
        logger.error(error_msg)
        raise NAIPClipError(error_msg) from e


def _save_clipped_raster(array: np.ndarray, metadata: dict, output_path: Path) -> Path:
    """
    Save clipped raster array to TIF file.

    Args:
        array: Clipped raster array
        metadata: Rasterio metadata dict
        output_path: Output file path

    Returns:
        Path to saved file

    Raises:
        NAIPClipError: If saving fails
    """
    try:
        with rasterio.open(output_path, "w", **metadata) as dest:
            dest.write(array)

        logger.info(f"Saved clipped raster: {array.shape[2]}x{array.shape[1]} pixels to {output_path.name}")
        return output_path

    except Exception as e:
        error_msg = f"Failed to save clipped raster to {output_path}: {str(e)}"
        logger.error(error_msg)
        raise NAIPClipError(error_msg) from e


def _create_rgb_preview(array: np.ndarray, output_path: Path) -> Path:
    """
    Create RGB preview PNG from multi-band array.

    Args:
        array: Multi-band raster array (bands, height, width)
        output_path: Output PNG path

    Returns:
        Path to saved PNG

    Raises:
        NAIPClipError: If preview creation fails
    """
    try:
        # Extract RGB bands (first 3 bands)
        rgb = array[[0, 1, 2], :, :].transpose(1, 2, 0)

        # Normalize to 8-bit
        if rgb.max() > 255:
            rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

        # Save as PNG
        Image.fromarray(rgb).save(output_path)
        logger.info(f"Saved RGB preview to {output_path.name}")

        return output_path

    except Exception as e:
        error_msg = f"Failed to create RGB preview {output_path}: {str(e)}"
        logger.error(error_msg)
        raise NAIPClipError(error_msg) from e


def clip_naip_to_building(
    naip_path: Path,
    building_polygon: gpd.GeoDataFrame,
    output_dir: Optional[Path] = None,
    padding_meters: float = NAIP_CLIP_PADDING_METERS
) -> Tuple[Path, Path, Dict]:
    """
    Clip NAIP imagery to building footprint with padding.

    Args:
        naip_path: Path to NAIP COG file
        building_polygon: Building geometry in WGS84
        output_dir: Output directory for clipped files
        padding_meters: Buffer padding in meters

    Returns:
        Tuple of (clipped_tif_path, rgb_png_path, metadata_dict)

    Raises:
        ValueError: If inputs are invalid
        NAIPClipError: If clipping fails

    Example:
        >>> from src.config import OUTPUT_DIR
        >>> tif, png, meta = clip_naip_to_building(naip_path, building_gdf, OUTPUT_DIR)
    """
    # Input validation. GDAL virtual paths (/vsis3/, /vsicurl/...) are remote
    # datasets opened by rasterio directly — a filesystem existence check
    # would wrongly reject them.
    if not str(naip_path).startswith("/vsi") and not naip_path.exists():
        raise ValueError(f"NAIP file does not exist: {naip_path}")
    if building_polygon.empty:
        raise ValueError("Building polygon GeoDataFrame is empty")
    if padding_meters < 0:
        raise ValueError(f"Padding must be non-negative, got {padding_meters}")

    # Use default output directory if not provided
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare geometry
    if hasattr(building_polygon, 'union_all'):
        building_geom = building_polygon.union_all()
    else:
        # Fallback for older geopandas versions
        building_geom = building_polygon.unary_union

    buffered_geom = _create_buffered_clip_geometry(building_geom, padding_meters)

    # Clip raster
    clipped_array, _, raster_meta = _clip_raster_to_geometry(
        naip_path, buffered_geom, "EPSG:4326"
    )

    # Save files
    clipped_tif = output_dir / "naip_clipped.tif"
    _save_clipped_raster(clipped_array, raster_meta, clipped_tif)

    clipped_png = output_dir / "naip_clipped_rgb.png"
    _create_rgb_preview(clipped_array, clipped_png)

    # Build metadata
    with rasterio.open(naip_path) as src:
        resolution = src.res[0]
        crs = str(src.crs)

    metadata = {
        "naip_source": str(naip_path.name),
        "building_center_wgs84": list(building_geom.centroid.coords[0]),
        "clipped_bounds_wgs84": list(buffered_geom.bounds),
        "padding_meters": padding_meters,
        "size_pixels": [int(clipped_array.shape[2]), int(clipped_array.shape[1])],
        "resolution_meters": resolution,
        "crs": crs
    }

    return clipped_tif, clipped_png, metadata


def get_naip_for_location(
    lat: float,
    lon: float,
    state: str,
    building_polygon: gpd.GeoDataFrame,
    preferred_year: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Tuple[Optional[Path], Optional[Path], Optional[Dict]]:
    """
    Complete NAIP acquisition workflow for a location.

    Args:
        lat: Latitude
        lon: Longitude
        state: State code (e.g., "fl")
        building_polygon: Building footprint in WGS84
        preferred_year: Preferred imagery year
        output_dir: Output directory for clipped files (defaults to OUTPUT_DIR from config)

    Returns:
        Tuple of (clipped_tif, rgb_png, metadata) or (None, None, None)

    Raises:
        NAIPNotFoundError: If NAIP data not available for location
        NAIPTileSearchError: If tile search fails
        NAIPDownloadError: If download fails
        NAIPClipError: If clipping fails

    Example:
        >>> tif, png, meta = get_naip_for_location(
        ...     28.1178, -82.3951, "fl", building_gdf, "2023"
        ... )
    """
    logger.info(f"Starting NAIP acquisition for location ({lat:.4f}, {lon:.4f})")

    # Use default output directory if not provided
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Normalize: S3 paths use lowercase state codes
    state = state.lower()

    # Get quadrangle
    quadrangle = latlon_to_naip_quadrangle(lat, lon)
    logger.info(f"NAIP quadrangle: {quadrangle} for state: {state}")

    # Resolve path
    resolved = resolve_naip_prefix(state, quadrangle, preferred_year)
    if not resolved:
        error_msg = f"No NAIP data found for {state}/{quadrangle}"
        logger.error(error_msg)
        raise NAIPNotFoundError(error_msg)

    logger.info(f"Using NAIP data: {resolved['year']}/{resolved['res']}/{resolved['prod']}")

    # List tiles
    tiles = list_naip_tiles(resolved["prefix"])
    if not tiles:
        error_msg = f"No tiles found in {resolved['prefix']}"
        logger.error(error_msg)
        raise NAIPNotFoundError(error_msg)

    # Find matching tile
    tile_key = find_tile_containing_point(tiles, lat, lon)
    if not tile_key:
        logger.warning(f"Point not in any tile bounds, using first tile as fallback")
        tile_key = tiles[0]

    # Clip. Prefer (1) an already-downloaded local tile, then (2) windowed
    # reads straight from S3 — the tiles are ~600 MB COGs but the chip window
    # is ~200x200 px, so range requests fetch only the needed blocks
    # (seconds vs ~8 min for a full-tile download). Full download remains
    # the fallback when the remote windowed read fails.
    local_tile = DATA_CACHE_DIR / "naip" / Path(tile_key).name
    if local_tile.exists():
        return clip_naip_to_building(local_tile, building_polygon, output_dir)

    try:
        vsipath = f"/vsis3/{NAIP_S3_BUCKET}/{tile_key}"
        aws_session = AWSSession(boto3.Session(), requester_pays=True)
        with rasterio.Env(aws_session):
            result = clip_naip_to_building(Path(vsipath), building_polygon, output_dir)
        logger.info(f"Clipped via windowed S3 read (no tile download): {Path(tile_key).name}")
        return result
    except Exception as e:
        logger.warning(f"Windowed S3 clip failed ({e}) — falling back to full tile download")

    naip_path = download_naip_tile(tile_key)
    return clip_naip_to_building(naip_path, building_polygon, output_dir)
