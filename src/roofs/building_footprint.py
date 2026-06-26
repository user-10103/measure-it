"""
Fetch a building footprint polygon for a given lat/lng using the
Microsoft Global ML Building Footprints dataset (130M US buildings).

The Microsoft dataset is organised by Bing Maps quadkeys at zoom=9.
Each tile is a small GeoJSON.gz (~50–300 KB) hosted on Azure Blob Storage.
We download, cache, and filter by proximity so each address costs one
small HTTP request (cached for the session).

Usage
-----
from src.roofs.building_footprint import get_footprint_pixels

# Returns a list of (x, y) pixel tuples in the chip's coordinate space,
# or None if no footprint was found within max_dist_m metres.
poly_px = get_footprint_pixels(
    lat=27.823, lon=-82.645,
    chip_lat_lon_bounds=(north, south, east, west),
    chip_w_px=512, chip_h_px=512,
    max_dist_m=50,
)
"""

import gzip
import json
import logging
import math
import os
import threading
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Microsoft dataset configuration ──────────────────────────────────────────
# The Global ML Building Footprints (2023+) tile index.
# Each quadkey at zoom=9 maps to a GeoJSON.gz on Azure Blob Storage.
_DATASET_INDEX_URL = (
    "https://minedbuildings.blob.core.windows.net/"
    "global-buildings/dataset-links.csv"
)
_CACHE_DIR = Path(os.environ.get("FOOTPRINT_CACHE_DIR", "/tmp/footprint_cache"))
_index: Optional[dict] = None  # quadkey → download_url
_index_lock = threading.Lock()


# ── Quadkey utilities ─────────────────────────────────────────────────────────

def _lat_lon_to_tile(lat: float, lon: float, zoom: int = 9) -> tuple[int, int]:
    """Convert WGS-84 lat/lon to Bing Maps tile (x, y) at given zoom."""
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_quadkey(x: int, y: int, zoom: int = 9) -> str:
    qk = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        qk.append(str(digit))
    return "".join(qk)


def _lat_lon_to_quadkey(lat: float, lon: float, zoom: int = 9) -> str:
    x, y = _lat_lon_to_tile(lat, lon, zoom)
    return _tile_to_quadkey(x, y, zoom)


# ── Index loading ─────────────────────────────────────────────────────────────

def _load_index() -> dict:
    global _index
    with _index_lock:
        if _index is not None:
            return _index

        index_cache = _CACHE_DIR / "dataset-links.csv"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not index_cache.exists():
            log.info("Downloading MS Building Footprints index (~5 MB)…")
            try:
                urllib.request.urlretrieve(_DATASET_INDEX_URL, index_cache)
            except Exception as e:
                log.warning(f"Could not download footprint index: {e}")
                _index = {}
                return _index

        mapping: dict[str, str] = {}
        with open(index_cache, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("QuadKey"):
                    continue
                parts = line.split(",", 1)
                if len(parts) == 2:
                    mapping[parts[0].strip()] = parts[1].strip()

        _index = mapping
        log.info(f"Loaded {len(_index)} quadkey entries")
        return _index


# ── Tile fetching ─────────────────────────────────────────────────────────────

def _fetch_tile(quadkey: str) -> list[dict]:
    """Download + cache one tile; return list of GeoJSON feature dicts."""
    index = _load_index()
    url = index.get(quadkey)
    if not url:
        # Try parent quadkey (zoom 8) as fallback
        parent = quadkey[:-1]
        url = index.get(parent)
        if not url:
            log.debug(f"No tile found for quadkey {quadkey}")
            return []

    cache_file = _CACHE_DIR / f"{quadkey}.geojson.gz"
    if not cache_file.exists():
        try:
            urllib.request.urlretrieve(url, cache_file)
        except Exception as e:
            log.warning(f"Could not download tile {quadkey}: {e}")
            return []

    try:
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("features", [])
    except Exception as e:
        log.warning(f"Could not parse tile {quadkey}: {e}")
        cache_file.unlink(missing_ok=True)
        return []


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _centroid(coords: list) -> tuple[float, float]:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _lon_lat_to_pixel(
    lon: float, lat: float,
    north: float, south: float, east: float, west: float,
    w_px: int, h_px: int,
) -> tuple[float, float]:
    x = (lon - west) / (east - west) * w_px
    y = (north - lat) / (north - south) * h_px
    return x, y


# ── Public API ────────────────────────────────────────────────────────────────

def get_footprint_latlon(
    lat: float,
    lon: float,
    max_dist_m: float = 60.0,
) -> Optional[list[tuple[float, float]]]:
    """
    Return the closest building footprint polygon as [(lat, lon), …] or None.

    Looks up the Microsoft Global ML Building Footprints tile that covers
    (lat, lon) and returns the polygon whose centroid is within max_dist_m.
    """
    qk = _lat_lon_to_quadkey(lat, lon, zoom=9)
    features = _fetch_tile(qk)
    if not features:
        return None

    best_dist = float("inf")
    best_poly: Optional[list] = None

    for feat in features:
        geom = feat.get("geometry", {})
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        rings = (
            geom["coordinates"]
            if geom["type"] == "Polygon"
            else geom["coordinates"][0]
        )
        outer = rings[0]
        clat, clon = _centroid(outer)
        dist = _haversine_m(lat, lon, clat, clon)
        if dist < best_dist:
            best_dist = dist
            best_poly = outer

    if best_dist > max_dist_m or best_poly is None:
        log.debug(f"Nearest footprint {best_dist:.0f} m away — skipping")
        return None

    return [(c[1], c[0]) for c in best_poly]  # (lat, lon)


def get_footprint_pixels(
    lat: float,
    lon: float,
    chip_bounds: tuple[float, float, float, float],
    chip_w_px: int = 512,
    chip_h_px: int = 512,
    max_dist_m: float = 60.0,
) -> Optional[list[tuple[float, float]]]:
    """
    Return the building footprint as pixel coordinates within the chip.

    Parameters
    ----------
    lat, lon : float
        Address coordinates (WGS-84).
    chip_bounds : (north, south, east, west) in decimal degrees.
    chip_w_px, chip_h_px : int
        Chip dimensions in pixels.
    max_dist_m : float
        Reject footprints whose centroid is further than this from (lat, lon).

    Returns
    -------
    List of (x, y) pixel tuples (closed ring, last == first) or None.
    """
    poly_latlon = get_footprint_latlon(lat, lon, max_dist_m=max_dist_m)
    if poly_latlon is None:
        return None

    north, south, east, west = chip_bounds
    pixels = [
        _lon_lat_to_pixel(plon, plat, north, south, east, west, chip_w_px, chip_h_px)
        for plat, plon in poly_latlon
    ]

    # Clip to chip bounds
    clipped = [
        (max(0.0, min(float(chip_w_px), x)), max(0.0, min(float(chip_h_px), y)))
        for x, y in pixels
    ]
    return clipped


def get_footprint_flat(
    lat: float,
    lon: float,
    chip_bounds: tuple[float, float, float, float],
    chip_w_px: int = 512,
    chip_h_px: int = 512,
    max_dist_m: float = 60.0,
) -> Optional[list[float]]:
    """Same as get_footprint_pixels but returns a flat [x, y, x, y, …] list
    compatible with COCO segmentation format."""
    poly = get_footprint_pixels(lat, lon, chip_bounds, chip_w_px, chip_h_px, max_dist_m)
    if poly is None:
        return None
    return [coord for xy in poly for coord in xy]
