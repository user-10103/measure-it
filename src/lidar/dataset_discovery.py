"""
LiDAR Dataset Discovery Module

Queries USGS 3DEP WESM spatial index to find LiDAR coverage for a location.
Single responsibility: find the best available LiDAR dataset and construct EPT URL.
"""

import datetime as _dt
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import boto3
import geopandas as gpd
import pandas as pd
from botocore.exceptions import ClientError
from shapely.geometry import Point

from src.config import WESM_INDEX_PATH

logger = logging.getLogger(__name__)

ENTWINE_RESOURCES_URL = "https://usgs.entwine.io/boundaries/resources.geojson"
ENTWINE_CACHE = Path(__file__).resolve().parents[2] / "data" / "ept_resources.geojson"
ENTWINE_TTL_DAYS = 30        # refresh the coverage index after this many days
USGS_EPT_BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
_YEAR_MIN = 1990             # no 3DEP-era survey predates this
# a year sitting immediately after one of these is a RELEASE year, not a
# collection year (e.g. ..._2017_LAS_2019 was collected 2017, released 2019)
_RELEASE_MARKERS = ("LAS", "PUBLISHED", "REL")


def _ensure_entwine_cache(refresh: Optional[bool] = None) -> bool:
    """Make sure the entwine coverage index on disk is present and fresh enough.

    Returns True when a usable cache exists.

    This used to be ``if not ENTWINE_CACHE.exists()`` — fetched once and then kept
    forever, so every survey USGS published after the first run was invisible and
    "newest dataset covering this point" silently meant "newest as of first run".
    Now it refreshes after ``ENTWINE_TTL_DAYS`` (override with
    MEASURE_IT_ENTWINE_TTL_DAYS); MEASURE_IT_ENTWINE_REFRESH=1 or ``refresh=True``
    forces it.

    A FAILED refresh deliberately falls back to the stale cache: an out-of-date
    index still finds LiDAR, whereas a hard failure loses pitch for the whole
    report. Only a missing cache AND a failed download is fatal.
    """
    import time

    if refresh is None:
        refresh = os.getenv("MEASURE_IT_ENTWINE_REFRESH", "0") == "1"
    try:
        ttl_days = float(os.getenv("MEASURE_IT_ENTWINE_TTL_DAYS", ENTWINE_TTL_DAYS))
    except (TypeError, ValueError):
        ttl_days = ENTWINE_TTL_DAYS

    exists = ENTWINE_CACHE.exists()
    age_s = (time.time() - ENTWINE_CACHE.stat().st_mtime) if exists else 0.0
    stale = exists and age_s > ttl_days * 86400.0
    if exists and not stale and not refresh:
        return True

    why = "forced" if refresh else ("stale" if stale else "missing")
    logger.info("Refreshing entwine EPT index (%s) -> %s", why, ENTWINE_CACHE)
    try:
        import requests
        resp = requests.get(ENTWINE_RESOURCES_URL, timeout=120)
        resp.raise_for_status()
        ENTWINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ENTWINE_CACHE.write_bytes(resp.content)
        return True
    except Exception as e:  # noqa: BLE001 — a stale index beats no LiDAR
        if exists:
            logger.warning("entwine index refresh failed (%s) — using stale cache "
                           "(%.0f days old)", e, age_s / 86400.0)
            return True
        logger.error("entwine index unavailable and nothing cached: %s", e)
        return False


def _collection_year(name: str) -> int:
    r"""Best estimate of the COLLECTION year from a USGS/entwine dataset name.

    Dataset names mix two different quantities, which the previous
    ``re.search(r"(\d{4})$", name)`` compared against each other:

        USGS_LPC_FL_Upper_Saint_Johns_2017_LAS_2019   collected 2017, released 2019
        FL_Peninsular_FDEM_Brevard_2018               collected 2018
        FL_Elgin_2006_2008                            collected across 2006-2008

    Taking the trailing year reads 2019 for the first and 2018 for the second, so a
    2017 survey outranks a 2018 one purely because its LAS release was later. That
    is how 1600 Sarno Rd drew Upper-Saint-Johns instead of Brevard.

    Rule: ignore any year directly preceded by a release marker (LAS/PUBLISHED/REL),
    then take the MAX of what remains — so a collection RANGE is represented by the
    year it finished, which is the fair thing to compare against a single-year name.
    Returns 0 when the name carries no plausible year.
    """
    current = _dt.date.today().year
    years: List[int] = []
    for m in re.finditer(r"(?<!\d)(\d{4})(?!\d)", name):
        y = int(m.group(1))
        if not (_YEAR_MIN <= y <= current + 1):
            continue
        prefix = name[: m.start()].rstrip("_-").upper()
        if any(prefix.endswith(mark) for mark in _RELEASE_MARKERS):
            continue                      # release/publication year, not collection
        years.append(y)
    return max(years) if years else 0


def _index_density(props: dict, geom) -> float:
    """Cheap quality proxy: indexed point count per unit of footprint area.

    Only meaningful when the entwine index carries a point count; returns 0.0
    otherwise, in which case ranking falls through to the name tiebreak. This is a
    COARSE signal — the real number that matters is points per square metre over
    the actual parcel (one Tampa roof came back at ~1.3 pts/m2), and that cannot be
    known without fetching the tiles. Proper per-parcel density gating is future work.
    """
    pts = props.get("points") or props.get("count")
    try:
        pts = float(pts)
    except (TypeError, ValueError):
        return 0.0
    try:
        area = float(geom.area)
    except Exception:  # noqa: BLE001
        return 0.0
    return pts / area if area > 0 else 0.0


def discover_ept_candidates(lat: float, lon: float,
                            refresh: Optional[bool] = None) -> List[dict]:
    """Every EPT dataset covering a point, best first.

    Returns dicts of ``{name, url, year, density}``. Ranked by collection year
    (newest first), then indexed point density (densest first), then name ascending.

    The name is the LAST tiebreak and exists only to make the order deterministic.
    Previously ``max((year, name))`` fell through to comparing names whenever years
    tied, which quietly made an alphabetical accident the selection criterion.

    Callers that want resilience should walk this list rather than taking [0] — a
    dataset can cover a point and still yield no usable returns over the building.
    """
    import json as _json

    from shapely.geometry import shape as _shape

    if not _ensure_entwine_cache(refresh):
        return []
    try:
        data = _json.load(open(ENTWINE_CACHE))
    except Exception as e:  # noqa: BLE001
        logger.error("entwine index unreadable (%s)", e)
        return []

    pt = Point(lon, lat)
    out: List[dict] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        name = props.get("name", "")
        if not name:
            continue
        try:
            geom = _shape(feat["geometry"])
            if not geom.contains(pt):
                continue
        except Exception:  # noqa: BLE001 — a broken feature shouldn't kill discovery
            continue
        out.append({
            "name": name,
            "url": f"{USGS_EPT_BASE}/{name}/ept.json",
            "year": _collection_year(name),
            "density": _index_density(props, geom),
        })

    out.sort(key=lambda d: (-d["year"], -d["density"], d["name"]))
    return out


def discover_ept_from_entwine(lat: float, lon: float,
                              refresh: Optional[bool] = None) -> Optional[str]:
    """Newest EPT dataset covering a point -> ept.json URL, or None.

    Thin wrapper over :func:`discover_ept_candidates` kept for callers that only
    want one answer. Prefer the candidate list where a fallback is possible.
    """
    cands = discover_ept_candidates(lat, lon, refresh=refresh)
    if not cands:
        logger.warning(f"No entwine EPT coverage at ({lat:.5f}, {lon:.5f})")
        return None
    best = cands[0]
    logger.info("Entwine EPT for (%.5f, %.5f): %s (%s)%s", lat, lon, best["name"],
                best["year"] or "year?",
                f" [+{len(cands) - 1} more covering]" if len(cands) > 1 else "")
    return best["url"]


def discover_lidar_dataset(lat: float, lon: float, index_path: str = None) -> Optional[dict]:
    """
    Query WESM spatial index for best LiDAR dataset covering a point location.

    Single responsibility: find the most suitable dataset based on:
    1. Has point cloud data available
    2. Most recent collection date
    3. Best quality level

    Args:
        lat: Latitude in WGS84 decimal degrees
        lon: Longitude in WGS84 decimal degrees
        index_path: Path to WESM.gpkg file (defaults to config value)

    Returns:
        Dict with dataset metadata:
        {
            "workunit": "FL_Peninsular_Hernando_2019",
            "lpc_link": "https://...",
            "sourcedem_link": "http://...",
            "collect_start": "2019-03-24",
            "collect_end": "2019-04-21",
            "quality_level": "QL 1",
            "dem_resolution": 1.0,
            "lpc_category": "Meets",
            "metadata_link": "https://..."
        }
        or None if no coverage found

    Raises:
        FileNotFoundError: If WESM index file does not exist
        RuntimeError: If spatial query fails
    """
    if index_path is None:
        index_path = WESM_INDEX_PATH

    if not index_path:
        error_msg = "WESM_INDEX_PATH not configured in .env file"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    index_file = Path(index_path)
    if not index_file.exists():
        error_msg = f"WESM index file not found at {index_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Querying WESM spatial index for location ({lat:.6f}, {lon:.6f})")

    try:
        # Use bbox filter for faster initial load (0.2 degree buffer ~ 22km)
        bbox = (lon - 0.2, lat - 0.2, lon + 0.2, lat + 0.2)
        gdf = gpd.read_file(index_path, bbox=bbox)

        if len(gdf) == 0:
            logger.warning(f"No datasets found in spatial index near ({lat:.6f}, {lon:.6f})")
            return None

        logger.debug(f"Found {len(gdf)} datasets in bbox search")

        # Create query point
        point = Point(lon, lat)

        # Spatial filter: datasets that contain the point
        matches = gdf[gdf.geometry.contains(point)]

        if len(matches) == 0:
            logger.warning(
                f"Point ({lat:.6f}, {lon:.6f}) not covered by any dataset. "
                f"Found {len(gdf)} nearby datasets but none contain this exact location"
            )
            return None

        logger.info(f"Found {len(matches)} datasets covering location")

        # Rank by: 1) Has LiDAR point cloud, 2) Newest collection date, 3) Best quality
        matches = matches.copy()
        matches['has_lpc'] = matches['lpc_link'].notna() & (matches['lpc_category'] != 'Not Applicable')
        matches['collect_end_date'] = pd.to_datetime(matches['collect_end'])

        # Quality level to numeric (QL 0 = best, QL 5 = worst)
        matches['ql_numeric'] = matches['ql'].apply(_parse_quality_level)

        # Sort: LPC available first, then newest, then best quality
        matches = matches.sort_values(
            by=['has_lpc', 'collect_end_date', 'ql_numeric'],
            ascending=[False, False, True]
        )

        best = matches.iloc[0]

        dataset_info = {
            "workunit": best['workunit'],
            "lpc_link": best.get('lpc_link'),
            "sourcedem_link": best.get('sourcedem_link'),
            "collect_start": str(best.get('collect_start')),
            "collect_end": str(best.get('collect_end')),
            "quality_level": best.get('ql'),
            "dem_resolution": best.get('dem_gsd_meters'),
            "lpc_category": best.get('lpc_category'),
            "metadata_link": best.get('metadata_link')
        }

        logger.info(
            f"Selected dataset: {dataset_info['workunit']} "
            f"(collected {dataset_info['collect_start']} to {dataset_info['collect_end']}, "
            f"quality {dataset_info['quality_level']})"
        )

        return dataset_info

    except Exception as e:
        error_msg = f"Failed to query WESM spatial index: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def normalize_workunit_for_s3(workunit: str) -> List[str]:
    """
    Generate S3 bucket key variants from WESM workunit name.

    WESM uses uppercase naming (FL_HILLSBOROUGHCO_LOT2_2011) but S3 bucket keys
    often use CamelCase with hyphens (FL_HillsboroughCo-Lot2_2011).

    Single responsibility: generate plausible S3 key variants to try.

    Args:
        workunit: Workunit name from WESM (e.g., "FL_HILLSBOROUGHCO_LOT2_2011")

    Returns:
        List of S3 key variants to try, ordered by likelihood
    """
    variants = [workunit]  # Original first

    # Pattern 1: Convert _LOT to -Lot (common pattern)
    if "_LOT" in workunit:
        variants.append(workunit.replace("_LOT", "-Lot"))

    # Pattern 2: Convert UPPERCASE county names to CamelCase
    # e.g., FL_HILLSBOROUGHCO_LOT2_2011 → FL_HillsboroughCo-Lot2_2011
    def to_camel_case(name: str) -> str:
        """Convert COUNTYNAME to CountyName preserving underscores."""
        parts = name.split("_")
        result = []
        for part in parts:
            # Skip state codes (2 uppercase letters) and years (4 digits)
            if len(part) == 2 and part.isupper():
                result.append(part)  # Keep state code as-is
            elif part.isdigit():
                result.append(part)  # Keep years as-is
            elif part.isupper() and len(part) > 2:
                # Handle county names ending in CO (e.g., HILLSBOROUGHCO → HillsboroughCo)
                if part.endswith("CO"):
                    # Split at CO suffix: HILLSBOROUGHCO → Hillsborough + Co
                    base = part[:-2].capitalize()  # Hillsborough
                    result.append(base + "Co")
                else:
                    result.append(part.capitalize())
            else:
                result.append(part)
        return "_".join(result)

    camel = to_camel_case(workunit)
    if camel != workunit and camel not in variants:
        variants.append(camel)

    # Pattern 3: CamelCase + hyphen for LOT
    # Check for both _LOT (uppercase) and _Lot (after CamelCase conversion)
    if "_LOT" in camel:
        camel_hyphen = camel.replace("_LOT", "-Lot")
        if camel_hyphen not in variants:
            variants.append(camel_hyphen)
    elif "_Lot" in camel:
        camel_hyphen = camel.replace("_Lot", "-Lot")
        if camel_hyphen not in variants:
            variants.append(camel_hyphen)

    # Pattern 4: Handle other underscore-to-hyphen cases
    # e.g., FL_ELGIN_2006_2008 → FL_Elgin_2006-2008
    if re.search(r'_(\d{4})_(\d{4})$', workunit):
        year_hyphen = re.sub(r'_(\d{4})_(\d{4})$', r'_\1-\2', workunit)
        if year_hyphen not in variants:
            variants.append(year_hyphen)
        # Also try CamelCase version
        camel_year = to_camel_case(year_hyphen)
        if camel_year not in variants:
            variants.append(camel_year)

    return variants


def construct_ept_url(dataset: dict) -> Optional[str]:
    """
    Construct and validate EPT endpoint URL from WESM dataset metadata.

    Single responsibility: find working EPT URL with validation and fallback.

    Fallback strategy:
    1. Try workunit name variants (handles WESM→S3 naming mismatches)
    2. If all variants fail, explore S3 for state's available datasets

    Args:
        dataset: Dataset dict from discover_lidar_dataset() with keys:
            - workunit: Dataset workunit name
            - lpc_category: Quality category
            (other fields ignored for URL construction)

    Returns:
        Valid EPT URL if found, None if no EPT data available

    Raises:
        ValueError: If dataset dict is missing required fields
    """
    if not dataset or "workunit" not in dataset:
        raise ValueError("Dataset dict must contain 'workunit' field")

    workunit = dataset["workunit"]
    base_url = "https://usgs-lidar-public.s3-us-west-2.amazonaws.com"

    # Tier 1: Try workunit name variants
    variants = normalize_workunit_for_s3(workunit)
    logger.info(f"Trying {len(variants)} EPT URL variants for workunit: {workunit}")

    for variant in variants:
        ept_url = f"{base_url}/{variant}/ept.json"
        logger.debug(f"Trying variant: {variant}")

        if validate_ept_url(ept_url):
            if variant != workunit:
                logger.info(f"EPT URL found with naming variant: {variant}")
            else:
                logger.info(f"EPT URL validated successfully")
            return ept_url

    # Tier 2: Fall back to S3 exploration
    logger.warning(f"EPT endpoint not found for any variant of {workunit}")

    # Extract state from workunit (e.g., "FL_..." -> "FL")
    state_match = re.match(r'^([A-Z]{2})_', workunit)
    if state_match:
        state = state_match.group(1)
        logger.info(f"Attempting S3 exploration fallback for state {state}")
        fallback_url = discover_ept_via_s3_exploration(state)

        if fallback_url:
            return fallback_url

    # All tiers failed
    logger.error(
        f"No EPT data available for dataset {workunit}. "
        f"This may be a legacy dataset not yet converted to EPT format."
    )
    return None


def validate_ept_url(ept_url: str) -> bool:
    """
    Validate that EPT endpoint exists and is accessible.

    Single responsibility: check if ept.json file exists at URL.

    Args:
        ept_url: EPT endpoint URL to validate

    Returns:
        True if EPT endpoint exists and is accessible, False otherwise
    """
    try:
        # Extract bucket and key from URL
        # Format: https://usgs-lidar-public.s3-us-west-2.amazonaws.com/{workunit}/ept.json
        if "s3" not in ept_url:
            logger.debug(f"URL is not an S3 URL: {ept_url}")
            return False

        parts = ept_url.replace("https://", "").split("/", 1)
        if len(parts) < 2:
            logger.debug(f"Could not parse bucket/key from URL: {ept_url}")
            return False

        bucket_domain = parts[0]
        key = parts[1]

        # Extract bucket name
        bucket = bucket_domain.split(".")[0]  # e.g., "usgs-lidar-public"

        # Check if object exists
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.head_object(Bucket=bucket, Key=key, RequestPayer="requester")

        logger.debug(f"EPT URL validated successfully: {ept_url}")
        return True

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404":
            logger.debug(f"EPT endpoint not found: {ept_url}")
        else:
            logger.debug(f"EPT validation failed ({error_code}): {ept_url}")
        return False
    except Exception as e:
        logger.debug(f"EPT validation error: {str(e)}")
        return False


def discover_ept_via_s3_exploration(state: str) -> Optional[str]:
    """
    Explore S3 bucket to find EPT datasets for a state.

    Single responsibility: list available EPT datasets in S3 for fallback.

    Args:
        state: State abbreviation (e.g., "FL")

    Returns:
        First valid EPT URL found for the state, or None if none exist
    """
    try:
        s3 = boto3.client("s3", region_name="us-west-2")
        bucket = "usgs-lidar-public"

        # List prefixes starting with state
        prefix = f"{state.upper()}_"

        logger.info(f"Exploring S3 bucket for {state.upper()} datasets...")

        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter="/",
            RequestPayer="requester",
            MaxKeys=100
        )

        prefixes = [p["Prefix"] for p in response.get("CommonPrefixes", [])]

        if not prefixes:
            logger.warning(f"No datasets found in S3 for state {state.upper()}")
            return None

        logger.info(f"Found {len(prefixes)} potential datasets in S3")

        # Try each prefix to find one with EPT
        for prefix in prefixes:
            ept_url = f"https://{bucket}.s3-us-west-2.amazonaws.com/{prefix}ept.json"
            if validate_ept_url(ept_url):
                logger.info(f"Found valid EPT dataset via S3 exploration: {ept_url}")
                return ept_url

        logger.warning(f"No valid EPT datasets found for state {state.upper()}")
        return None

    except Exception as e:
        logger.error(f"S3 exploration failed: {str(e)}")
        return None


def _parse_quality_level(ql_str) -> int:
    """
    Convert quality level string to numeric value for sorting.

    Args:
        ql_str: Quality level string (e.g., "QL 1", "QL2")

    Returns:
        Numeric quality level (0 = best, 99 = unknown)
    """
    if pd.isna(ql_str) or not isinstance(ql_str, str):
        return 99

    match = re.search(r'(\d+)', ql_str)
    return int(match.group(1)) if match else 99
