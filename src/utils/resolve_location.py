"""Robust user-input -> coordinates for the report frontend.

One entry point, ``resolve_location(text)``, that accepts anything a
non-technical user might paste into the box:

  * decimal coordinates      "28.0303, -80.69809"  /  "28.0303 -80.69809"
  * a Google Maps link       ".../@28.0303,-80.698,19z"  /  "?q=28.03,-80.69"
  * a street address         "909 Spring Island Way, Melbourne, FL 32940"
  * a landmark               "Hemingway House, Key West"

Lessons baked in from the Colab runs (2026-07-04): Nominatim geocoded
"909 Spring Island Way, Melbourne, FL" but FAILED the same address with the
ZIP appended, so every geocoder is retried over ADDRESS VARIANTS (as given,
without ZIP, with ", USA"). Provider chain (first hit wins):

  parse coords -> cache -> Amazon Location (if configured) -> Census -> Nominatim

Amazon Location activates when the ``AWS_LOCATION_PLACE_INDEX`` env var names a
place index (commercial Esri/HERE data — primary in the AWS deployment).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

from src.utils.geocode import (
    GeocodingError,
    _cache_get,
    _cache_put,
    _geocode_census,
    _geocode_nominatim,
    validate_coordinates,
)

logger = logging.getLogger(__name__)


class LocationError(GeocodingError):
    """User-facing resolution failure (friendly message, no stack trace)."""


FRIENDLY_MISS = (
    "We couldn't find that location. Try the full street address with city and "
    "state (e.g. \"909 Spring Island Way, Melbourne, FL\"), or paste coordinates "
    "from Google Maps (right-click the roof and click the numbers that appear)."
)

# ── 1. coordinate / maps-link parsing ────────────────────────────────────────
# "28.0303, -80.69809"  |  "28.0303 -80.69809"  |  "(28.03, -80.69)"
_COORD_PAIR = re.compile(
    r"^\s*\(?\s*(-?\d{1,3}(?:\.\d+)?)\s*[,;\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*\)?\s*$"
)
# Google Maps URL forms: .../@28.0303,-80.698,19z   ?q=28.03,-80.69   ?ll=...
_MAPS_AT = re.compile(r"@(-?\d{1,3}\.\d+),(-?\d{1,3}\.\d+)")
_MAPS_QUERY = re.compile(r"[?&](?:q|ll|query)=(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)")


def parse_coords(text: str) -> Optional[Dict[str, float]]:
    """Extract an explicit lat/lon from raw text or a maps link, else None."""
    text = text.strip()
    m = _COORD_PAIR.match(text) or _MAPS_AT.search(text) or _MAPS_QUERY.search(text)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not validate_coordinates(lat, lon) or (abs(lat) < 0.01 and abs(lon) < 0.01):
        return None
    return {"lat": lat, "lon": lon}


# ── 2. address variants (the ZIP lesson) ─────────────────────────────────────
_ZIP_RE = re.compile(r"[,\s]+\d{5}(?:-\d{4})?\s*$")


def address_variants(address: str) -> List[str]:
    """Orderings a geocoder is retried with: as given, ZIP stripped, ', USA'."""
    base = " ".join(address.strip().split())
    variants = [base]
    no_zip = _ZIP_RE.sub("", base).rstrip(",").strip()
    if no_zip and no_zip != base:
        variants.append(no_zip)
    if not base.lower().endswith(("usa", "united states")):
        variants.append(f"{base}, USA")
    return variants


# ── 3. providers ─────────────────────────────────────────────────────────────
def _geocode_amazon(address: str) -> Optional[Dict[str, float]]:
    """Amazon Location Service (primary in the AWS deployment).

    Needs the AWS_LOCATION_PLACE_INDEX env var + standard AWS credentials
    (instance IAM role in production). Silently unavailable otherwise.
    """
    index = os.getenv("AWS_LOCATION_PLACE_INDEX")
    if not index:
        return None
    try:
        import boto3
        client = boto3.client(
            "location", region_name=os.getenv("AWS_REGION", "us-west-2")
        )
        resp = client.search_place_index_for_text(
            IndexName=index, Text=address, MaxResults=1,
            FilterCountries=["USA"],
        )
        results = resp.get("Results", [])
        if results:
            lon, lat = results[0]["Place"]["Geometry"]["Point"]
            return {"lat": float(lat), "lon": float(lon)}
    except Exception as e:  # noqa: BLE001 — degrade to the free chain
        logger.warning("Amazon Location failed (%s) — falling back", e)
    return None


_PROVIDERS = [
    ("amazon", _geocode_amazon),
    ("census", _geocode_census),      # authoritative for US street addresses
    ("nominatim", _geocode_nominatim),  # handles landmarks ("Hemingway House")
]


# ── entry point ──────────────────────────────────────────────────────────────
def resolve_location(text: str) -> Dict:
    """User input -> {"lat", "lon", "source"}.

    Raises LocationError with a friendly, user-facing message on total miss.
    """
    if not text or not text.strip():
        raise LocationError(FRIENDLY_MISS)
    text = text.strip()

    coords = parse_coords(text)
    if coords:
        logger.info("resolve_location: parsed coordinates %s", coords)
        return {**coords, "source": "coordinates"}

    cached = _cache_get(text)
    if cached:
        return {**cached, "source": "cache"}

    for variant in address_variants(text):
        for name, fn in _PROVIDERS:
            coords = fn(variant)
            if coords and validate_coordinates(coords["lat"], coords["lon"]):
                logger.info("resolve_location: %s matched %r -> (%.6f, %.6f)",
                            name, variant, coords["lat"], coords["lon"])
                _cache_put(text, coords)   # cache under what the USER typed
                return {**coords, "source": name}

    raise LocationError(FRIENDLY_MISS)
