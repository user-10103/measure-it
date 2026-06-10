"""
Address geocoding utilities.
Converts street addresses to lat/lon coordinates using Nominatim (OpenStreetMap).
No API key required.
"""
import logging
from typing import Dict, Optional
import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "measure-it-roof-pipeline/1.0"}
CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"


class GeocodingError(Exception):
    """Raised when geocoding fails."""
    pass


def _geocode_nominatim(address: str) -> Optional[Dict[str, float]]:
    """Try Nominatim; return coords dict or None on miss."""
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"q": address, "format": "json", "limit": 1, "addressdetails": 0},
            headers=NOMINATIM_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        results = r.json()
        if results:
            return {"lat": float(results[0]["lat"]), "lon": float(results[0]["lon"])}
    except Exception:
        pass
    return None


def _geocode_census(address: str) -> Optional[Dict[str, float]]:
    """Try US Census Bureau Geocoder; return coords dict or None on miss."""
    try:
        r = requests.get(
            CENSUS_GEOCODER_URL,
            params={"address": address, "benchmark": "Public_AR_Current", "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        matches = r.json().get("result", {}).get("addressMatches", [])
        if matches:
            coords = matches[0]["coordinates"]
            return {"lat": float(coords["y"]), "lon": float(coords["x"])}
    except Exception:
        pass
    return None


def geocode_address(address: str) -> Dict[str, float]:
    """
    Convert street address to latitude/longitude. No API key required.

    Tries Nominatim (OpenStreetMap) first, then falls back to the US Census
    Bureau Geocoder which has complete coverage of all US addresses.

    Args:
        address: Full street address (e.g., "123 Main St, City, State ZIP")

    Returns:
        Dictionary with 'lat' and 'lon' keys

    Raises:
        GeocodingError: If both geocoders return no results

    Example:
        >>> coords = geocode_address("16347 Heathrow Dr, Tampa, FL 33647")
        >>> print(coords)
        {'lat': 28.1178764, 'lon': -82.3951068}
    """
    logger.info(f"Geocoding: {address}")

    coords = _geocode_nominatim(address)
    if coords:
        logger.info(f"Nominatim -> ({coords['lat']:.6f}, {coords['lon']:.6f})")
        return coords

    logger.info("Nominatim miss — trying US Census Geocoder")
    coords = _geocode_census(address)
    if coords:
        logger.info(f"Census -> ({coords['lat']:.6f}, {coords['lon']:.6f})")
        return coords

    raise GeocodingError(
        f"Could not geocode address: '{address}'. "
        "Check that the address is a valid US street address."
    )


def get_coordinates(
    address: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Dict[str, float]:
    """
    Get coordinates from either address or explicit lat/lon.

    Priority:
    1. If lat/lon provided, use them directly
    2. Otherwise geocode the address with Nominatim

    Args:
        address: Street address to geocode
        lat: Explicit latitude (overrides address)
        lon: Explicit longitude (overrides address)

    Returns:
        Dictionary with 'lat' and 'lon' keys

    Raises:
        ValueError: If neither address nor lat/lon provided
        GeocodingError: If geocoding fails
    """
    if lat is not None and lon is not None:
        logger.info(f"Using explicit coordinates: ({lat:.6f}, {lon:.6f})")
        return {"lat": lat, "lon": lon}

    if address:
        return geocode_address(address)

    raise ValueError("Must provide either address or both lat and lon")


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within valid ranges.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        True if valid, False otherwise
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180
