"""
Address geocoding utilities.
Converts street addresses to lat/lon coordinates using Google Maps Geocoding API.
"""
import logging
from typing import Dict, Optional, Tuple
import requests
from src.config import GOOGLE_MAPS_API_KEY

logger = logging.getLogger(__name__)


class GeocodingError(Exception):
    """Raised when geocoding fails."""
    pass


def geocode_address(address: str, api_key: Optional[str] = None) -> Dict[str, float]:
    """
    Convert street address to latitude/longitude coordinates.

    Args:
        address: Full street address (e.g., "123 Main St, City, State ZIP")
        api_key: Google Maps API key (defaults to config value)

    Returns:
        Dictionary with 'lat' and 'lon' keys

    Raises:
        GeocodingError: If geocoding fails or returns no results

    Example:
        >>> coords = geocode_address("16347 Heathrow Dr, Tampa, FL 33647")
        >>> print(coords)
        {'lat': 28.1178764, 'lon': -82.3951068}
    """
    if api_key is None:
        api_key = GOOGLE_MAPS_API_KEY

    if not api_key:
        raise GeocodingError(
            "Google Maps API key not configured. "
            "Set GOOGLE_MAPS_API_KEY in your .env file."
        )

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    try:
        logger.info(f"Geocoding address: {address}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK":
            raise GeocodingError(
                f"Geocoding failed with status: {data.get('status')}. "
                f"Message: {data.get('error_message', 'No error message')}"
            )

        if not data.get("results"):
            raise GeocodingError(f"No results found for address: {address}")

        location = data["results"][0]["geometry"]["location"]
        coords = {
            "lat": location["lat"],
            "lon": location["lng"]
        }

        logger.info(f"Geocoded to: ({coords['lat']:.6f}, {coords['lon']:.6f})")
        return coords

    except requests.RequestException as e:
        raise GeocodingError(f"Network error during geocoding: {str(e)}") from e


def get_coordinates(
    address: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None
) -> Dict[str, float]:
    """
    Get coordinates from either address or explicit lat/lon.

    Priority:
    1. If lat/lon provided, use them
    2. Otherwise, geocode the address

    Args:
        address: Street address to geocode
        lat: Explicit latitude (overrides address)
        lon: Explicit longitude (overrides address)

    Returns:
        Dictionary with 'lat' and 'lon' keys

    Raises:
        ValueError: If neither address nor lat/lon provided
        GeocodingError: If geocoding fails

    Example:
        >>> # Using explicit coordinates
        >>> coords = get_coordinates(lat=28.1178764, lon=-82.3951068)
        >>>
        >>> # Using address
        >>> coords = get_coordinates(address="16347 Heathrow Dr, Tampa, FL 33647")
    """
    # If lat/lon provided, use them
    if lat is not None and lon is not None:
        logger.info(f"Using explicit coordinates: ({lat:.6f}, {lon:.6f})")
        return {"lat": lat, "lon": lon}

    # Otherwise geocode the address
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
