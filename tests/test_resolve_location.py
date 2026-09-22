"""resolve_location: coords parsing, ZIP-variant retry, provider fallback."""
import pytest

from src.utils import resolve_location as rl
from src.utils.resolve_location import (
    LocationError, address_variants, parse_coords, resolve_location,
)


# ── coordinate parsing ───────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "28.0303, -80.69809",
    "28.0303,-80.69809",
    "28.0303 -80.69809",
    "(28.0303, -80.69809)",
    "  28.0303 ; -80.69809 ",
])
def test_parse_decimal_pairs(text):
    c = parse_coords(text)
    assert c and abs(c["lat"] - 28.0303) < 1e-6 and abs(c["lon"] + 80.69809) < 1e-6


@pytest.mark.parametrize("url", [
    "https://www.google.com/maps/place/Melbourne/@28.0303,-80.69809,19z",
    "https://maps.google.com/?q=28.0303,-80.69809",
    "https://www.google.com/maps?ll=28.0303,-80.69809&z=18",
])
def test_parse_maps_links(url):
    c = parse_coords(url)
    assert c and abs(c["lat"] - 28.0303) < 1e-6


@pytest.mark.parametrize("text", [
    "909 Spring Island Way, Melbourne, FL",   # address, not coords
    "99, 200",                                 # out of range
    "0.0, 0.0",                                # null island
    "",
])
def test_parse_rejects_non_coords(text):
    assert parse_coords(text) is None


# ── the ZIP lesson from Colab ────────────────────────────────────────────────
def test_variants_strip_zip():
    v = address_variants("909 Spring Island Way, Melbourne, FL 32940")
    assert v[0] == "909 Spring Island Way, Melbourne, FL 32940"
    assert "909 Spring Island Way, Melbourne, FL" in v      # ZIP-stripped retry


def test_variants_zip4_and_usa():
    v = address_variants("1 Main St, Tampa, FL 33647-1234")
    assert "1 Main St, Tampa, FL" in v
    assert v[-1].endswith(", USA")


# ── provider chain (mocked — no network) ─────────────────────────────────────
@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr(rl, "_cache_get", lambda a: None)
    monkeypatch.setattr(rl, "_cache_put", lambda a, c: None)


def test_coords_bypass_geocoders(no_cache, monkeypatch):
    def boom(_): raise AssertionError("geocoder must not be called for coords")
    monkeypatch.setattr(rl, "_PROVIDERS", [("boom", boom)])
    out = resolve_location("28.0303, -80.69809")
    assert out["source"] == "coordinates"


def test_zip_failure_recovers_via_variant(no_cache, monkeypatch):
    """The exact Colab bug: geocoder misses WITH the ZIP, hits WITHOUT it."""
    def zip_hater(addr):
        return {"lat": 28.0303, "lon": -80.69809} if "32940" not in addr else None
    monkeypatch.setattr(rl, "_PROVIDERS", [("mock", zip_hater)])
    out = resolve_location("909 Spring Island Way, Melbourne, FL 32940")
    assert out["source"] == "mock" and abs(out["lat"] - 28.0303) < 1e-6


def test_fallback_order_first_hit_wins(no_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(rl, "_PROVIDERS", [
        ("first", lambda a: calls.append("first") or None),
        ("second", lambda a: calls.append("second") or {"lat": 25.0, "lon": -81.0}),
        ("third", lambda a: calls.append("third") or {"lat": 1.0, "lon": 1.0}),
    ])
    out = resolve_location("Hemingway House, Key West")
    assert out["source"] == "second"
    assert "third" not in calls


def test_total_miss_is_friendly(no_cache, monkeypatch):
    monkeypatch.setattr(rl, "_PROVIDERS", [("mock", lambda a: None)])
    with pytest.raises(LocationError) as e:
        resolve_location("123 Nowhere Blvd, Atlantis")
    assert "Google Maps" in str(e.value)       # actionable guidance, not a trace


def test_empty_input(no_cache):
    with pytest.raises(LocationError):
        resolve_location("   ")


def test_amazon_skipped_without_config(no_cache, monkeypatch):
    monkeypatch.delenv("AWS_LOCATION_PLACE_INDEX", raising=False)
    assert rl._geocode_amazon("909 Spring Island Way") is None
