"""Choose the imagery source for a location: GIS orthophoto first, NAIP last.

The facet model is fine-tuned on county GIS orthophotos (3-6 inch), but serving
called ``report_service.fetch_chip`` (NAIP, 30-100 cm) unconditionally — a
train/serve resolution gap, and not a decision anyone made: ``fetch_chip_gis``
was written and tested but wired into nothing.

This module picks the highest-resolution source actually available for a point
and records WHICH ONE was used, so a report never silently hides the fact that
it was measured off imagery 4x coarser than the model expects.

Coverage, stated plainly: ``county_imagery.COUNTY_ENDPOINTS`` currently holds
three Florida counties, with a Florida-only statewide fallback. Outside Florida
this resolver has nothing to offer and returns NAIP — so wiring GIS closes the
gap in Florida and leaves it open everywhere else until the endpoint registry
grows. That is a data-collection task, not an engineering one.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_CENSUS_COORD = ("https://geocoding.geo.census.gov/geocoder/geographies/"
                 "coordinates")
_UA = {"User-Agent": "Mozilla/5.0 (measure-it imagery-select)"}

# An endpoint flagged unreachable is still worth ONE fast attempt: the flag was
# measured from a single datacenter IP. But a reachable county answers quickly
# (Sarasota: 1.1 s), so a long wait is evidence of a block, not of slowness.
# Six seconds distinguishes them and bounds the cost.
UNVERIFIED_TIMEOUT_S = 6
VERIFIED_TIMEOUT_S = 60

# Endpoints that failed in THIS process. A 20-address sweep must not pay the
# same dead endpoint twenty times; the first failure is the measurement.
_DEAD_ENDPOINTS: set = set()

_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "District Of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR",
}


def state_county_for(lat: float, lon: float,
                     timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """lat/lon -> (2-letter state, county BASENAME) from the US Census geocoder.

    Both come back in ONE request because both are needed and neither is worth
    a second round trip. The state matters as much as the county: the serving
    entry point takes `state` for the NAIP archive, and every caller was
    defaulting it to "FL" — so the pipeline had literally never been asked for
    imagery outside one state until the national probe forced the question.

    Returns (None, None) rather than raising: an unknown location degrades the
    imagery choice, and imagery selection must never be what fails a report.
    """
    q = urllib.parse.urlencode({
        "x": lon, "y": lat, "benchmark": "Public_AR_Current",
        "vintage": "Current_Current", "layers": "Counties,States",
        "format": "json"})
    try:
        req = urllib.request.Request(f"{_CENSUS_COORD}?{q}", headers=_UA)
        geo = json.load(urllib.request.urlopen(req, timeout=timeout))["result"]["geographies"]
    except Exception as e:  # noqa: BLE001 - advisory lookup
        logger.info("census lookup failed for (%.5f, %.5f): %s", lat, lon, e)
        return None, None
    counties = geo.get("Counties") or []
    states = geo.get("States") or []
    county = counties[0].get("BASENAME") if counties else None
    # STUSAB is the USPS abbreviation; older vintages only carry the full name.
    state = None
    if states:
        state = states[0].get("STUSAB") or _STATE_ABBR.get(
            (states[0].get("BASENAME") or "").title())
    return state, county


def county_for(lat: float, lon: float, timeout: int = 30) -> Optional[str]:
    """Back-compat wrapper: county only."""
    return state_county_for(lat, lon, timeout)[1]


def _usable(ep: dict) -> bool:
    """False for an endpoint that cannot answer without a token we do not have.

    FCDOP was listed as public and returns 499 Token Required to everything, so
    every Florida address outside the three registered counties spent a request
    and a timeout on it before degrading to NAIP. Trying an endpoint that CAN
    answer is worth a timeout; trying one that provably cannot is not.
    """
    import os

    if not ep.get("requires_token"):
        return True
    return bool(os.getenv(ep.get("token_env") or ""))


def candidate_sources(lat: float, lon: float, state: str,
                      county: Optional[str] = None) -> List[Tuple[str, dict]]:
    """Imagery sources to try, best resolution first.

    ``reachable`` is a RANKING hint, not a veto: it was measured from one
    datacenter IP, and an endpoint that blocks that IP may well answer from
    the machine actually running this. So an unreachable-flagged county is
    still attempted, and the tier says so ("county-3in-unverified").

    That rationale holds. What did NOT hold is the sentence this docstring used
    to carry — "the cost of being wrong is one timeout". Measured: a blocked
    county hangs for the full 45 s while Sarasota answers in 1.1 s, and a sweep
    pays that on EVERY address in that county before degrading to NAIP. The
    attempt is right; its cost was unbounded. See UNVERIFIED_TIMEOUT_S and the
    per-process dead-endpoint memo in fetch_chip_best.
    """
    from src.ingestion.county_imagery import COUNTY_ENDPOINTS, FCDOP_FALLBACK

    ranked: List[Tuple[str, dict]] = []
    ep = COUNTY_ENDPOINTS.get(county) if county else None
    if ep:
        tier = "county-3in" if ep.get("reachable", True) else "county-3in-unverified"
        ranked.append((tier, ep))
    if str(state).upper() == "FL" and _usable(FCDOP_FALLBACK):
        ranked.append(("fl-statewide", FCDOP_FALLBACK))
    ranked.append(("naip", {}))
    return ranked


def fetch_chip_best(lat: float, lon: float, state: str, out_dir,
                    chip_buffer_m: Optional[float] = None,
                    county: Optional[str] = None):
    """``fetch_chip``-compatible 5-tuple from the best available source.

    Tries each candidate in turn and falls through on failure, so a county
    server that is down degrades to statewide, then to NAIP, instead of failing
    the report. ``meta`` gains ``imagery_source``/``imagery_gsd_m``/
    ``imagery_year`` so the sweep can bucket results by what was actually used.
    """
    from src.ingestion.gis_chip import fetch_chip_gis
    from src.serve.report_service import fetch_chip as fetch_chip_naip

    if county is None:
        resolved_state, county = state_county_for(lat, lon)
        state = state or resolved_state
    attempts = []
    for tier, ep in candidate_sources(lat, lon, state, county):
        try:
            if tier == "naip":
                out = fetch_chip_naip(lat, lon, state, out_dir,
                                      chip_buffer_m=chip_buffer_m)
                gsd, year = None, None
            else:
                url = ep.get("url")
                if url in _DEAD_ENDPOINTS:
                    attempts.append(f"{tier}: skipped — failed earlier in this "
                                    f"process")
                    continue
                # Bound the wait on an endpoint we already believe is blocked.
                tmo = (UNVERIFIED_TIMEOUT_S if tier.endswith("-unverified")
                       else VERIFIED_TIMEOUT_S)
                out = fetch_chip_gis(lat, lon, state, out_dir,
                                     chip_buffer_m=chip_buffer_m, endpoint=ep,
                                     timeout=tmo)
                gsd, year = ep.get("gsd_m"), ep.get("year")
        except Exception as e:  # noqa: BLE001 - try the next source
            if tier != "naip" and ep.get("url"):
                _DEAD_ENDPOINTS.add(ep["url"])
            attempts.append(f"{tier}: {type(e).__name__}: {e}")
            logger.info("imagery %s unavailable at (%.5f, %.5f): %s",
                        tier, lat, lon, e)
            continue
        chip, transform, png, anchor, meta = out
        meta = dict(meta)
        meta["imagery_source"] = tier
        if gsd is None:
            # NAIP's meta carries no gsd_m, so the warning that exists to say
            # WHAT resolution we fell back to printed "(None m/px)". The pixel
            # size is right there on the affine - transform.a is the x scale.
            gsd = meta.get("gsd_m")
            if gsd is None and transform is not None:
                try:
                    gsd = abs(float(transform.a))
                except Exception:  # noqa: BLE001 - advisory
                    gsd = None
        meta["imagery_gsd_m"] = gsd
        meta["imagery_year"] = year
        meta["imagery_county"] = county
        meta["imagery_attempts"] = attempts
        if attempts:
            # A DEGRADED source is a warning, not chatter: every better option
            # failed and the report's area and pitch accuracy are now worse than
            # they had to be. INFO was the wrong level -- demo_lib.live_report
            # mutes every logger to ERROR, so the one message that explains a
            # 30 cm chip where a 15 cm one was available was being emitted into
            # silence and then discarded with the stdout buffer.
            logger.warning(
                "imagery DEGRADED to %s (%s m/px) at (%.5f, %.5f) county=%s "
                "- declined: %s", tier, meta["imagery_gsd_m"], lat, lon,
                county, "; ".join(attempts))
        else:
            logger.info("imagery: %s (%s m/px, %s) for (%.5f, %.5f) county=%s",
                        tier, meta["imagery_gsd_m"], year, lat, lon, county)
        return chip, transform, png, anchor, meta
    raise RuntimeError(
        f"No imagery source succeeded at ({lat}, {lon}): " + "; ".join(attempts))
