"""GIS-or-nothing: NAIP must not be reachable when the caller forbade it.

NAIP is 0.6 m/px 4-band; the facet model is fine-tuned on 7-15 cm three-band
orthophotos. A NAIP-sourced report is out-of-domain inference that produces the
same six pages with the same confident numbers, so the fallback has to be
absent, not merely deprecated.
"""
import pytest

from src.ingestion.imagery_select import (
    NoGisImagery, candidate_sources,
)

TAMPA = (27.9506, -82.4572)          # Hillsborough — in the registry
REMOTE = (30.4, -83.2)               # Madison — not in the registry


def test_naip_is_offered_by_default():
    tiers = [t for t, _ in candidate_sources(*TAMPA, "FL", county="Hillsborough")]
    assert tiers[-1] == "naip"
    assert any(t.startswith("county") for t in tiers)


def test_require_gis_removes_naip_entirely():
    tiers = [t for t, _ in candidate_sources(*TAMPA, "FL", county="Hillsborough",
                                             require_gis=True)]
    assert "naip" not in tiers
    assert tiers, "a registered county must still offer its GIS endpoint"


def test_uncovered_county_raises_instead_of_degrading_silently():
    """The 62-of-67 case. Without require_gis this returns NAIP and says little."""
    tiers = [t for t, _ in candidate_sources(*REMOTE, "FL", county="Madison")]
    assert tiers == ["naip"], "Madison has no GIS endpoint today"

    with pytest.raises(NoGisImagery, match="no GIS imagery source"):
        candidate_sources(*REMOTE, "FL", county="Madison", require_gis=True)


def test_error_names_what_is_missing_not_just_that_it_failed():
    with pytest.raises(NoGisImagery) as e:
        candidate_sources(*REMOTE, "FL", county="Madison", require_gis=True)
    msg = str(e.value)
    assert "COUNTY_ENDPOINTS holds" in msg     # how many counties are covered
    assert "Madison" in msg


def test_out_of_state_with_require_gis_raises():
    with pytest.raises(NoGisImagery):
        candidate_sources(39.7, -104.9, "CO", county="Denver", require_gis=True)
