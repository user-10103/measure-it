"""Tests for the settled pitch policy (src/roofs/pitch_policy.py)."""

import math

from src.roofs.pitch_policy import (
    DEFAULT_PITCH_X12, classify_pitch, is_flat, roof_pitch_prior, slope_to_x12,
    bayesian_snap_pitch, enforce_symmetric_pitch,
)


def test_flat_by_slope():
    assert is_flat(slope_deg=2.0) is True
    assert is_flat(slope_deg=18.0) is False


def test_flat_by_area_ratio_beats_slope():
    # independent surface/plan ratio ~1.0 => flat, even if slope says otherwise
    assert is_flat(slope_deg=25.0, surface_area_m2=100.0, plan_area_m2=99.5) is True
    # a 4:12 facet (ratio ~1.054) is pitched
    assert is_flat(slope_deg=25.0, surface_area_m2=105.4, plan_area_m2=100.0) is False


def test_pitched_facet_takes_prior_not_noisy_slope():
    # slope says ~7:12 but policy ignores per-facet magnitude -> roof prior
    ann = classify_pitch(slope_deg=30.0, roof_pitch_x12=DEFAULT_PITCH_X12)
    assert ann.is_flat is False
    assert ann.pitch_string == "4:12"
    assert ann.source == "prior"


def test_steep_outlier_flagged_for_review():
    ann = classify_pitch(slope_deg=40.0, roof_pitch_x12=DEFAULT_PITCH_X12)
    assert ann.needs_review is True
    ann2 = classify_pitch(slope_deg=20.0, roof_pitch_x12=DEFAULT_PITCH_X12)
    assert ann2.needs_review is False


def test_roof_prior_defaults_when_facets_disagree():
    # scattered (noisy) facet slopes -> fall back to the 4:12 default
    assert roof_pitch_prior([6.0, 25.0, 40.0, 12.0]) == DEFAULT_PITCH_X12


def test_roof_prior_trusts_consistent_facets():
    # facets agree tightly around ~27 deg (~6:12) -> trust their median
    prior = roof_pitch_prior([26.0, 27.0, 27.5, 26.5])
    assert prior == slope_to_x12(27.0) == 6


def test_flat_facets_ignored_in_prior():
    # flats (<5 deg) excluded; remaining pitched facets agree at ~18 deg (4:12)
    assert roof_pitch_prior([1.0, 0.0, 18.0, 18.4]) == 4


# ---------------------------------------------------------------------------
# Bayesian MAP pitch snapping (Fix 2)
# ---------------------------------------------------------------------------

def test_bayesian_snap_4_12_is_dominant():
    # 4:12 is at ~18.4 deg; tight sigma → snaps to 4:12
    assert bayesian_snap_pitch(18.4, sigma_deg=1.5) == 4


def test_bayesian_snap_prior_breaks_geometric_tie():
    # 4:12 (14.0 deg) and 5:12 (22.6 deg) are ~equidistant from 18.3 deg
    # but the FL prior heavily favours 4:12 so MAP picks 4:12
    assert bayesian_snap_pitch(18.3, sigma_deg=4.0) == 4


def test_bayesian_snap_clear_6_12():
    # 6:12 is at ~26.6 deg; with tight sigma it wins over 4:12
    assert bayesian_snap_pitch(26.6, sigma_deg=1.5) == 6


def test_bayesian_snap_very_wide_sigma_falls_back_to_prior():
    # When sigma is huge, likelihood is nearly flat → prior dominates → 4:12
    assert bayesian_snap_pitch(30.0, sigma_deg=20.0) == 4


def test_bayesian_snap_flat_slope_gives_low_pitch():
    # A slope near 0 should snap to pitch 2 (lowest in the prior), not flat
    result = bayesian_snap_pitch(5.0, sigma_deg=2.5)
    assert result <= 3


def test_roof_prior_uses_bayesian_snap():
    # Consistent facets at ~26.6 deg (6:12). bayesian_snap should return 6.
    prior = roof_pitch_prior([26.0, 26.5, 27.0, 26.8], sigma_deg=1.5)
    assert prior == 6


# ---------------------------------------------------------------------------
# Symmetric pitch enforcement (Fix 3)
# ---------------------------------------------------------------------------

class _MockFacet:
    """Minimal facet-like object for testing enforce_symmetric_pitch."""
    def __init__(self, slope_deg, aspect_bin):
        self.slope_deg = slope_deg
        self.aspect_bin = aspect_bin


def test_enforce_symmetric_corrects_noisy_pair():
    # N and S facets on a hip roof must share pitch. If they diverge by >8 deg
    # enforce_symmetric_pitch should pull both to their mean.
    n = _MockFacet(slope_deg=10.0, aspect_bin="N")
    s = _MockFacet(slope_deg=26.0, aspect_bin="S")
    enforce_symmetric_pitch([n, s])
    assert n.slope_deg == s.slope_deg == 18.0


def test_enforce_symmetric_leaves_consistent_pair_untouched():
    # N and S already agree within CONSISTENT_SPREAD_DEG — no correction.
    n = _MockFacet(slope_deg=18.0, aspect_bin="N")
    s = _MockFacet(slope_deg=20.0, aspect_bin="S")
    enforce_symmetric_pitch([n, s])
    assert n.slope_deg == 18.0
    assert s.slope_deg == 20.0


def test_enforce_symmetric_skips_flat_facets():
    # Flat facets (slope < FLAT_SLOPE_DEG) must not be modified.
    n = _MockFacet(slope_deg=2.0, aspect_bin="N")
    s = _MockFacet(slope_deg=24.0, aspect_bin="S")
    enforce_symmetric_pitch([n, s])
    assert n.slope_deg == 2.0   # flat — untouched


def test_enforce_symmetric_handles_diagonal_pairs():
    # NE/SW is also a valid symmetric pair on a hip roof.
    ne = _MockFacet(slope_deg=8.0, aspect_bin="NE")
    sw = _MockFacet(slope_deg=28.0, aspect_bin="SW")
    enforce_symmetric_pitch([ne, sw])
    assert ne.slope_deg == sw.slope_deg == 18.0


def test_enforce_symmetric_unmatched_aspect_unchanged():
    # A facet whose opposite aspect isn't present should be unchanged.
    n = _MockFacet(slope_deg=15.0, aspect_bin="N")
    e = _MockFacet(slope_deg=30.0, aspect_bin="E")  # no W facet
    enforce_symmetric_pitch([n, e])
    assert e.slope_deg == 30.0
