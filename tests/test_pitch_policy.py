"""Tests for the settled pitch policy (src/roofs/pitch_policy.py)."""

from src.roofs.pitch_policy import (
    DEFAULT_PITCH_X12, classify_pitch, is_flat, roof_pitch_prior, slope_to_x12,
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
