"""Tests for the monocular pitch estimator (fallback + math; no torch needed)."""
import math

import numpy as np

from src.roofs.pitch_mono import (
    MonocularPitchEstimator, _plane_slope_from_depth_patch, slope_to_pitch_string,
)


def test_lidar_fallback():
    est = MonocularPitchEstimator()
    r = est.estimate_facet_pitch(np.zeros((4, 4, 3), np.uint8), None, 0.3,
                                 lidar_slope_deg=22.0)
    assert r.source == "lidar_fallback"
    assert r.slope_deg == 22.0


def test_default_fallback():
    est = MonocularPitchEstimator(default_slope_deg=18.0)
    r = est.estimate_facet_pitch(np.zeros((4, 4, 3), np.uint8), None, 0.3)
    assert r.source == "default"
    assert r.slope_deg == 18.0


def test_ensure_model_returns_false_without_torch():
    est = MonocularPitchEstimator()
    # torch/transformers absent in this env -> graceful False, no raise
    assert est._ensure_model() is False


def test_plane_slope_from_depth_patch():
    # depth rises 0.5 m per ground-metre in the column direction -> slope=atan(0.5)
    gsd = 0.3
    cols = np.arange(20)
    # depth(col) = 0.5 * (col * gsd)  => per-metre gradient 0.5
    patch = np.zeros((20, 20))
    for c in cols:
        patch[:, c] = 0.5 * (c * gsd)
    mask = np.ones((20, 20), dtype=bool)
    slope = _plane_slope_from_depth_patch(patch, mask, gsd)
    assert math.isclose(slope, math.degrees(math.atan(0.5)), abs_tol=1.0)


def test_slope_to_pitch_string():
    assert slope_to_pitch_string(18.0) == "4:12"   # tan(18)*12 ~= 3.9
    assert slope_to_pitch_string(0.0) == "0:12"
