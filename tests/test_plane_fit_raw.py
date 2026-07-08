"""Tests for the raw-EPT plane fitting and ridge-eave pitch functions."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from src.roofs.plane_fit import (
    fit_plane_to_raw_points,
    compute_pitch_from_ridge_eave,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_roof_points(
    slope_deg: float,
    aspect_deg: float = 180.0,
    n: int = 200,
    noise_m: float = 0.05,
    ground_z: float = 10.0,
    height_above_ground: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Build a synthetic roof point cloud over a 10×10 m facet in the XY plane.

    The facet is centred at (0, 0) with ridge at y=+5 and eave at y=-5.
    slope_deg is the true roof pitch; aspect_deg controls downslope direction
    (180 = south-facing, i.e. ridge at north).
    """
    rng = np.random.default_rng(seed)
    # Random XY within a 10×10 m square
    x = rng.uniform(-5.0, 5.0, n)
    y = rng.uniform(-5.0, 5.0, n)

    # True plane: slope in the aspect direction
    g = math.tan(math.radians(slope_deg))
    phi = math.radians(90.0 - aspect_deg)  # compass → math angle
    # gradient points uphill (opposite to downslope direction)
    a = g * math.cos(phi)
    b = g * math.sin(phi)
    z_base = ground_z + height_above_ground
    z = a * x + b * y + z_base + rng.normal(0, noise_m, n)

    dtype = np.dtype([("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    pts = np.zeros(n, dtype=dtype)
    pts["X"], pts["Y"], pts["Z"] = x, y, z
    return pts


# ---------------------------------------------------------------------------
# fit_plane_to_raw_points
# ---------------------------------------------------------------------------

class TestFitPlaneToRawPoints:

    def _facet_poly(self):
        return Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])

    def test_recovers_known_slope(self):
        pts = _make_roof_points(slope_deg=18.4, noise_m=0.03)
        plane = fit_plane_to_raw_points(
            pts,
            facet_polygon=self._facet_poly(),
            ground_z=10.0,
            dist_thresh=0.10,
            min_inlier_ratio=0.65,
        )
        assert plane is not None
        assert plane.success
        recovered = math.degrees(math.atan(plane.gradient_magnitude))
        assert abs(recovered - 18.4) < 2.0, f"slope error: {abs(recovered - 18.4):.2f}°"

    def test_returns_none_for_empty_input(self):
        dtype = np.dtype([("X", "f8"), ("Y", "f8"), ("Z", "f8")])
        empty = np.zeros(0, dtype=dtype)
        result = fit_plane_to_raw_points(empty, self._facet_poly(), ground_z=10.0)
        assert result is None

    def test_returns_none_when_all_below_ground(self):
        # Use a flat roof (slope=0) so all z ≈ ground_z + 0.5.
        # With min_height_above_ground=1.5 every point is stripped → None.
        pts = _make_roof_points(slope_deg=0.0, ground_z=10.0, height_above_ground=0.5)
        result = fit_plane_to_raw_points(
            pts, self._facet_poly(), ground_z=10.0, min_height_above_ground=1.5
        )
        assert result is None

    def test_accepts_lowercase_xyz_fields(self):
        """Raster-path arrays use lowercase x/y/z — function must handle both."""
        pts_upper = _make_roof_points(slope_deg=14.0, noise_m=0.04)
        dtype_lower = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8")])
        pts_lower = np.zeros(len(pts_upper), dtype=dtype_lower)
        pts_lower["x"] = pts_upper["X"]
        pts_lower["y"] = pts_upper["Y"]
        pts_lower["z"] = pts_upper["Z"]

        plane = fit_plane_to_raw_points(
            pts_lower, self._facet_poly(), ground_z=10.0
        )
        assert plane is not None
        assert plane.success

    def test_points_outside_polygon_excluded(self):
        """Points outside a tiny polygon should yield None (< 10 inliers)."""
        pts = _make_roof_points(slope_deg=18.4, n=200)
        tiny_poly = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
        result = fit_plane_to_raw_points(
            pts, tiny_poly, ground_z=10.0, min_height_above_ground=0.1
        )
        assert result is None

    def test_tighter_thresholds_than_raster_path(self):
        """The raw-EPT path should succeed at dist_thresh=0.10 m with low noise."""
        pts = _make_roof_points(slope_deg=22.6, noise_m=0.04, n=300)
        plane = fit_plane_to_raw_points(
            pts, self._facet_poly(), ground_z=10.0,
            dist_thresh=0.10, min_inlier_ratio=0.65,
        )
        assert plane is not None and plane.success
        assert plane.residual_median < 0.10


# ---------------------------------------------------------------------------
# compute_pitch_from_ridge_eave
# ---------------------------------------------------------------------------

def _make_ridge_eave_points(
    slope_deg: float,
    ground_z: float = 10.0,
    n_per_line: int = 60,
    noise_m: float = 0.05,
    seed: int = 7,
) -> tuple:
    """Return (raw_points, ridge_line, eave_line) for a simple gable.

    Ridge runs E-W at y=0, z=apex.  Eave runs E-W at y=5 m (south side), z=base.
    """
    rng = np.random.default_rng(seed)
    span = 5.0  # horizontal distance ridge → eave
    rise = span * math.tan(math.radians(slope_deg))
    apex_z = ground_z + 4.0 + rise
    base_z = ground_z + 4.0

    # Ridge points: scattered around y=0
    rx = rng.uniform(-5.0, 5.0, n_per_line)
    ry = rng.uniform(-0.5, 0.5, n_per_line)
    rz = apex_z + rng.normal(0, noise_m, n_per_line)

    # Eave points: scattered around y=5
    ex = rng.uniform(-5.0, 5.0, n_per_line)
    ey = rng.uniform(4.5, 5.5, n_per_line)
    ez = base_z + rng.normal(0, noise_m, n_per_line)

    dtype = np.dtype([("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    pts = np.zeros(n_per_line * 2, dtype=dtype)
    pts["X"] = np.concatenate([rx, ex])
    pts["Y"] = np.concatenate([ry, ey])
    pts["Z"] = np.concatenate([rz, ez])

    ridge = LineString([(-5, 0), (5, 0)])
    eave = LineString([(-5, 5), (5, 5)])
    return pts, ridge, eave


class TestComputePitchFromRidgeEave:

    def test_recovers_known_pitch(self):
        for slope_deg in (14.0, 18.4, 22.6, 26.6):
            pts, ridge, eave = _make_ridge_eave_points(slope_deg)
            result = compute_pitch_from_ridge_eave(pts, ridge, eave, sample_radius_m=1.5)
            assert result is not None, f"Got None for slope={slope_deg}"
            assert abs(result - slope_deg) < 2.5, (
                f"slope_deg={slope_deg}: expected ≈{slope_deg}, got {result:.2f}"
            )

    def test_returns_none_for_empty_points(self):
        dtype = np.dtype([("X", "f8"), ("Y", "f8"), ("Z", "f8")])
        pts = np.zeros(0, dtype=dtype)
        ridge = LineString([(-5, 0), (5, 0)])
        eave = LineString([(-5, 5), (5, 5)])
        assert compute_pitch_from_ridge_eave(pts, ridge, eave) is None

    def test_returns_none_when_ridge_below_eave(self):
        """Inverted geometry (eave above ridge) should return None."""
        pts, ridge, eave = _make_ridge_eave_points(18.4)
        # Swap ridge and eave to invert
        result = compute_pitch_from_ridge_eave(pts, eave, ridge, sample_radius_m=1.5)
        assert result is None

    def test_returns_none_when_too_few_points(self):
        """Only 2 points near lines — below the minimum of 3."""
        dtype = np.dtype([("X", "f8"), ("Y", "f8"), ("Z", "f8")])
        pts = np.zeros(2, dtype=dtype)
        pts["X"] = [0.0, 0.0]
        pts["Y"] = [0.0, 5.0]
        pts["Z"] = [15.0, 12.0]
        ridge = LineString([(-5, 0), (5, 0)])
        eave = LineString([(-5, 5), (5, 5)])
        assert compute_pitch_from_ridge_eave(pts, ridge, eave, sample_radius_m=0.2) is None
