"""Unit tests for src/roofs/facet_reconstruct.py (Phase 0/1).

Pure-geometry tests — no LiDAR, no model, no network. Runs on the base geo
stack (shapely). Verifies the design invariants that let the module drop into
pipeline.py:576-887 safely:

- facet count is preserved,
- the regularized set still tiles the footprint,
- edges come out straighter (fewer near-collinear kinks),
- total area does not drift materially.
"""
import math

import pytest
from shapely import affinity
from shapely.geometry import Polygon

from src.roofs.facet_reconstruct import (
    reconstruct_facets,
    regularize_facets,
    _principal_axis_angle,
)


def _jag(poly: Polygon, amp: float = 0.15, n: int = 3) -> Polygon:
    """Insert small zig-zag noise along each edge to simulate jagged
    LiDAR/segmentation boundaries."""
    coords = list(poly.exterior.coords)[:-1]
    out = []
    for k in range(len(coords)):
        ax, ay = coords[k]
        bx, by = coords[(k + 1) % len(coords)]
        out.append((ax, ay))
        for j in range(1, n):
            t = j / n
            # midpoint + perpendicular jitter
            mx, my = ax + (bx - ax) * t, ay + (by - ay) * t
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy) or 1.0
            px, py = -dy / L, dx / L
            wob = amp * (1 if j % 2 else -1)
            out.append((mx + px * wob, my + py * wob))
    return Polygon(out)


def _n_vertices(p: Polygon) -> int:
    return len(p.exterior.coords) - 1


def _max_edge_angle_dev(p: Polygon) -> float:
    """Largest deviation (deg) of any edge from the nearest 0/90 axis, in the
    polygon's own principal-axis frame. Small => cleanly rectilinear."""
    theta = math.radians(_principal_axis_angle(p))
    rot = affinity.rotate(p, -math.degrees(theta), origin="centroid")
    coords = list(rot.exterior.coords)
    worst = 0.0
    for a, b in zip(coords[:-1], coords[1:]):
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90.0
        dev = min(ang, 90.0 - ang)
        worst = max(worst, dev)
    return worst


@pytest.fixture
def footprint():
    # 20m x 10m rectangle, rotated 20 deg so tests exercise the axis logic.
    rect = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    return affinity.rotate(rect, 20, origin="centroid")


@pytest.fixture
def jagged_facets(footprint):
    # Two facets that split the (unrotated) rectangle down the middle, rotated
    # to match the footprint, then roughened.
    left = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    right = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
    left = affinity.rotate(left, 20, origin=(10, 5))
    right = affinity.rotate(right, 20, origin=(10, 5))
    # align rotation origin with footprint centroid rotation
    left = affinity.rotate(left, 0)
    return [(1, _jag(left)), (2, _jag(right))]


def test_count_preserved(footprint, jagged_facets):
    facets, boundaries, planes = reconstruct_facets(footprint, jagged_facets)
    assert len(facets) == len(jagged_facets)
    assert set(boundaries.keys()) == {fid for fid, _ in facets}


def test_all_valid_polygons(footprint, jagged_facets):
    facets, _, _ = reconstruct_facets(footprint, jagged_facets)
    for fid, poly in facets:
        assert isinstance(poly, Polygon)
        assert poly.is_valid and not poly.is_empty
        assert poly.area > 0.5


def test_edges_are_straighter(footprint, jagged_facets):
    facets, _, _ = reconstruct_facets(footprint, jagged_facets)
    for (fid_in, jag), (fid_out, reg) in zip(jagged_facets, facets):
        # Regularized facet should have far fewer vertices than the jagged input
        assert _n_vertices(reg) <= _n_vertices(jag)
        # ...and edges close to rectilinear in its own axis frame.
        assert _max_edge_angle_dev(reg) < 8.0


def test_tiling_preserved(footprint, jagged_facets):
    facets, _, _ = reconstruct_facets(footprint, jagged_facets)
    from shapely.ops import unary_union
    covered = unary_union([p for _, p in facets])
    # Regularized facets should cover ~the whole footprint with negligible gap.
    coverage = covered.area / footprint.area
    assert 0.95 <= coverage <= 1.02
    # And not overlap materially (sum of parts ~= union).
    parts = sum(p.area for _, p in facets)
    assert abs(parts - covered.area) / footprint.area < 0.05


def test_area_not_drifting(footprint, jagged_facets):
    facets, _, _ = reconstruct_facets(footprint, jagged_facets)
    total = sum(p.area for _, p in facets)
    assert abs(total - footprint.area) / footprint.area < 0.05


def test_plane_priority_and_alignment(footprint, jagged_facets):
    # Fake planes: facet 2 has more inliers, so it should win contested area
    # (claimed first). Planes must stay index-aligned to returned facets.
    class _Plane:
        def __init__(self, n):
            self.inlier_count = n

    planes = [_Plane(10), _Plane(500)]  # facet 2 dominant
    facets, boundaries, out_planes = reconstruct_facets(
        footprint, jagged_facets, planes_for_poly=planes)
    assert len(out_planes) == len(facets)
    # dominant facet (id 2) should retain the larger regularized area
    by_id = dict(facets)
    assert by_id[2].area >= by_id[1].area * 0.8  # not starved by trimming


def test_empty_input():
    facets, boundaries, planes = reconstruct_facets(None, [])
    assert facets == [] and boundaries == {} and planes is None


def test_no_outline_partitions_against_each_other(jagged_facets):
    # outline=None path must still return regularized, count-preserved facets.
    facets, _, _ = reconstruct_facets(None, jagged_facets)
    assert len(facets) == len(jagged_facets)
    for _, p in facets:
        assert p.is_valid and p.area > 0.5


def test_regularize_facets_direct(footprint):
    jag = _jag(footprint, amp=0.2, n=4)
    out = regularize_facets([jag], footprint)
    assert len(out) == 1
    assert _n_vertices(out[0]) < _n_vertices(jag)
    assert _max_edge_angle_dev(out[0]) < 8.0


# ── Phase 4: reconstruct_edges ────────────────────────────────────────────────

import numpy as np  # noqa: E402
from src.roofs.facet_reconstruct import reconstruct_edges  # noqa: E402
from src.roofs.edges import EdgeType  # noqa: E402
from src.roofs.plane_fit import PlaneModel  # noqa: E402


def _plane(a, b, c):
    """Synthetic plane z = a*x + b*y + c (LiDAR-style, real intercept)."""
    return PlaneModel(a=a, b=b, c=c, inlier_mask=np.ones(1, bool),
                      inlier_count=100, residual_median=0.01, success=True)


def _gable(ridge_x=10.0, right_x0=10.0):
    """A simple gable: 20x10 roof, ridge along y at x=ridge_x. Two planes slope
    DOWN away from the ridge (opposed gradients -> RIDGE). right_x0 lets a test
    introduce a small gap between the two facets."""
    left = Polygon([(0, 0), (ridge_x, 0), (ridge_x, 10), (0, 10)])
    right = Polygon([(right_x0, 0), (20, 0), (20, 10), (right_x0, 10)])
    facets = [(1, left), (2, right)]
    planes = [_plane(0.5, 0.0, 0.0),      # rises toward x=10 (high at ridge)
              _plane(-0.5, 0.0, 10.0)]    # rises toward x=10 from the right
    outline = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    return facets, planes, outline


def test_gable_yields_single_ridge():
    facets, planes, outline = _gable()
    edges = reconstruct_edges(facets, planes, outline=outline)
    ridges = [e for e in edges if e.edge_type == EdgeType.RIDGE]
    assert len(ridges) == 1, [e.edge_type for e in edges]
    assert abs(ridges[0].length_m - 10.0) < 1.0


def test_edge_count_collapses():
    # A clean gable should produce only a handful of edges (ridge + eaves/rakes),
    # nowhere near the ~60/roof jagged baseline.
    facets, planes, outline = _gable()
    edges = reconstruct_edges(facets, planes, outline=outline)
    assert 1 <= len(edges) <= 10


def test_shared_boundary_coincidence_snap():
    # Introduce a 0.02 m gap between the two facets. Without snapping the exact
    # boundary.intersection would find nothing -> ridge lost. reconstruct_edges
    # grid-snaps (0.05 m) so the ridge is still detected.
    facets, planes, outline = _gable(right_x0=10.02)
    edges = reconstruct_edges(facets, planes, outline=outline, grid_m=0.05)
    ridges = [e for e in edges if e.edge_type == EdgeType.RIDGE]
    assert len(ridges) == 1


def test_phase2_to_phase4_integration(footprint):
    # End-to-end: jagged facets -> reconstruct_facets (straighten) -> assign
    # planes -> reconstruct_edges. Must not crash and must stay clean.
    facets_in, planes, outline = _gable()
    jagged = [(fid, _jag(poly, amp=0.12, n=3)) for fid, poly in facets_in]
    clean, _, _ = reconstruct_facets(outline, jagged, planes_for_poly=planes)
    assert len(clean) == 2
    edges = reconstruct_edges(clean, planes, outline=outline)
    assert 1 <= len(edges) <= 12
    # the interior seam should still classify as a ridge (opposed planes)
    assert any(e.edge_type == EdgeType.RIDGE for e in edges)
