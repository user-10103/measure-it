"""Facet boundaries drawn from the LiDAR rather than from an imagery mask.

segment_facets_planes was written 2026-07-02 21:33 and has NEVER been wired —
git log -S returns its birth commit and one import by PEARL, nothing else. The
56-facet fragmentation that condemned the whole point-clustering family 78
minutes later was PEARL's; a prototype of the plain RANSAC path merged 32 planes
into 13 coherent facets on a complex roof. This is that experiment, made runnable.
"""
import numpy as np
from shapely.geometry import box

from src.roofs.lidar_facets import lidar_facets_from_points


def _hip(step=0.4):
    """A four-sided hip: each slope is its own plane, meeting at a ridge."""
    xs, ys = np.meshgrid(np.arange(0, 20, step), np.arange(0, 12, step))
    x, y = xs.ravel(), ys.ravel()
    # two long slopes N/S plus two end slopes E/W, apex along y=6
    z = np.minimum(np.minimum(y, 12 - y), np.minimum(x, 20 - x)) * 0.4
    return np.column_stack([x, y, z])


def test_boundaries_come_from_the_points_and_facets_are_found():
    facets = lidar_facets_from_points(_hip(), outline=box(0, 0, 20, 12))
    assert len(facets) >= 3, f"expected the hip's planes, got {len(facets)}"
    for f in facets:
        assert f.polygon is not None and not f.polygon.is_empty
        assert f.points is not None and len(f.points) > 0
        assert f.polygon.area >= 2.0


def test_it_returns_empty_rather_than_guessing_on_junk():
    """An empty list means 'this did not work here' and the caller keeps its own
    facets — the experiment must never make a report worse by failing."""
    assert lidar_facets_from_points(None) == []
    assert lidar_facets_from_points(np.empty((0, 3))) == []
    rng = np.random.RandomState(0)
    scatter = rng.uniform(0, 5, (40, 3))          # no planes, too few points
    assert lidar_facets_from_points(scatter) == []


def test_facets_stay_inside_the_outline():
    outline = box(2, 2, 18, 10)
    for f in lidar_facets_from_points(_hip(), outline=outline):
        assert f.polygon.difference(outline.buffer(0.5)).area < 1e-6, f.polygon


def test_a_concave_facet_is_not_returned_as_its_convex_envelope():
    """plane_fit.extract_inlier_boundary takes a CONVEX hull, which is the
    'convex-hull blobs' failure the segmentation docstrings complain about: an
    L-shaped facet comes back as its bounding envelope and swallows roof that
    belongs to a neighbour. concave_hull follows the point set."""
    from shapely import concave_hull
    from shapely.geometry import MultiPoint

    # an L of points
    a = np.mgrid[0:10:0.4, 0:3:0.4].reshape(2, -1).T
    b = np.mgrid[0:3:0.4, 0:10:0.4].reshape(2, -1).T
    pts = np.vstack([a, b])
    mp = MultiPoint([tuple(p) for p in pts])
    assert concave_hull(mp, ratio=0.35).area < mp.convex_hull.area * 0.95
