"""cgal_segment (RANSAC-fallback path, no cgal): outputs clean geometric polygons."""
import numpy as np
from shapely.geometry import box
from src.roofs.cgal_segment import segment_facets_cgal, _merge_coplanar, _polygonize


def _hip(W=40, H=40, s=0.3, n=80, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, W, n * n); ys = rng.uniform(0, H, n * n)
    z = s * np.minimum.reduce([xs, W - xs, ys, H - ys])
    a = np.zeros(len(xs), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    a["x"], a["y"], a["z"] = xs, ys, z + 20
    return a


def test_outputs_geometric_polygons_not_points():
    fp = box(0, 0, 40, 40)
    facets = segment_facets_cgal(_hip(), footprint=fp, min_points=60)
    assert 3 <= len(facets) <= 10
    for f in facets:
        # THE requirement: every facet is a real polygon, not a dot cluster
        assert f.polygon is not None and f.polygon.is_valid and not f.polygon.is_empty
        assert f.polygon.geom_type == "Polygon"
        assert f.polygon.area > 1.0
        assert f.points is not None            # points kept (for pitch), but polygon drawn


def test_polygonize_clips_to_footprint():
    pts = _hip()
    poly = _polygonize(pts, footprint=box(0, 0, 40, 40))
    assert poly.geom_type == "Polygon" and poly.area > 1.0


def test_empty():
    assert segment_facets_cgal(None) == []
