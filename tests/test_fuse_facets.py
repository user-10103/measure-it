"""fuse_geometric_facets: imagery facet SHAPES + LiDAR points -> facets w/ pitch."""
import numpy as np
from shapely.geometry import Polygon, box
from src.roofs.segment import Facet
from src.roofs.fuse_facets import fuse_geometric_facets


def _hip_points(W=40, H=40, s=0.3, n=70, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, W, n * n); ys = rng.uniform(0, H, n * n)
    z = s * np.minimum.reduce([xs, W - xs, ys, H - ys])
    a = np.zeros(len(xs), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    a["x"], a["y"], a["z"] = xs, ys, z + 20
    return a


def test_fuse_assigns_pitch_and_keeps_shapes():
    C = (20, 20)
    # 4 imagery facet polygons (hip wedges seen from above)
    tris = [Polygon([(0, 0), (40, 0), C]), Polygon([(40, 0), (40, 40), C]),
            Polygon([(40, 40), (0, 40), C]), Polygon([(0, 40), (0, 0), C])]
    image_facets = [Facet(facet_id=i + 1, polygon=p) for i, p in enumerate(tris)]
    footprint = box(0, 0, 40, 40)

    out = fuse_geometric_facets(image_facets, _hip_points(), footprint=footprint,
                                regularize=True, min_points=10)
    assert len(out) == 4                              # shapes preserved (no clustering)
    npitched = sum(1 for f in out if f.slope_deg is not None)
    assert npitched >= 3                              # LiDAR supplied pitch per facet
    for f in out:
        assert f.polygon is not None and f.polygon.is_valid


def test_empty():
    assert fuse_geometric_facets([], _hip_points()) == []
