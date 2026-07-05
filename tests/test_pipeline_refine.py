"""Pipeline refinements, anchored to the Holland-vs-Roofr criteria:
coplanar merge (evidence-based, area-conserving), flashing relabel at
pitched<->flat junctions (valleys -> ~0 on Holland), true 3D edge lengths
(hips 122.5' plan -> ~127' at 3:12)."""
import math

import numpy as np
from shapely.geometry import box

from src.roofs.fuse_sam_lidar import annotate_facets_with_lidar, merge_coplanar_facets
from src.roofs.geom_edges import (
    apply_3d_edge_lengths,
    relabel_flat_junction_flashing,
)
from src.roofs.segment import Facet


def _grid(poly, z_fn, step=0.5):
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = np.meshgrid(np.arange(minx + 0.25, maxx, step),
                         np.arange(miny + 0.25, maxy, step))
    x, y = xs.ravel(), ys.ravel()
    return np.column_stack([x, y, z_fn(x, y)])


# ── coplanar merge ───────────────────────────────────────────────────────────
def test_bands_on_one_plane_merge():
    # the 909 failure mode: one slope split into two parallel bands
    f1 = Facet(facet_id=1, polygon=box(0, 0, 20, 5))
    f2 = Facet(facet_id=2, polygon=box(0, 5, 20, 10))
    pts = np.vstack([_grid(f.polygon, lambda x, y: 0.5 * y) for f in (f1, f2)])
    ann = annotate_facets_with_lidar([f1, f2], pts)
    merged, changed = merge_coplanar_facets([f1, f2], pts, ann)
    assert changed and len(merged) == 1
    # area conserved
    assert abs(merged[0].polygon.area - 200.0) < 2.0


def test_gable_slopes_never_merge():
    # two faces at the same |slope| but opposite aspects = a real ridge
    f1 = Facet(facet_id=1, polygon=box(0, 0, 20, 5))
    f2 = Facet(facet_id=2, polygon=box(0, 5, 20, 10))
    pts = np.vstack([_grid(f1.polygon, lambda x, y: 0.5 * y),
                     _grid(f2.polygon, lambda x, y: 0.5 * (10 - y))])
    ann = annotate_facets_with_lidar([f1, f2], pts)
    merged, changed = merge_coplanar_facets([f1, f2], pts, ann)
    assert not changed and len(merged) == 2


def test_flat_never_merges_with_pitched():
    # Holland criterion: the flat section must stay separate from the 3:12 wing
    f1 = Facet(facet_id=1, polygon=box(0, 0, 20, 5))       # pitched
    f2 = Facet(facet_id=2, polygon=box(0, 5, 20, 10))      # flat
    pts = np.vstack([_grid(f1.polygon, lambda x, y: 0.25 * y),
                     _grid(f2.polygon, lambda x, y: np.full_like(x, 1.25))])
    ann = annotate_facets_with_lidar([f1, f2], pts)
    merged, changed = merge_coplanar_facets([f1, f2], pts, ann)
    assert not changed


def test_same_slope_different_height_never_merges():
    # a step: same gradient, offset by 2 m — the combined-fit check must reject
    f1 = Facet(facet_id=1, polygon=box(0, 0, 20, 5))
    f2 = Facet(facet_id=2, polygon=box(0, 5, 20, 10))
    pts = np.vstack([_grid(f1.polygon, lambda x, y: 0.3 * x),
                     _grid(f2.polygon, lambda x, y: 0.3 * x + 2.0)])
    ann = annotate_facets_with_lidar([f1, f2], pts)
    merged, changed = merge_coplanar_facets([f1, f2], pts, ann)
    assert not changed and len(merged) == 2


# ── flashing relabel ─────────────────────────────────────────────────────────
def test_pitched_flat_junction_becomes_flashing():
    f1 = Facet(facet_id=1, polygon=box(0, 0, 20, 5))       # pitched
    f2 = Facet(facet_id=2, polygon=box(0, 5, 20, 10))      # flat
    ann = {1: {"is_flat": False}, 2: {"is_flat": True}}
    edges = [{"edge_type": "valley", "length_m": 20.0,
              "geometry_xy": [[0, 5], [20, 5]]}]           # their shared seam
    relabel_flat_junction_flashing(edges, [f1, f2], ann)
    assert edges[0]["edge_type"] == "wall_flashing"


def test_pitched_pitched_valley_stays_valley():
    f1 = Facet(facet_id=1, polygon=box(0, 0, 20, 5))
    f2 = Facet(facet_id=2, polygon=box(0, 5, 20, 10))
    ann = {1: {"is_flat": False}, 2: {"is_flat": False}}
    edges = [{"edge_type": "valley", "length_m": 20.0,
              "geometry_xy": [[0, 5], [20, 5]]}]
    relabel_flat_junction_flashing(edges, [f1, f2], ann)
    assert edges[0]["edge_type"] == "valley"


# ── aspect-based internal classification ────────────────────────────────────
def _ann(fid_aspect_flat):
    return {fid: {"aspect_deg": a, "is_flat": fl}
            for fid, (a, fl) in fid_aspect_flat.items()}


def test_gable_seam_is_ridge():
    from src.roofs.geom_edges import classify_internal_edges
    f1 = Facet(facet_id=1, polygon=box(0, 0, 40, 10))    # slopes S (180)
    f2 = Facet(facet_id=2, polygon=box(0, 10, 40, 20))   # slopes N (0)
    edges = [{"edge_type": "hip", "length_m": 40.0,       # wrong on purpose
              "geometry_xy": [[0, 10], [40, 10]]}]
    out = classify_internal_edges(edges, [f1, f2], _ann({1: (180, False), 2: (0, False)}))
    assert out[0]["edge_type"] == "ridge"                 # diverging + level


def test_butterfly_seam_is_valley():
    from src.roofs.geom_edges import classify_internal_edges
    f1 = Facet(facet_id=1, polygon=box(0, 0, 40, 10))    # slopes N (0): INTO seam
    f2 = Facet(facet_id=2, polygon=box(0, 10, 40, 20))   # slopes S (180): INTO seam
    edges = [{"edge_type": "ridge", "length_m": 40.0,
              "geometry_xy": [[0, 10], [40, 10]]}]
    out = classify_internal_edges(edges, [f1, f2], _ann({1: (0, False), 2: (180, False)}))
    assert out[0]["edge_type"] == "valley"                # converging drainage


def test_corner_to_corner_seam_splits_hip_ridge_hip():
    # the Holland inflation case: one long seam between two big opposing
    # facets, running corner -> ridge -> corner. Diagonal runs must be HIP
    # (sloped), only the level middle is RIDGE.
    from src.roofs.geom_edges import classify_internal_edges
    from shapely.geometry import Polygon
    front = Facet(facet_id=1, polygon=Polygon([(0, 0), (40, 0), (30, 10), (10, 10)]))
    back = Facet(facet_id=2, polygon=Polygon([(0, 0), (10, 10), (30, 10), (40, 0),
                                              (40, 20), (0, 20)]))
    edges = [{"edge_type": "hip", "length_m": 48.28,
              "geometry_xy": [[0, 0], [10, 10], [30, 10], [40, 0]]}]
    out = classify_internal_edges(edges, [front, back],
                                  _ann({1: (180, False), 2: (0, False)}))
    kinds = {}
    for e in out:
        kinds[e["edge_type"]] = kinds.get(e["edge_type"], 0.0) + e["length_m"]
    assert abs(kinds.get("ridge", 0) - 20.0) < 0.5        # the level middle
    assert abs(kinds.get("hip", 0) - 2 * 14.142) < 0.5    # both diagonals


def test_flat_junction_segment_is_flashing():
    from src.roofs.geom_edges import classify_internal_edges
    f1 = Facet(facet_id=1, polygon=box(0, 0, 20, 10))     # pitched
    f2 = Facet(facet_id=2, polygon=box(0, 10, 20, 20))    # flat
    ann = {1: {"aspect_deg": 180.0, "is_flat": False}, 2: {"is_flat": True}}
    edges = [{"edge_type": "valley", "length_m": 20.0,
              "geometry_xy": [[0, 10], [20, 10]]}]
    out = classify_internal_edges(edges, [f1, f2], ann)
    assert out[0]["edge_type"] == "wall_flashing"


def test_two_story_is_relative_not_absolute():
    # a tall SINGLE-level building (all eaves ~5.5 m) must NOT be two-story;
    # a building with a roof a level above its lowest eave MUST be.
    tallflat1 = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    tallflat2 = Facet(facet_id=2, polygon=box(10, 0, 20, 10))
    pts = np.vstack([_grid(tallflat1.polygon, lambda x, y: np.full_like(x, 5.6)),
                     _grid(tallflat2.polygon, lambda x, y: np.full_like(x, 5.5))])
    ann = annotate_facets_with_lidar([tallflat1, tallflat2], pts, ground_z=0.0)
    assert not ann[1]["is_two_story"] and not ann[2]["is_two_story"]   # Holland case

    lower = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    upper = Facet(facet_id=2, polygon=box(20, 0, 30, 10))
    pts2 = np.vstack([_grid(lower.polygon, lambda x, y: 3.0 + 0.1 * y),
                      _grid(upper.polygon, lambda x, y: 5.8 + 0.1 * y)])
    ann2 = annotate_facets_with_lidar([lower, upper], pts2)   # even w/o ground
    assert not ann2[1]["is_two_story"] and ann2[2]["is_two_story"]


# ── 3D lengths ───────────────────────────────────────────────────────────────
def test_3d_lengths_slope_aware():
    # facet slopes along +y with gradient 0.5 (6:12)
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 10))
    grads = {1: (0.0, 0.5)}
    edges = [
        {"edge_type": "rake", "length_m": 10.0,             # runs WITH the slope
         "geometry_xy": [[0, 0], [0, 10]]},
        {"edge_type": "eave", "length_m": 20.0,             # level, perpendicular
         "geometry_xy": [[0, 0], [20, 0]]},
    ]
    apply_3d_edge_lengths(edges, [f], grads)
    assert abs(edges[0]["length_m"] - 10.0 * math.sqrt(1.25)) < 1e-6   # +11.8%
    assert abs(edges[1]["length_m"] - 20.0) < 1e-6                     # unchanged


def test_3d_lengths_no_grad_no_change():
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 10))
    edges = [{"edge_type": "hip", "length_m": 7.0,
              "geometry_xy": [[0, 0], [5, 5]]}]
    apply_3d_edge_lengths(edges, [f], {})                   # no LiDAR planes
    assert edges[0]["length_m"] == 7.0
