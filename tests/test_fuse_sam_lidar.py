"""Fusion is read-only annotation: known synthetic slopes -> exact pitch."""
import math

import numpy as np
import pytest
from shapely.geometry import box

from src.roofs.fuse_sam_lidar import (
    annotate_facets_with_lidar,
    fuse_into_report_input,
    split_multiplane_facets,
)
from src.roofs.segment import Facet


def _grid_points(poly, z_fn, step=0.5):
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = np.meshgrid(np.arange(minx + 0.25, maxx, step),
                         np.arange(miny + 0.25, maxy, step))
    x, y = xs.ravel(), ys.ravel()
    return np.column_stack([x, y, z_fn(x, y)])


def test_six_twelve_pitch_recovered_exactly():
    # z = 0.5*x  -> gradient 0.5 -> rise 6 per 12 run -> "6:12", slope 26.57 deg
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x)
    ann = annotate_facets_with_lidar([f], pts)
    a = ann[1]
    assert a["pitch_string"] == "6:12"
    assert abs(a["slope_deg"] - math.degrees(math.atan(0.5))) < 0.5
    # sloped area = plan / cos(theta): 100 / cos(26.57deg) ~ 111.8
    assert abs(a["surface_area_m2"] - 100.0 / math.cos(math.atan(0.5))) < 1.0
    assert not a["is_flat"]


def test_flat_roof_detected():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0))
    a = annotate_facets_with_lidar([f], pts)[1]
    assert a["is_flat"]
    assert abs(a["surface_area_m2"] - 100.0) < 0.5      # flat: sloped == plan


def test_two_facets_annotated_independently():
    f1 = Facet(facet_id=1, polygon=box(0, 0, 10, 10))     # 6:12
    f2 = Facet(facet_id=2, polygon=box(20, 0, 30, 10))    # flat
    pts = np.vstack([
        _grid_points(f1.polygon, lambda x, y: 0.5 * x),
        _grid_points(f2.polygon, lambda x, y: np.full_like(x, 3.0)),
    ])
    ann = annotate_facets_with_lidar([f1, f2], pts)
    assert ann[1]["pitch_string"] == "6:12" and not ann[1]["is_flat"]
    assert ann[2]["is_flat"]


def test_too_few_points_stays_unspecified():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [3.0, 1.0, 0.5]])
    assert annotate_facets_with_lidar([f], pts) == {}    # absent, not wrong


def test_fusion_never_touches_geometry():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x)
    before = list(f.polygon.exterior.coords)
    annotate_facets_with_lidar([f], pts)
    assert list(f.polygon.exterior.coords) == before     # frozen shapes


def test_split_multiplane_facet_at_the_ridge():
    # one facet the model returned as a blob, but the points form a ridge at x=5:
    # left half z=0.5*x, right half z=0.5*(10-x) -> two planes ~53deg apart
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: np.where(x < 5, 0.5 * x, 0.5 * (10 - x)),
                       step=0.3)
    out, changed = split_multiplane_facets([f], pts)
    assert changed and len(out) == 2
    assert abs(sum(g.polygon.area for g in out) - 100.0) < 1.0   # area conserved
    assert all(g.polygon.area > 20 for g in out)                 # two real halves


def test_single_plane_facet_not_split():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.4 * x + 0.1 * y + 2.0, step=0.3)
    out, changed = split_multiplane_facets([f], pts)
    assert not changed and len(out) == 1


def test_ground_returns_excluded_from_fit_and_eave():
    # a facet with roof points ~5-9 m up, plus driveway returns at z~0.2 bleeding in
    f = Facet(facet_id=1, polygon=box(0, 0, 8, 8))
    roof = _grid_points(f.polygon, lambda x, y: 0.4 * x + 5.0, step=0.4)
    ground = np.column_stack([np.linspace(0, 8, 60), np.linspace(0, 8, 60),
                              np.full(60, 0.2)])
    pts = np.vstack([roof, ground])
    ann = annotate_facets_with_lidar([f], pts, ground_z=0.0)
    assert 1 in ann
    # eave computed on roof points only -> well above ground, not dragged to ~0.2
    assert ann[1]["eave_height_m"] > 3.0


def test_fuse_into_report_input_fills_only_annotated():
    ri = {"facets": [
        {"facet_id": 1, "polygon_xy": [[0, 0]], "plan_area_m2": 100.0,
         "pitch_string": None, "slope_deg": None, "aspect_bin": None,
         "is_flat": False, "surface_area_m2": None},
        {"facet_id": 2, "polygon_xy": [[5, 5]], "plan_area_m2": 50.0,
         "pitch_string": None, "slope_deg": None, "aspect_bin": None,
         "is_flat": False, "surface_area_m2": None},
    ]}
    ann = {1: {"pitch_string": "6:12", "slope_deg": 26.57, "aspect_bin": "E",
               "is_flat": False, "surface_area_m2": 111.8}}
    out = fuse_into_report_input(ri, ann)
    assert out["facets"][0]["pitch_string"] == "6:12"
    assert out["facets"][0]["polygon_xy"] == [[0, 0]]     # geometry untouched
    assert out["facets"][1]["pitch_string"] is None       # stays unspecified
    # the unresolved facet must be FLAGGED, not left silent (gate honesty guard)
    assert out["facets"][1]["needs_review"] is True
    assert out["facets"][0].get("needs_review") is not True  # resolved -> not flagged


def test_unresolved_pitch_facet_is_flagged_so_gate_passes():
    """A facet whose plane fit failed (no annotation) is flagged needs_review, so
    report_qc.pitch_resolved / slope_applied treat it as honestly disclosed rather
    than a silent plan-area-as-surface-area error. This is the recurring gate FAIL."""
    from src.output.report_qc import score_report
    ri = {
        "address": "x", "report_id": "MI-TEST",
        "outline_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "facets": [
            {"facet_id": 1, "polygon_xy": [[0, 0], [10, 0], [10, 5], [0, 5]],
             "plan_area_m2": 50.0, "pitch_string": None, "slope_deg": None,
             "aspect_bin": None, "is_flat": False, "surface_area_m2": None},
        ],
        "edges": [],
    }
    # no annotation -> facet 1 is unresolved
    fuse_into_report_input(ri, {})
    assert ri["facets"][0]["needs_review"] is True
    res = score_report(ri)
    checks = {c["id"]: c for c in res["checks"]}
    assert checks["pitch_resolved"]["ok"], checks["pitch_resolved"]["detail"]
    assert checks["slope_applied"]["ok"], checks["slope_applied"]["detail"]
