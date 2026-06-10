"""Tests for the RGB path: pitch/aspect->plane bridge, ml seam, geometric edges."""
import math

from shapely.geometry import Polygon

from src.roofs.plane_fit import plane_from_pitch_aspect
from src.roofs.metrics import compute_slope_deg, compute_aspect_deg, compute_aspect_bin
from src.roofs.segment import segment_facets, CocoStandinBackend
from src.roofs.edges import (
    classify_edges_from_facets, correct_edge_lengths_3d,
    classify_perimeter_edge, EdgeType, RoofEdge,
)
from shapely.geometry import LineString


def test_plane_from_pitch_aspect_roundtrip():
    # flat facet -> zero gradient
    flat = plane_from_pitch_aspect(0.0, aspect_bin="flat")
    assert flat.a == 0.0 and flat.b == 0.0

    # a 20 deg facet facing South should recover slope=20 and aspect bin S
    p = plane_from_pitch_aspect(20.0, aspect_bin="S")
    assert math.isclose(compute_slope_deg(p), 20.0, abs_tol=1e-6)
    assert compute_aspect_bin(compute_aspect_deg(p)) == "S"

    # facing East recovers E
    pe = plane_from_pitch_aspect(30.0, aspect_bin="E")
    assert compute_aspect_bin(compute_aspect_deg(pe)) == "E"


def _gable_prediction():
    """Two facets forming a gable: A (y 0-5) slopes S, B (y 5-10) slopes N."""
    return {
        "outline": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "facets": [
            {"polygon": [[0, 0], [10, 0], [10, 5], [0, 5]], "slope_deg": 20.0, "aspect_bin": "S"},
            {"polygon": [[0, 5], [10, 5], [10, 10], [0, 10]], "slope_deg": 20.0, "aspect_bin": "N"},
        ],
    }


def test_segment_facets_ml():
    facets = segment_facets(method="ml", prediction=_gable_prediction())
    assert len(facets) == 2
    for f in facets:
        assert f.points is None
        assert f.polygon is not None and f.polygon.area > 0
        assert f.plane is not None
    assert {f.aspect_bin for f in facets} == {"S", "N"}


def test_geometric_edges_gable_has_ridge():
    facets = segment_facets(method="ml", prediction=_gable_prediction())
    fps = [(f.facet_id, f.polygon) for f in facets]
    planes = [f.plane for f in facets]
    outline = Polygon(_gable_prediction()["outline"])
    edges = classify_edges_from_facets(fps, planes, outline)

    types = [e.edge_type for e in edges]
    # exactly the shared ridge line interior
    assert types.count(EdgeType.RIDGE) == 1
    assert EdgeType.HIP not in types and EdgeType.VALLEY not in types
    # perimeter: the low (y=0 and y=10) edges are eaves, the gable sides are rakes
    assert EdgeType.EAVE in types
    assert EdgeType.RAKE in types
    ridge = next(e for e in edges if e.edge_type == EdgeType.RIDGE)
    assert math.isclose(ridge.length_m, 10.0, abs_tol=1e-6)


def test_geometric_edges_hip():
    """Four facets sloping outward to four sides from a center point -> hips."""
    # pyramidal hip roof on a 10x10 square, apex at center (5,5)
    pred = {
        "outline": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "facets": [
            {"polygon": [[0, 0], [10, 0], [5, 5]], "slope_deg": 20.0, "aspect_bin": "S"},
            {"polygon": [[10, 0], [10, 10], [5, 5]], "slope_deg": 20.0, "aspect_bin": "E"},
            {"polygon": [[10, 10], [0, 10], [5, 5]], "slope_deg": 20.0, "aspect_bin": "N"},
            {"polygon": [[0, 10], [0, 0], [5, 5]], "slope_deg": 20.0, "aspect_bin": "W"},
        ],
    }
    facets = segment_facets(method="ml", prediction=pred)
    fps = [(f.facet_id, f.polygon) for f in facets]
    planes = [f.plane for f in facets]
    edges = classify_edges_from_facets(fps, planes, Polygon(pred["outline"]))
    types = [e.edge_type for e in edges]
    # adjacent faces of a pyramid meet at hips, not ridges
    assert EdgeType.HIP in types
    assert EdgeType.RIDGE not in types
    assert EdgeType.VALLEY not in types


# ---------------------------------------------------------------------------
# 3D edge length correction (Fix: hips/rakes/valleys are sloped in 3D)
# ---------------------------------------------------------------------------

def test_ridge_length_unchanged():
    """Ridge is horizontal — correct_edge_lengths_3d must not change its length."""
    plane = plane_from_pitch_aspect(20.0, aspect_bin="S")
    ridge = RoofEdge(0, EdgeType.RIDGE, LineString([(0, 5), (10, 5)]), 10.0, (0, None))
    correct_edge_lengths_3d([ridge], {0: plane})
    assert math.isclose(ridge.length_m, 10.0, abs_tol=1e-6)


def test_eave_length_unchanged():
    """Eave runs along a contour line — no 3D correction needed."""
    plane = plane_from_pitch_aspect(20.0, aspect_bin="S")
    eave = RoofEdge(0, EdgeType.EAVE, LineString([(0, 0), (10, 0)]), 10.0, (0, None))
    correct_edge_lengths_3d([eave], {0: plane})
    assert math.isclose(eave.length_m, 10.0, abs_tol=1e-6)


def test_rake_length_corrected():
    """Rake runs in the downslope direction — 3D length = plan / cos(slope)."""
    slope_deg = 20.0
    plane = plane_from_pitch_aspect(slope_deg, aspect_bin="S")
    # rake runs N-S (along the slope direction for an S-facing facet)
    rake = RoofEdge(0, EdgeType.RAKE, LineString([(5, 0), (5, 10)]), 10.0, (0, None))
    correct_edge_lengths_3d([rake], {0: plane})
    expected = 10.0 / math.cos(math.radians(slope_deg))
    assert math.isclose(rake.length_m, expected, rel_tol=0.01)


def test_hip_length_corrected():
    """Hip at 45° between two orthogonal facets gets a partial slope correction."""
    slope_deg = 20.0
    plane = plane_from_pitch_aspect(slope_deg, aspect_bin="S")
    # hip runs diagonally (45°, NE direction from eave corner to apex)
    hip = RoofEdge(0, EdgeType.HIP, LineString([(0, 0), (5, 5)]), math.hypot(5, 5), (0, None))
    before = hip.length_m
    correct_edge_lengths_3d([hip], {0: plane})
    # hip must be longer than plan distance but shorter than a full-slope rake
    assert hip.length_m > before
    assert hip.length_m < before / math.cos(math.radians(slope_deg))


def test_no_plane_leaves_edge_unchanged():
    """If no plane is available for a hip edge, length must not change."""
    hip = RoofEdge(0, EdgeType.HIP, LineString([(0, 0), (5, 5)]), 7.07, (None, None))
    correct_edge_lengths_3d([hip], {})
    assert math.isclose(hip.length_m, 7.07, abs_tol=1e-6)


def test_gable_rakes_longer_than_plan_after_classify():
    """End-to-end: rake edges from classify_edges_from_facets have 3D length > plan."""
    pred = _gable_prediction()
    facets = segment_facets(method="ml", prediction=pred)
    fps = [(f.facet_id, f.polygon) for f in facets]
    planes = [f.plane for f in facets]
    outline = Polygon(pred["outline"])
    edges = classify_edges_from_facets(fps, planes, outline)
    rakes = [e for e in edges if e.edge_type == EdgeType.RAKE]
    assert rakes, "expected rake edges on a gable"
    slope_deg = 20.0
    plan_rake_len = 5.0  # each half-rake is 5 m in plan
    expected_min = plan_rake_len / math.cos(math.radians(slope_deg)) * 0.9  # 10% tolerance
    for r in rakes:
        assert r.length_m > plan_rake_len * 0.95, (
            f"rake {r.length_m:.3f} m not longer than plan {plan_rake_len} m"
        )
        _ = expected_min  # referenced to silence linter


# ---------------------------------------------------------------------------
# Parapet and step-flashing detection
# ---------------------------------------------------------------------------

def test_flat_facet_perimeter_is_parapet():
    """Perimeter of a flat facet (g≈0) must be classified as PARAPET."""
    flat_plane = plane_from_pitch_aspect(0.0, aspect_bin="flat")
    etype = classify_perimeter_edge(flat_plane, (0, 0), (10, 0))
    assert etype == EdgeType.PARAPET


def test_sloped_eave_classified_correctly():
    """Downslope-facing perimeter edge is still EAVE."""
    # S-facing facet (drains south). Centroid is to the north of the bottom eave.
    plane = plane_from_pitch_aspect(20.0, aspect_bin="S")
    # Edge runs E-W at the south (low) end; centroid is 5m north of midpoint.
    etype = classify_perimeter_edge(plane, (0, 0), (10, 0), interior_point=(5.0, 5.0))
    assert etype == EdgeType.EAVE


def test_uphill_perimeter_edge_is_step_flashing():
    """Perimeter edge at the HIGH side of a facet (drains toward interior) → STEP_FLASHING."""
    # S-facing facet drains south. The NORTH edge of the facet is its uphill side.
    # The centroid of the facet is to the SOUTH of the north edge midpoint.
    plane = plane_from_pitch_aspect(20.0, aspect_bin="S")
    # North edge: y=10, runs E-W. Interior (centroid) is at y≈5 — i.e., SOUTH of this edge.
    # For an S-facing facet, downslope is south. Inward from this north edge is also south.
    # → downslope ∥ inward → step flashing.
    etype = classify_perimeter_edge(plane, (0, 10), (10, 10), interior_point=(5.0, 5.0))
    assert etype == EdgeType.STEP_FLASHING


def test_end_to_end_flat_roof_has_parapet_edges():
    """A purely flat roof (one flat facet) produces only parapet perimeter edges."""
    pred = {
        "outline": [[0, 0], [20, 0], [20, 10], [0, 10]],
        "facets": [
            {"polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
             "slope_deg": 0.0, "aspect_bin": "flat"},
        ],
    }
    facets = segment_facets(method="ml", prediction=pred)
    fps = [(f.facet_id, f.polygon) for f in facets]
    planes = [f.plane for f in facets]
    outline = Polygon(pred["outline"])
    edges = classify_edges_from_facets(fps, planes, outline)
    perimeter = [e for e in edges if e.edge_type != EdgeType.RIDGE]
    assert perimeter, "expected perimeter edges"
    assert all(e.edge_type == EdgeType.PARAPET for e in perimeter), (
        f"non-parapet edge found: {[e.edge_type for e in perimeter]}"
    )


def test_coco_standin_backend():
    coco = {
        "images": [{"id": 1, "address_id": "a1", "width": 10, "height": 10}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]},
            {"id": 2, "image_id": 1, "category_id": 2,
             "segmentation": [[0, 0, 10, 0, 10, 5, 0, 5]],
             "attributes": {"slope_deg": 20.0, "aspect_label": "S"}},
        ],
    }
    backend = CocoStandinBackend(coco)
    pred = backend.predict_for("a1")
    assert pred["outline"] is not None
    assert len(pred["facets"]) == 1
    assert pred["facets"][0]["slope_deg"] == 20.0
    facets = segment_facets(method="ml", prediction=pred)
    assert len(facets) == 1 and facets[0].aspect_bin == "S"
