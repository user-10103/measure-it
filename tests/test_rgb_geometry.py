"""Tests for the RGB path: pitch/aspect->plane bridge, ml seam, geometric edges."""
import math

from shapely.geometry import Polygon

from src.roofs.plane_fit import plane_from_pitch_aspect
from src.roofs.metrics import compute_slope_deg, compute_aspect_deg, compute_aspect_bin
from src.roofs.segment import segment_facets, CocoStandinBackend
from src.roofs.edges import classify_edges_from_facets, EdgeType


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
