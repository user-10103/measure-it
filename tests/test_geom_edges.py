"""Geometry-only typed edge graph from outline + facets (no pitch)."""
from collections import Counter

from shapely.geometry import Polygon, box

from src.roofs.geom_edges import edges_from_outline_and_facets


def test_hip_roof_edges():
    outline = box(0, 0, 40, 20)
    facets = [Polygon([(0, 0), (40, 0), (30, 10), (10, 10)]),   # front trapezoid
              Polygon([(0, 20), (40, 20), (30, 10), (10, 10)]),  # back trapezoid
              Polygon([(0, 0), (0, 20), (10, 10)]),              # left triangle
              Polygon([(40, 0), (40, 20), (30, 10)])]            # right triangle
    counts = Counter(e["edge_type"] for e in edges_from_outline_and_facets(outline, facets))
    assert counts["eave"] == 4
    assert counts["ridge"] == 1        # interior seam, no perimeter contact
    assert counts["hip"] == 4          # seams from the convex corners
    assert "valley" not in counts


def test_valley_from_reflex_corner():
    # L-shaped roof -> the reflex (concave) corner produces a valley
    outline = Polygon([(0, 0), (40, 0), (40, 20), (20, 20), (20, 40), (0, 40)])
    f1 = Polygon([(0, 0), (40, 0), (40, 20), (20, 20), (0, 20)])
    f2 = Polygon([(0, 20), (20, 20), (20, 40), (0, 40)])
    types = [e["edge_type"] for e in edges_from_outline_and_facets(outline, f1_f2 := [f1, f2])]
    assert "valley" in types


def test_all_edges_have_length_and_geometry():
    outline = box(0, 0, 20, 20)
    facets = [Polygon([(0, 0), (20, 0), (10, 10)]), Polygon([(0, 0), (10, 10), (0, 20)])]
    for e in edges_from_outline_and_facets(outline, facets):
        assert e["length_m"] > 0
        assert len(e["geometry_xy"]) >= 2
        assert e["edge_type"] in ("eave", "ridge", "hip", "valley")


def test_no_outline_no_edges():
    assert edges_from_outline_and_facets(None, []) == []
