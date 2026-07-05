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
    edges = edges_from_outline_and_facets(outline, [f1, f2])
    assert "valley" in [e["edge_type"] for e in edges]


def test_all_edges_have_length_and_geometry():
    outline = box(0, 0, 20, 20)
    facets = [Polygon([(0, 0), (20, 0), (10, 10)]), Polygon([(0, 0), (10, 10), (0, 20)])]
    for e in edges_from_outline_and_facets(outline, facets):
        assert e["length_m"] > 0
        assert len(e["geometry_xy"]) >= 2
        assert e["edge_type"] in ("eave", "ridge", "hip", "valley")


def test_no_outline_no_edges():
    assert edges_from_outline_and_facets(None, []) == []


def test_rake_relabel_gable():
    # gable: two facets sloping N/S -> the E/W gable-end edges become RAKES,
    # the N/S gutter edges stay EAVES
    from src.roofs.geom_edges import relabel_rakes
    from src.roofs.segment import Facet

    outline = box(0, 0, 40, 20)
    front = Facet(facet_id=1, polygon=box(0, 0, 40, 10))    # slopes south
    back = Facet(facet_id=2, polygon=box(0, 10, 40, 20))    # slopes north
    edges = edges_from_outline_and_facets(outline, [front.polygon, back.polygon])
    relabel_rakes(edges, [front, back], {1: 180.0, 2: 0.0})
    by_seg = {}
    for e in edges:
        if e["edge_type"] in ("eave", "rake"):
            (x0, y0), (x1, y1) = e["geometry_xy"][0], e["geometry_xy"][-1]
            horiz = abs(x1 - x0) > abs(y1 - y0)
            by_seg.setdefault(e["edge_type"], []).append(horiz)
    assert all(by_seg["eave"])                 # horizontal edges stayed eaves
    assert not any(by_seg["rake"])             # vertical gable ends -> rakes
    assert len(by_seg["rake"]) == 2


def test_rake_relabel_hip_roof_has_no_rakes():
    # hip: every facet slopes toward its own eave -> perimeter stays all eaves
    from src.roofs.geom_edges import relabel_rakes
    from src.roofs.segment import Facet

    outline = box(0, 0, 40, 20)
    polys = [Polygon([(0, 0), (40, 0), (30, 10), (10, 10)]),    # slopes S (180)
             Polygon([(0, 20), (40, 20), (30, 10), (10, 10)]),   # slopes N (0)
             Polygon([(0, 0), (0, 20), (10, 10)]),               # slopes W (270)
             Polygon([(40, 0), (40, 20), (30, 10)])]             # slopes E (90)
    facets = [Facet(facet_id=i + 1, polygon=p) for i, p in enumerate(polys)]
    edges = edges_from_outline_and_facets(outline, polys)
    relabel_rakes(edges, facets, {1: 180.0, 2: 0.0, 3: 270.0, 4: 90.0})
    assert not any(e["edge_type"] == "rake" for e in edges)


def test_rake_relabel_without_aspect_is_noop():
    from src.roofs.geom_edges import relabel_rakes
    from src.roofs.segment import Facet

    outline = box(0, 0, 40, 20)
    f = Facet(facet_id=1, polygon=box(0, 0, 40, 20))
    edges = edges_from_outline_and_facets(outline, [f.polygon])
    before = [e["edge_type"] for e in edges]
    relabel_rakes(edges, [f], {})              # no LiDAR aspects
    assert [e["edge_type"] for e in edges] == before


def test_corner_to_corner_seam_keeps_its_ridge():
    # Two big facets whose shared seam runs corner-to-corner (hip+ridge+hip in
    # one path — common when raster partitions absorb the small end triangles).
    # The middle run must still classify as RIDGE, not be lumped into hip.
    outline = box(0, 0, 40, 20)
    front = Polygon([(0, 0), (40, 0), (30, 10), (10, 10)])
    back = Polygon([(0, 0), (10, 10), (30, 10), (40, 0), (40, 20), (0, 20)])
    totals = {}
    for e in edges_from_outline_and_facets(outline, [front, back]):
        totals[e["edge_type"]] = totals.get(e["edge_type"], 0.0) + e["length_m"]
    assert abs(totals.get("ridge", 0.0) - 20.0) < 1.0     # the 20-unit ridge
    assert totals.get("hip", 0.0) > 20.0                   # both diagonals
