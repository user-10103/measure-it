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


def test_seam_survives_a_sub_centimetre_gap():
    """Facet polygons are simplified independently and pushed through union/buffer
    in the coplanarity merge, so adjacent facets routinely end up millimetres apart.
    An EXACT boundary intersection returns nothing for a 1 cm gap, which deletes
    every internal edge from the report: 1600 Sarno shipped with
    edge_totals_m == {"eave": 65.28} — no ridge, no hip, nothing mislabelled,
    just absent. Snapping within tolerance recovers the seam at its true length."""
    from shapely.geometry import box
    from src.roofs.geom_edges import _shared_seams

    for a, b, label in [
        (box(0, 0, 5, 10), box(5, 0, 10, 10), "touching"),
        (box(0, 0, 5, 10), box(5.008, 0, 10, 10), "8 mm gap"),
        (box(0, 0, 5.008, 10), box(5, 0, 10, 10), "8 mm overlap"),
    ]:
        got = [s.length for s in _shared_seams([a, b])]
        assert got and abs(got[0] - 10.0) < 0.05, f"{label}: {got}"

    # genuinely separate facets must NOT acquire a seam
    assert _shared_seams([box(0, 0, 5, 10), box(10, 0, 15, 10)]) == []
    # nor should two facets that merely touch at a corner
    assert _shared_seams([box(0, 0, 5, 5), box(5, 5, 10, 10)]) == []


def test_flat_sections_at_different_heights_type_as_parapet():
    """On a commercial roof the seam between two FLAT sections is not a ridge —
    neither side drains. What matters is whether they sit at different heights:
    a step is a parapet / level change and a chargeable edge. 2725 Judge Fran
    reported 43,029 sqft with zero internal edges of any kind."""
    from shapely.geometry import box
    from src.roofs.geom_edges import classify_internal_edges
    from src.roofs.segment import Facet

    lo = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    hi = Facet(facet_id=2, polygon=box(10, 0, 20, 10))
    seam = [{"edge_type": "ridge", "length_m": 10.0,
             "geometry_xy": [[10, 0], [10, 10]]}]

    def ann(z):
        return {"is_flat": True, "aspect_deg": 0.0, "median_z": z,
                "grad": (0.0, 0.0), "slope_deg": 0.0}

    # 1.4 m step -> parapet
    out = classify_internal_edges(list(seam), [lo, hi], {1: ann(5.0), 2: ann(6.4)})
    assert {e["edge_type"] for e in out} == {"parapet"}

    # same height -> just a segmentation seam, not a chargeable parapet
    out = classify_internal_edges(list(seam), [lo, hi], {1: ann(5.0), 2: ann(5.05)})
    assert {e["edge_type"] for e in out} == {"transition"}
