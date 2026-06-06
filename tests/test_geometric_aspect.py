"""Tests for geometric aspect derivation (downslope = facet -> eave)."""

from shapely.geometry import Polygon

from src.roofs.geometric_aspect import facet_aspect_deg


# A gable roof, outline = full 10x10 square; two facets meeting at the y=5 ridge.
OUTLINE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SOUTH_FACET = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])   # eave on south (y=0)
NORTH_FACET = Polygon([(0, 5), (10, 5), (10, 10), (0, 10)])  # eave on north (y=10)


def _bin(deg):
    from src.roofs.metrics import compute_aspect_bin
    return compute_aspect_bin(deg)


def test_south_facet_faces_south():
    deg = facet_aspect_deg(SOUTH_FACET, OUTLINE)
    assert deg is not None
    assert _bin(deg) == "S"          # downslope points toward the south eave


def test_north_facet_faces_north():
    deg = facet_aspect_deg(NORTH_FACET, OUTLINE)
    assert _bin(deg) == "N"


def test_opposite_facets_are_opposite():
    # the key property the ridge classifier needs (LS labels failed this)
    s = facet_aspect_deg(SOUTH_FACET, OUTLINE)
    n = facet_aspect_deg(NORTH_FACET, OUTLINE)
    diff = abs((s - n + 180) % 360 - 180)
    assert diff > 150               # ~180 deg apart


def test_no_outline_returns_none():
    assert facet_aspect_deg(SOUTH_FACET, None) is None


def test_interior_facet_uses_centroid_bearing():
    """Interior facet (no outline boundary) falls back to roof_centroid->facet_centroid bearing."""
    from shapely.geometry import Point
    # 2x2 outline centred at (5,5); interior facet is NE of centre -> downslope NE ~ 45 deg
    interior = Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])
    outline = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    roof_c = Point(5, 5)  # outline centroid
    # Without roof_centroid: returns None (interior, no eave)
    assert facet_aspect_deg(interior, outline) is None
    # With roof_centroid: returns ~45 deg (NE)
    deg = facet_aspect_deg(interior, outline, roof_centroid=roof_c)
    assert deg is not None
    assert abs((deg - 45) % 360) < 10, f"expected ~45, got {deg:.1f}"


def test_interior_facet_south_of_centre():
    """Interior facet SW of roof centre -> downslope bearing ~225 deg."""
    from shapely.geometry import Point
    interior = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    outline = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    roof_c = Point(5, 5)
    deg = facet_aspect_deg(interior, outline, roof_centroid=roof_c)
    assert deg is not None
    assert abs((deg - 225) % 360) < 10, f"expected ~225, got {deg:.1f}"
