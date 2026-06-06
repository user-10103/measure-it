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
