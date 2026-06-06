"""Tests for the facet boundary-snap / tiling post-process."""

from shapely.geometry import Polygon

from src.roofs.tiling import tile_facets


def _poly(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_gap_between_facets_is_closed_and_shared():
    # two facets with a 1m gap between them, inside a 10x5 outline
    a = _poly(0, 0, 4, 5)
    b = _poly(5, 0, 10, 5)            # gap x in [4,5]
    outline = _poly(0, 0, 10, 5)
    before_a, before_b = a, b
    assert before_a.boundary.intersection(before_b.boundary).length == 0  # no shared edge

    out = tile_facets([(0, a), (1, b)], outline)
    pa = dict(out)[0]
    pb = dict(out)[1]
    # they now share a real boundary segment...
    shared = pa.boundary.intersection(pb.boundary)
    assert shared.length > 1.0
    # ...and together cover the whole outline (gap filled)
    assert pa.union(pb).area >= outline.area * 0.98


def test_order_and_count_preserved():
    a = _poly(0, 0, 4, 5)
    b = _poly(5, 0, 10, 5)
    out = tile_facets([(7, a), (3, b)], _poly(0, 0, 10, 5))
    assert [fid for fid, _ in out] == [7, 3]


def test_single_facet_unchanged():
    a = _poly(0, 0, 4, 5)
    out = tile_facets([(0, a)], _poly(0, 0, 4, 5))
    assert out == [(0, a)]


def test_no_outline_falls_back_to_union():
    a = _poly(0, 0, 4, 5)
    b = _poly(5, 0, 10, 5)
    out = tile_facets([(0, a), (1, b)], outline=None)
    assert len(out) == 2
    # still produces a shared boundary via the buffered-union fallback outline
    assert dict(out)[0].boundary.intersection(dict(out)[1].boundary).length > 1.0
