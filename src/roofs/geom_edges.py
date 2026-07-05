"""Derive the roof's typed edge graph from geometry alone — no pitch/LiDAR.

The "unfold": the zero-shot roof outline is a clean closed polygon (the roof's
perimeter graph); the facet partition adds the internal seams. Union the two and
classify each edge by geometry:

  * perimeter segment            -> eave   (eave-vs-rake needs pitch/slope)
  * internal seam, fully interior -> ridge
  * internal seam touching a CONVEX outline corner  -> hip
  * internal seam touching a REFLEX (concave) corner -> valley

Everything is 2-D (pixel or metric, whatever the caller's CRS is). Lengths are
plan lengths; LiDAR upgrades these to true sloped lengths and splits eave/rake.
"""
from __future__ import annotations

from typing import List

from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient


def _reflex_flags(ring: List[tuple]) -> List[bool]:
    """For a CCW ring (no repeated last point), True where the vertex is reflex
    (interior angle > 180 deg)."""
    n = len(ring)
    flags = []
    for i in range(n):
        ax, ay = ring[(i - 1) % n]
        bx, by = ring[i]
        cx, cy = ring[(i + 1) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        flags.append(cross < 0)      # CCW ring: reflex when the turn is clockwise
    return flags


def _shared_seams(facets) -> List[LineString]:
    """Boundaries shared between facet pairs = the internal seams."""
    segs: List[LineString] = []
    polys = [f for f in facets if f is not None and not f.is_empty]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = polys[i].boundary.intersection(polys[j].boundary)
            for g in getattr(inter, "geoms", [inter]):
                if g.geom_type == "LineString" and g.length > 1e-6:
                    segs.append(g)
    return segs


def edges_from_outline_and_facets(outline, facets, touch_tol: float = 2.0) -> List[dict]:
    """outline (Polygon) + facets (list of Polygon) -> report-ready edge dicts:
    ``{edge_type, length_m, geometry_xy}``. Types: eave / ridge / hip / valley."""
    edges: List[dict] = []
    if outline is None or outline.is_empty:
        return edges

    poly = orient(outline, 1.0)                 # CCW so reflex test is consistent
    ring = list(poly.exterior.coords)[:-1]
    reflex = _reflex_flags(ring)
    verts = [Point(p) for p in ring]
    ext = poly.exterior

    # perimeter -> eaves
    closed = ring + [ring[0]]
    for a, b in zip(closed[:-1], closed[1:]):
        seg = LineString([a, b])
        if seg.length <= 1e-6:
            continue
        edges.append({"edge_type": "eave", "length_m": float(seg.length),
                      "geometry_xy": [list(a), list(b)]})

    # internal seams -> ridge / hip / valley
    for seg in _shared_seams(facets):
        ends = [Point(seg.coords[0]), Point(seg.coords[-1])]
        kinds = []
        for e in ends:
            if ext.distance(e) > touch_tol:
                kinds.append("interior")
                continue
            k = min(range(len(verts)), key=lambda idx: verts[idx].distance(e))
            if verts[k].distance(e) <= touch_tol * 2.5:
                kinds.append("valley" if reflex[k] else "hip")
            else:
                kinds.append("hip")             # meets the perimeter mid-segment
        etype = "valley" if "valley" in kinds else "hip" if "hip" in kinds else "ridge"
        edges.append({"edge_type": etype, "length_m": float(seg.length),
                      "geometry_xy": [list(c) for c in seg.coords]})
    return edges


def relabel_rakes(edges: List[dict], facets, aspects_deg,
                  angle_thresh_deg: float = 45.0, touch_tol: float = 1.5) -> List[dict]:
    """Split perimeter "eave" edges into eave vs RAKE using slope direction.

    Both are perimeter edges; the difference is orientation to the adjacent
    facet's downslope direction (LiDAR aspect): an eave runs PERPENDICULAR to
    the slope (the gutter edge), a rake runs ALONG it (the gable-end edge).
    Pure relabel — geometry and lengths never change. Facets without an aspect
    (flat / no LiDAR) keep their edges as eaves.

    Args:
        edges: report edge dicts (mutated in place and returned).
        facets: objects with ``.facet_id`` and ``.polygon`` (the SAM facets).
        aspects_deg: {facet_id: downslope compass degrees from North}.
    """
    import math

    from shapely.geometry import LineString, Point

    cos_thr = math.cos(math.radians(angle_thresh_deg))
    lookup = [(f.facet_id, f.polygon) for f in facets
              if f.polygon is not None and not f.polygon.is_empty]
    for e in edges:
        if e.get("edge_type") != "eave" or len(e.get("geometry_xy", [])) < 2:
            continue
        seg = LineString(e["geometry_xy"])
        mid = seg.interpolate(0.5, normalized=True)
        # adjacent facet = nearest boundary to the segment midpoint
        best_id, best_d = None, float("inf")
        for fid, poly in lookup:
            d = poly.boundary.distance(mid)
            if d < best_d:
                best_id, best_d = fid, d
        if best_id is None or best_d > touch_tol:
            continue
        aspect = aspects_deg.get(best_id)
        if aspect is None:
            continue
        (x0, y0), (x1, y1) = e["geometry_xy"][0], e["geometry_xy"][-1]
        elen = math.hypot(x1 - x0, y1 - y0)
        if elen <= 1e-9:
            continue
        ex, ey = (x1 - x0) / elen, (y1 - y0) / elen
        a = math.radians(aspect)                      # compass: 0=N, 90=E
        dx, dy = math.sin(a), math.cos(a)             # downslope unit vector
        if abs(ex * dx + ey * dy) >= cos_thr:         # parallel to slope -> rake
            e["edge_type"] = "rake"
    return edges
