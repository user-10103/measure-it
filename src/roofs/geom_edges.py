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


def _nearest_facets(mid, lookup, touch_tol):
    """Facet ids whose boundary passes within touch_tol of a point, nearest first."""
    hits = sorted(((poly.boundary.distance(mid), fid) for fid, poly in lookup),
                  key=lambda t: t[0])
    return [fid for d, fid in hits if d <= touch_tol]


def relabel_flat_junction_flashing(edges: List[dict], facets, annotations,
                                   touch_tol: float = 1.5) -> List[dict]:
    """A seam between a PITCHED facet and a FLAT facet is wall flashing, not a
    valley/hip/ridge (the Holland-vs-Roofr criterion: Roofr books those
    junctions under flashing; a valley needs two PITCHED faces draining into
    it). Pure relabel — geometry untouched; needs LiDAR is_flat per facet."""
    from shapely.geometry import LineString

    lookup = [(f.facet_id, f.polygon) for f in facets
              if f.polygon is not None and not f.polygon.is_empty]
    for e in edges:
        if e.get("edge_type") not in ("valley", "hip", "ridge"):
            continue
        g = e.get("geometry_xy", [])
        if len(g) < 2:
            continue
        mid = LineString(g).interpolate(0.5, normalized=True)
        near = _nearest_facets(mid, lookup, touch_tol)[:2]
        if len(near) < 2:
            continue
        flats = [annotations.get(fid, {}).get("is_flat") for fid in near]
        if None in flats:
            continue
        if flats[0] != flats[1]:                 # one pitched, one flat
            e["edge_type"] = "wall_flashing"
    return edges


def classify_internal_edges(edges: List[dict], facets, annotations,
                            touch_tol: float = 1.5,
                            level_dot: float = 0.45) -> List[dict]:
    """Reclassify internal seams using the LiDAR aspects — physics, not corners.

    Per straight SEGMENT (junction-safe), from the two flanking facets:
      * one flat, one pitched          -> wall_flashing (Roofr books it there)
      * both flat                      -> transition
      * both pitched:
          - segment LEVEL (runs perpendicular to both downslopes):
              downslopes diverge -> RIDGE, converge -> VALLEY
          - sloped segment: drainage decides — both downslopes pointing INTO
            the seam -> VALLEY, else HIP
          - aspects nearly equal (<45 deg) -> transition (a step, not a break)

    Fixes the Holland regression where post-merge seam footage inflated the
    ridge/hip buckets (157'/195' vs Roofr's 59'/127'). Perimeter edges pass
    through untouched. Returns a NEW edge list.
    """
    import math

    from shapely.geometry import LineString

    lookup = [(f.facet_id, f.polygon) for f in facets
              if f.polygon is not None and not f.polygon.is_empty]
    cent = {fid: poly.centroid for fid, poly in lookup}

    def _down(ann):
        r = math.radians(ann["aspect_deg"])
        return (math.sin(r), math.cos(r))          # unit downslope (x=E, y=N)

    out: List[dict] = []
    for e in edges:
        if e.get("edge_type") not in ("ridge", "hip", "valley") \
                or len(e.get("geometry_xy", [])) < 2:
            out.append(e)
            continue
        coords = [list(c) for c in e["geometry_xy"]]
        runs: List = []
        for a, b in zip(coords[:-1], coords[1:]):
            seg = LineString([a, b])
            L = seg.length
            if L <= 1e-9:
                continue
            mid = seg.interpolate(0.5, normalized=True)
            near = _nearest_facets(mid, lookup, touch_tol)[:2]
            etype = e["edge_type"]                  # geometric fallback
            a1 = annotations.get(near[0]) if len(near) > 0 else None
            a2 = annotations.get(near[1]) if len(near) > 1 else None
            if len(near) == 2 and (a1 is None or a2 is None):
                # a flank we couldn't measure -> don't guess ridge/hip; say so
                etype = "unspecified"
            elif a1 and a2:
                fl1, fl2 = bool(a1.get("is_flat")), bool(a2.get("is_flat"))
                if fl1 != fl2:
                    etype = "wall_flashing"
                elif fl1 and fl2:
                    etype = "transition"
                elif "aspect_deg" in a1 and "aspect_deg" in a2:
                    d1, d2 = _down(a1), _down(a2)
                    spread = abs(a1["aspect_deg"] - a2["aspect_deg"]) % 360
                    spread = min(spread, 360 - spread)
                    dirv = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)

                    def _toward(fid, d):
                        c = cent[fid]
                        vx, vy = mid.x - c.x, mid.y - c.y
                        n = math.hypot(vx, vy) or 1.0
                        return (d[0] * vx + d[1] * vy) / n

                    t1, t2 = _toward(near[0], d1), _toward(near[1], d2)
                    level = max(abs(dirv[0] * d1[0] + dirv[1] * d1[1]),
                                abs(dirv[0] * d2[0] + dirv[1] * d2[1])) <= level_dot
                    converge = t1 > 0.2 and t2 > 0.2
                    if spread < 45:
                        etype = "transition"
                    elif spread >= 135 and level:
                        etype = "valley" if converge else "ridge"
                    else:
                        etype = "valley" if converge else "hip"
            if runs and runs[-1][0] == etype and runs[-1][1][-1] == a:
                runs[-1][1].append(b)
            else:
                runs.append((etype, [a, b]))
        for etype, pts in runs:
            out.append({"edge_type": etype,
                        "length_m": float(LineString(pts).length),
                        "geometry_xy": pts})
    return out


def apply_3d_edge_lengths(edges: List[dict], facets, grads,
                          touch_tol: float = 1.5) -> List[dict]:
    """Plan lengths -> true sloped lengths using the adjacent facet's plane.

    An edge on plane z = a*x + b*y + c with unit direction d gains
    sqrt(1 + (grad . d)^2). Level edges (eaves, ridges) get factor ~1
    automatically because they run perpendicular to the gradient; hips,
    valleys, and rakes lengthen — e.g. Holland's hips 122.5' plan -> ~127'
    true at 3:12 (the Roofr number).
    """
    import math

    from shapely.geometry import LineString

    lookup = [(f.facet_id, f.polygon) for f in facets
              if f.polygon is not None and not f.polygon.is_empty]
    for e in edges:
        g = e.get("geometry_xy", [])
        if len(g) < 2:
            continue
        (x0, y0), (x1, y1) = g[0], g[-1]
        plan = math.hypot(x1 - x0, y1 - y0)
        if plan <= 1e-9:
            continue
        d = ((x1 - x0) / plan, (y1 - y0) / plan)
        mid = LineString(g).interpolate(0.5, normalized=True)
        near = _nearest_facets(mid, lookup, touch_tol)
        grad = next((grads[fid] for fid in near if fid in grads), None)
        if grad is None:
            continue
        dz = abs(grad[0] * d[0] + grad[1] * d[1])
        e["length_m"] = float(e["length_m"] * math.sqrt(1.0 + dz * dz))
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
