"""Clean facet reconstruction — regularize jagged facet polygons into
straight-edged, footprint-tiling facets.

This is Phase 0/1 of the facet-reconstruct plan. It consolidates the scattered
regularization logic that previously lived in prototypes and half-wired helpers
into one importable, unit-tested module:

- ``naip_segment._regularize_polys_native`` (axis-snap + gap-fill + overlap-trim)
- ``prototype_edges_v3.fit_segment``           (PCA straight-line fit, for edges)

and adds an optional pass through ``buildingregulariser`` (ArcGIS-style
"Regularize Building Footprint") when that package is installed.

Everything operates in a **projected metric CRS** (UTM metres) — never lat/lon.
The public entry point :func:`reconstruct_facets` mirrors the input/output
contract of ``src/pipeline.py`` lines 576-887 so it can drop in as a replacement
in Phase 2 without changing any downstream consumer (metrics, edge
classification, GeoJSON export, PDF).

Design invariants (kept identical to the code this replaces):
- The facet **count is preserved** — regularization never merges or drops a
  facet, so per-facet planes/pitch/aspect stay aligned by index/id.
- Any facet that would degenerate falls back to a simplified original.
- The regularized set still **tiles the footprint** (gap-fill + overlap-trim).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Snap grid for axis regularization (metres). Matches naip_segment.SNAP_GRID_M.
SNAP_GRID_M = 0.5

# Optional ArcGIS-style regularizer. If present we prefer it; otherwise we use
# the built-in axis-snap regularizer (no hard dependency).
try:  # pragma: no cover - availability depends on environment
    from buildingregulariser import regularize_geodataframe as _br_regularize  # type: ignore
    import geopandas as _gpd  # type: ignore
    _HAS_BUILDINGREGULARISER = True
except Exception:  # pragma: no cover
    _HAS_BUILDINGREGULARISER = False


# ── geometry helpers ──────────────────────────────────────────────────────────

def _valid(g: Optional[Polygon]) -> Optional[Polygon]:
    """Repair invalid polygons with the standard buffer(0) trick."""
    if g is None:
        return None
    return g if g.is_valid else g.buffer(0)


def _principal_axis_angle(footprint: Polygon) -> float:
    """Angle (degrees, east=0, CCW) of the footprint's longest axis, taken from
    its minimum rotated rectangle. Facet edges are snapped parallel/perpendicular
    to this direction."""
    rect = footprint.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]
    a, b = edges[lengths.index(max(lengths))]
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _snap_poly_to_axis(poly: Polygon, cos_t: float, sin_t: float,
                       snap_m: float) -> Polygon:
    """Rotate *poly* into the building-axis frame, snap vertices to a *snap_m*
    grid, rotate back. Falls back to a simplified original on any failure so a
    facet is never lost to a regularization error."""
    from shapely.affinity import affine_transform

    try:
        # Forward rotation: x' = cos·x + sin·y, y' = -sin·x + cos·y
        rotated = affine_transform(poly, [cos_t, sin_t, -sin_t, cos_t, 0, 0])
        simplified = rotated.simplify(snap_m * 0.9, preserve_topology=True)
        if simplified.geom_type != "Polygon" or simplified.is_empty:
            return poly.simplify(snap_m, preserve_topology=True)

        raw = list(simplified.exterior.coords)
        snapped = [(round(x / snap_m) * snap_m, round(y / snap_m) * snap_m)
                   for x, y in raw]
        deduped = [snapped[0]]
        for pt in snapped[1:]:
            if pt != deduped[-1]:
                deduped.append(pt)
        if deduped[0] != deduped[-1]:
            deduped.append(deduped[0])
        if len(deduped) < 4:
            return poly.simplify(snap_m, preserve_topology=True)

        reg = Polygon(deduped)
        if not reg.is_valid:
            reg = reg.buffer(0)
        if reg.is_empty or reg.area < 0.01:
            return poly.simplify(snap_m, preserve_topology=True)

        # Inverse rotation: x = cos·x' - sin·y', y = sin·x' + cos·y'
        return affine_transform(reg, [cos_t, -sin_t, sin_t, cos_t, 0, 0])
    except Exception:
        try:
            return poly.simplify(snap_m, preserve_topology=True)
        except Exception:
            return poly


def _br_straighten(polys: Sequence[Polygon], snap_m: float) -> List[Polygon]:
    """Straighten each polygon with buildingregulariser (ArcGIS-style). Row
    order/count is preserved; any degenerate result falls back to the original.
    Raises if the package changed the row count (caller falls back)."""
    gdf = _gpd.GeoDataFrame(geometry=list(polys))
    out = _br_regularize(
        gdf,
        allow_45_degree=True,      # hips/valleys often run at 45°
        allow_circles=False,       # roof facets are not circular
        simplify_tolerance=max(0.25, snap_m),
        parallel_threshold=1.0,
    )
    geoms = list(out.geometry)
    if len(geoms) != len(polys):
        raise ValueError("buildingregulariser changed row count")
    result: List[Polygon] = []
    for orig, g in zip(polys, geoms):
        g = _keep_polygon(g) if g is not None else None
        if g is None or g.is_empty or not g.is_valid or g.area < 0.01:
            g = orig
        result.append(g)
    return result


def _straighten_each(polys: Sequence[Polygon], footprint: Polygon,
                     snap_m: float) -> List[Polygon]:
    """Straighten each polygon independently — via buildingregulariser when
    available, else the built-in principal-axis snap. Never changes count."""
    if _HAS_BUILDINGREGULARISER:
        try:
            return _br_straighten(polys, snap_m)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("buildingregulariser failed (%s) — axis-snap fallback", e)
    theta = math.radians(_principal_axis_angle(footprint))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return [_snap_poly_to_axis(p, cos_t, sin_t, snap_m) for p in polys]


def regularize_facets(polys: Sequence[Polygon], footprint: Polygon,
                      snap_m: float = SNAP_GRID_M) -> List[Polygon]:
    """Straighten a set of facet polygons into clean edges, while still tiling
    the footprint.

    Count-preserving: returns exactly ``len(polys)`` polygons (index-aligned to
    the input), so per-facet plane/pitch/aspect attributes remain valid.
    """
    if not polys:
        return list(polys)

    regularized = _straighten_each(polys, footprint, snap_m)

    # Clip to footprint so snapping can't push a facet outside the building.
    clipped: List[Polygon] = []
    for orig, reg in zip(polys, regularized):
        try:
            c = reg.intersection(footprint)
            clipped.append(c if (not c.is_empty and c.area > 0.01) else orig)
        except Exception:
            clipped.append(orig)
    regularized = clipped

    # Gap-fill: any footprint area left uncovered after snapping is merged into
    # the nearest facet (by centroid distance) so the tiling stays complete.
    covered = unary_union(regularized)
    gap = footprint.difference(covered)
    if not gap.is_empty:
        gap_parts = (list(gap.geoms)
                     if gap.geom_type in ("MultiPolygon", "GeometryCollection")
                     else [gap])
        for piece in gap_parts:
            if piece.is_empty or piece.area < 1e-8:
                continue
            idx = min(range(len(regularized)),
                      key=lambda i: piece.centroid.distance(regularized[i].centroid))
            regularized[idx] = regularized[idx].union(piece)

    # Overlap-trim: largest facets keep their full area; smaller facets are
    # trimmed away from already-claimed regions (no double-counted area).
    order = sorted(range(len(regularized)), key=lambda i: -regularized[i].area)
    trimmed: List[Optional[Polygon]] = [None] * len(regularized)
    claimed = None
    for i in order:
        p = regularized[i]
        if claimed is not None:
            try:
                diff = p.difference(claimed)
                p = diff if (not diff.is_empty and diff.area > 0.01) else p
            except Exception:
                pass
        trimmed[i] = p
        claimed = unary_union([x for x in trimmed if x is not None])

    return [t if t is not None else o for t, o in zip(trimmed, polys)]


def _keep_polygon(g) -> Polygon:
    """Coerce a set-op result (may be Multi/GeometryCollection) to the single
    largest Polygon component."""
    if g is None or g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return g
    polys = [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
    return max(polys, key=lambda p: p.area) if polys else g


# ── public entry point (mirrors pipeline.py:576-887 contract) ──────────────────

def reconstruct_facets(
    outline_native: Optional[Polygon],
    facet_polygons: List[Tuple[int, Polygon]],
    planes_for_poly: Optional[List] = None,
    snap_m: float = SNAP_GRID_M,
) -> Tuple[List[Tuple[int, Polygon]], Dict[int, Polygon], Optional[List]]:
    """Partition candidate facet boundaries against the roof outline and
    regularize them into clean, straight-edged, tiling facets.

    Parameters
    ----------
    outline_native : Polygon | None
        Roof outline in the projected metric CRS. If ``None``, facets are only
        partitioned against each other (no outline clip) — matches the pipeline
        fallback when outline reprojection fails.
    facet_polygons : list[(facet_id, Polygon)]
        Candidate boundaries (convex hulls or NAIP cells), metric CRS.
    planes_for_poly : list[PlaneModel] | None
        Parallel to ``facet_polygons``. If provided, facets are claimed in order
        of descending ``inlier_count`` (dominant planes win contested area);
        otherwise by descending area. Returned pruned/aligned to the output.
    snap_m : float
        Regularization snap grid in metres.

    Returns
    -------
    (facet_polygons, facet_boundaries_by_id, planes_for_poly)
        Same shapes/types the downstream pipeline expects.
    """
    if not facet_polygons:
        return [], {}, planes_for_poly

    ids = [fid for fid, _ in facet_polygons]
    polys = [_valid(p) for _, p in facet_polygons]
    outline = _valid(outline_native) if outline_native is not None else None

    # Claim priority: dominant planes (most inliers) first, else largest area.
    if planes_for_poly and len(planes_for_poly) == len(polys):
        def _priority(i: int) -> float:
            return float(getattr(planes_for_poly[i], "inlier_count", 0) or 0)
    else:
        def _priority(i: int) -> float:
            return polys[i].area if polys[i] is not None else 0.0

    order = sorted(range(len(polys)), key=lambda i: -_priority(i))

    # Partition: each facet is clipped to the outline, then to the area not yet
    # claimed by a higher-priority facet, so the result tiles without overlap.
    claimed = None
    partitioned: Dict[int, Polygon] = {}
    for i in order:
        p = polys[i]
        if p is None or p.is_empty:
            continue
        if outline is not None:
            try:
                p = _keep_polygon(p.intersection(outline))
            except Exception:
                pass
        if p is None or p.is_empty:
            continue
        if claimed is not None:
            try:
                d = _keep_polygon(p.difference(claimed))
                if not d.is_empty and d.area > 0.01:
                    p = d
            except Exception:
                pass
        if p is None or p.is_empty or p.area <= 0.01:
            continue
        partitioned[i] = p
        claimed = p if claimed is None else unary_union([claimed, p])

    if not partitioned:
        return [], {}, planes_for_poly

    kept = sorted(partitioned.keys())
    tiling_target = outline if outline is not None else unary_union(
        [partitioned[i] for i in kept])

    reg = regularize_facets([partitioned[i] for i in kept], tiling_target, snap_m)

    out_facets = [(ids[i], reg[j]) for j, i in enumerate(kept)]
    out_boundaries = dict(out_facets)
    out_planes = ([planes_for_poly[i] for i in kept]
                  if (planes_for_poly and len(planes_for_poly) == len(polys))
                  else planes_for_poly)

    if outline is not None:
        _tiled = sum(p.area for _, p in out_facets)
        logger.info("reconstruct_facets: %d facet(s) tile %.1f / %.1f sq m",
                    len(out_facets), _tiled, outline.area)

    return out_facets, out_boundaries, out_planes


# ── Phase 4: clean typed edges from clean facets ───────────────────────────────

def reconstruct_edges(
    facet_polygons: List[Tuple[int, Polygon]],
    planes: List,
    outline: Optional[Polygon] = None,
    step_height_m: Optional[float] = None,
    grid_m: float = 0.05,
    angle_tol_deg: float = 15.0,
    snap_tol_m: float = 0.5,
):
    """Derive clean, de-duplicated, typed roof edges from clean facets.

    The existing ``classify_edges_from_facets`` finds interior ridge/hip/valley
    edges via an **exact** ``boundary.intersection`` of adjacent facets — so if
    two neighbours are not precisely coincident on their shared edge, that edge
    is silently lost. This wrapper first **snaps every facet to a common grid**
    (``set_precision``) so shared boundaries become exactly coincident, then runs
    the existing classifier and collapses collinear runs with roof-tuned
    tolerances. Net effect: the jagged ~60-edges/roof output falls toward the
    real ~8-20 clean typed edges, with ridge/hip/valley reliably detected.

    Returns a list of ``RoofEdge`` (import kept local to avoid a module-load
    cycle and to keep this file usable without the edge stack at import time).
    """
    from shapely import set_precision
    from src.roofs.edges import (
        classify_edges_from_facets,
        merge_collinear_edges,
    )

    def _snap(poly: Optional[Polygon]) -> Optional[Polygon]:
        if poly is None:
            return None
        try:
            sp = _keep_polygon(set_precision(poly, grid_m))
            if sp is None or sp.is_empty or not sp.is_valid:
                return poly
            return sp
        except Exception:
            return poly

    snapped = [(fid, _snap(poly)) for fid, poly in facet_polygons]
    snap_outline = _snap(outline) if outline is not None else None

    edges = classify_edges_from_facets(
        snapped, planes, outline=snap_outline, step_height_m=step_height_m
    )
    edges = merge_collinear_edges(edges, angle_tol_deg=angle_tol_deg,
                                  snap_tol_m=snap_tol_m)
    logger.info("reconstruct_edges: %d typed edge(s) after merge", len(edges))
    return edges
