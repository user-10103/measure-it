"""
Facet boundary-snap / tiling post-process.

Hand-drawn (and model-predicted) facet polygons don't share exact boundaries, so
the geometric edge classifier finds no interior ridge/hip/valley edges (it reads
them from boundary.intersection of adjacent facets) and the map shows white gaps.

This snaps the facets into a clean planar partition of the roof outline:
neighbouring facets share EXACT edges, gaps are filled, areas sum to the outline
-- which unblocks interior edge typing and finishes the map.

Method: snap all boundary linework to a grid (cleans tiny misalignments), then
node + polygonize the combined linework into cells. Assign each cell to the facet
it most overlaps (a gap cell, overlapping no facet, goes to the nearest facet),
then merge the cells per facet. Because every cell comes from one noded
arrangement, adjacent merged facets share exact boundary segments.
"""
import logging
from typing import List, Optional, Tuple

from shapely import set_precision
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

logger = logging.getLogger(__name__)

SNAP_GRID_M = 0.1
GAP_CLOSE_M = 0.5


def _largest(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return None


def tile_facets(
    facet_polys: List[Tuple[int, Polygon]],
    outline: Optional[Polygon] = None,
    grid_size: float = SNAP_GRID_M,
) -> List[Tuple[int, Polygon]]:
    """Snap facets into a shared-boundary partition of the outline.

    Returns a list parallel to the input (same ids, same order); a facet that
    can't be re-tiled keeps its original polygon. Never raises -- falls back to
    the input on any failure.
    """
    valid = [(fid, p) for fid, p in facet_polys
             if p is not None and not p.is_empty and p.is_valid and p.area > 0]
    if len(valid) < 2:
        return facet_polys

    if outline is None or outline.is_empty or not outline.is_valid:
        merged = unary_union([p for _, p in valid]).buffer(GAP_CLOSE_M).buffer(-GAP_CLOSE_M)
        outline = _largest(merged)
    if outline is None:
        return facet_polys

    try:
        lines = [set_precision(outline.boundary, grid_size)]
        lines += [set_precision(p.boundary, grid_size) for _, p in valid]
        noded = unary_union(lines)
        cells = [c for c in polygonize(noded) if c.area > 0]
    except Exception as e:
        logger.warning("tile_facets noding failed (%s); keeping original facets", e)
        return facet_polys
    if not cells:
        return facet_polys

    keep = outline.buffer(grid_size)
    assigned = {fid: [] for fid, _ in valid}
    for cell in cells:
        rp = cell.representative_point()
        if not keep.contains(rp):
            continue                                   # outside the roof
        best, best_ov = None, 0.0
        for fid, p in valid:
            if cell.intersects(p):
                ov = cell.intersection(p).area
                if ov > best_ov:
                    best, best_ov = fid, ov
        if best is None:                               # gap cell -> nearest facet
            best = min(valid, key=lambda fp: fp[1].distance(rp))[0]
        assigned[best].append(cell)

    snapped = {fid: _largest(unary_union(cs)) for fid, cs in assigned.items() if cs}

    result = []
    for fid, p in facet_polys:
        sp = snapped.get(fid)
        result.append((fid, sp if (sp is not None and sp.area > 0) else p))
    return result
