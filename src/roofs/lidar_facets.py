"""Facet boundaries drawn from the LiDAR, not from an imagery mask.

THE EXPERIMENT THIS EXISTS TO RUN
---------------------------------
``plane_segment.segment_facets_planes`` — sequential RANSAC plus a DBSCAN split
into spatially-connected parts, "the method the pro roof tools (roofer/3DBAG)
use" — was written on 2026-07-02 at 21:33 and has never once been wired. Its
commit touches only the module and its tests; the message says "pipeline fits
plane + boundary downstream", and that follow-up never came.

Sixteen minutes later ``pearl_segment`` was written to fix a defect in it. Sixty
two minutes after that, the whole point-clustering family was abandoned:

    "The point-clustering track fragments (56 facets on a simple roof) [...]
     No clustering."                                          -- 2049627

Those 56 speckle facets are PEARL's. A prototype of the plain sequential-RANSAC
path on the same class of roof produced 32 planes that merged to 13 coherent
facets. So the method that survives to production -- SAM draws the boundary,
LiDAR may only annotate it -- rests on a failure that belonged to a DIFFERENT
algorithm, and the simpler one was condemned by association without ever running.

This module is that experiment, wired behind MEASURE_IT_LIDAR_FACETS=1 so it can
be measured against the SAM facets on the same roof rather than argued about.

WHY concave_hull AND NOT extract_inlier_boundary
------------------------------------------------
plane_fit.extract_inlier_boundary takes a CONVEX hull, which is the
"convex-hull blobs" failure the segmentation docstrings complain about: an
L-shaped or notched facet comes back as its bounding envelope, swallowing roof
that belongs to a neighbour. shapely.concave_hull follows the point set.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# concave_hull ratio: 0 hugs the points (noisy), 1 is the convex hull (blobs).
CONCAVE_RATIO = 0.35
MIN_FACET_AREA_M2 = 2.0


def lidar_facets_from_points(points: np.ndarray,
                             outline=None,
                             ratio: float = CONCAVE_RATIO,
                             min_area_m2: float = MIN_FACET_AREA_M2,
                             regularize: bool = True) -> List:
    """Roof point cloud -> facets whose boundaries come from the POINTS.

    Returns Facet objects carrying both ``points`` and ``polygon``, so they drop
    into the same downstream annotate/merge/edge path the SAM facets use. An
    empty list means "this did not work here" — the caller keeps its own facets.
    """
    from shapely import concave_hull
    from shapely.geometry import MultiPoint

    from src.roofs.plane_segment import segment_facets_planes
    from src.roofs.segment import Facet

    if points is None or len(points) == 0:
        return []
    xyz = np.asarray(points, float)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return []

    raw = segment_facets_planes(xyz)
    if not raw:
        logger.info("lidar facets: plane segmentation found nothing")
        return []

    out: List = []
    for f in raw:
        pts = getattr(f, "points", None)
        if pts is None or len(pts) < 3:
            continue
        try:
            poly = concave_hull(MultiPoint([tuple(p[:2]) for p in pts]),
                                ratio=ratio)
        except Exception as e:  # noqa: BLE001 - one bad facet must not fail the roof
            logger.info("lidar facets: hull failed for a cluster (%s)", e)
            continue
        if poly is None or poly.is_empty or poly.geom_type != "Polygon":
            continue
        if outline is not None:
            try:
                poly = poly.intersection(outline)
            except Exception:  # noqa: BLE001
                pass
            if poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        if poly.geom_type != "Polygon" or poly.area < min_area_m2:
            continue
        out.append(Facet(facet_id=len(out) + 1, points=pts, polygon=poly))

    if regularize and out:
        # The same straightening the imagery facets get (mask_facets uses it),
        # so a comparison between the two paths is not confounded by one side
        # having straight edges and the other raw hulls.
        try:
            from src.roofs.facet_reconstruct import regularize_facets
            fp = outline if outline is not None else _union([f.polygon for f in out])
            if fp is not None:
                straight = regularize_facets([f.polygon for f in out], fp)
                for f, p in zip(out, straight):
                    if p is not None and not p.is_empty and p.geom_type == "Polygon":
                        f.polygon = p
        except Exception as e:  # noqa: BLE001 - regularization is cosmetic
            logger.info("lidar facets: regularize skipped (%s)", e)

    logger.info("lidar facets: %d plane cluster(s) -> %d facet(s) with boundaries",
                len(raw), len(out))
    return out


def _union(polys) -> Optional[object]:
    from shapely.ops import unary_union
    try:
        u = unary_union([p for p in polys if p is not None and not p.is_empty])
        return u if not u.is_empty else None
    except Exception:  # noqa: BLE001
        return None
