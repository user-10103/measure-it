"""Geometric facet fusion: shapes from IMAGERY, pitch from LiDAR.

The lesson from the point-clustering attempts: clustering noisy LiDAR points into
facets is fragile and fragments (56 speckle-facets on a simple roof). But the
facet *shapes* are clearly visible in the imagery — so take the BOUNDARIES from
the image (a model's facet polygons, or classical line detection) and use LiDAR
only to MEASURE each facet's plane (pitch/aspect). Then draw clean straight-edged
polygons. "Draw where visible, measure in 3D."

fuse_geometric_facets: attach a LiDAR-fit plane to each image facet, regularize
the boundary to clean geometry. No point-clustering — the boundaries come from
the visible structure, LiDAR just supplies the slope.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.roofs.segment import Facet

logger = logging.getLogger(__name__)


def fuse_geometric_facets(image_facets: List[Facet], lidar_points,
                          footprint=None, regularize: bool = True,
                          min_points: int = 10) -> List[Facet]:
    """Fuse imagery facet SHAPES with LiDAR PITCH into clean geometric facets.

    Parameters
    ----------
    image_facets : list[Facet]
        Facets whose ``.polygon`` is the shape seen in imagery (from the RF-DETR
        model, or classical line detection). ``.points`` may be None.
    lidar_points : structured array (x, y, z) or None
        Roof LiDAR points, in the same metric CRS as the polygons. Used only to
        fit each facet's plane (pitch/aspect) — NOT to find boundaries.
    footprint : shapely Polygon or None
        Roof outline; used to regularize boundaries and as the tiling reference.
    """
    from src.roofs.plane_fit import fit_plane_ransac
    from src.roofs.model_segment import assign_lidar_points

    facets = [f for f in image_facets if f.polygon is not None
              and not f.polygon.is_empty]
    if not facets:
        return []

    # 1. attach LiDAR points to each image facet (point-in-polygon)
    if lidar_points is not None and len(lidar_points) > 0:
        facets = assign_lidar_points(facets, lidar_points, min_points=min_points)

    # 2. fit a plane per facet from its LiDAR points -> pitch/aspect (measurement)
    from src.roofs.metrics import compute_slope_deg, compute_aspect_bin
    for f in facets:
        if f.points is not None and f.count >= min_points:
            try:
                p = fit_plane_ransac(f.points)
                if p.success:
                    f.plane = p
                    f.slope_deg = compute_slope_deg(p)
                    f.aspect_bin = compute_aspect_bin(f.slope_deg, p)
            except Exception:
                pass

    # 3. regularize boundaries -> clean straight-edged geometric polygons
    if regularize and footprint is not None and facets:
        try:
            from src.roofs.facet_reconstruct import regularize_facets
            polys = regularize_facets([f.polygon for f in facets], footprint)
            for f, poly in zip(facets, polys):
                if poly is not None and not poly.is_empty and poly.is_valid:
                    f.polygon = poly
        except Exception as e:  # noqa: BLE001
            logger.warning("regularize failed (%s) — keeping raw polygons", e)

    logger.info("fuse_geometric_facets: %d facet(s) — shapes from imagery, "
                "pitch from LiDAR", len(facets))
    return facets
