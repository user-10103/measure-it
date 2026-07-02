"""Sequential-RANSAC roof facet segmentation on the LiDAR POINT CLOUD.

The robust way to find roof facets in real (noisy) LiDAR is plane detection on
the points themselves — RANSAC ignores noise/outliers by construction. This is
what the professional roof-reconstruction tools (TU Delft roofer / 3DBAG, CGAL
region growing) do, and it's fundamentally different from what failed for us:

  - k-means clusters by HEIGHT -> blobs, no plane structure
  - nDSM gradient clustering -> amplifies grid noise -> over/under-segments
  - plane-intersection arrangement -> ill-conditioned ridge position

``segment_facets_planes`` repeatedly fits the dominant plane, splits its inliers
into spatially-connected facets, removes them, and repeats. Returns Facet objects
carrying points (the existing pipeline fits the plane + extracts the boundary
downstream), so it slots in as a ``segment_method``.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.roofs.segment import Facet
from src.roofs.plane_fit import fit_plane_ransac

logger = logging.getLogger(__name__)


def _xy(points: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(points["x"], dtype="float64"),
                            np.asarray(points["y"], dtype="float64")])


def segment_facets_planes(points: np.ndarray, dist_thresh: float = 0.15,
                          min_inliers: int = 60, max_planes: int = 20,
                          split_eps: float = 1.5,
                          min_inlier_ratio: float = 0.10) -> List[Facet]:
    """Segment roof points into facets by sequential RANSAC plane detection.

    Parameters
    ----------
    dist_thresh : inlier distance to the plane (m). ~0.15 suits roof LiDAR.
    min_inliers : minimum points for a facet.
    max_planes  : safety cap on iterations.
    split_eps   : DBSCAN radius (m) to split one plane into separate roof parts.
    """
    from sklearn.cluster import DBSCAN

    if points is None or len(points) < min_inliers:
        return []
    remaining = points
    facets: List[Facet] = []
    fid = 0
    for _ in range(max_planes):
        if remaining is None or len(remaining) < min_inliers:
            break
        try:
            plane = fit_plane_ransac(remaining, dist_thresh=dist_thresh,
                                     min_inlier_ratio=min_inlier_ratio)
        except Exception:
            break
        if not getattr(plane, "success", False) or plane.inlier_count < min_inliers:
            break
        mask = np.asarray(plane.inlier_mask, dtype=bool)
        if mask.sum() < min_inliers:
            break
        inliers = remaining[mask]
        # One geometric plane can appear on separate roof parts (e.g. two wings
        # with the same slope/aspect). Split spatially so each is its own facet.
        try:
            labels = DBSCAN(eps=split_eps, min_samples=5).fit_predict(_xy(inliers))
        except Exception:
            labels = np.zeros(len(inliers), dtype=int)
        for lab in sorted(set(labels)):
            if lab == -1:
                continue
            comp = inliers[labels == lab]
            if len(comp) >= min_inliers:
                fid += 1
                facets.append(Facet(facet_id=fid, points=comp, label=fid))
        remaining = remaining[~mask]

    logger.info("plane segmentation: %d facet(s) via sequential RANSAC "
                "(%d pts unassigned)", len(facets),
                0 if remaining is None else len(remaining))
    return facets
