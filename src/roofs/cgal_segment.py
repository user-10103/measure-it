"""Clean GEOMETRIC facet segmentation from LiDAR — not dots/blobs.

Replaces the k-means gap-fill (which clusters by height -> convex-hull blobs) with
the best validated LiDAR method: CGAL region-growing + coplanar merge, and returns
each facet as a REGULARIZED straight-edged polygon (concave boundary -> straighten
-> clip to footprint), so the pipeline draws geometry, not point clusters.

Falls back to sequential RANSAC when the ``cgal`` package isn't installed, so it
works everywhere; CGAL (GPL, server-side) is used when present for the cleanest
planes. The geometric-drawing steps (merge, polygonize, regularize) are pure
numpy/shapely and are unit-tested.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.roofs.segment import Facet
from src.roofs.plane_fit import fit_plane_ransac

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. per-point plane labels — CGAL region-growing, or RANSAC fallback
# --------------------------------------------------------------------------- #
def _cgal_labels(points: np.ndarray, min_points: int = 25, k: int = 12):
    """CGAL region-growing plane index per point, or None if cgal unavailable."""
    try:
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Point_set_3 import Point_set_3
        from CGAL.CGAL_Point_set_processing_3 import jet_estimate_normals
        from CGAL.CGAL_Shape_detection import region_growing
    except Exception:
        return None
    X = np.asarray(points["x"], float); Y = np.asarray(points["y"], float)
    Z = np.asarray(points["z"], float)
    ox, oy = X.min(), Y.min()
    ps = Point_set_3(); ps.add_normal_map()
    for x, y, z in zip(X - ox, Y - oy, Z):
        ps.insert(Point_3(float(x), float(y), float(z)))
    jet_estimate_normals(ps, 24)
    pmap = ps.add_int_map("plane_index")
    region_growing(ps, pmap, min_points=min_points, k=k)
    return np.array([pmap.get(i) for i in ps.indices()])


def _ransac_labels(points: np.ndarray, dist_thresh: float = 0.15,
                   min_points: int = 25, max_planes: int = 20) -> np.ndarray:
    """Fallback: sequential-RANSAC plane index per point (no cgal needed)."""
    labels = np.full(len(points), -1, dtype=int)
    idx = np.arange(len(points))
    lab = 0
    for _ in range(max_planes):
        if len(idx) < min_points:
            break
        try:
            p = fit_plane_ransac(points[idx], dist_thresh=dist_thresh,
                                 min_inlier_ratio=0.05)
        except Exception:
            break
        if not getattr(p, "success", False) or p.inlier_count < min_points:
            break
        m = np.asarray(p.inlier_mask, dtype=bool)
        labels[idx[m]] = lab; lab += 1
        idx = idx[~m]
    return labels


# --------------------------------------------------------------------------- #
# 2. spatial split + coplanar merge (over-segment -> merge, the pro recipe)
# --------------------------------------------------------------------------- #
def _connected_clusters(points, labels, split_eps=1.5, min_points=25):
    """Split each plane label into spatially-connected components (index arrays)."""
    from sklearn.cluster import DBSCAN
    out = []
    for lb in np.unique(labels):
        if lb < 0:
            continue
        gi = np.where(labels == lb)[0]
        if len(gi) < min_points:
            continue
        xy = np.column_stack([points["x"][gi], points["y"][gi]])
        cl = DBSCAN(eps=split_eps, min_samples=5).fit_predict(xy)
        for c in np.unique(cl):
            if c < 0:
                continue
            comp = gi[cl == c]
            if len(comp) >= min_points:
                out.append(comp)
    return out


def _merge_coplanar(points, clusters, angle_tol_deg=12.0, adj_dist=1.5):
    """Merge clusters that BOTH face the same way (coplanar) AND touch (adjacent)."""
    from sklearn.neighbors import NearestNeighbors
    if len(clusters) <= 1:
        return clusters
    nrm, xy = [], []
    for ci in clusters:
        P = np.column_stack([points["x"][ci], points["y"][ci], points["z"][ci]])
        A = P - P.mean(0)
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        v = Vt[-1]; nrm.append((-v if v[2] < 0 else v) / (np.linalg.norm(v) + 1e-12))
        xy.append(P[:, :2])
    nn = [NearestNeighbors(n_neighbors=1).fit(p) for p in xy]
    parent = list(range(len(clusters)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    cos_tol = np.cos(np.radians(angle_tol_deg))
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if abs(nrm[i] @ nrm[j]) < cos_tol:
                continue
            if nn[i].kneighbors(xy[j])[0].min() < adj_dist:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(clusters)):
        groups.setdefault(find(i), []).append(i)
    return [np.concatenate([clusters[i] for i in g]) for g in groups.values()]


# --------------------------------------------------------------------------- #
# 3. clean GEOMETRIC polygon per facet (concave -> regularize -> clip)
# --------------------------------------------------------------------------- #
def _polygonize(sub: np.ndarray, footprint=None, ratio: float = 0.35):
    from shapely.geometry import MultiPoint
    import shapely
    mp = MultiPoint(list(zip(np.asarray(sub["x"], float),
                             np.asarray(sub["y"], float))))
    try:
        poly = shapely.concave_hull(mp, ratio=ratio)      # tighter than convex hull
    except Exception:
        poly = mp.convex_hull
    if poly is None or poly.is_empty or poly.geom_type != "Polygon":
        poly = mp.convex_hull
    if footprint is not None and not footprint.is_empty:
        try:
            clip = poly.intersection(footprint)
            if not clip.is_empty and clip.geom_type in ("Polygon", "MultiPolygon"):
                poly = clip
        except Exception:
            pass
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def segment_facets_cgal(points: np.ndarray, footprint=None, min_points: int = 25,
                        angle_tol_deg: float = 12.0, adj_dist: float = 1.5,
                        regularize: bool = True) -> List[Facet]:
    """Segment LiDAR into clean GEOMETRIC facets (regularized polygons, not dots).

    CGAL region-growing (or RANSAC fallback) -> connected clusters -> coplanar
    merge -> concave polygon -> regularize -> clip to footprint.
    """
    if points is None or len(points) < min_points:
        return []
    labels = _cgal_labels(points, min_points=min_points)
    if labels is None:
        logger.info("cgal unavailable — RANSAC fallback for facet segmentation")
        labels = _ransac_labels(points, min_points=min_points)
    clusters = _connected_clusters(points, labels, min_points=min_points)
    clusters = _merge_coplanar(points, clusters, angle_tol_deg, adj_dist)

    facets: List[Facet] = []
    for fid, ci in enumerate(clusters, start=1):
        sub = points[ci]
        try:
            plane = fit_plane_ransac(sub)
        except Exception:
            plane = None
        poly = _polygonize(sub, footprint)
        if poly is None or poly.is_empty:
            continue
        facets.append(Facet(facet_id=fid, points=sub, label=fid,
                            polygon=poly, plane=plane))

    if regularize and footprint is not None and facets:      # straight edges
        try:
            from src.roofs.facet_reconstruct import regularize_facets
            polys = regularize_facets([f.polygon for f in facets], footprint)
            for f, p in zip(facets, polys):
                if p is not None and not p.is_empty and p.is_valid:
                    f.polygon = p
        except Exception as e:  # noqa: BLE001
            logger.warning("regularize failed (%s) — keeping concave polygons", e)

    logger.info("cgal_segment: %d clean geometric facet(s)", len(facets))
    return facets
