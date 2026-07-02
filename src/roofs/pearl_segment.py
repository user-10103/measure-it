"""PEARL-style energy-minimization multi-plane fitting for roof facets.

Globally assigns roof points to planes by minimizing
    E = sum_i data(i, label_i)                     # point fits its plane
      + w * sum_(i,j) [label_i != label_j]         # spatial smoothness (Potts)
      + labelcost * (#distinct planes used)        # MDL: fewest facets
over candidate planes proposed by RANSAC. This fixes sequential-RANSAC's failures:
smoothness snaps facet boundaries to real creases (kills slope-band bleed), and
the label cost absorbs noise slivers (correct facet count).

Ref: Isack & Boykov, "Energy-Based Geometric Multi-Model Fitting" (IJCV 2012).

The optimizer is a dependency-free ICM (iterated conditional modes) that runs and
tests anywhere; if ``pygco`` is installed, alpha-expansion is used instead (a
better global optimum). Candidates come from our own RANSAC — no CGAL needed.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.roofs.segment import Facet
from src.roofs.plane_fit import fit_plane_ransac

logger = logging.getLogger(__name__)


def generate_candidates(points: np.ndarray, n_candidates: int = 40,
                        dist_thresh: float = 0.15, min_inliers: int = 40) -> list:
    """Oversample candidate planes by peeling RANSAC planes (keep many)."""
    planes = []
    rem = points
    for _ in range(n_candidates):
        if rem is None or len(rem) < min_inliers:
            break
        try:
            p = fit_plane_ransac(rem, dist_thresh=dist_thresh, min_inlier_ratio=0.03)
        except Exception:
            break
        if not getattr(p, "success", False) or p.inlier_count < min_inliers:
            break
        planes.append(p)
        rem = rem[~np.asarray(p.inlier_mask, dtype=bool)]
    return planes


def estimate_normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-point surface normals via local PCA (smallest-eigenvector of the
    neighbourhood covariance), oriented upward. This is the ingredient Efficient
    RANSAC (Schnabel 2007) adds over plain distance RANSAC — it lets us separate
    spatially-adjacent facets that face different directions."""
    from sklearn.neighbors import NearestNeighbors
    xyz = np.column_stack([np.asarray(points["x"], float),
                           np.asarray(points["y"], float),
                           np.asarray(points["z"], float)])
    kk = min(k + 1, len(xyz))
    _, idx = NearestNeighbors(n_neighbors=kk).fit(xyz).kneighbors(xyz)
    normals = np.zeros((len(xyz), 3))
    for i in range(len(xyz)):
        nbr = xyz[idx[i]]
        c = nbr - nbr.mean(0)
        w, v = np.linalg.eigh(c.T @ c)
        n = v[:, 0]                                     # smallest eigenvalue -> normal
        normals[i] = -n if n[2] < 0 else n             # orient up
    return normals


def _plane_normal(p) -> np.ndarray:
    n = np.array([-p.a, -p.b, 1.0])
    return n / (np.linalg.norm(n) + 1e-12)


def _data_cost(points: np.ndarray, planes: list, normals: np.ndarray = None,
               normal_weight: float = 0.5) -> np.ndarray:
    """Data term = |vertical residual| + normal-disagreement penalty (Efficient-
    RANSAC-style). Normal term separates facets by ORIENTATION, not just height."""
    x = np.asarray(points["x"], float); y = np.asarray(points["y"], float)
    z = np.asarray(points["z"], float)
    R = np.empty((len(points), len(planes)), dtype="float32")
    for j, p in enumerate(planes):
        dist = np.abs(p.a * x + p.b * y + p.c - z)
        if normals is not None and normal_weight > 0:
            agree = np.abs(normals @ _plane_normal(p))   # 1 = aligned, 0 = perp
            dist = dist + normal_weight * (1.0 - agree)   # penalise misaligned normals
        R[:, j] = dist
    return R


def _knn_neighbors(points: np.ndarray, k: int = 8):
    from sklearn.neighbors import NearestNeighbors
    xy = np.column_stack([np.asarray(points["x"], float),
                          np.asarray(points["y"], float)])
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy))).fit(xy)
    _, idx = nn.kneighbors(xy)
    return idx[:, 1:]                                   # drop self


def _icm(R: np.ndarray, neigh: np.ndarray, labels: np.ndarray,
         smooth: float, label_cost: float, sweeps: int = 6) -> np.ndarray:
    """Iterated conditional modes: data + Potts smoothness + label (MDL) cost."""
    N, L = R.shape
    for _ in range(sweeps):
        used = np.bincount(labels, minlength=L)
        changed = 0
        for i in range(N):
            nl = labels[neigh[i]]
            # smoothness cost of assigning label l = smooth * (#neighbors != l)
            smooth_cost = smooth * (nl.shape[0] - np.bincount(nl, minlength=L))
            # label (MDL) cost: pay label_cost only to OPEN a currently-unused plane
            mdl = np.where(used > 0, 0.0, label_cost)
            mdl[labels[i]] = 0.0                        # its current label is open
            total = R[i] + smooth_cost + mdl
            best = int(np.argmin(total))
            if best != labels[i]:
                used[labels[i]] -= 1
                used[best] += 1
                labels[i] = best
                changed += 1
        if changed == 0:
            break
    return labels


def _alpha_expansion(R, edges, labels, smooth):
    """Global optimizer via pygco alpha-expansion (data + Potts smoothness).

    NOTE: pygco's cut_general_graph has NO label-cost parameter, so the MDL /
    "fewest facets" term is enforced by the caller pruning under-supported planes
    between iterations (the refit loop drops planes below min_inliers).
    """
    import pygco
    unary = np.ascontiguousarray((R * 50.0).astype(np.int32))
    ew = np.full(len(edges), max(1, int(smooth * 50)), dtype=np.int32)
    L = R.shape[1]
    pw = (1 - np.eye(L)).astype(np.int32)                # Potts smoothness
    out = pygco.cut_general_graph(
        np.ascontiguousarray(edges.astype(np.int32)), ew, unary, pw,
        n_iter=-1, algorithm="expansion",
        init_labels=np.ascontiguousarray(labels.astype(np.int32)))
    return np.asarray(out).astype(int)


def segment_facets_pearl(points: np.ndarray, n_candidates: int = 40,
                         dist_thresh: float = 0.15, smooth: float = 0.4,
                         label_cost: float = 3.0, iters: int = 3,
                         min_inliers: int = 40, split_eps: float = 1.5,
                         normal_weight: float = 0.5,
                         use_graphcut: bool = True) -> List[Facet]:
    """Segment roof points into facets by PEARL energy minimization.

    Data term uses distance + normal-agreement (Efficient-RANSAC-style) so facets
    are separated by ORIENTATION, not just height."""
    from sklearn.cluster import DBSCAN
    if points is None or len(points) < min_inliers:
        return []
    planes = generate_candidates(points, n_candidates, dist_thresh, min_inliers)
    if not planes:
        return []

    normals = estimate_normals(points)
    neigh = _knn_neighbors(points)
    R = _data_cost(points, planes, normals, normal_weight)
    labels = R.argmin(1)
    for _ in range(iters):
        if use_graphcut:
            try:
                edges = np.array([(i, int(j)) for i in range(len(neigh))
                                  for j in neigh[i] if i < j])
                labels = _alpha_expansion(R, edges, labels, smooth)
            except Exception as _ge:
                logger.info("graph-cut unavailable (%s) — ICM", _ge)
                labels = _icm(R, neigh, labels, smooth, label_cost)
        else:
            labels = _icm(R, neigh, labels, smooth, label_cost)
        # re-fit each used plane to its assigned points
        new = []
        remap = {}
        for lab in np.unique(labels):
            pts = points[labels == lab]
            if len(pts) < min_inliers:
                continue
            try:
                p = fit_plane_ransac(pts, dist_thresh=dist_thresh, min_inlier_ratio=0.2)
                if p.success:
                    remap[lab] = len(new); new.append(p)
            except Exception:
                pass
        if not new:
            break
        planes = new
        # relabel to compact plane set + recompute data cost (dist + normals)
        R = _data_cost(points, planes, normals, normal_weight)
        labels = R.argmin(1)

    # build facets; spatially split each plane into connected roof parts
    facets: List[Facet] = []
    fid = 0
    for lab in np.unique(labels):
        pts = points[labels == lab]
        if len(pts) < min_inliers:
            continue
        xy = np.column_stack([pts["x"], pts["y"]])
        cl = DBSCAN(eps=split_eps, min_samples=5).fit_predict(xy)
        for c in sorted(set(cl)):
            if c == -1:
                continue
            comp = pts[cl == c]
            if len(comp) >= min_inliers:
                fid += 1
                facets.append(Facet(facet_id=fid, points=comp, label=fid))
    logger.info("PEARL segmentation: %d candidate plane(s) -> %d facet(s)",
                len(planes), len(facets))
    return facets
