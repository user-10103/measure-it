"""Facet extraction from the nDSM height surface.

The roof's structure is far crisper in the *height* map than in the flat photo:
each facet is a smooth planar region, and the ridges/hips/valleys are the crease
lines where the surface gradient (slope direction) changes. So we segment by
**gradient direction**, not by height — pixels on the same plane share a gradient
vector, and cluster boundaries fall exactly on the crease (ridge/hip) lines.

This is the principled route to "facets as they are": clean planar regions bounded
by the real ridge lines, instead of k-means blobs of similar-height points.

``ndsm_facets`` returns polygons (in the nDSM raster CRS) + a fitted plane per
facet. ``visualize`` renders the height heatmap with facet boundaries so the
result can be eyeballed before wiring it into the pipeline.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def segment_by_gradient(ndsm: np.ndarray, sigma: float = 1.5,
                        max_planes: int = 8, merge_deg: float = 20.0,
                        min_region_px: int = 40,
                        valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Label each nDSM pixel with a facet id by clustering surface gradient.

    Returns an int label array (0 = background/invalid, 1..N = facets).
    """
    from scipy.ndimage import gaussian_filter, label as cc_label
    from sklearn.cluster import KMeans

    h, w = ndsm.shape
    if valid_mask is None:
        valid_mask = np.isfinite(ndsm) & (ndsm > 0.3)      # above-ground roof pixels
    sm = gaussian_filter(np.where(valid_mask, ndsm, 0.0), sigma)
    gy, gx = np.gradient(sm)

    idx = np.flatnonzero(valid_mask)
    if idx.size < min_region_px:
        return np.zeros((h, w), dtype=np.int32)
    feat = np.stack([gx.ravel()[idx], gy.ravel()[idx]], 1)

    k = int(min(max_planes, max(2, idx.size // 400)))
    km = KMeans(k, n_init=5, random_state=0).fit(feat)
    # merge clusters whose mean gradient points ~the same way (same plane)
    cen = km.cluster_centers_
    remap = list(range(k))
    for i in range(k):
        for j in range(i):
            gi, gj = cen[i], cen[j]
            ni, nj = np.hypot(*gi), np.hypot(*gj)
            if ni < 1e-6 or nj < 1e-6:
                continue
            cos = float(np.dot(gi, gj) / (ni * nj))
            if cos > math.cos(math.radians(merge_deg)) and abs(ni - nj) < 0.05:
                remap[i] = remap[j]
                break
    grad_lab = np.zeros(h * w, dtype=np.int32)
    grad_lab[idx] = [remap[c] + 1 for c in km.labels_]
    grad_lab = grad_lab.reshape(h, w)

    # split spatially-disconnected same-gradient regions; drop slivers
    out = np.zeros((h, w), dtype=np.int32)
    nxt = 1
    for g in range(1, k + 1):
        comp, n = cc_label(grad_lab == g)
        for c in range(1, n + 1):
            m = comp == c
            if int(m.sum()) < min_region_px:
                continue
            out[m] = nxt
            nxt += 1
    return out


def _fit_plane(ndsm: np.ndarray, mask: np.ndarray, transform):
    """Least-squares plane z = a*x + b*y + c over a region's world coords."""
    rows, cols = np.nonzero(mask)
    xs, ys = transform * (cols + 0.5, rows + 0.5)          # pixel centres -> world
    xs, ys = np.asarray(xs), np.asarray(ys)
    zs = ndsm[rows, cols].astype(float)
    A = np.c_[xs, ys, np.ones_like(xs)]
    try:
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        a, b, c = map(float, coef)
    except Exception:
        a = b = 0.0
        c = float(zs.mean())
    slope = math.degrees(math.atan(math.hypot(a, b)))
    aspect = math.degrees(math.atan2(-b, -a)) % 360.0
    return a, b, c, slope, aspect


def _keep_polygon(g):
    """Coerce a set-op result to its single largest Polygon component."""
    if g is None or g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return g
    polys = [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
    return max(polys, key=lambda p: p.area) if polys else None


def _aspect_diff(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _merge_coplanar_adjacent(facets: List[dict], aspect_tol: float = 25.0,
                             slope_tol: float = 8.0, adj_len_m: float = 0.5) -> List[dict]:
    """Union facets that are ADJACENT and share ~the same plane — i.e. one
    physical facet that noise split into pieces. Keeps the largest piece's plane."""
    from shapely.ops import unary_union
    n = len(facets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = facets[i], facets[j]
            if _aspect_diff(fi["aspect_deg"], fj["aspect_deg"]) > aspect_tol:
                continue
            if abs(fi["slope_deg"] - fj["slope_deg"]) > slope_tol:
                continue
            try:
                shared = fi["polygon"].boundary.intersection(
                    fj["polygon"].boundary).length
            except Exception:
                shared = 0.0
            if shared >= adj_len_m:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: List[dict] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(facets[members[0]])
            continue
        merged = _keep_polygon(unary_union([facets[m]["polygon"] for m in members]))
        if merged is None:
            out.extend(facets[m] for m in members)
            continue
        big = max(members, key=lambda m: facets[m]["polygon"].area)
        f = dict(facets[big])
        f["polygon"] = merged
        f["area_px"] = sum(facets[m]["area_px"] for m in members)
        out.append(f)
    return out


def _merge_slivers(facets: List[dict], min_area_m2: float) -> List[dict]:
    """Merge any facet below min_area_m2 into the neighbour it shares the most
    boundary with (or drop it if it has no neighbour)."""
    from shapely.ops import unary_union
    facets = [dict(f) for f in facets]
    while True:
        small = [i for i, f in enumerate(facets) if f["polygon"].area < min_area_m2]
        if not small or len(facets) <= 1:
            break
        i = min(small, key=lambda i: facets[i]["polygon"].area)
        best_j, best_len = None, 0.0
        for j in range(len(facets)):
            if j == i:
                continue
            try:
                s = facets[i]["polygon"].boundary.intersection(
                    facets[j]["polygon"].boundary).length
            except Exception:
                s = 0.0
            if s > best_len:
                best_len, best_j = s, j
        if best_j is None:
            facets.pop(i)
            continue
        merged = _keep_polygon(unary_union([facets[best_j]["polygon"],
                                            facets[i]["polygon"]]))
        if merged is not None:
            facets[best_j]["polygon"] = merged
            facets[best_j]["area_px"] += facets[i]["area_px"]
        facets.pop(i)
    return facets


def cleanup_facets(facets: List[dict], min_area_m2: float = 3.0,
                   regularize: bool = True) -> List[dict]:
    """Post-process raw nDSM facets: merge noise-split coplanar fragments, merge
    remaining slivers, then straighten boundaries."""
    if not facets:
        return facets
    facets = _merge_coplanar_adjacent(facets)
    facets = _merge_slivers(facets, min_area_m2)
    if regularize and facets:
        try:
            from shapely.ops import unary_union
            from src.roofs.facet_reconstruct import regularize_facets
            footprint = unary_union([f["polygon"] for f in facets])
            reg = regularize_facets([f["polygon"] for f in facets], footprint)
            for f, p in zip(facets, reg):
                if p is not None and not p.is_empty and p.is_valid:
                    f["polygon"] = p
        except Exception as e:  # noqa: BLE001
            logger.warning("regularize failed (%s) — keeping unregularized", e)
    facets.sort(key=lambda f: -f["area_px"])
    return facets


def ndsm_facets(ndsm: np.ndarray, transform, simplify_m: float = 0.5,
                clean: bool = True, min_area_m2: float = 3.0,
                **seg_kw) -> List[dict]:
    """Extract clean facet polygons + planes from an nDSM raster.

    Returns a list of dicts: ``{polygon, a, b, c, slope_deg, aspect_deg,
    area_px}`` with polygon in the raster's world CRS. With ``clean=True`` the
    raw regions are de-fragmented, de-slivered, and boundary-regularized.
    """
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shp_shape

    labels = segment_by_gradient(ndsm, **seg_kw)
    facets: List[dict] = []
    for geom, val in rio_shapes(labels.astype(np.int32), mask=labels > 0,
                                transform=transform):
        poly = shp_shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        if simplify_m:
            poly = poly.simplify(simplify_m, preserve_topology=True)
        mask = labels == int(val)
        a, b, c, slope, aspect = _fit_plane(ndsm, mask, transform)
        facets.append({"polygon": poly, "a": a, "b": b, "c": c,
                       "slope_deg": round(slope, 1),
                       "aspect_deg": round(aspect, 1),
                       "area_px": int(mask.sum())})
    facets.sort(key=lambda f: -f["area_px"])
    n_raw = len(facets)
    if clean:
        facets = cleanup_facets(facets, min_area_m2=min_area_m2)
    logger.info("ndsm_facets: %d raw -> %d facet(s) from height surface",
                n_raw, len(facets))
    return facets


def visualize(ndsm: np.ndarray, transform, out_png: str,
              facets: Optional[List[dict]] = None, title: str = "nDSM facets"):
    """Render the height heatmap with facet boundaries overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.transform import array_bounds

    if facets is None:
        facets = ndsm_facets(ndsm, transform)
    h, w = ndsm.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(np.where(ndsm > 0.3, ndsm, np.nan),
              extent=(left, right, bottom, top), cmap="turbo", origin="upper")
    for f in facets:
        xs, ys = f["polygon"].exterior.xy
        ax.plot(xs, ys, color="white", linewidth=2)
    ax.set_title(f"{title} — {len(facets)} facets")
    ax.set_axis_off()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png
