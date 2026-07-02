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


def ndsm_facets(ndsm: np.ndarray, transform, simplify_m: float = 0.5,
                **seg_kw) -> List[dict]:
    """Extract clean facet polygons + planes from an nDSM raster.

    Returns a list of dicts: ``{polygon, a, b, c, slope_deg, aspect_deg,
    area_px}`` with polygon in the raster's world CRS.
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
    logger.info("ndsm_facets: %d facet(s) from height surface", len(facets))
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
