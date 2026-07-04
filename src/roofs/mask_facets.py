"""Turn SAM 3's raw promptable instance masks into a clean facet polygon map.

SAM 3 (fine-tuned on "roof facet") emits N *independent, overlapping* soft
instance masks per roof — not a partition. This module collapses those into a
clean, mutually-exclusive set of straight-edged facet polygons:

    threshold (score) -> mask-NMS (drop overlapping dupes) -> greedy argmax
    partition (each pixel -> highest-score mask) -> clip to roof outline ->
    polygonize -> regularize (straight edges).

Everything works in **image pixel coordinates** (x=col, y=row), so the output
polygons overlay directly on the NAIP chip. Georeferencing (pixel -> metric CRS
via the chip Affine) is a downstream step, same as the LiDAR path.

Reuses ``regularize_facets`` from facet_reconstruct so the imagery facets get
the same straightening as the LiDAR facets in ``cgal_segment``.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.roofs.segment import Facet

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# mask ops
# --------------------------------------------------------------------------- #
def _as_bool_masks(masks) -> np.ndarray:
    """N x H x W bool. Accepts list / ndarray / torch tensor of bool|prob|logit."""
    arr = np.asarray([np.asarray(m) for m in masks])
    if arr.size == 0:
        return arr.reshape(0, 0, 0).astype(bool)
    if arr.dtype != bool:
        # probabilities or logits -> threshold at 0.5 / 0.0
        thr = 0.5 if (arr.min() >= 0.0 and arr.max() <= 1.0) else 0.0
        arr = arr > thr
    return arr


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def _nms(masks: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy mask NMS. Returns kept indices in descending-score order.

    A candidate is dropped if it overlaps an already-kept mask with IoU > iou_thr
    OR is >80% contained in one (a small facet swallowed by a big duplicate).
    """
    order = list(np.argsort(-scores))
    kept: List[int] = []
    for i in order:
        mi = masks[i]
        ai = mi.sum()
        drop = False
        for j in kept:
            mj = masks[j]
            inter = np.logical_and(mi, mj).sum()
            if inter == 0:
                continue
            iou = inter / (np.logical_or(mi, mj).sum())
            contain = inter / max(ai, 1)
            if iou > iou_thr or contain > 0.80:
                drop = True
                break
        if not drop:
            kept.append(i)
    return kept


def _partition(masks: np.ndarray, kept: Sequence[int], hw: Tuple[int, int]) -> np.ndarray:
    """Greedy argmax partition: each pixel -> the highest-score mask covering it.

    ``kept`` is already in descending-score order, so assigning in that order and
    only writing not-yet-claimed pixels realises the per-pixel score-argmax.
    Returns an int label map (0 = background, 1..K = facet id).
    """
    lbl = np.zeros(hw, dtype=np.int32)
    for fid, idx in enumerate(kept, start=1):
        lbl[np.logical_and(masks[idx], lbl == 0)] = fid
    return lbl


# --------------------------------------------------------------------------- #
# raster -> polygon
# --------------------------------------------------------------------------- #
def _region_to_polygon(mask: np.ndarray, simplify_px: float):
    """Largest connected polygon of a boolean region, in pixel (x=col,y=row)."""
    from rasterio import features
    from shapely.geometry import shape as shp_shape

    polys = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask):
        if val != 1:
            continue
        p = shp_shape(geom)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        polys.append(p)
    if not polys:
        return None
    poly = max(polys, key=lambda g: g.area)
    if simplify_px and simplify_px > 0:
        poly = poly.simplify(simplify_px, preserve_topology=True)
    if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def outline_polygon(mask, simplify_px: float = 2.0):
    """Largest polygon of a single boolean region (e.g. a SAM 'roof' mask), in
    pixel coords. Public entry for the roof-outline path."""
    return _region_to_polygon(np.asarray(mask, dtype=bool), simplify_px)


def _outline_mask(outline, hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Rasterise a roof-outline polygon (pixel coords) to a bool mask, or pass a
    mask through unchanged. Returns None if no outline given."""
    if outline is None:
        return None
    if isinstance(outline, np.ndarray):
        return outline.astype(bool)
    from rasterio import features
    m = features.rasterize([(outline, 1)], out_shape=hw, dtype=np.uint8)
    return m.astype(bool)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def masks_to_facets(
    masks,
    scores,
    outline=None,
    score_thr: float = 0.65,
    iou_thr: float = 0.5,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.9,
    simplify_px: float = 2.0,
    fill_to_outline: bool = True,
    regularize: bool = True,
    snap_px: float = 3.0,
) -> Tuple[List[Facet], np.ndarray]:
    """Collapse overlapping SAM3 instance masks into clean facet polygons.

    Args:
        masks:  N x H x W (bool | prob | logit); list / ndarray / tensor.
        scores: length-N confidences.
        outline: optional roof outline to clip to — a shapely Polygon in PIXEL
            coords (x=col, y=row) or a bool mask. Removes off-roof spillover and
            enables edge regularization. If None, no clip (partition + NMS +
            min-area still de-duplicate heavily).
        score_thr:  drop masks below this score first (kills weak / false pos).
        iou_thr:    mask-NMS overlap threshold (higher = keep more).
        min_area_frac: drop facets smaller than this fraction of the roof area
            (outline area if given, else image area).
        simplify_px: Douglas-Peucker tolerance in pixels.
        regularize: straighten edges via facet_reconstruct.regularize_facets
            (requires a polygon outline).
        snap_px: regularization snap grid, in pixels.

    Returns:
        (facets, label_map). ``facets`` carry pixel-space ``polygon``; ``points``
        is None (imagery path). ``label_map`` is the int partition for viz.
    """
    masks = _as_bool_masks(masks)
    scores = np.asarray(scores, dtype=float)
    if masks.ndim != 3 or len(masks) == 0:
        return [], np.zeros((1, 1), np.int32)
    hw = masks.shape[1:]

    # 1. score threshold
    keep0 = np.where(scores >= score_thr)[0]
    if len(keep0) == 0:
        return [], np.zeros(hw, np.int32)
    masks, scores = masks[keep0], scores[keep0]

    # rasterise the outline once (used by the area filters + clipping below)
    omask = _outline_mask(outline, hw)
    roof_px = float(omask.sum()) if omask is not None else float(hw[0] * hw[1])

    # 1b. drop whole-roof masks. SAM often emits a mask spanning (nearly) the
    # entire roof alongside the individual facets; in the score-argmax partition
    # that super-mask swallows every facet, collapsing the roof to one region.
    # Remove masks larger than max_area_frac of the roof — but never all of them
    # (a genuinely single-facet roof keeps its one mask).
    if max_area_frac and roof_px > 0 and len(masks) > 1:
        areas = masks.reshape(len(masks), -1).sum(axis=1).astype(float)
        small = areas <= max_area_frac * roof_px
        if small.any():
            masks, scores = masks[small], scores[small]

    # 2. mask-NMS
    kept = _nms(masks, scores, iou_thr)

    # 3. greedy argmax partition
    lbl = _partition(masks, kept, hw)

    # 4. clip to roof outline (drop off-roof spillover)
    if omask is not None:
        lbl[~omask] = 0

    # 4b. tile the outline: fill unclaimed interior with the nearest facet
    # (constrained Voronoi) so facets partition the roof with no gaps. Requires
    # an outline (the tiling target) and at least one facet to expand from.
    if fill_to_outline and omask is not None and (lbl > 0).any():
        holes = omask & (lbl == 0)
        if holes.any():
            try:
                from scipy.ndimage import distance_transform_edt
                # for each pixel, index of the nearest non-background (facet) pixel
                inds = distance_transform_edt(lbl == 0, return_distances=False,
                                              return_indices=True)
                nearest = lbl[tuple(inds)]
                lbl = np.where(holes, nearest, lbl)
            except Exception as e:  # noqa: BLE001
                logger.warning("outline tiling fill skipped (%s)", e)

    # 5. drop tiny facets
    roof_px = float(omask.sum()) if omask is not None else float(hw[0] * hw[1])
    min_px = max(1, int(min_area_frac * roof_px))
    fids = [f for f in np.unique(lbl) if f != 0 and (lbl == f).sum() >= min_px]

    # 6. polygonize each facet region
    outline_poly = outline if (outline is not None and not isinstance(outline, np.ndarray)) else None
    facets: List[Facet] = []
    for new_id, fid in enumerate(sorted(fids), start=1):
        poly = _region_to_polygon(lbl == fid, simplify_px)
        if poly is None or poly.is_empty:
            continue
        facets.append(Facet(facet_id=new_id, points=None, label=int(fid), polygon=poly))

    # 7. regularize (straight edges) — reuse the LiDAR-path regularizer.
    # If the outline came in as a mask, derive its polygon so regularize still runs.
    if regularize and outline_poly is None and omask is not None:
        outline_poly = _region_to_polygon(omask, simplify_px)
    if regularize and outline_poly is not None and facets:
        try:
            from src.roofs.facet_reconstruct import regularize_facets
            polys = regularize_facets([f.polygon for f in facets], outline_poly, snap_m=snap_px)
            for f, p in zip(facets, polys):
                if p is not None and not p.is_empty and p.is_valid:
                    f.polygon = p
        except Exception as e:  # noqa: BLE001 — mirror cgal_segment: keep raw on failure
            logger.warning("regularize failed (%s) — keeping raster polygons", e)

    logger.info("mask_facets: %d raw masks -> %d clean facet(s)", len(masks), len(facets))
    return facets, lbl


# --------------------------------------------------------------------------- #
# viz (partition is non-overlapping, unlike the raw-mask overlay)
# --------------------------------------------------------------------------- #
def render_facets(image_rgb: np.ndarray, facets: List[Facet], label_map=None,
                  out_path: Optional[str] = None, alpha: float = 0.55, title: str = ""):
    """Overlay the clean partition + polygon edges on the chip. Saves if out_path."""
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    img = np.asarray(image_rgb)
    ov = img.astype(float).copy()
    if label_map is not None:
        cols = (np.asarray(colormaps["tab20"].colors) * 255)
        for k, fid in enumerate([f for f in np.unique(label_map) if f != 0]):
            ov[label_map == fid] = (1 - alpha) * ov[label_map == fid] + alpha * cols[k % len(cols)]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img); ax[0].set_title("NAIP chip"); ax[0].axis("off")
    ax[1].imshow(ov.astype(np.uint8))
    for f in facets:                                   # draw straight edges
        geoms = f.polygon.geoms if f.polygon.geom_type == "MultiPolygon" else [f.polygon]
        for g in geoms:
            x, y = g.exterior.xy
            ax[1].plot(x, y, "-", lw=1.6, color="k")
    ax[1].set_title(f"{title} | {len(facets)} facets"); ax[1].axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
        return out_path
    plt.show()
    return None
