"""SAM 3 imagery front-end: roof outline + facet polygons in one pass.

The "revamp" engine — promote SAM 3 to the primary imagery source:
  * ``"roof"`` prompt      -> whole-roof OUTLINE polygon (zero-shot; excellent).
  * ``"roof facet"`` prompt -> per-facet masks -> clean partition (fine-tuned).

Boundaries come from imagery (SAM 3); PITCH + sloped area come from LiDAR
downstream (see ``fuse_facets.fuse_geometric_facets``). This module only owns
the imagery step: raw SAM masks -> georeferenced facet geometry.

**Decoupled from the SAM 3 API on purpose.** The caller injects a
``predict_masks(chip, concept) -> (masks[N,H,W] bool, scores[N])`` callable, so
this module has no torch / sam3 dependency and is fully unit-testable. Wire your
real SAM 3 predictor in at the call site.

Coordinates: SAM masks are pixel (x=col, y=row). If a chip ``transform``
(rasterio ``Affine``) is given, polygons are mapped to the chip's world CRS so
edge lengths / areas are metric; otherwise they stay in pixel space (viz/tests).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.roofs.mask_facets import masks_to_facets, outline_polygon
from src.roofs.segment import Facet

logger = logging.getLogger(__name__)

# predict_masks(chip, concept) -> (masks: N x H x W bool, scores: N floats)
PredictFn = Callable[[np.ndarray, str], Tuple[np.ndarray, np.ndarray]]


@dataclass
class SamRoof:
    """Result of the SAM imagery pass."""
    outline: object = None                 # shapely Polygon (world or pixel CRS)
    facets: List[Facet] = field(default_factory=list)   # facet polygons, same CRS
    label_map: Optional[np.ndarray] = None  # int partition (pixel space) for viz
    georeferenced: bool = False            # True if mapped to world CRS


def _to_world(poly, transform):
    """Map a pixel-space (x=col, y=row) shapely polygon to world coords via an
    affine transform (rasterio Affine: x = a*col + b*row + c, y = d*col+e*row+f)."""
    if poly is None or transform is None:
        return poly
    from shapely.affinity import affine_transform
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    return affine_transform(poly, [a, b, d, e, c, f])   # [a, b, d, e, xoff, yoff]


def segment_roof_sam(
    predict_masks: PredictFn,
    chip: np.ndarray,
    transform=None,
    *,
    predict_outline: Optional[PredictFn] = None,
    anchor_mask: Optional[np.ndarray] = None,
    roof_concept: str = "roof",
    facet_concept: str = "roof facet",
    score_thr: float = 0.65,
    iou_thr: float = 0.5,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.9,
    simplify_px: float = 2.0,
    fill_to_outline: bool = True,
    regularize: bool = True,
    snap_px: float = 3.0,
) -> SamRoof:
    """Run SAM 3 for roof outline + facet polygons.

    Args:
        predict_masks: FACET predictor ``(chip, concept) -> (masks[N,H,W] bool,
            scores[N])`` — use the FINE-TUNED SAM 3 here.
        chip: H x W x 3 image.
        transform: optional rasterio ``Affine`` -> output in world CRS (metric).
        predict_outline: optional separate predictor for the roof OUTLINE — pass
            ZERO-SHOT (base) SAM 3 here. If None, the facet predictor is used for
            the outline too (single-model mode).
        anchor_mask: optional H x W bool mask of the TARGET building footprint
            (pixel space). On a loose crop several roofs are visible and the
            top-scoring "roof" mask can be a NEIGHBOR's — the anchor picks the
            roof mask that actually covers the target instead.
        roof_concept / facet_concept: the two text prompts.
        score_thr / iou_thr / min_area_frac / simplify_px / regularize / snap_px:
            forwarded to ``masks_to_facets`` for the facet partition.

    Returns:
        ``SamRoof`` with ``outline`` (Polygon), ``facets`` (List[Facet]),
        ``label_map`` (pixel partition), ``georeferenced``.
    """
    # 1. roof outline — top-scoring "roof" mask.
    # Use predict_outline if given (the design: ZERO-SHOT base SAM 3 for the
    # outline, which it nails, and the fine-tuned model for facets only —
    # fine-tuning on facets can drift the "roof" concept). Falls back to the
    # facet predictor when a separate outline model isn't supplied.
    _outline_predict = predict_outline or predict_masks
    roof_mask = None
    outline_px = None
    try:
        r_masks, r_scores = _outline_predict(chip, roof_concept)
        r_masks = np.asarray([np.asarray(m, bool) for m in r_masks])
        if len(r_masks):
            r_scores = np.asarray(r_scores, dtype=float)
            idx = int(np.argmax(r_scores))
            if anchor_mask is not None and anchor_mask.any():
                # fraction of the target footprint each candidate covers
                anchor = np.asarray(anchor_mask, bool)
                cover = np.array([(m & anchor).sum() / float(anchor.sum())
                                  for m in r_masks])
                best = int(np.lexsort((r_scores, cover))[-1])
                if cover[best] >= 0.2:
                    if best != idx:
                        logger.info(
                            "outline anchor override: mask %d (cover %.0f%%, "
                            "score %.2f) over top-score mask %d (cover %.0f%%)",
                            best, 100 * cover[best], r_scores[best],
                            idx, 100 * cover[idx])
                    idx = best
                else:
                    logger.warning(
                        "no roof mask covers the target footprint (best "
                        "%.0f%%) — falling back to top score", 100 * cover.max())
            roof_mask = r_masks[idx]
            outline_px = outline_polygon(roof_mask, simplify_px)
    except Exception as e:  # noqa: BLE001
        logger.warning("SAM roof-outline prompt failed (%s)", e)

    # 2. facets — "roof facet" masks -> clean partition, clipped to the outline
    f_masks, f_scores = predict_masks(chip, facet_concept)
    facets, lbl = masks_to_facets(
        f_masks, f_scores, outline=roof_mask,
        score_thr=score_thr, iou_thr=iou_thr, min_area_frac=min_area_frac,
        max_area_frac=max_area_frac, simplify_px=simplify_px,
        fill_to_outline=fill_to_outline, regularize=regularize, snap_px=snap_px,
    )

    # 3. georeference (pixel -> world) if a transform is supplied
    georef = transform is not None
    if georef:
        outline_px = _to_world(outline_px, transform)
        for f in facets:
            f.polygon = _to_world(f.polygon, transform)

    logger.info("segment_roof_sam: outline=%s, %d facet(s)%s",
                "yes" if outline_px is not None else "none", len(facets),
                " (world CRS)" if georef else " (pixel)")
    return SamRoof(outline=outline_px, facets=facets, label_map=lbl, georeferenced=georef)
