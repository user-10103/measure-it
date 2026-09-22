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

from src.roofs.building_select import select_building_mask
from src.roofs.mask_facets import (_outline_mask, masks_to_facets,
                                   outline_polygon)
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
    # Which candidate became the outline, and how well it fitted the target
    # footprint. None when no anchor was supplied — and that is the state in
    # which a multi-building outline goes unnoticed, so report_qc treats a
    # missing selection as a finding rather than as nothing to say.
    selection: object = None


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
    expand_to_anchor: bool = True,
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
        expand_to_anchor: union the chosen roof mask with the footprint. Tree
            and self-shadows erode the imagery mask (sawtooth eaves, missing
            wings — observed on the Holland Ln benchmark); the footprint is
            shadow-immune, so the union restores the true building extent.
            When the roof prompt finds nothing at all, the footprint alone
            becomes the outline.
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
    selection = None
    roof_mask = None
    outline_px = None
    try:
        r_masks, r_scores = _outline_predict(chip, roof_concept)
        r_masks = np.asarray([np.asarray(m, bool) for m in r_masks])
        if len(r_masks):
            r_scores = np.asarray(r_scores, dtype=float)
            idx = int(np.argmax(r_scores))
            if anchor_mask is not None and anchor_mask.any():
                # Rank by IoU against the footprint (dilated for eaves) and
                # strip blobs not connected to it. The old rule ranked by
                # `cover` alone — recall of the anchor, which a mask spanning
                # the neighbours satisfies just as well as the right one — and
                # then broke the 100%-vs-100% tie on SAM's score, which favours
                # the big blob. See src/roofs/building_select.py.
                anchor = np.asarray(anchor_mask, bool)
                px_m = abs(float(transform.a)) if transform is not None else None
                sel = select_building_mask(r_masks, r_scores, anchor, px_m)
                selection = sel
                if sel.index >= 0:
                    idx = sel.index
                    roof_mask = sel.mask      # neighbours already stripped
                else:
                    # No candidate covers the target footprint (the muddy "best
                    # 19% mask" case, typically when the FACET model doubles as the
                    # outline model). A wrong roof mask is worse than none — use the
                    # shadow-immune MS footprint as the outline directly instead of
                    # falling back to a top-score mask that's on the wrong building.
                    logger.info(
                        "no roof mask covers the footprint — using the MS "
                        "footprint as the outline (%s)",
                        "; ".join(sel.notes) or "no qualifying candidate")
                    roof_mask = anchor
            else:
                roof_mask = r_masks[idx]
    except Exception as e:  # noqa: BLE001
        logger.warning("SAM roof-outline prompt failed (%s)", e)

    # heal shadow erosion: the footprint can't be fooled by tree/self-shadow,
    # so the outline is the union of what the model saw and what the footprint
    # guarantees. No roof mask at all -> the footprint alone is the outline.
    if expand_to_anchor and anchor_mask is not None and anchor_mask.any():
        anchor = np.asarray(anchor_mask, bool)
        if roof_mask is not None:
            healed = int((anchor & ~roof_mask).sum())
            if healed:
                logger.info("outline expanded to footprint: +%d px healed "
                            "(shadow/occlusion)", healed)
            roof_mask = roof_mask | anchor
        else:
            logger.info("no roof mask — using the footprint as the outline")
            roof_mask = anchor
    if roof_mask is not None:
        outline_px = outline_polygon(roof_mask, simplify_px)
        # WHAT WE DRAW IS WHAT WE MEASURED. outline_polygon returns the LARGEST
        # polygon of the mask, while the clip below used the WHOLE mask — so a
        # mask spanning four detached buildings drew ONE and measured FOUR. The
        # facet count, every area, and every edge total carried the neighbours
        # while the diagram showed the target alone. Measured on a synthetic
        # four-building row: the report displayed 25% of what it measured, a 4x
        # overstatement with nothing on the page to reveal it. It is also why
        # 1845 Morrill St broke the perimeter-per-ksqft invariant at 30.6
        # against a floor of 56 — area from four buildings, perimeter from one.
        #
        # Clipping to the polygon we actually draw makes the two agree by
        # construction rather than by the mask happening to be single-part.
        # (It also hands regularize the real polygon instead of making
        # masks_to_facets re-derive it from a raster.)
        if outline_px is not None:
            dropped = int(roof_mask.sum()) - int(_outline_mask(
                outline_px, roof_mask.shape[:2]).sum())
            if dropped > 0.02 * roof_mask.sum():
                logger.warning(
                    "outline: %d px (%.0f%% of the roof mask) lie outside the "
                    "drawn outline and will NOT be measured — the mask is "
                    "multi-part, most likely neighbouring buildings",
                    dropped, 100 * dropped / roof_mask.sum())

    # 2. facets — "roof facet" masks -> clean partition, clipped to the outline
    # we draw (outline_px), never the raw mask.
    f_masks, f_scores = predict_masks(chip, facet_concept)
    facets, lbl = masks_to_facets(
        f_masks, f_scores, outline=(outline_px if outline_px is not None
                                    else roof_mask),
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
    return SamRoof(outline=outline_px, facets=facets, label_map=lbl,
                   georeferenced=georef, selection=selection)
