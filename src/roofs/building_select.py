"""Pick the TARGET building's roof mask out of a chip that shows several.

THE DEFECT THIS REPLACES
------------------------
``sam_segment`` ranked outline candidates by

    cover = (mask & anchor).sum() / anchor.sum()

which is *recall of the target footprint and nothing else*. A mask spanning the
target plus two neighbours covers 100% of the anchor. So does the correct mask.
``lexsort((scores, cover))`` then broke the tie on SAM's score, and the large
multi-building blob usually scores higher than the tight single roof. Adding
neighbours could never lower ``cover``, so over-inclusion was invisible to the
criterion that was supposed to catch it.

It then got worse downstream: ``masks_to_facets(outline=roof_mask,
fill_to_outline=True)`` TILES the outline, filling unclaimed interior with the
nearest facet. A three-building outline does not merely fail to exclude the
neighbours — it actively partitions them into facets and puts them in the table.

WHAT REPLACES IT
----------------
Two stages, both of which can only ever remove wrongly-included area:

1. ANCHOR-CONNECTED COMPONENT. Keep only the parts of a candidate that are
   physically connected to the target footprint. Separate buildings are separate
   blobs, so this alone resolves the common case and needs no tuning.
2. IoU AGAINST THE DILATED ANCHOR. For neighbours that genuinely touch (row
   housing, an attached unit), connectivity cannot separate them, so rank by
   overlap in BOTH directions. A three-unit mask scores ~0.33 against a one-unit
   anchor; the correct mask scores high.

The anchor is dilated by ``eave_allowance_m`` first, because a roof legitimately
overhangs its footprint and an undilated IoU would punish the correct mask for
having eaves — the mirror-image error of the one being fixed.

Pure numpy + scipy: no model, no network, no imagery. Fully testable offline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# A roof overhangs its footprint, and the footprint itself (MS/parcel) carries
# registration error. Dilate by this before measuring spill so eaves are not
# scored as trespass.
#
# THIS IS A CLIFF, NOT A COMFORT MARGIN. Measured on a 6 m building at 0.15 m/px
# (IoU of the correct mask vs an attached three-unit row):
#     0.00 m -> 0.826 vs 0.333, but a footprint-TIGHT mask wins at 1.000:
#               the eaves get amputated, and eaves are what the report measures
#     0.45 m -> 0.924 vs 0.364   margin 0.56
#     0.75 m -> 0.787 vs 0.380   margin 0.41   <- default
#     1.50 m -> 0.557 vs 0.408   margin 0.15
#     2.00 m -> 0.466 vs 0.418   margin 0.05   effectively a coin toss
# Dilating too little amputates the eaves; dilating too much dissolves the gap
# that distinguishes the target from the neighbour attached to it. Raising this
# "to be safe" walks toward the wrong-building failure, not away from it, and
# test_raising_the_eave_allowance_erodes_the_margin pins that so the cost is
# visible to whoever next reaches for the knob.
EAVE_ALLOWANCE_M = 0.75

MIN_COVER = 0.2          # below this a candidate is not this building at all
SPILL_WARN = 0.35        # flag a winner that is still mostly off-footprint


@dataclass
class Selection:
    """Which candidate won, and the evidence for second-guessing it."""
    index: int                       # -1 == no candidate qualified
    mask: Optional[np.ndarray]
    cover: float                     # fraction of anchor the winner covers
    spill: float                     # fraction of winner outside dilated anchor
    iou: float                       # IoU vs dilated anchor — the rank key
    n_candidates: int
    runner_up_iou: float = 0.0
    components_dropped: int = 0      # disconnected blobs removed from the winner
    overrode_top_score: bool = False
    notes: List[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        """Flat fields for report meta / QC. Absence of these is itself a signal."""
        return {"select_mask_cover": round(float(self.cover), 4),
                "select_mask_spill": round(float(self.spill), 4),
                "select_mask_iou": round(float(self.iou), 4),
                "select_mask_runner_up_iou": round(float(self.runner_up_iou), 4),
                "select_mask_candidates": int(self.n_candidates),
                "select_mask_components_dropped": int(self.components_dropped),
                "select_mask_overrode_top_score": bool(self.overrode_top_score)}


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation by a disc of ``radius_px`` (identity when radius < 1)."""
    if radius_px < 1:
        return np.asarray(mask, bool)
    from scipy import ndimage
    r = int(radius_px)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (yy * yy + xx * xx) <= r * r
    return ndimage.binary_dilation(np.asarray(mask, bool), structure=disc)


def anchor_connected(mask: np.ndarray, anchor: np.ndarray) -> tuple:
    """Keep only the blobs of ``mask`` that touch ``anchor``.

    Returns ``(kept_mask, n_dropped)``. This is what removes the detached
    neighbour that ``cover`` could never see, and it needs no threshold: two
    buildings with ground between them are two components.
    """
    from scipy import ndimage
    m = np.asarray(mask, bool)
    a = np.asarray(anchor, bool)
    if not m.any():
        return m, 0
    lab, n = ndimage.label(m)
    # NO early return for n == 1. A single-component mask sitting entirely on
    # the neighbour is precisely the confident-wrong-building case, and an
    # `if n <= 1: return m, 0` shortcut hands it straight back unexamined. The
    # anchor test has to run for every candidate, however few blobs it has.
    touching = set(np.unique(lab[a & m]))
    touching.discard(0)
    if not touching:
        # Nothing in this candidate reaches the target at all — it is entirely
        # a neighbour. Return empty rather than the largest blob: a confident
        # wrong building is the failure mode that ships looking clean.
        return np.zeros_like(m), n
    kept = np.isin(lab, list(touching))
    return kept, n - len(touching)


# An eave is roughly this fraction of a building's width. sqrt(footprint area)
# approximates that width, so the product is a scale-free stand-in for the eave
# allowance when no transform is available: ~0.6 m on a 12 m house, ~1.5 m on a
# 30 m commercial building. Rough, but bounded on both sides — unlike 0, which
# is not an approximation of the allowance but a reversal of the ranking.
EAVE_FRAC_OF_WIDTH = 0.05


def _fallback_radius_px(anchor: np.ndarray) -> int:
    """Dilation radius in pixels when the ground sample distance is unknown."""
    return max(1, int(round(EAVE_FRAC_OF_WIDTH * np.sqrt(float(anchor.sum())))))


def select_building_mask(masks: Sequence[np.ndarray],
                         scores: Optional[Sequence[float]],
                         anchor: np.ndarray,
                         px_m: Optional[float] = None,
                         eave_allowance_m: float = EAVE_ALLOWANCE_M,
                         min_cover: float = MIN_COVER) -> Selection:
    """Choose the candidate mask that is the TARGET building.

    Args:
        masks: candidate H x W bool masks from the outline prompt.
        scores: model confidences, used only to report whether the geometric
            pick disagreed with the top-scoring one.
        anchor: H x W bool footprint of the target building.
        px_m: ground sample distance in metres per pixel. When None the
            allowance is estimated from the anchor's own size instead (see
            ``_fallback_radius_px``) — NOT skipped. Skipping it inverts the
            ranking: with no dilation a footprint-tight mask scores IoU 1.0 and
            beats the real roof, amputating the eaves the report measures.
            I first wrote that the ranking "still works" without a scale; the
            wiring test disproved it immediately. Pass px_m when a transform
            exists; the estimate is a floor, not an equal.
    """
    a = np.asarray(anchor, bool)
    n = len(masks)
    if not a.any() or n == 0:
        return Selection(-1, None, 0.0, 0.0, 0.0, n,
                         notes=["no anchor" if not a.any() else "no candidates"])

    if px_m:
        radius = int(round(eave_allowance_m / px_m))
    else:
        radius = _fallback_radius_px(a)
        logger.info("building select: no pixel scale — eave allowance estimated "
                    "at %d px from the footprint's own size", radius)
    a_d = dilate_mask(a, radius)
    a_sum = float(a.sum())

    best = -1
    best_iou = -1.0
    second_iou = 0.0
    best_cover = best_spill = 0.0
    best_dropped = 0
    best_mask = None

    for i, m_raw in enumerate(masks):
        m, dropped = anchor_connected(np.asarray(m_raw, bool), a)
        m_sum = float(m.sum())
        if m_sum == 0:
            continue
        cover = float((m & a).sum()) / a_sum
        spill = float((m & ~a_d).sum()) / m_sum
        inter = float((m & a_d).sum())
        union = float((m | a_d).sum())
        iou = inter / union if union else 0.0
        if cover < min_cover:
            continue
        if iou > best_iou:
            second_iou = best_iou if best_iou > 0 else second_iou
            best, best_iou, best_mask = i, iou, m
            best_cover, best_spill, best_dropped = cover, spill, dropped
        elif iou > second_iou:
            second_iou = iou

    if best < 0:
        return Selection(-1, None, 0.0, 0.0, 0.0, n,
                         notes=[f"no candidate covers >={min_cover:.0%} of the "
                                f"footprint"])

    overrode = False
    if scores is not None and len(scores) == n:
        top = int(np.argmax(np.asarray(scores, dtype=float)))
        overrode = top != best
        if overrode:
            logger.info(
                "building select: geometry chose mask %d (IoU %.2f, cover %.0f%%, "
                "spill %.0f%%) over top-scoring mask %d — the old rule took the "
                "high-score mask whenever cover tied at 100%%",
                best, best_iou, 100 * best_cover, 100 * best_spill, top)

    sel = Selection(best, best_mask, best_cover, best_spill, best_iou, n,
                    runner_up_iou=max(second_iou, 0.0),
                    components_dropped=best_dropped,
                    overrode_top_score=overrode)
    if best_dropped:
        logger.info("building select: dropped %d detached blob(s) from the "
                    "chosen mask (neighbouring buildings)", best_dropped)
        sel.notes.append(f"{best_dropped} detached blob(s) removed")
    if best_spill > SPILL_WARN:
        logger.warning(
            "building select: winner is %.0f%% outside the target footprint "
            "(+%.1f m eave allowance). The chip may show attached neighbours "
            "that connectivity cannot separate; facets will be tiled across "
            "whatever the outline covers.", 100 * best_spill, eave_allowance_m)
        sel.notes.append(f"spill {best_spill:.0%} above {SPILL_WARN:.0%}")
    return sel
