"""Three buildings in one chip, and only one of them is the customer's.

The scene in every test: a 200x200 chip at 0.15 m/px with three 40x40 roofs in
a row, 20 px of ground between them. The anchor is the middle one.
"""
import numpy as np
import pytest

from src.roofs.building_select import (
    MIN_COVER, anchor_connected, dilate_mask, select_building_mask,
)

PX_M = 0.15


def scene():
    """(anchor, correct_mask, three_building_mask, neighbour_only_mask)."""
    anchor = np.zeros((200, 200), bool)
    anchor[80:120, 80:120] = True                 # middle building

    correct = np.zeros((200, 200), bool)
    correct[78:122, 78:122] = True                # middle roof, eaves overhang

    three = np.zeros((200, 200), bool)
    for x0 in (20, 80, 140):                      # all three, 20px gaps
        three[80:120, x0:x0 + 40] = True

    neighbour = np.zeros((200, 200), bool)
    neighbour[80:120, 20:60] = True               # the left one only
    return anchor, correct, three, neighbour


def old_rule(masks, scores, anchor):
    """The criterion being replaced, verbatim from sam_segment.py:117-131."""
    cover = np.array([(m & anchor).sum() / float(anchor.sum()) for m in masks])
    return int(np.lexsort((np.asarray(scores, float), cover))[-1])


def test_the_old_rule_picks_the_three_building_blob():
    """cover is recall of the anchor: both masks score 1.0, score breaks the tie.

    This is the bug. Extra buildings cannot lower cover, so the criterion that
    was supposed to reject them is blind to them, and SAM scores the big
    coherent blob higher than the tight roof.
    """
    anchor, correct, three, _ = scene()
    masks = [correct, three]
    scores = [0.71, 0.88]                          # blob scores higher
    cover = [(m & anchor).sum() / anchor.sum() for m in masks]
    assert cover == [1.0, 1.0]                     # tied, so score decides
    assert old_rule(masks, scores, anchor) == 1    # picks the blob


def test_new_rule_picks_the_target_despite_the_lower_score():
    anchor, correct, three, _ = scene()
    sel = select_building_mask([correct, three], [0.71, 0.88], anchor, PX_M)
    assert sel.index == 0
    assert sel.overrode_top_score is True
    assert sel.cover == 1.0
    assert sel.spill < 0.05                        # eaves only
    assert sel.iou > 0.7   # 0.787 at the 0.75 m default


def test_detached_neighbours_are_removed_from_the_winning_mask():
    """Even if the blob is the ONLY candidate, its neighbours are stripped.

    The chosen mask becomes the outline, and masks_to_facets tiles the outline —
    so a blob passed through here is partitioned into facets across all three
    buildings and put in the customer's table.
    """
    anchor, _, three, _ = scene()
    sel = select_building_mask([three], [0.9], anchor, PX_M)
    assert sel.index == 0
    assert sel.components_dropped == 2
    assert sel.mask.sum() == 40 * 40               # exactly the middle roof
    assert not (sel.mask & ~anchor).any()


def test_a_mask_entirely_on_a_neighbour_is_refused_not_shrunk():
    """Returning the biggest blob here would be a confident wrong-building report."""
    anchor, _, _, neighbour = scene()
    kept, dropped = anchor_connected(neighbour, anchor)
    assert not kept.any() and dropped == 1
    sel = select_building_mask([neighbour], [0.99], anchor, PX_M)
    assert sel.index == -1 and sel.mask is None


def test_eave_overhang_is_not_scored_as_trespass():
    """The mirror-image error: punishing the correct mask for having eaves.

    An undilated IoU would rank a footprint-tight mask above the real roof,
    amputating exactly the overhang the eave measurements come from.
    """
    anchor, correct, _, _ = scene()
    tight = anchor.copy()
    with_allowance = select_building_mask([correct, tight], [0.8, 0.8], anchor, PX_M)
    assert with_allowance.index == 0               # the real roof, eaves and all

    undilated = select_building_mask([correct, tight], [0.8, 0.8], anchor,
                                     px_m=PX_M, eave_allowance_m=0.0)
    assert undilated.index == 1                    # would pick the amputated one


def test_under_covering_candidates_are_disqualified():
    """A tiny clean mask must not beat the roof by spilling nothing."""
    anchor, correct, _, _ = scene()
    sliver = np.zeros((200, 200), bool)
    sliver[95:100, 95:100] = True                  # 25px inside the anchor
    assert (sliver & anchor).sum() / anchor.sum() < MIN_COVER
    sel = select_building_mask([sliver, correct], [0.99, 0.5], anchor, PX_M)
    assert sel.index == 1


def test_attached_row_housing_ranks_by_iou_when_connectivity_cannot_help():
    """Townhouses share walls, so there is one component. IoU still separates."""
    anchor = np.zeros((200, 200), bool)
    anchor[80:120, 80:120] = True
    row = np.zeros((200, 200), bool)
    row[80:120, 40:160] = True                     # three attached units
    unit = np.zeros((200, 200), bool)
    unit[80:120, 80:120] = True
    kept, dropped = anchor_connected(row, anchor)
    assert dropped == 0 and kept.sum() == row.sum()      # connectivity is no help
    sel = select_building_mask([row, unit], [0.95, 0.6], anchor, PX_M)
    assert sel.index == 1
    assert sel.runner_up_iou < sel.iou


def test_no_anchor_and_no_candidates_are_reported_not_guessed():
    anchor, correct, _, _ = scene()
    assert select_building_mask([correct], [0.9], np.zeros((200, 200), bool),
                                PX_M).index == -1
    assert select_building_mask([], [], anchor, PX_M).index == -1


def test_selection_meta_is_flat_and_json_safe():
    import json
    anchor, correct, three, _ = scene()
    meta = select_building_mask([correct, three], [0.7, 0.9], anchor, PX_M).to_meta()
    assert json.loads(json.dumps(meta))["select_mask_overrode_top_score"] is True
    assert set(meta) == {
        "select_mask_cover", "select_mask_spill", "select_mask_iou",
        "select_mask_runner_up_iou", "select_mask_candidates",
        "select_mask_components_dropped", "select_mask_overrode_top_score"}


def test_dilate_is_identity_below_one_pixel():
    a = np.zeros((20, 20), bool); a[8:12, 8:12] = True
    assert (dilate_mask(a, 0) == a).all()
    assert dilate_mask(a, 2).sum() > a.sum()


def test_raising_the_eave_allowance_erodes_the_margin():
    """The knob has a cliff on BOTH sides. Pin it so the cost stays visible.

    Someone will eventually widen this to forgive a badly registered footprint.
    At 2.0 m the correct mask and an attached three-unit row are 0.05 apart —
    a coin toss that produces a confident report on the wrong building.
    """
    anchor = np.zeros((200, 200), bool)
    anchor[80:120, 80:120] = True
    correct = np.zeros((200, 200), bool)
    correct[78:122, 78:122] = True
    row = np.zeros((200, 200), bool)
    row[80:120, 40:160] = True

    def margin(e):
        a = select_building_mask([correct], [0.9], anchor, PX_M, eave_allowance_m=e)
        b = select_building_mask([row], [0.9], anchor, PX_M, eave_allowance_m=e)
        return a.iou - b.iou

    assert margin(0.45) > 0.5
    assert margin(0.75) > 0.4          # the default still has real headroom
    assert margin(1.5) < 0.2
    assert margin(2.0) < 0.1           # the cliff

    # And at zero the ranking inverts: a footprint-tight mask beats the roof.
    tight = anchor.copy()
    assert select_building_mask([correct, tight], [0.8, 0.8], anchor, PX_M,
                                eave_allowance_m=0.0).index == 1


def test_missing_pixel_scale_estimates_the_allowance_rather_than_dropping_it():
    """px_m=None must not silently become eave_allowance=0.

    Zero dilation is not a conservative default — it inverts the ranking, so a
    footprint-tight mask beats the real roof and the eaves are amputated. The
    unscaled path is reached whenever segment_roof_sam runs without a transform.
    """
    from src.roofs.building_select import _fallback_radius_px
    anchor, correct, _, _ = scene()
    tight = anchor.copy()

    assert _fallback_radius_px(anchor) == 2            # 5% of sqrt(1600)
    sel = select_building_mask([correct, tight], [0.8, 0.8], anchor, px_m=None)
    assert sel.index == 0, "unscaled path picked the footprint-tight mask"
    assert _fallback_radius_px(np.ones((10, 10), bool)) >= 1
