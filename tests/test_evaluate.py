"""
Unit tests for the contract evaluation harness (src/eval/evaluate.py).

These tests build tiny in-memory COCO dicts (no network, no files) and assert
the metric and gate behavior described by the client contract.
"""

import copy

import pytest

from src.eval.evaluate import (
    AREA_TOL_PCT,
    EvalResult,
    OUTLINE_IOU_MIN,
    PITCH_TOL_DEG,
    evaluate,
)


def _rect(x0, y0, x1, y1):
    """Flat COCO ring for an axis-aligned rectangle."""
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _build_gt():
    """
    Tiny GT COCO: 1 image (100x100), one outline square (0,0)-(100,100),
    two facet squares splitting it (left/right halves) with slopes 20 and 25,
    plus two edges.
    """
    return {
        "images": [{"id": 1, "address_id": "addr-001", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,  # roof_polygon outline
                "segmentation": [_rect(0, 0, 100, 100)],
                "attributes": {"slope_deg": None},
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,  # facet (left half)
                "segmentation": [_rect(0, 0, 50, 100)],
                "attributes": {"slope_deg": 20.0},
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 2,  # facet (right half)
                "segmentation": [_rect(50, 0, 100, 100)],
                "attributes": {"slope_deg": 25.0},
            },
            {
                "id": 4,
                "image_id": 1,
                "category_id": 8,  # eave
                "segmentation": [[0, 0, 100, 0]],
                "attributes": {"slope_deg": None},
            },
            {
                "id": 5,
                "image_id": 1,
                "category_id": 10,  # ridge
                "segmentation": [[0, 50, 100, 50]],
                "attributes": {"slope_deg": None},
            },
        ],
        "categories": [
            {"id": 1, "name": "roof_polygon"},
            {"id": 2, "name": "facet"},
            {"id": 8, "name": "eave"},
            {"id": 10, "name": "ridge"},
        ],
    }


def test_case_a_perfect_match():
    """pred == GT: F1 == 1.0, outline IoU ~ 1.0, area/pitch ~ 0, passed True."""
    gt = _build_gt()
    pred = copy.deepcopy(gt)

    result = evaluate(pred, gt)

    assert isinstance(result, EvalResult)
    assert result.n_images == 1
    assert result.facet_f1 == pytest.approx(1.0)
    assert result.facet_precision == pytest.approx(1.0)
    assert result.facet_recall == pytest.approx(1.0)
    assert result.facet_mean_iou == pytest.approx(1.0)
    assert result.outline_iou == pytest.approx(1.0, abs=1e-6)
    assert result.area_err_pct == pytest.approx(0.0, abs=1e-6)
    assert result.edge_len_err_pct == pytest.approx(0.0, abs=1e-6)
    assert result.pitch_err_deg == pytest.approx(0.0, abs=1e-6)
    assert result.passed is True
    assert all(result.gates.values())


def test_case_b_shifted_facet_drops_recall():
    """One facet shifted so its best IoU < 0.5: recall drops, F1 < 1.0."""
    gt = _build_gt()
    pred = copy.deepcopy(gt)

    # Shift the right-half facet far to the right so it barely overlaps GT.
    # New rect (90,0)-(140,100): overlap with GT right half (50..100) is x in
    # [90,100] -> area 1000; union = 5000+5000-1000 = 9000 -> IoU ~0.11 < 0.5.
    for ann in pred["annotations"]:
        if ann["id"] == 3:
            ann["segmentation"] = [_rect(90, 0, 140, 100)]

    result = evaluate(pred, gt)

    assert result.facet_recall < 1.0
    assert result.facet_f1 < 1.0
    # One TP (left facet), one FN (unmatched GT right facet), one FP (shifted).
    assert result.facet_recall == pytest.approx(0.5)
    assert not result.gates["facet_f1"]
    assert result.passed is False


def test_case_c_pitch_off_fails_gate():
    """A facet slope off by 10 deg makes the pitch gate fail (err ~10 > 5)."""
    gt = _build_gt()
    pred = copy.deepcopy(gt)

    for ann in pred["annotations"]:
        if ann["id"] == 2:  # left facet GT slope 20 -> 30
            ann["attributes"]["slope_deg"] = 30.0

    result = evaluate(pred, gt)

    # Two matched facets: diffs 10 and 0 -> mean 5.0. Make it clearly fail by
    # also nudging the other; first confirm the single-off case is ~5.
    assert result.facet_f1 == pytest.approx(1.0)  # geometry unchanged
    assert result.pitch_err_deg == pytest.approx(5.0)

    # Now push both facets off by 10 so the mean is unambiguously > tolerance.
    pred2 = copy.deepcopy(gt)
    for ann in pred2["annotations"]:
        if ann["category_id"] == 2 and ann["attributes"]["slope_deg"] is not None:
            ann["attributes"]["slope_deg"] += 10.0
    result2 = evaluate(pred2, gt)
    assert result2.pitch_err_deg == pytest.approx(10.0)
    assert result2.pitch_err_deg > PITCH_TOL_DEG
    assert not result2.gates["pitch_err_deg"]
    assert result2.passed is False


def test_ids_filter_restricts_evaluation():
    """The ids filter scores only the requested address_ids."""
    gt = _build_gt()
    # Add a second image/address with a wildly wrong prediction.
    gt["images"].append(
        {"id": 2, "address_id": "addr-002", "width": 100, "height": 100}
    )
    gt["annotations"].extend(
        [
            {
                "id": 10,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [_rect(0, 0, 100, 100)],
                "attributes": {"slope_deg": None},
            },
            {
                "id": 11,
                "image_id": 2,
                "category_id": 2,
                "segmentation": [_rect(0, 0, 100, 100)],
                "attributes": {"slope_deg": 20.0},
            },
        ]
    )
    pred = copy.deepcopy(gt)
    # Corrupt addr-002's facet so it would fail if scored.
    for ann in pred["annotations"]:
        if ann["id"] == 11:
            ann["segmentation"] = [_rect(0, 0, 5, 5)]

    # Restrict to addr-001 only -> corruption ignored, should pass.
    res_filtered = evaluate(pred, gt, ids=["addr-001"])
    assert res_filtered.n_images == 1
    assert res_filtered.passed is True

    # Restrict to addr-002 only -> corruption scored, facet IoU < 0.5.
    res_bad = evaluate(pred, gt, ids=["addr-002"])
    assert res_bad.n_images == 1
    assert res_bad.facet_recall < 1.0
