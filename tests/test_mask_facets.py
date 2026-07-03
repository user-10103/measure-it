"""Tests for src/roofs/mask_facets.py — SAM3 mask -> clean facet polygons."""
import numpy as np

from src.roofs.mask_facets import masks_to_facets


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_duplicate_masks_collapse_via_nms():
    # two near-identical high-score masks of the same facet -> 1 facet
    h = w = 64
    a = _rect(h, w, 10, 50, 10, 50)
    b = _rect(h, w, 11, 51, 11, 51)          # ~IoU 0.9 with a
    facets, lbl = masks_to_facets([a, b], [0.9, 0.85], iou_thr=0.5,
                                  regularize=False, min_area_frac=0.0)
    assert len(facets) == 1
    assert (lbl > 0).sum() > 0


def test_two_partial_overlap_partition_to_two_disjoint():
    # two overlapping masks -> partition into 2 mutually-exclusive facets
    h = w = 64
    a = _rect(h, w, 10, 40, 10, 40)          # higher score claims the overlap
    b = _rect(h, w, 30, 60, 30, 60)
    facets, lbl = masks_to_facets([a, b], [0.9, 0.7], iou_thr=0.9,
                                  regularize=False, min_area_frac=0.0)
    assert len(facets) == 2
    # partition is disjoint: the two facet polygons must not overlap in area
    assert facets[0].polygon.intersection(facets[1].polygon).area < 1.0
    # higher-score facet keeps the contested overlap region
    assert facets[0].polygon.area > facets[1].polygon.area


def test_score_threshold_drops_weak_mask():
    h = w = 64
    strong = _rect(h, w, 10, 50, 10, 50)
    weak = _rect(h, w, 5, 15, 55, 62)
    facets, _ = masks_to_facets([strong, weak], [0.9, 0.4],
                                score_thr=0.65, regularize=False, min_area_frac=0.0)
    assert len(facets) == 1


def test_outline_clip_removes_spillover():
    # a mask spilling outside the roof outline gets clipped to it
    from shapely.geometry import box
    h = w = 80
    roof = box(20, 20, 60, 60)               # pixel-coord outline
    spill = _rect(h, w, 25, 55, 25, 75)      # extends past x=60 onto "grass"
    facets, lbl = masks_to_facets([spill], [0.9], outline=roof,
                                  regularize=False, min_area_frac=0.0)
    assert len(facets) == 1
    # nothing kept outside the outline
    assert lbl[:, 61:].sum() == 0
    assert facets[0].polygon.bounds[2] <= 61  # max-x clipped to the roof


def test_min_area_filters_specks():
    h = w = 64
    big = _rect(h, w, 5, 60, 5, 60)
    speck = _rect(h, w, 0, 2, 0, 2)          # 4 px
    facets, _ = masks_to_facets([big, speck], [0.9, 0.8],
                                min_area_frac=0.01, regularize=False)
    assert len(facets) == 1


def test_empty_and_all_below_threshold():
    assert masks_to_facets([], [])[0] == []
    m = np.zeros((16, 16), bool); m[2:6, 2:6] = True
    facets, _ = masks_to_facets([m], [0.1], score_thr=0.65)
    assert facets == []
