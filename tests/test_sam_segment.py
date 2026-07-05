"""Tests for src/roofs/sam_segment.py — SAM3 imagery front-end (injected predict)."""
import numpy as np

from src.roofs.sam_segment import segment_roof_sam


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def _fake_predictor(roof, facets, roof_score=0.95, facet_scores=None):
    """Build a predict_masks(chip, concept) that returns canned masks per concept."""
    fs = facet_scores if facet_scores is not None else [0.9] * len(facets)

    def predict_masks(chip, concept):
        if concept == "roof":
            return np.asarray([roof]), np.asarray([roof_score])
        return np.asarray(facets), np.asarray(fs)
    return predict_masks


def test_outline_and_facets_pixel_space():
    h = w = 80
    roof = _rect(h, w, 20, 60, 20, 60)
    f1 = _rect(h, w, 20, 40, 20, 60)          # top half
    f2 = _rect(h, w, 40, 60, 20, 60)          # bottom half
    chip = np.zeros((h, w, 3), np.uint8)
    res = segment_roof_sam(_fake_predictor(roof, [f1, f2]), chip,
                           regularize=False, min_area_frac=0.0)
    assert res.outline is not None and res.outline.area > 0
    assert len(res.facets) == 2
    assert not res.georeferenced
    # facets are disjoint and together roughly cover the roof
    inter = res.facets[0].polygon.intersection(res.facets[1].polygon).area
    assert inter < 1.0


def test_facets_clipped_to_outline():
    # a facet mask spilling past the roof outline is clipped to it
    h = w = 100
    roof = _rect(h, w, 30, 70, 30, 70)
    spill = _rect(h, w, 35, 65, 35, 95)       # extends to x=95, past roof x=70
    chip = np.zeros((h, w, 3), np.uint8)
    res = segment_roof_sam(_fake_predictor(roof, [spill]), chip,
                           regularize=False, min_area_frac=0.0)
    assert len(res.facets) == 1
    assert res.facets[0].polygon.bounds[2] <= 71   # clipped to roof outline


def test_georeference_scales_coords():
    from affine import Affine
    h = w = 40
    roof = _rect(h, w, 10, 30, 10, 30)
    f1 = _rect(h, w, 10, 30, 10, 30)
    chip = np.zeros((h, w, 3), np.uint8)
    # 0.5 m/px, origin at (1000, 2000); note rasterio Affine has negative e (y down)
    tr = Affine(0.5, 0.0, 1000.0, 0.0, -0.5, 2000.0)
    res = segment_roof_sam(_fake_predictor(roof, [f1]), chip, transform=tr,
                           regularize=False, min_area_frac=0.0)
    assert res.georeferenced
    minx, miny, maxx, maxy = res.outline.bounds
    assert 1000.0 <= minx <= 1020.0            # world X near origin+10*0.5
    assert maxx <= 1016.0                       # ~ origin + 30*0.5
    # area scales by (0.5 m/px)^2 vs the ~20x20 px pixel box (~400 px -> ~100 m^2)
    assert 60.0 < res.outline.area < 130.0


def test_separate_outline_and_facet_predictors():
    # zero-shot model handles "roof" (outline); fine-tuned handles "roof facet"
    h = w = 60
    roof = _rect(h, w, 10, 50, 10, 50)
    f1 = _rect(h, w, 10, 30, 10, 50)
    f2 = _rect(h, w, 30, 50, 10, 50)
    chip = np.zeros((h, w, 3), np.uint8)
    calls = {"outline": [], "facet": []}

    def outline_predict(chip, concept):      # the "zero-shot" model
        calls["outline"].append(concept)
        return np.asarray([roof]), np.asarray([0.95])

    def facet_predict(chip, concept):        # the "fine-tuned" model
        calls["facet"].append(concept)
        return np.asarray([f1, f2]), np.asarray([0.9, 0.9])

    res = segment_roof_sam(facet_predict, chip, predict_outline=outline_predict,
                           regularize=False, min_area_frac=0.0)
    assert calls["outline"] == ["roof"]          # outline model got ONLY the roof prompt
    assert calls["facet"] == ["roof facet"]       # facet model got ONLY the facet prompt
    assert res.outline is not None and len(res.facets) == 2


def test_anchor_picks_target_roof_not_neighbor():
    # THE 909 BUG: loose crop shows the neighbor's bigger roof, which outscores
    # the target. The footprint anchor must override the score ranking.
    h = w = 100
    neighbor = _rect(h, w, 0, 30, 10, 90)      # wide band at the top, score 0.97
    target = _rect(h, w, 50, 85, 30, 70)       # the actual house, score 0.80
    facet = _rect(h, w, 50, 85, 30, 70)
    chip = np.zeros((h, w, 3), np.uint8)
    anchor = _rect(h, w, 55, 80, 35, 65)       # building footprint (inside target)

    def outline_predict(chip, concept):
        return np.asarray([neighbor, target]), np.asarray([0.97, 0.80])

    def facet_predict(chip, concept):
        return np.asarray([facet]), np.asarray([0.9])

    res = segment_roof_sam(facet_predict, chip, predict_outline=outline_predict,
                           anchor_mask=anchor, regularize=False, min_area_frac=0.0)
    minx, miny, maxx, maxy = res.outline.bounds
    assert miny >= 45 and maxy <= 90          # the TARGET roof, not the top band
    # without the anchor, the neighbor wins (documents the old behavior)
    res2 = segment_roof_sam(facet_predict, chip, predict_outline=outline_predict,
                            regularize=False, min_area_frac=0.0)
    assert res2.outline.bounds[3] <= 35        # top band


def test_anchor_ignored_when_nothing_covers_it():
    # anchor off in the weeds -> fall back to top score, don't return nothing
    h = w = 60
    roof = _rect(h, w, 10, 40, 10, 50)
    anchor = _rect(h, w, 50, 58, 50, 58)       # no candidate covers this
    res = segment_roof_sam(_fake_predictor(roof, [roof]), np.zeros((h, w, 3), np.uint8),
                           anchor_mask=anchor, regularize=False, min_area_frac=0.0)
    assert res.outline is not None             # graceful fallback


def test_no_roof_mask_still_returns_facets():
    # roof prompt yields nothing -> no outline, but facets still produced (unclipped)
    h = w = 60
    f1 = _rect(h, w, 10, 50, 10, 50)
    chip = np.zeros((h, w, 3), np.uint8)

    def predict_masks(chip, concept):
        if concept == "roof":
            return np.zeros((0, h, w), bool), np.zeros((0,))
        return np.asarray([f1]), np.asarray([0.9])

    res = segment_roof_sam(predict_masks, chip, regularize=False, min_area_frac=0.0)
    assert res.outline is None
    assert len(res.facets) == 1
