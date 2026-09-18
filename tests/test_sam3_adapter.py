"""The adapter exists to stop evaluate() being fed data it silently misreads.

Every test here is a failure that produced a plausible number, not an error.
No cv2/pycocotools/torch: these cover the schema translation, which is where
all three silent failures live.
"""
import pytest

from src.eval.evaluate import CAT_FACET, CAT_OUTLINE, evaluate
from src.eval.sam3_adapter import (
    address_id_for, assert_scoreable, gt_to_eval_coco, masks_to_pred_coco,
)

SQ = [[0, 0, 40, 0, 40, 40, 0, 40]]          # 40x40 box
SQ2 = [[50, 50, 90, 50, 90, 90, 50, 90]]     # disjoint second box


def prep_gt(**kw):
    """GT exactly as prep_sam3_facets.py writes it: facet=1, outline=2."""
    d = {"images": [{"id": 1, "file_name": "b2__roof_a.png",
                     "height": 128, "width": 128}],
         "annotations": [
             {"id": 1, "image_id": 1, "category_id": 1, "segmentation": SQ},
             {"id": 2, "image_id": 1, "category_id": 2, "segmentation": SQ2}],
         "categories": [{"id": 1, "name": "roof facet"},
                        {"id": 2, "name": "roof"}]}
    d.update(kw)
    return d


def test_raw_prep_gt_scores_zero_images_and_passes_three_gates():
    """The bug being fixed. Untranslated GT is not an error — it is a PASS.

    No address_id -> 0 images scored -> area/edge errors average empty lists to
    0.0 -> their gates pass. A gate report reading '0.00% area error' on a run
    that scored nothing at all.
    """
    r = evaluate({"images": [], "annotations": []}, prep_gt())
    assert r.n_images == 0
    assert r.area_err_pct == 0.0 and r.edge_len_err_pct == 0.0
    assert r.gates["area_err_pct"] and r.gates["edge_len_err_pct"]
    assert sum(r.gates.values()) == 3


def test_category_ids_are_inverted_between_the_two_dialects():
    """prep writes facet=1/outline=2; evaluate declares outline=1/facet=2."""
    assert CAT_OUTLINE == 1 and CAT_FACET == 2
    gt = gt_to_eval_coco(prep_gt())
    facets = [a for a in gt["annotations"] if a["category_id"] == CAT_FACET]
    outlines = [a for a in gt["annotations"] if a["category_id"] == CAT_OUTLINE]
    assert len(facets) == 1 and len(outlines) == 1
    # The facet is the 40x40 box, NOT the one that was sitting at id 2.
    assert facets[0]["segmentation"][0][:4] == [0, 0, 40, 0]


def test_refuses_to_guess_the_facet_class():
    """A corpus with no facet-named category must raise, never keep-everything.

    'Keep everything' is how outlines get admitted as facets and inflate counts.
    """
    bad = prep_gt(categories=[{"id": 1, "name": "roof_polygon"},
                              {"id": 2, "name": "building"}])
    with pytest.raises(ValueError, match="Refusing to guess"):
        gt_to_eval_coco(bad)


def test_refuses_when_redialect_yields_no_facets():
    gt = prep_gt(annotations=[{"id": 1, "image_id": 1, "category_id": 2,
                               "segmentation": SQ2}])
    with pytest.raises(RuntimeError, match="0 facet annotations"):
        gt_to_eval_coco(gt)


def test_address_id_collision_refused_on_both_sides():
    """evaluate.index_images does out[addr] = id — last writer wins, silently."""
    gt = prep_gt(images=[{"id": 1, "file_name": "a/roof.png", "height": 8, "width": 8},
                         {"id": 2, "file_name": "a/roof.png", "height": 8, "width": 8}])
    with pytest.raises(ValueError, match="collision"):
        gt_to_eval_coco(gt)
    with pytest.raises(ValueError, match="collision"):
        masks_to_pred_coco([({"id": 1, "file_name": "x.png"}, []),
                            ({"id": 2, "file_name": "x.png"}, [])])


def test_assert_scoreable_refuses_disjoint_keys():
    gt = gt_to_eval_coco(prep_gt())
    pred = masks_to_pred_coco([({"id": 9, "file_name": "somewhere_else.png"}, [])])
    with pytest.raises(RuntimeError, match="no address_id shared"):
        assert_scoreable(pred, gt)
    assert address_id_for("b2/roof_a.png") == "b2__roof_a"


def _pred(*segs):
    """Prediction COCO keyed to match prep_gt()'s single image."""
    return {"images": [{"id": 1, "file_name": "b2__roof_a.png",
                        "address_id": "b2__roof_a"}],
            "annotations": [{"id": i + 1, "image_id": 1,
                             "category_id": CAT_FACET, "segmentation": s}
                            for i, s in enumerate(segs)],
            "categories": []}


def test_translated_gt_actually_scores():
    gt = gt_to_eval_coco(prep_gt())
    assert assert_scoreable(_pred(SQ), gt) == 1
    r = evaluate(_pred(SQ), gt)
    assert r.n_images == 1
    assert r.facet_recall == 1.0 and r.facet_precision == 1.0
    assert r.facet_f1 == 1.0


def test_duplicate_predictions_now_cost_precision():
    """The whole point. Under the old metric, 9 copies scored like 1.

    max(iou(g, p) for p in pred) per GT cannot fall when predictions are added,
    so over-segmentation was invisible. evaluate()'s one-to-one greedy match
    charges every unmatched duplicate as a false positive.
    """
    gt = gt_to_eval_coco(prep_gt())
    one = evaluate(_pred(SQ), gt)
    nine = evaluate(_pred(*([SQ] * 9)), gt)
    assert nine.facet_recall == one.facet_recall == 1.0   # recall unchanged
    assert nine.facet_precision < one.facet_precision      # precision collapses
    assert nine.facet_precision == pytest.approx(1 / 9)
    assert nine.facet_f1 < one.facet_f1


def test_geom_to_rings_takes_geometry_as_geometry():
    """pipeline mode yields shapely polygons, raw mode yields masks.

    masks_to_facets returns Facet objects carrying a pixel-space `polygon` and
    no `.mask`. Rasterizing that polygon only to re-contour it would discard the
    regularized edges it just produced, so geometry goes straight to rings.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from src.eval.sam3_adapter import geom_to_rings

    assert geom_to_rings(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])) == [
        [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 0.0, 0.0]]
    assert len(geom_to_rings(MultiPolygon([
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])]))) == 2
    assert geom_to_rings(Polygon()) == []


def test_pipeline_polygons_survive_into_a_scoreable_coco():
    """End to end on the pipeline dialect, with no cv2 in the loop."""
    from shapely.geometry import Polygon
    gt = gt_to_eval_coco(prep_gt())
    pred = masks_to_pred_coco([
        ({"id": 1, "file_name": "b2__roof_a.png"},
         [Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])])])
    assert assert_scoreable(pred, gt) == 1
    r = evaluate(pred, gt)
    assert r.facet_f1 == 1.0


def test_pipeline_mode_uses_the_threshold_that_actually_ships():
    """masks_to_facets defaults to 0.65; production sets 0.15. Nothing uses 0.65.

    Scoring a 'pipeline' mode at the library default would describe a third
    configuration nobody deploys, while carrying the name of the deployed one.
    """
    import inspect
    from src.eval import sam3_adapter
    from src.roofs.mask_facets import masks_to_facets
    from src.serve.report_service import SCORE_THR

    assert sam3_adapter.SCORE_THR is SCORE_THR
    assert SCORE_THR == 0.15
    assert inspect.signature(masks_to_facets).parameters["score_thr"].default == 0.65
    assert inspect.signature(
        sam3_adapter.run_split).parameters["score_thr"].default == SCORE_THR
