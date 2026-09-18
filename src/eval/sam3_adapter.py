#!/usr/bin/env python3
"""Run SAM 3 over a val split and emit COCO that ``src/eval/evaluate.py`` scores.

WHY THIS EXISTS
---------------
The facet gate has been run with a standalone script that reports recall@50,
mean IoU and a "count bias". Those three numbers cannot decide anything:

  * Its matcher is ``max(iou(g, p) for p in pred)`` per GT, independently. There
    is no one-to-one assignment and no precision term, so **adding predictions
    can never lower either quality metric** (measured: 0/300 trials where
    injecting pure random junk masks hurt mean IoU or the match count). One mask
    may also satisfy every GT, so nine copies of a mega-facet score like one
    clean facet. A model that over-segments is structurally rewarded.
  * ``count_bias`` is computed on the RAW predictor output. ``load_sam3_predictors``
    defaults ``confidence_threshold=0.1`` precisely because ``masks_to_facets``
    is supposed to do the real filtering. The standalone gate never calls it, and
    discards ``scores`` entirely — so the figure describes the population the
    pipeline is designed to throw away, not the facets that reach a report.

    Note ``masks_to_facets``'s own default is ``score_thr=0.65``, but NOTHING in
    production uses it: ``report_service.SCORE_THR`` is **0.15** ("in-training
    checkpoint scores low; NMS dedupes"). Scoring a "pipeline" mode at 0.65 would
    describe a third configuration that nobody ships. This module imports the
    serving constant so the number cannot drift from the deployed path.

``evaluate()`` already computes precision, recall, F1, outline IoU, area and
edge-length error against the shipped tolerances. It just could not be fed:
the trainer-side COCO is in a different dialect. This module is the translator.

THE THREE DIALECT MISMATCHES, EACH OF WHICH FAILS QUIETLY
---------------------------------------------------------
1. CATEGORY IDS ARE EXACTLY INVERTED. ``prep_sam3_facets.py`` writes
   ``category_id = 1 if is_facet else 2`` with categories
   ``[{1: "roof facet"}, {2: "roof"}]``. ``evaluate.py`` declares
   ``CAT_OUTLINE = 1`` and ``CAT_FACET = 2``. Fed straight in, every facet is
   scored as an outline and every outline as a facet. Nothing raises. So this
   module resolves the facet class BY NAME via ``coco_schema.resolve_facet_ids``,
   which refuses to guess, and never by position.

2. ``evaluate.index_images`` keys on ``img["address_id"]`` and SKIPS any image
   without one. Trainer COCO has no such key, so every image is dropped and
   ``n_images`` is 0 — at which point ``area_err_pct`` and ``edge_len_err_pct``
   average empty lists to ``0.0`` and their gates PASS. An empty run reports
   "0.00% area error" and three of five gates green. Verified on real GT.

3. Trainer GT segmentation is compressed RLE (a dict). ``_segmentation_rings``
   does ``seg[0]``, which on a dict is ``KeyError: 0``. That one is at least
   loud, but it means RLE must be decoded to rings here.

Every one of those is checked and refused rather than worked around silently.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.eval.evaluate import CAT_FACET, CAT_OUTLINE  # noqa: E402
from src.serve.report_service import SCORE_THR         # noqa: E402  the SHIPPING threshold
from training.coco_schema import resolve_facet_ids     # noqa: E402

logger = logging.getLogger(__name__)

MIN_RING_AREA_PX = 4.0     # below this a contour is noise, not a facet
EPSILON_FRAC = 0.01        # Douglas-Peucker tolerance, fraction of perimeter


# --- mask / segmentation -> COCO polygon rings ------------------------------
def mask_to_rings(mask, epsilon_frac: float = EPSILON_FRAC) -> List[List[float]]:
    """Boolean mask -> list of flat COCO rings ``[x0,y0,x1,y1,...]``.

    ALL external contours are returned, not just the largest. ``evaluate``
    unions an annotation's rings before matching, so a facet that the model
    emitted in two pieces stays one annotation with two rings. Keeping only the
    largest would silently shrink such a facet and inflate its area error.
    """
    import cv2      # lazy: GPU box only, not a dev-machine dependency
    import numpy as np

    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if m.ndim != 2 or not m.any():
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rings: List[List[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_RING_AREA_PX:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon_frac * cv2.arcLength(cnt, True), True)
        if len(approx) < 3:
            continue
        rings.append([float(v) for pt in approx for v in (pt[0][0], pt[0][1])])
    return rings


def geom_to_rings(obj) -> List[List[float]]:
    """Boolean mask OR shapely geometry -> flat COCO rings.

    The two prediction modes produce different types: ``raw`` gives the
    predictor's boolean masks, ``pipeline`` gives the pixel-space polygons that
    ``masks_to_facets`` built. Converting the polygon back to a raster and
    re-contouring it would throw away the regularized edges, so geometry is
    taken as geometry.
    """
    if hasattr(obj, "geom_type"):
        geoms = (list(obj.geoms) if obj.geom_type.startswith("Multi")
                 else [obj])
        rings: List[List[float]] = []
        for g in geoms:
            if g.is_empty or g.area <= 0:
                continue
            rings.append([float(v) for xy in g.exterior.coords for v in xy])
        return rings
    return mask_to_rings(obj)


def ann_to_rings(ann: dict, height: Optional[int] = None,
                 width: Optional[int] = None) -> List[List[float]]:
    """Any COCO segmentation -> flat rings. Handles polygon AND compressed RLE."""
    seg = ann.get("segmentation")
    if not seg:
        return []
    if isinstance(seg, dict):                       # RLE
        try:
            from pycocotools import mask as mask_utils
        except ImportError as e:                    # never silently yield []
            raise RuntimeError(
                "annotation carries RLE segmentation but pycocotools is not "
                "installed; refusing to return an empty facet") from e
        rle = dict(seg)
        if isinstance(rle.get("counts"), list):     # uncompressed
            if height is None or width is None:
                raise ValueError("uncompressed RLE needs image height/width")
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_to_rings(mask_utils.decode(rle))
    if isinstance(seg[0], (int, float)):            # single flat ring
        return [list(seg)]
    return [list(r) for r in seg if r]


# --- image keying -----------------------------------------------------------
def address_id_for(file_name: str) -> str:
    """Stable ``address_id`` for a trainer image: its path-flattened stem."""
    return os.path.splitext(file_name.replace("/", "__"))[0]


def _index_by_address(images: Sequence[dict], *, side: str) -> Dict[str, dict]:
    """Map address_id -> image record, REFUSING on collision.

    ``evaluate.index_images`` does ``out[addr] = img["id"]`` — last writer wins,
    silently, and the loser's annotations are never scored. Two chips whose
    stems collide would quietly remove one roof from the gate.
    """
    out: Dict[str, dict] = {}
    for img in images:
        addr = address_id_for(img["file_name"])
        if addr in out:
            raise ValueError(
                f"{side}: address_id collision on {addr!r} between "
                f"{out[addr]['file_name']!r} and {img['file_name']!r}; "
                f"evaluate() would silently score only one of them")
        out[addr] = img
    return out


# --- GT: trainer dialect -> evaluate dialect --------------------------------
def gt_to_eval_coco(gt: dict) -> dict:
    """Re-dialect a trainer split so ``evaluate()`` can actually read it.

    Resolves the facet class BY NAME (refusing to guess), remaps it to
    ``CAT_FACET``, treats every other declared class as the outline, decodes RLE
    to rings, and stamps ``address_id`` on each image.
    """
    cats = gt.get("categories") or []
    facet_ids = resolve_facet_ids(cats, strict=True)      # raises rather than guess
    outline_ids = {c["id"] for c in cats} - facet_ids
    logger.info("GT categories %s -> facet ids %s, outline ids %s",
                [(c["id"], c.get("name")) for c in cats],
                sorted(facet_ids), sorted(outline_ids))

    by_addr = _index_by_address(gt.get("images", []), side="gt")
    dims = {im["id"]: (im.get("height"), im.get("width"))
            for im in gt.get("images", [])}

    images = [{**im, "address_id": address_id_for(im["file_name"])}
              for im in gt.get("images", [])]
    anns: List[dict] = []
    for a in gt.get("annotations", []):
        if a["category_id"] in facet_ids:
            cat = CAT_FACET
        elif a["category_id"] in outline_ids:
            cat = CAT_OUTLINE
        else:
            continue
        h, w = dims.get(a["image_id"], (None, None))
        rings = ann_to_rings(a, h, w)
        if not rings:
            continue
        anns.append({"id": len(anns) + 1, "image_id": a["image_id"],
                     "category_id": cat, "segmentation": rings})

    n_facets = sum(1 for a in anns if a["category_id"] == CAT_FACET)
    if not n_facets:
        raise RuntimeError(
            f"GT re-dialect produced 0 facet annotations from "
            f"{len(gt.get('annotations', []))} source annotations. Scoring "
            f"against this yields recall 0 and a vacuous area/edge pass.")
    logger.info("GT: %d images, %d facets, %d outlines",
                len(images), n_facets, len(anns) - n_facets)
    assert by_addr is not None
    return {"images": images, "annotations": anns,
            "categories": [{"id": CAT_OUTLINE, "name": "roof_polygon"},
                           {"id": CAT_FACET, "name": "facet"}]}


# --- predictions -> evaluate dialect ----------------------------------------
def masks_to_pred_coco(per_image: Sequence[Tuple[dict, Sequence]],
                       outlines: Optional[Dict[int, object]] = None) -> dict:
    """``[(image_record, [mask, ...]), ...]`` -> a COCO evaluate() can score."""
    _index_by_address([im for im, _ in per_image], side="pred")
    images = [{**im, "address_id": address_id_for(im["file_name"])}
              for im, _ in per_image]
    anns: List[dict] = []
    for im, masks in per_image:
        for m in masks:
            rings = geom_to_rings(m)
            if rings:
                anns.append({"id": len(anns) + 1, "image_id": im["id"],
                             "category_id": CAT_FACET, "segmentation": rings})
        if outlines and im["id"] in outlines:
            rings = geom_to_rings(outlines[im["id"]])
            if rings:
                anns.append({"id": len(anns) + 1, "image_id": im["id"],
                             "category_id": CAT_OUTLINE, "segmentation": rings})
    return {"images": images, "annotations": anns,
            "categories": [{"id": CAT_OUTLINE, "name": "roof_polygon"},
                           {"id": CAT_FACET, "name": "facet"}]}


def assert_scoreable(pred: dict, gt: dict) -> int:
    """Refuse before scoring if evaluate() would silently score nothing.

    At ``n_images == 0`` evaluate returns area_err 0.0 and edge_err 0.0, whose
    gates then PASS — an empty run reports perfect area accuracy and three of
    five gates green. That must never reach a decision.
    """
    g = {i.get("address_id") for i in gt.get("images", []) if i.get("address_id")}
    p = {i.get("address_id") for i in pred.get("images", []) if i.get("address_id")}
    if not g:
        raise RuntimeError("GT has no address_id on any image; evaluate() would "
                           "score 0 images and pass 3 of 5 gates vacuously")
    overlap = g & p
    if not overlap:
        raise RuntimeError(
            f"no address_id shared between {len(p)} predicted and {len(g)} GT "
            f"images — every facet would count as a false negative. "
            f"pred sample: {sorted(p)[:3]}, gt sample: {sorted(g)[:3]}")
    if len(overlap) < len(g):
        logger.warning("%d of %d GT images have no prediction; they score as "
                       "all-FN, which is correct but worth knowing",
                       len(g) - len(overlap), len(g))
    return len(overlap)


# --- CLI --------------------------------------------------------------------
def run_split(ckpt: str, data_dir: str, concept: str = "roof facet",
              limit: Optional[int] = None, mode: str = "both",
              score_thr: float = SCORE_THR) -> dict:
    """Predict over a split and score each mode with ``evaluate()``.

    ``mode``:
      raw       — predictor output as-is (``confidence_threshold=0.1``). This is
                  what the model emits, and what the old gate measured.
      pipeline  — the same masks through ``masks_to_facets`` at the SERVING
                  threshold (``report_service.SCORE_THR``, currently 0.15) plus
                  NMS and area bounds. Closest thing to what reaches a report.
                  Not identical: serving also passes ``outline=roof_mask``,
                  which clips spillover and enables edge regularization. Without
                  a footprint here, pipeline facets are slightly LOOSER than
                  production, so treat its precision as a lower bound.
      both      — default. The gap between the two IS the finding; reporting one
                  alone is how "count bias 8.66" got read as a pipeline defect.
    """
    import numpy as np
    from PIL import Image
    from src.eval.evaluate import evaluate
    from src.roofs.sam3_predictors import load_sam3_predictors

    gt_raw = json.load(open(os.path.join(data_dir, "_annotations.coco.json")))
    gt = gt_to_eval_coco(gt_raw)

    with_anns = {a["image_id"] for a in gt["annotations"]}
    images = [i for i in gt["images"] if i["id"] in with_anns]
    if limit:
        images = images[:limit]
    logger.info("scoring %d images from %s", len(images), data_dir)

    predict_facets, _ = load_sam3_predictors(ckpt, use_zeroshot_outline=False)
    modes = ["raw", "pipeline"] if mode == "both" else [mode]
    collected: Dict[str, List[Tuple[dict, list]]] = {m: [] for m in modes}
    score_stats: List[float] = []

    for k, im in enumerate(images):
        arr = np.array(Image.open(
            os.path.join(data_dir, im["file_name"])).convert("RGB"))
        masks, scores = predict_facets(arr, concept)
        masks = [] if masks is None else list(masks)
        scores = np.asarray(scores).reshape(-1) if len(masks) else np.zeros(0)
        score_stats.extend(scores.tolist())
        if "raw" in collected:
            collected["raw"].append((im, masks))
        if "pipeline" in collected:
            # The scores the old gate discarded are exactly what decides this.
            from src.roofs.mask_facets import masks_to_facets
            facets, _lm = (masks_to_facets(masks, scores,
                                           score_thr=score_thr, iou_thr=0.5)
                           if masks else ([], None))
            # Facet carries a pixel-space shapely polygon and NO mask. Take the
            # geometry directly: rasterizing it only to re-contour would lose
            # the regularized edges masks_to_facets just produced.
            collected["pipeline"].append((im, [f.polygon for f in facets
                                               if f.polygon is not None]))
        if k % 20 == 0:
            logger.info("  [%d/%d] %s raw=%d", k, len(images),
                        im["file_name"], len(masks))

    out: Dict[str, dict] = {}
    for m, per_image in collected.items():
        pred = masks_to_pred_coco(per_image)
        n = assert_scoreable(pred, gt)
        res = evaluate(pred, gt).to_dict()
        res["n_scoreable"] = n
        res["pred_facets"] = sum(1 for a in pred["annotations"]
                                 if a["category_id"] == CAT_FACET)
        res["gt_facets"] = sum(1 for a in gt["annotations"]
                               if a["category_id"] == CAT_FACET)
        out[m] = res
    if score_stats:
        import numpy as _np
        s = _np.asarray(score_stats)
        out["raw_score_distribution"] = {
            "n": int(s.size), "min": float(s.min()), "max": float(s.max()),
            "median": float(_np.median(s)),
            "serving_threshold": score_thr,
            "frac_above_serving_thr": float((s >= score_thr).mean()),
            "frac_above_0.65": float((s >= 0.65).mean()),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True,
                    help="split dir holding _annotations.coco.json + images")
    ap.add_argument("--concept", default="roof facet")
    ap.add_argument("--mode", choices=("raw", "pipeline", "both"), default="both")
    # NO default limit. The old gate defaulted to 40 and took the FIRST 40 in
    # file order, which is how a quarter-sample (0.964/0.854) was compared
    # against a full run (0.9153/0.8128) as though they were the same number.
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N images (default: all)")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR,
                    help=f"pipeline-mode score threshold (default {SCORE_THR}, "
                         f"the value report_service ships)")
    ap.add_argument("--out", default="facet_gate.json")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_split(a.ckpt, a.data, a.concept, a.limit, a.mode, a.score_thr)
    results["checkpoint"] = a.ckpt
    json.dump(results, open(a.out, "w"), indent=2)

    print("\n=== FACET GATE ===")
    for m in ("raw", "pipeline"):
        if m not in results:
            continue
        r = results[m]
        print(f"\n[{m}]  images={r['n_images']}  "
              f"pred={r['pred_facets']}  gt={r['gt_facets']}")
        print(f"  precision {r['facet_precision']:.4f}   "
              f"recall {r['facet_recall']:.4f}   F1 {r['facet_f1']:.4f}")
        print(f"  matched IoU {r['facet_mean_iou']:.4f}   "
              f"outline IoU {r['outline_iou']:.4f}")
        print(f"  area err {r['area_err_pct']:.2f}%   "
              f"edge err {r['edge_len_err_pct']:.2f}%")
        print(f"  gates: {r['gates']}  passed={r['passed']}")
    if "raw_score_distribution" in results:
        d = results["raw_score_distribution"]
        print(f"\nraw mask scores: n={d['n']} range [{d['min']:.3f}, "
              f"{d['max']:.3f}] median {d['median']:.3f}")
        print(f"  {d['frac_above_serving_thr']:.1%} clear the SERVING threshold "
              f"{d['serving_threshold']} (report_service.SCORE_THR)")
        print(f"  {d['frac_above_0.65']:.1%} clear masks_to_facets' 0.65 default "
              f"(which production does NOT use)")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
