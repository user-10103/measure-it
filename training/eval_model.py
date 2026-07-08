#!/usr/bin/env python3
"""Honest accuracy: run the trained model over held-out chips → emit COCO preds
→ score vs the GT annotations with src/eval/evaluate.py.

Measures MODEL-vs-ANNOTATION geometry on the DISJOINT held-out split (no leak):
facet F1 (IoU>=0.5 greedy match), outline IoU, plan-area %. Edge-length and pitch
gates pass vacuously because the GT COCO carries no edge/pitch annotations — those
are derived downstream, not predicted by the 2-class model.

This is the deliverable's first real accuracy number. Run on the pod after the
clean retrain:
  python training/eval_model.py \
      --checkpoint output/checkpoint_best_regular.pth \
      --gt   training/roof_dataset_clean/valid/_annotations.coco.json \
      --chips training/roof_dataset_clean/valid
"""
import argparse
import json
import os

import numpy as np

from src.eval.evaluate import evaluate, format_report


def build_pred_coco(gt, chips_dir, backend):
    """Run the model over each GT image's chip and return a prediction COCO.

    Predictions are scaled from PNG-pixel space into the GT COCO's width/height
    so pred and GT polygons live in the same coordinate frame.
    """
    from PIL import Image
    pred_images, pred_anns = [], []
    aid = 1
    missing = 0
    for im in gt["images"]:
        path = os.path.join(chips_dir, im["file_name"])
        if not os.path.exists(path):
            missing += 1
            continue                                  # GT image with no chip -> all-FN
        arr = np.array(Image.open(path).convert("RGB"), dtype="uint8")
        ph, pw = arr.shape[:2]
        sx, sy = im["width"] / pw, im["height"] / ph   # PNG px -> GT coco px
        pred = backend.predict(arr)
        pred_images.append({"id": im["id"], "address_id": im["address_id"],
                            "file_name": im["file_name"],
                            "width": im["width"], "height": im["height"]})

        def add(poly, cat):
            nonlocal aid
            ring = [v for pt in poly for v in (pt[0] * sx, pt[1] * sy)]
            if len(ring) >= 6:
                pred_anns.append({"id": aid, "image_id": im["id"],
                                  "category_id": cat, "segmentation": [ring]})
                aid += 1

        if pred.get("outline"):
            add(pred["outline"], 1)
        for f in pred.get("facets", []):
            add(f["polygon"], 2)
    return {"images": pred_images, "annotations": pred_anns,
            "categories": gt["categories"]}, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gt", required=True, help="GT COCO json (held-out split)")
    ap.add_argument("--chips", required=True, help="dir of chip PNGs for the GT images")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--infer-shape", type=int, default=624,
                    help="Inference shape passed to model.predict(); must be divisible by 24 (default 624 matches training)")
    ap.add_argument("--mask-epsilon", type=float, default=0.025)
    ap.add_argument("--out-pred", default=None, help="optional: write pred COCO here")
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    # evaluate() keys images on address_id; the roof_dataset COCO has none -> add stem
    for im in gt["images"]:
        im.setdefault("address_id", os.path.splitext(im["file_name"])[0])

    from src.roofs.rfdetr_backend import RFDETRBackend
    backend = RFDETRBackend(args.checkpoint, threshold=args.threshold,
                            infer_shape=args.infer_shape,
                            mask_epsilon=args.mask_epsilon)

    pred_coco, missing = build_pred_coco(gt, args.chips, backend)
    if args.out_pred:
        json.dump(pred_coco, open(args.out_pred, "w"))
    if missing:
        print(f"WARNING: {missing} GT chips had no PNG in {args.chips} "
              f"(scored as full misses — fetch them or they drag recall down)")

    result = evaluate(pred_coco, gt)
    print(format_report(result))
    print("\nNOTE: MODEL-vs-ANNOTATION accuracy on the held-out split. Edge-length &")
    print("pitch gates pass vacuously (GT has no edge/pitch anns). Facet F1 / outline")
    print("IoU / area% are the real numbers. (Annotation itself has error — this is")
    print("the first rung; absolute-truth vs LiDAR/survey is a further step.)")


if __name__ == "__main__":
    main()
