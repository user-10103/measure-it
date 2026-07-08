#!/usr/bin/env python3
"""
Generate pseudo-labels for unannotated NAIP chips using a trained checkpoint.

Runs the model on every chip listed in chips_needed.txt that does NOT already
appear in the existing COCO annotations. Predictions that pass confidence and
quality gates are saved as a new COCO JSON that can be merged into the next
training run via merge_datasets.py.

Usage:
    # 1. Generate pseudo-labels
    python training/pseudo_label.py \
        --checkpoint /content/drive/MyDrive/roof_training/roof_dataset_v6b_run1/checkpoint_10.ckpt \
        --chips-dir  training/naip_dataset/train \
        --gt-json    training/naip_dataset/train/_annotations.coco.json \
        --output     training/pseudo_dataset \
        --threshold  0.65 \
        --min-facets 2

    # 2. Inspect the output, then add to next training merge:
    python training/merge_datasets.py \
        --source training/naip_dataset    phase1 \
        --source training/pseudo_dataset  pseudo \
        --source training/rid2_dataset    rid2 \
        ...
        --output training/roof_dataset_v7

Quality gates (a chip is accepted only if ALL pass):
  --threshold   minimum confidence score for every predicted facet  (default 0.65)
  --min-facets  minimum number of accepted facets per chip          (default 2)
  --max-area-ratio  max ratio of any facet area to total roof area  (default 0.90)
                    guards against single-facet chips where the model collapsed
                    all planes into one
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


CATEGORIES = [
    {"id": 1, "name": "roof_polygon"},
    {"id": 2, "name": "facet"},
]


def polygon_area(pts):
    """Shoelace formula — polygon area in pixel²."""
    pts = np.array(pts).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def mask_to_polygon(mask: np.ndarray, simplify_eps: float = 1.5):
    """Binary mask → flat [x,y,x,y,...] polygon (largest contour only)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    eps = simplify_eps * cv2.arcLength(c, True) / 100
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2).tolist()
    if len(pts) < 3:
        return None
    return [coord for xy in pts for coord in xy]


def bbox_from_polygon(seg):
    """COCO bbox [x, y, w, h] from flat polygon."""
    pts = np.array(seg).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def main():
    ap = argparse.ArgumentParser(description="Pseudo-label unannotated NAIP chips")
    ap.add_argument("--checkpoint",    required=True,
                    help=".ckpt or .pth from a trained RFDETRSeg model")
    ap.add_argument("--chips-dir",     required=True,
                    help="directory containing chip PNGs + chips_needed.txt")
    ap.add_argument("--gt-json",       required=True,
                    help="existing COCO JSON for this split (to skip already-annotated chips)")
    ap.add_argument("--output",        default="training/pseudo_dataset",
                    help="output dataset dir (same layout as naip_dataset)")
    ap.add_argument("--threshold",     type=float, default=0.65,
                    help="min confidence for every facet in a chip (default 0.65)")
    ap.add_argument("--min-facets",    type=int,   default=2,
                    help="min facets per accepted chip (default 2)")
    ap.add_argument("--max-area-ratio",type=float, default=0.90,
                    help="max single-facet/total-roof area ratio (default 0.90)")
    ap.add_argument("--split",         default="train",
                    help="output split name (default: train)")
    ap.add_argument("--workers",       type=int,   default=1,
                    help="inference workers (default 1; keep at 1 for GPU)")
    args = ap.parse_args()

    chips_dir = Path(args.chips_dir)
    gt_json   = Path(args.gt_json)
    out_dir   = Path(args.output) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load already-annotated filenames so we skip them
    with open(gt_json) as f:
        gt = json.load(f)
    annotated = {img["file_name"] for img in gt["images"]}
    print(f"Already annotated: {len(annotated)} chips")

    # Find unannotated chips
    all_chips = sorted(chips_dir.glob("*.png"))
    unannotated = [c for c in all_chips if c.name not in annotated]
    print(f"Unannotated chips to process: {len(unannotated)}")
    if not unannotated:
        print("Nothing to pseudo-label — all chips are already annotated.")
        sys.exit(0)

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    import torch
    torch.backends.cudnn.enabled = False
    for _n in ["float8_e4m3fn", "float8_e4m3fnuz",
               "float8_e5m2",   "float8_e5m2fnuz", "float8_e8m0fnu"]:
        if not hasattr(torch, _n):
            setattr(torch, _n, torch.float32)

    from src.roofs.rfdetr_backend import RFDETRBackend
    backend = RFDETRBackend(args.checkpoint, threshold=args.threshold)

    # Inference loop
    images_out = []
    anns_out   = []
    next_img_id = 1
    next_ann_id = 1
    accepted = rejected_thresh = rejected_facets = rejected_area = 0

    for chip_path in unannotated:
        bgr = cv2.imread(str(chip_path))
        if bgr is None:
            continue
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            pred = backend.predict(img_rgb)
        except Exception as e:
            print(f"  SKIP {chip_path.name}: inference error {e}")
            continue

        facets = pred.get("facets", [])

        # Gate 1: minimum facet count
        if len(facets) < args.min_facets:
            rejected_facets += 1
            continue

        # Gate 2: all facet scores above threshold (backend already filtered by
        # threshold on masks, but scores may still be available in raw output)
        # If backend doesn't expose scores, this is a no-op — rely on threshold
        # set at model load time.

        # Build facet polygons
        facet_polys = []
        for f in facets:
            poly = f.get("polygon")
            if poly and len(poly) >= 3:
                flat = [coord for xy in poly for coord in xy]
                facet_polys.append(flat)

        if len(facet_polys) < args.min_facets:
            rejected_facets += 1
            continue

        # Gate 3: no single facet dominates (catches collapsed predictions)
        areas = [polygon_area(p) for p in facet_polys]
        total_area = sum(areas)
        if total_area > 0 and max(areas) / total_area > args.max_area_ratio:
            rejected_area += 1
            continue

        # Accepted — write image + annotations
        h, w = bgr.shape[:2]
        img_entry = {
            "id":        next_img_id,
            "file_name": chip_path.name,
            "width":     w,
            "height":    h,
            "source":    "pseudo",
        }
        images_out.append(img_entry)

        # roof_polygon: convex hull of all facet points
        outline = pred.get("outline")
        if outline and len(outline) >= 3:
            flat_outline = [coord for xy in outline for coord in xy]
            area_outline = polygon_area(flat_outline)
            anns_out.append({
                "id":          next_ann_id,
                "image_id":    next_img_id,
                "category_id": 1,  # roof_polygon
                "segmentation":[flat_outline],
                "area":        float(area_outline),
                "bbox":        bbox_from_polygon(flat_outline),
                "iscrowd":     0,
            })
            next_ann_id += 1

        for flat_poly, area in zip(facet_polys, areas):
            anns_out.append({
                "id":          next_ann_id,
                "image_id":    next_img_id,
                "category_id": 2,  # facet
                "segmentation":[flat_poly],
                "area":        float(area),
                "bbox":        bbox_from_polygon(flat_poly),
                "iscrowd":     0,
            })
            next_ann_id += 1

        # Symlink chip into output dir
        dst = out_dir / chip_path.name
        if not dst.exists():
            dst.symlink_to(chip_path.resolve())

        next_img_id += 1
        accepted += 1

        if accepted % 50 == 0:
            total_seen = accepted + rejected_thresh + rejected_facets + rejected_area
            print(f"  {total_seen}/{len(unannotated)} processed — "
                  f"{accepted} accepted, {rejected_facets} too few facets, "
                  f"{rejected_area} area dominated")

    # Write COCO JSON
    coco_out = {
        "images":      images_out,
        "annotations": anns_out,
        "categories":  CATEGORIES,
    }
    out_json = out_dir / "_annotations.coco.json"
    with open(out_json, "w") as f:
        json.dump(coco_out, f)

    # Write chips_needed.txt for fetch_chips.py (in case chips aren't already local)
    (out_dir / "chips_needed.txt").write_text(
        "\n".join(img["file_name"] for img in images_out)
    )

    total_processed = accepted + rejected_thresh + rejected_facets + rejected_area
    n_facet_anns = sum(1 for a in anns_out if a["category_id"] == 2)
    print(f"\n{'='*60}")
    print(f"Pseudo-labeling complete")
    print(f"  Processed:          {total_processed} chips")
    print(f"  Accepted:           {accepted} chips ({100*accepted/max(total_processed,1):.1f}%)")
    print(f"  Rejected (facets):  {rejected_facets}")
    print(f"  Rejected (area):    {rejected_area}")
    print(f"  Facet annotations:  {n_facet_anns}")
    print(f"  Output:             {out_json}")
    print(f"\nNext: add to merge_datasets.py as --source {args.output} pseudo")


if __name__ == "__main__":
    main()
