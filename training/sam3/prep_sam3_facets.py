#!/usr/bin/env python3
"""Prepare NAIP facet COCO annotations for SAM 3 fine-tuning.

SAM 3 fine-tuning (sam3/train/train.py) expects a Roboflow-VL layout:
    <out>/<supercategory>/train/_annotations.coco.json  (+ image files)
    <out>/<supercategory>/test/_annotations.coco.json   (+ image files)

Key facts baked in here:
  - The COCO **category name becomes the text/concept prompt**, so we rename the
    facet category to "roof facet" (what we prompt with at inference).
  - We keep ONLY facet annotations (drop the roof_polygon/outline class).
  - The training config uses a DecodeRle transform, so segmentation is converted
    to RLE (COCO polygons -> RLE via pycocotools).
  - Only images that actually contain >=1 facet are kept.

Usage:
  python prep_sam3_facets.py \
    --train-coco roof_dataset/train/_annotations.coco.json --train-images roof_dataset/train \
    --val-coco   roof_dataset/valid/_annotations.coco.json --val-images   roof_dataset/valid \
    --out sam3_roof_data
"""
import argparse
import json
import os
import shutil

FACET_NAMES = {"facet", "roof_facet", "roof facet", "roof face"}


def _to_rle(seg, h, w):
    """COCO polygon (or uncompressed RLE) -> compressed RLE with str counts."""
    from pycocotools import mask as mask_utils
    if isinstance(seg, dict):                       # already RLE
        rle = seg
        if isinstance(rle.get("counts"), list):     # uncompressed -> compressed
            rle = mask_utils.frPyObjects(rle, h, w)
    else:                                           # list of polygons
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    if isinstance(rle.get("counts"), bytes):
        rle = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
    return rle


def prep_split(coco_json, images_dir, out_dir, concept):
    os.makedirs(out_dir, exist_ok=True)
    d = json.load(open(coco_json))
    cats = d.get("categories", [])
    facet_ids = {c["id"] for c in cats if c["name"].lower() in FACET_NAMES}
    if not facet_ids:                               # no explicit facet class -> keep all
        facet_ids = {c["id"] for c in cats}

    dims = {im["id"]: (im.get("height"), im.get("width")) for im in d.get("images", [])}
    anns = []
    for a in d.get("annotations", []):
        if a["category_id"] not in facet_ids:
            continue
        a = dict(a); a["category_id"] = 1
        h, w = dims.get(a["image_id"], (None, None))
        if a.get("segmentation") is not None and h and w:
            try:
                a["segmentation"] = _to_rle(a["segmentation"], h, w)
                a["iscrowd"] = 0
            except Exception:
                a.pop("segmentation", None)
        anns.append(a)

    used = {a["image_id"] for a in anns}
    imgs = []
    for im in d.get("images", []):
        if im["id"] not in used:
            continue
        src = os.path.join(images_dir, im["file_name"])
        if not os.path.exists(src):
            continue
        base = os.path.basename(im["file_name"])
        shutil.copy(src, os.path.join(out_dir, base))
        im = dict(im); im["file_name"] = base
        imgs.append(im)

    out = {"images": imgs, "annotations": anns,
           "categories": [{"id": 1, "name": concept, "supercategory": concept}]}
    json.dump(out, open(os.path.join(out_dir, "_annotations.coco.json"), "w"))
    return len(imgs), len(anns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-coco", required=True)
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--val-coco", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--supercategory", default="roof")
    ap.add_argument("--concept", default="roof facet")
    args = ap.parse_args()

    base = os.path.join(args.out, args.supercategory)
    ni, na = prep_split(args.train_coco, args.train_images,
                        os.path.join(base, "train"), args.concept)
    print(f"train: {ni} images, {na} facet annotations")
    vi, va = prep_split(args.val_coco, args.val_images,
                        os.path.join(base, "test"), args.concept)
    print(f"test:  {vi} images, {va} facet annotations")
    print(f"\nReady: {base}/{{train,test}}  (concept='{args.concept}')")
    print(f"Set  paths.roboflow_vl_100_root: {os.path.abspath(args.out)}")
    print(f"     roboflow_train.supercategory: {args.supercategory}")


if __name__ == "__main__":
    main()
