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


def prep_split(coco_json, images_dir, out_dir, concept, keep_names=None):
    os.makedirs(out_dir, exist_ok=True)
    d = json.load(open(coco_json))
    cats = d.get("categories", [])
    # Optional readiness gate: restrict to file_names that passed
    # training.label_readiness (clean facets only). Roofs with no/garbage facet
    # labels are excluded so the retrain never sees empty or jagged targets.
    from training.coco_split import _relpath   # shared <batch>/<name> normalization
    keep_ids = None
    if keep_names:
        # Match in the normalized namespace on BOTH sides so raw-export keep
        # names and split-normalized image names always agree (else keep_ids
        # empties silently and prep writes zero annotations).
        keep_names = {_relpath(n) for n in keep_names}
        keep_ids = {im["id"] for im in d.get("images", [])
                    if _relpath(im.get("file_name", "")) in keep_names}
    facet_ids = {c["id"] for c in cats if c["name"].lower() in FACET_NAMES}
    if not facet_ids:                               # no explicit facet class -> keep all
        facet_ids = {c["id"] for c in cats}

    # FAIL LOUD if pycocotools is missing: a broken import here is what silently
    # dropped EVERY mask in the original prep run and produced box-only data that
    # trained the mask losses to NaN. Never let that be silent again.
    from pycocotools import mask as _mu
    dims = {im["id"]: (im.get("height"), im.get("width")) for im in d.get("images", [])}
    anns = []
    dropped = 0
    seg_src = 0   # annotations that carried a source mask
    seg_ok = 0    # ... that successfully converted to RLE
    for a in d.get("annotations", []):
        if a["category_id"] not in facet_ids:
            continue
        if keep_ids is not None and a["image_id"] not in keep_ids:
            continue
        # Drop degenerate annotations (zero-area bbox). A zero-area target box
        # yields NaN in the Hungarian giou cost and crashes the matcher mid-train
        # (scipy: "matrix contains invalid numeric entries"). Filter at the source.
        bx = a.get("bbox")
        if bx is not None and (bx[2] <= 1 or bx[3] <= 1):
            dropped += 1
            continue
        a = dict(a); a["category_id"] = 1
        h, w = dims.get(a["image_id"], (None, None))
        if a.get("segmentation") is not None and h and w:
            seg_src += 1
            # NO bare except: a conversion failure must surface, never silently
            # yield a mask-less annotation (the epoch-9 NaN root cause).
            a["segmentation"] = _to_rle(a["segmentation"], h, w)
            a["iscrowd"] = 0
            if _mu.area(a["segmentation"]) < 1:   # empty mask -> 0/0 dice -> NaN
                dropped += 1
                continue
            seg_ok += 1
        anns.append(a)
    # Hard guard: a segmentation-bearing source that yields zero masks is the
    # silent-drop failure. Refuse to write box-only data that NaNs the mask loss.
    if seg_src > 0 and seg_ok == 0:
        raise RuntimeError(
            f"{coco_json}: {seg_src} source masks but 0 converted to RLE — "
            "pycocotools/_to_rle broken in this env; refusing to write mask-less data")
    if dropped:
        print(f"  dropped {dropped} degenerate annotation(s) (zero-area bbox / empty mask)")
    if seg_src:
        print(f"  converted {seg_ok}/{seg_src} source masks -> RLE")

    used = {a["image_id"] for a in anns}
    img_root = os.path.realpath(images_dir)
    imgs = []
    for im in d.get("images", []):
        if im["id"] not in used:
            continue
        # Resolve + contain: reject file_name that escapes images_dir via
        # ../ or an absolute path (path-traversal from an untrusted COCO json).
        src = os.path.realpath(os.path.join(images_dir, im["file_name"]))
        if not (src == img_root or src.startswith(img_root + os.sep)):
            continue
        if not os.path.exists(src):
            continue
        # Collision-safe flat name: basenames COLLIDE across batches (batch2/
        # roof_001.png vs batch3/roof_001.png), and a bare basename copy silently
        # overwrites — image records then train against the wrong pixels. Encode
        # the batch into the name and REFUSE to overwrite.
        base = _relpath(im["file_name"]).replace("/", "__")
        dst = os.path.join(out_dir, base)
        if os.path.exists(dst):
            raise RuntimeError(
                f"image name collision on {base!r} in {out_dir} — two source "
                "images map to one output name; aborting rather than overwrite")
        shutil.copy(src, dst)
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
    ap.add_argument("--keep", default=None,
                    help="readiness manifest {file_names:[...]} from "
                         "training.label_readiness; restricts to clean-facet roofs")
    args = ap.parse_args()

    keep_names = None
    if args.keep:
        keep_names = json.load(open(args.keep)).get("file_names")
        print(f"readiness gate: restricting to {len(keep_names)} clean-facet roofs")

    base = os.path.join(args.out, args.supercategory)
    ni, na = prep_split(args.train_coco, args.train_images,
                        os.path.join(base, "train"), args.concept, keep_names)
    print(f"train: {ni} images, {na} facet annotations")
    vi, va = prep_split(args.val_coco, args.val_images,
                        os.path.join(base, "test"), args.concept, keep_names)
    print(f"test:  {vi} images, {va} facet annotations")
    print(f"\nReady: {base}/{{train,test}}  (concept='{args.concept}')")
    print(f"Set  paths.roboflow_vl_100_root: {os.path.abspath(args.out)}")
    print(f"     roboflow_train.supercategory: {args.supercategory}")


if __name__ == "__main__":
    main()
