#!/usr/bin/env python3
"""Build the RF-DETR-Seg training dataset from the cleaned Label Studio export.

Runs LOCALLY (CPU) -- dataset prep doesn't belong on the GPU. Produces the
Roboflow/RF-DETR COCO layout:

  roof_dataset/
    train/_annotations.coco.json   + chips_needed.txt
    valid/_annotations.coco.json   + chips_needed.txt
    test/_annotations.coco.json    + chips_needed.txt

Only TWO categories are emitted -- roof_outline + facet. Pitch is applied by the
deliverable's policy and aspect is derived geometrically, so the model never has
to learn them (much easier target, and it matches what we proved about the
labels). Then fetch the chip PNGs into each split dir (see README) and upload
roof_dataset/ to RunPod.

Usage (with measure-it importable):
  PYTHONPATH=/home/salter/Desktop/measure-it/measure-it-main \
    python build_dataset.py --ls /path/to/project-1-export.json --out roof_dataset
"""
import argparse
import json
import os

from src.data.ls_to_coco import ls_export_to_coco, make_splits


def _subset(coco, image_ids):
    keep = set(image_ids)
    imgs = [im for im in coco["images"] if im["id"] in keep]
    anns = [a for a in coco["annotations"] if a["image_id"] in keep]
    # re-index image + annotation ids to be contiguous per split
    id_map = {im["id"]: i + 1 for i, im in enumerate(imgs)}
    for im in imgs:
        im["id"] = id_map[im["id"]]
    for j, a in enumerate(anns):
        a["id"] = j + 1
        a["image_id"] = id_map[a["image_id"]]
    return {"images": imgs, "annotations": anns, "categories": coco["categories"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls", required=True, help="Label Studio JSON export")
    ap.add_argument("--out", default="roof_dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco = ls_export_to_coco(json.load(open(args.ls)))
    print(f"full: {len(coco['images'])} images, {len(coco['annotations'])} anns, "
          f"cats={[c['name'] for c in coco['categories']]}")

    # --- corrected split: group by unique chip stem, not per-annotation image id ---
    # make_splits dedupes internally (sorted(set(...))), so passing chip stems
    # produces 956 unique chips split 80/10/10 with no chip crossing splits.
    chip_stems = [os.path.splitext(im["file_name"])[0] for im in coco["images"]]
    stem_splits = make_splits(chip_stems, seed=args.seed)
    # convert back to image-id sets so subset() still works
    stem_to_ids = {}
    for im in coco["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        stem_to_ids.setdefault(stem, []).append(im["id"])
    splits = {sp: [iid for stem in stems for iid in stem_to_ids.get(stem, [])]
              for sp, stems in stem_splits.items()}
    # RF-DETR expects folder names train/valid/test
    name_map = {"train": "train", "val": "valid", "test": "test"}
    for split, ids in splits.items():
        folder = name_map.get(split, split)
        d = os.path.join(args.out, folder)
        os.makedirs(d, exist_ok=True)
        sub = _subset(json.loads(json.dumps(coco)), ids)
        json.dump(sub, open(os.path.join(d, "_annotations.coco.json"), "w"))
        with open(os.path.join(d, "chips_needed.txt"), "w") as f:
            f.write("\n".join(im["file_name"] for im in sub["images"]))
        print(f"  {folder}: {len(sub['images'])} images, {len(sub['annotations'])} anns")
    print(f"\nwrote {args.out}/  -> next: fetch chip PNGs into each split dir (see README)")


if __name__ == "__main__":
    main()
