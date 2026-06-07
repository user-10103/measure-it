#!/usr/bin/env python3
"""
rebuild_clean_dataset.py  --  regenerate roof_dataset/ with clean disjoint splits.

Usage (from workspace/measure-it/):
    python3 training/rebuild_clean_dataset.py \
        --ls  /path/to/project-1-export.json \
        --out training/roof_dataset_clean

Then run fetch_chips.py to pull PNGs into each split dir, and retrain.
The new splits are disjoint by chip: every annotation of a chip lands
in exactly one split (train/valid/test).
"""
import argparse, json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data.ls_to_coco import ls_export_to_coco, make_splits

def subset(coco, image_ids):
    keep = set(image_ids)
    imgs = [im for im in coco["images"] if im["id"] in keep]
    anns = [a for a in coco["annotations"] if a["image_id"] in keep]
    id_map = {im["id"]: i + 1 for i, im in enumerate(imgs)}
    for im in imgs:
        im["id"] = id_map[im["id"]]
    for j, a in enumerate(anns):
        a["id"] = j + 1
        a["image_id"] = id_map[a["image_id"]]
    return {"images": imgs, "annotations": anns, "categories": coco["categories"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls",   required=True, help="Label Studio JSON export")
    ap.add_argument("--out",  default="training/roof_dataset_clean")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco = ls_export_to_coco(json.load(open(args.ls)))
    print(f"full: {len(coco['images'])} images, {len(coco['annotations'])} anns")

    # Split by unique chip stem — no chip crosses splits
    chip_stems = [os.path.splitext(im["file_name"])[0] for im in coco["images"]]
    stem_splits = make_splits(chip_stems, seed=args.seed)

    stem_to_ids = {}
    for im in coco["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        stem_to_ids.setdefault(stem, []).append(im["id"])

    splits = {sp: [iid for stem in stems for iid in stem_to_ids.get(stem, [])]
              for sp, stems in stem_splits.items()}

    name_map = {"train": "train", "val": "valid", "test": "test"}
    for split, ids in splits.items():
        folder = name_map.get(split, split)
        d = os.path.join(args.out, folder)
        os.makedirs(d, exist_ok=True)
        sub = subset(json.loads(json.dumps(coco)), ids)
        json.dump(sub, open(os.path.join(d, "_annotations.coco.json"), "w"))
        with open(os.path.join(d, "chips_needed.txt"), "w") as f:
            f.write("\n".join(im["file_name"] for im in sub["images"]))
        unique_chips = len({os.path.splitext(im["file_name"])[0] for im in sub["images"]})
        print(f"  {folder}: {len(sub['images'])} images ({unique_chips} unique chips), "
              f"{len(sub['annotations'])} anns")

    print(f"\nWrote {args.out}/  -> next: fetch_chips.py --bucket <bucket> "
          f"--dataset {args.out}")

if __name__ == "__main__":
    main()
