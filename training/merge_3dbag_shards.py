#!/usr/bin/env python3
"""
Merge 3DBAG shard COCO JSONs into one _annotations.coco.json per split.

Each shard wrote _annotations_shard{N}.coco.json.  This script combines
them with sequential IDs and writes the final _annotations.coco.json.
Also merges chips_needed_3dbag_shard{N}.txt into chips_needed_3dbag.txt.

Usage:
    python training/merge_3dbag_shards.py \\
        --input training/3dbag_dataset \\
        --num-shards 4
"""

import argparse
import json
from pathlib import Path

CATEGORIES = [
    {"id": 1, "name": "roof_polygon"},
    {"id": 2, "name": "facet"},
]


def merge_shards(input_dir: Path, num_shards: int):
    for split in ("train", "valid"):
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"  [{split}] directory not found — skipping")
            continue

        all_images, all_anns = [], []
        img_id = 1; ann_id = 1

        for shard in range(num_shards):
            jpath = split_dir / f"_annotations_shard{shard}.coco.json"
            if not jpath.exists():
                print(f"  [{split}] shard {shard}: {jpath.name} not found — skipping")
                continue

            with open(jpath) as f:
                coco = json.load(f)

            old_img_to_new = {}
            for img in coco.get("images", []):
                old_img_to_new[img["id"]] = img_id
                all_images.append({**img, "id": img_id, "shard": shard})
                img_id += 1

            for ann in coco.get("annotations", []):
                new_img_id = old_img_to_new.get(ann["image_id"])
                if new_img_id is None:
                    continue
                all_anns.append({**ann, "id": ann_id, "image_id": new_img_id})
                ann_id += 1

            print(f"  [{split}] shard {shard}: "
                  f"{len(coco.get('images', []))} imgs, "
                  f"{len(coco.get('annotations', []))} anns")

        merged = {"images": all_images, "annotations": all_anns,
                  "categories": CATEGORIES}
        out_path = split_dir / "_annotations.coco.json"
        with open(out_path, "w") as f:
            json.dump(merged, f)

        # Merge chips_needed files
        all_names: set = set()
        for shard in range(num_shards):
            txt = split_dir / f"chips_needed_3dbag_shard{shard}.txt"
            if txt.exists():
                all_names.update(l.strip() for l in txt.read_text().splitlines() if l.strip())
        with open(split_dir / "chips_needed_3dbag.txt", "w") as f:
            f.write("".join(n + "\n" for n in sorted(all_names)))

        n_facets = sum(1 for a in all_anns if a["category_id"] == 2)
        print(f"  [{split}] MERGED: {len(all_images)} images, {n_facets} facets → {out_path}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True, help="3dbag_dataset directory")
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    print(f"Merging {args.num_shards} shards from {args.input} …\n")
    merge_shards(Path(args.input), args.num_shards)
    print("Done.")


if __name__ == "__main__":
    main()
