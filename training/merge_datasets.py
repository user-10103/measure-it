#!/usr/bin/env python3
"""
Merge multiple COCO dataset directories into a single roof_dataset_v4.

Each source directory must have the structure:
    {src_dir}/
        train/_annotations.coco.json
        train/chips_needed_{prefix}.txt   (or chips_needed.txt)
        valid/_annotations.coco.json
        valid/chips_needed_{prefix}.txt

Image IDs and annotation IDs are renumbered sequentially to guarantee uniqueness.
Chips are symlinked (not copied) to save disk space.
A chips_needed_{prefix}.txt is written per source per split so fetch_chips.py
can download each source independently.

Usage:
    python training/merge_datasets.py \\
        --source training/roof_dataset_clean  phase1 \\
        --source training/rid2_dataset        rid2 \\
        --source training/3dbag_dataset       3dbag \\
        --output training/roof_dataset_v4

Then in Colab (or EC2), fetch all chips:
    python training/fetch_chips.py --bucket florida-roofs-v4 --prefix phase1 \\
        --dataset training/roof_dataset_v4 --listing chips_needed_phase1.txt
    python training/fetch_chips.py --bucket florida-roofs-v4 --prefix rid2 \\
        --dataset training/roof_dataset_v4 --listing chips_needed_rid2.txt
    python training/fetch_chips.py --bucket florida-roofs-v4 --prefix 3dbag \\
        --dataset training/roof_dataset_v4 --listing chips_needed_3dbag.txt
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


CATEGORIES = [
    {"id": 1, "name": "roof_polygon"},
    {"id": 2, "name": "facet"},
]


def merge(sources: list[tuple[Path, str]], output_dir: Path):
    """
    sources: list of (dataset_dir, prefix) tuples
    output_dir: destination directory
    """
    for split in ("train", "valid", "test"):
        out_split = output_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        merged_images: list[dict] = []
        merged_anns:   list[dict] = []
        next_img_id = 1
        next_ann_id = 1

        chips_by_prefix: dict[str, list[str]] = {}

        for src_dir, prefix in sources:
            split_dir = src_dir / split
            json_path = split_dir / "_annotations.coco.json"

            if not json_path.exists():
                print(f"  [{split}] {prefix}: no annotations.json — skipping")
                continue

            with open(json_path) as f:
                coco = json.load(f)

            src_images = coco.get("images", [])
            src_anns   = coco.get("annotations", [])

            # Build old-id → new-id maps
            old_img_id_to_new: dict[int, int] = {}
            for img in src_images:
                old_img_id_to_new[img["id"]] = next_img_id
                new_img = dict(img)
                new_img["id"] = next_img_id
                new_img["source"] = prefix
                merged_images.append(new_img)
                next_img_id += 1

            for ann in src_anns:
                new_ann = dict(ann)
                new_ann["id"]       = next_ann_id
                new_ann["image_id"] = old_img_id_to_new.get(ann["image_id"], -1)
                if new_ann["image_id"] == -1:
                    continue
                new_ann["source"] = prefix
                merged_anns.append(new_ann)
                next_ann_id += 1

            # Symlink image files into output split dir
            linked = already = missing = 0
            for img in src_images:
                fn  = img["file_name"]
                src = split_dir / fn
                dst = out_split / fn
                if dst.exists():
                    already += 1
                    continue
                if src.exists():
                    dst.symlink_to(src.resolve())
                    linked += 1
                else:
                    missing += 1

            # chips_needed file for this prefix+split
            # try prefix-specific file first, fall back to generic
            for candidate in [
                split_dir / f"chips_needed_{prefix}.txt",
                split_dir / "chips_needed.txt",
            ]:
                if candidate.exists():
                    names = [l.strip() for l in candidate.read_text().splitlines() if l.strip()]
                    chips_by_prefix[prefix] = chips_by_prefix.get(prefix, []) + names
                    break
            else:
                # No chips_needed file: use all filenames from the JSON
                chips_by_prefix[prefix] = chips_by_prefix.get(prefix, []) + \
                    [img["file_name"] for img in src_images]

            n_facets = sum(1 for a in src_anns if a.get("category_id") == 2)
            print(
                f"  [{split}] {prefix:<12} "
                f"{len(src_images):>5} imgs  {n_facets:>6} facets  "
                f"({linked} linked, {already} already present, {missing} missing)"
            )

        # Write merged COCO JSON
        merged_coco = {
            "images":      merged_images,
            "annotations": merged_anns,
            "categories":  CATEGORIES,
        }
        with open(out_split / "_annotations.coco.json", "w") as f:
            json.dump(merged_coco, f)

        # Write per-prefix chips_needed files
        for prefix, names in chips_by_prefix.items():
            out_txt = out_split / f"chips_needed_{prefix}.txt"
            with open(out_txt, "w") as f:
                for name in sorted(set(names)):
                    f.write(name + "\n")

        total_facets = sum(1 for a in merged_anns if a.get("category_id") == 2)
        print(
            f"  [{split}] TOTAL: {len(merged_images)} images, "
            f"{total_facets} facets, {len(merged_anns)} annotations\n"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Merge multiple COCO roof datasets into roof_dataset_v4"
    )
    ap.add_argument(
        "--source", nargs=2, action="append", metavar=("DIR", "PREFIX"),
        help="Dataset directory + S3 prefix. Repeat for each source. "
             "Order matters: first source is the highest-priority (e.g. hand-labeled).",
        required=True,
    )
    ap.add_argument(
        "--output", default="training/roof_dataset_v4",
        help="Output directory (default: training/roof_dataset_v4)",
    )
    args = ap.parse_args()

    sources = [(Path(d), p) for d, p in args.source]
    output  = Path(args.output)

    print(f"\n{'='*60}")
    print(f"Merging {len(sources)} dataset(s) → {output}")
    print(f"{'='*60}")
    for d, p in sources:
        print(f"  {p:<12} {d}")
    print()

    for d, p in sources:
        if not d.exists():
            sys.exit(f"ERROR: source directory '{d}' does not exist")

    merge(sources, output)

    print(f"{'='*60}")
    print(f"Done → {output}")
    print()
    print("Next step — fetch all chips in Colab or EC2:")
    for _, p in sources:
        print(
            f"  python training/fetch_chips.py "
            f"--bucket YOUR_BUCKET --prefix {p} "
            f"--dataset {output} --listing chips_needed_{p}.txt"
        )


if __name__ == "__main__":
    main()
