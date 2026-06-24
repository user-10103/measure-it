#!/usr/bin/env python3
"""
Oversample minority sources in a merged COCO dataset to fix domain imbalance.

Reads an existing COCO dataset (e.g. roof_dataset_v4) and writes a new one
(e.g. roof_dataset_v5) where specified sources appear N times per epoch.
All image files are symlinked (not copied) — no re-downloading needed.

Why this exists: when one source dominates training (e.g. switzerland=69% of
images), the model optimises for that domain and stagnates on minority domains
that carry the key signal (e.g. NAIP facets, RID2 facets).

Usage:
    # Oversample NAIP and RID2 3x, keep switzerland/carecamp93 at 1x
    python training/oversample_dataset.py \\
        --input  training/roof_dataset_v4 \\
        --output training/roof_dataset_v5 \\
        --repeat phase1:3 rid2:3

    # To verify the resulting balance without writing files:
    python training/oversample_dataset.py \\
        --input training/roof_dataset_v4 --output /tmp/dry --repeat phase1:3 rid2:3 \\
        --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

CATEGORIES = [
    {"id": 1, "name": "roof_polygon"},
    {"id": 2, "name": "facet"},
]


def oversample(input_dir: Path, output_dir: Path, repeats: dict[str, int], dry_run: bool):
    for split in ("train", "valid", "test"):
        in_split  = input_dir  / split
        out_split = output_dir / split
        ann_path  = in_split   / "_annotations.coco.json"

        if not ann_path.exists():
            print(f"  [{split}] no annotations — skipping")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        src_images = coco["images"]
        src_anns   = coco["annotations"]

        # Build annotation lookup by image_id
        anns_by_img: dict[int, list[dict]] = {}
        for a in src_anns:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        # Expand: repeat each image N times based on its source tag
        out_images: list[dict] = []
        out_anns:   list[dict] = []
        next_img_id = 1
        next_ann_id = 1

        source_counts: dict[str, int] = {}

        for img in src_images:
            src = img.get("source", "unknown")
            n   = repeats.get(src, 1)
            source_counts[src] = source_counts.get(src, 0) + n

            orig_anns = anns_by_img.get(img["id"], [])

            for _ in range(n):
                new_img = dict(img)
                new_img["id"] = next_img_id
                out_images.append(new_img)

                old_img_id = img["id"]
                for a in orig_anns:
                    new_ann = dict(a)
                    new_ann["id"]       = next_ann_id
                    new_ann["image_id"] = next_img_id
                    out_anns.append(new_ann)
                    next_ann_id += 1

                next_img_id += 1

        # Print balance summary
        total = sum(source_counts.values())
        print(f"\n  [{split}] source balance after oversampling:")
        for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            mult = repeats.get(src, 1)
            pct  = 100 * count / total if total else 0
            tag  = f"×{mult}" if mult > 1 else "×1"
            print(f"    {src:<14} {count:>6} imgs  {pct:>5.1f}%  ({tag})")
        n_facets = sum(1 for a in out_anns if a.get("category_id") == 2)
        print(f"    {'TOTAL':<14} {total:>6} imgs  100.0%")
        print(f"    facet annotations: {n_facets}")

        if dry_run:
            continue

        out_split.mkdir(parents=True, exist_ok=True)

        # Write merged JSON
        merged = {"images": out_images, "annotations": out_anns, "categories": CATEGORIES}
        with open(out_split / "_annotations.coco.json", "w") as f:
            json.dump(merged, f)

        # Symlink all unique chips from input (files already downloaded in v4)
        linked = already = missing = 0
        seen_files: set[str] = set()
        for img in src_images:  # use original list — unique filenames
            fn  = img["file_name"]
            if fn in seen_files:
                continue
            seen_files.add(fn)
            src_file = in_split  / fn
            dst_file = out_split / fn
            if dst_file.exists():
                already += 1
            elif src_file.exists():
                dst_file.symlink_to(src_file.resolve())
                linked += 1
            else:
                missing += 1
        print(f"    chips: {linked} symlinked, {already} already present, {missing} missing")


def main():
    ap = argparse.ArgumentParser(description="Oversample minority COCO sources")
    ap.add_argument("--input",   required=True, help="existing dataset dir (e.g. roof_dataset_v4)")
    ap.add_argument("--output",  required=True, help="output dataset dir (e.g. roof_dataset_v5)")
    ap.add_argument("--repeat",  nargs="+", metavar="SOURCE:N", required=True,
                    help="source prefix and repeat count, e.g. phase1:3 rid2:3")
    ap.add_argument("--dry-run", action="store_true",
                    help="print balance summary without writing any files")
    args = ap.parse_args()

    repeats: dict[str, int] = {}
    for spec in args.repeat:
        try:
            src, n = spec.rsplit(":", 1)
            repeats[src] = int(n)
        except ValueError:
            sys.exit(f"ERROR: --repeat format must be SOURCE:N, got '{spec}'")

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        sys.exit(f"ERROR: input dir '{input_dir}' does not exist")

    print(f"\nOversampling {input_dir} → {output_dir}")
    print(f"Repeat factors: { {k: f'×{v}' for k, v in repeats.items()} }")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    oversample(input_dir, output_dir, repeats, args.dry_run)

    if not args.dry_run:
        print(f"\nDone → {output_dir}")
        print("Next: re-run training with --dataset pointing at the new dir")


if __name__ == "__main__":
    main()
