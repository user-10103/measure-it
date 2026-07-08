#!/usr/bin/env python3
"""
Convert RID2 (Roof Information Dataset 2) to COCO instance segmentation format,
then optionally merge into an existing roof_dataset_clean split.

RID2 mask pixel values (grayscale PNG, background = 5):
    0=North  1=East  2=South  3=West  4=Flat  5=Background

Conversion logic:
    1. For each chip, run connected-component labelling on each orientation class.
       Each connected blob → one facet instance (category_id=2).
    2. Derive roof_polygon (category_id=1) as the convex union of all non-background
       pixels — one polygon per chip.
    3. Discard chips with fewer than MIN_FACETS clean instances (too noisy / empty).

Sources used:
    • Grid chips listed in training_split_512.csv          (4,764 images, 512×512)
    • Roof-centred chips in case_study_roof_centered/      (1,819 images, 512×512)
      (these are RID2's own test set but are free for our training use)

Usage:
    # Step 1 — build the RID2-derived COCO dataset
    python training/build_rid2_dataset.py \\
        --rid2-root /data/roof_information_dataset_2 \\
        --output    training/rid2_dataset \\
        --val-frac  0.10

    # Full Colab workflow — convert + upload to S3 + merge in one command:
    python training/build_rid2_dataset.py \\
        --rid2-root  /content/rid2/roof_information_dataset_2 \\
        --output     training/rid2_dataset \\
        --merge-into training/roof_dataset_clean \\
        --s3-bucket  florida-roofs-v2-chips \\
        --s3-prefix  rid2

    After running, commit the updated annotations + chips_needed.txt to git and
    add a second fetch_chips call in the Colab training notebook:
        python training/fetch_chips.py --bucket florida-roofs-v2-chips --prefix rid2 \\
            --dataset training/roof_dataset_clean

Requirements (all present in pdal-env):
    scikit-image shapely numpy tqdm boto3

Install missing deps with:
    pip install scikit-image shapely tqdm boto3
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from skimage.measure import find_contours, label as cc_label
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

# RID2 mask class definitions (pixel value → class name)
RID2_FACET_VALS = [0, 1, 2, 3, 4]   # N, E, S, W, Flat
RID2_BG_VAL     = 5

# Quality gates
MIN_FACET_AREA_PX   = 150    # ~10 cm/px × 150 px ≈ 1.2 m² min facet area
MIN_FACETS_PER_CHIP = 2      # discard chips with fewer surviving facet instances
MIN_ROOF_AREA_PX    = 500    # discard chips where total roof area is tiny
MIN_POLYGON_COORDS  = 6      # ≥ 3 points (6 floats) after simplification

SIMPLIFY_EPSILON = 1.5       # Douglas-Peucker tolerance in pixels

# COCO category IDs — must match roof_dataset_clean
CAT_ROOF_POLYGON = 1
CAT_FACET        = 2


# ── Geometry helpers ──────────────────────────────────────────────────────────

def mask_to_polygon(binary: np.ndarray) -> list | None:
    """
    Largest-contour → simplified polygon as flat COCO [x,y,...] list, or None.

    Uses skimage.find_contours (returns row,col = y,x) + shapely simplification.
    """
    contours = find_contours(binary.astype(np.uint8), level=0.5)
    if not contours:
        return None

    # find_contours returns (row, col) = (y, x); pick the largest by area
    def contour_area(c):
        try:
            return ShapelyPolygon(zip(c[:, 1], c[:, 0])).area
        except Exception:
            return 0.0

    contour = max(contours, key=contour_area)
    if contour_area(contour) < MIN_FACET_AREA_PX:
        return None

    # Build shapely polygon (x=col, y=row) and simplify
    try:
        poly = ShapelyPolygon(zip(contour[:, 1], contour[:, 0]))
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = poly.simplify(SIMPLIFY_EPSILON, preserve_topology=True)
        if poly.is_empty or poly.geom_type != "Polygon":
            return None
        coords_xy = list(poly.exterior.coords)
    except Exception:
        return None

    if len(coords_xy) < 3:
        return None

    # Flatten to [x1, y1, x2, y2, ...]
    flat = [v for pt in coords_xy for v in (float(pt[0]), float(pt[1]))]
    if len(flat) < MIN_POLYGON_COORDS:
        return None
    return flat


def mask_to_bbox(binary: np.ndarray) -> list:
    """Return [x, y, w, h] bounding box."""
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


# ── Core chip processor ───────────────────────────────────────────────────────

def process_chip(img_path: Path, mask_path: Path, image_id: int, ann_id: int):
    """
    Convert one RID2 image+mask pair to COCO image+annotation records.

    Returns (image_dict, [annotation_dicts], next_ann_id) or (None, [], ann_id)
    if the chip is rejected.
    """
    try:
        from skimage.io import imread as sk_imread
        mask = sk_imread(str(mask_path), as_gray=False)
        # skimage returns float [0,1] for grayscale if as_gray=True; load as uint8
        if mask.ndim == 3:
            mask = mask[:, :, 0]          # take first channel
        mask = mask.astype(np.uint8)
    except Exception:
        return None, [], ann_id

    h, w = mask.shape
    roof_pixels = (mask != RID2_BG_VAL)

    if roof_pixels.sum() < MIN_ROOF_AREA_PX:
        return None, [], ann_id      # essentially empty chip

    annotations = []

    # ── 1. Facet instances ────────────────────────────────────────────────────
    for class_val in RID2_FACET_VALS:
        class_mask = (mask == class_val)
        if class_mask.sum() < MIN_FACET_AREA_PX:
            continue

        labeled = cc_label(class_mask)
        for comp_id in range(1, int(labeled.max()) + 1):
            comp = (labeled == comp_id)
            if comp.sum() < MIN_FACET_AREA_PX:
                continue

            polygon = mask_to_polygon(comp)
            if polygon is None:
                continue

            annotations.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": CAT_FACET,
                "segmentation": [polygon],
                "area":        float(comp.sum()),
                "bbox":        mask_to_bbox(comp),
                "iscrowd":     0,
            })
            ann_id += 1

    # Reject chips that didn't produce enough clean facets
    facet_count = sum(1 for a in annotations if a["category_id"] == CAT_FACET)
    if facet_count < MIN_FACETS_PER_CHIP:
        return None, [], ann_id - len(annotations)   # roll back ann_id

    # ── 2. Roof polygon — union of all non-background pixels ─────────────────
    roof_polygon = mask_to_polygon(roof_pixels)
    if roof_polygon is not None:
        annotations.append({
            "id":          ann_id,
            "image_id":    image_id,
            "category_id": CAT_ROOF_POLYGON,
            "segmentation": [roof_polygon],
            "area":        float(roof_pixels.sum()),
            "bbox":        mask_to_bbox(roof_pixels),
            "iscrowd":     0,
        })
        ann_id += 1

    image_dict = {
        "id":        image_id,
        "file_name": img_path.name,
        "width":     w,
        "height":    h,
    }
    return image_dict, annotations, ann_id


# ── Dataset builder ───────────────────────────────────────────────────────────

def gather_rid2_pairs(rid2_root: Path) -> list[tuple[Path, Path]]:
    """
    Collect (image_path, mask_path) pairs from:
      • Grid chips listed in training_split_512.csv
      • All roof-centred chips (case_study_roof_centered/)
    """
    pairs = []

    # Grid chips — read CSV to get training filenames
    csv_path = rid2_root / "training_split_512.csv"
    img_dir  = rid2_root / "images"
    mask_dir = rid2_root / "masks" / "masks_segments"

    if csv_path.exists() and img_dir.exists() and mask_dir.exists():
        with open(csv_path) as f:
            filenames = [ln.strip() for ln in f if ln.strip()]
        for fname in filenames:
            # CSV may or may not include extension
            stem = Path(fname).stem
            img  = img_dir  / f"{stem}.png"
            msk  = mask_dir / f"{stem}.png"
            if img.exists() and msk.exists():
                pairs.append((img, msk))
        print(f"Grid chips (training split): {len(pairs)}")
    else:
        print(f"WARNING: grid chip paths not found under {rid2_root}")

    # Roof-centred chips
    rc_img_dir  = rid2_root / "case_study_roof_centered" / "images_roof_centered"
    rc_mask_dir = rid2_root / "case_study_roof_centered" / "masks_roof_centered" / "masks_segments"
    if rc_img_dir.exists() and rc_mask_dir.exists():
        rc_pairs = [
            (img, rc_mask_dir / img.name)
            for img in sorted(rc_img_dir.glob("*.png"))
            if (rc_mask_dir / img.name).exists()
        ]
        print(f"Roof-centred chips: {len(rc_pairs)}")
        pairs.extend(rc_pairs)
    else:
        print(f"NOTE: roof-centred directory not found under {rid2_root}")

    return pairs


def s3_upload_chips(
    dataset_dir: Path,
    bucket: str,
    prefix: str,
    workers: int = 16,
):
    """
    Upload all PNG chips in dataset_dir/{train,valid}/ to s3://{bucket}/{prefix}/
    and write a chips_needed.txt per split (for fetch_chips.py compatibility).
    Skips chips already present in S3 (HEAD check).
    """
    import boto3
    import concurrent.futures

    s3 = boto3.client("s3")
    pfx = prefix.strip("/")

    for split in ("train", "valid"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        images = sorted(split_dir.glob("*.png"))
        jobs = []
        for img in images:
            key = f"{pfx}/{img.name}" if pfx else img.name
            jobs.append((str(img), key))

        print(f"  {split}: uploading {len(jobs)} chips to s3://{bucket}/{pfx}/ …")

        ok = fail = skip = 0
        def upload_one(job):
            local, key = job
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return "skip"
            except s3.exceptions.ClientError:
                pass
            except Exception:
                pass
            try:
                s3.upload_file(local, bucket, key)
                return "ok"
            except Exception as e:
                print(f"  FAIL {key}: {e}")
                return "fail"

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(upload_one, jobs):
                if r == "ok":   ok   += 1
                elif r == "skip": skip += 1
                else:           fail += 1

        print(f"    done: {ok} uploaded, {skip} already present, {fail} failed")

        # Write chips_needed.txt (using the RID2-specific prefix name so
        # fetch_chips.py --prefix rid2 can download them later)
        chips_txt = split_dir / "chips_needed.txt"
        with open(chips_txt, "w") as f:
            for img in images:
                f.write(img.name + "\n")
        print(f"    wrote {chips_txt}")


def build_dataset(
    rid2_root: Path,
    output_dir: Path,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Build COCO JSON splits from RID2.

    Returns {"train": coco_dict, "valid": coco_dict}.
    Also copies image files into output_dir/{train,valid}/.
    """
    pairs = gather_rid2_pairs(rid2_root)
    if not pairs:
        sys.exit("ERROR: No image/mask pairs found. Check --rid2-root path.")

    categories = [
        {"id": CAT_ROOF_POLYGON, "name": "roof_polygon"},
        {"id": CAT_FACET,        "name": "facet"},
    ]

    all_images      = []
    all_annotations = []
    image_id  = 1
    ann_id    = 1
    rejected  = 0

    print(f"\nProcessing {len(pairs)} chips …")
    for img_path, mask_path in tqdm(pairs):
        img_dict, anns, ann_id = process_chip(img_path, mask_path, image_id, ann_id)
        if img_dict is None:
            rejected += 1
            continue
        all_images.append(img_dict)
        all_annotations.extend(anns)
        image_id += 1

    facet_count = sum(1 for a in all_annotations if a["category_id"] == CAT_FACET)
    print(f"\nAccepted: {len(all_images)} chips  |  Rejected (too few facets): {rejected}")
    print(f"Total facet instances: {facet_count}")
    print(f"Total roof_polygon instances: {sum(1 for a in all_annotations if a['category_id']==CAT_ROOF_POLYGON)}")

    # ── Train / val split ────────────────────────────────────────────────────
    rng = random.Random(seed)
    indices = list(range(len(all_images)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_frac))
    val_idx   = set(indices[:n_val])
    train_idx = set(indices[n_val:])

    def build_split(idx_set: set, split_name: str) -> dict:
        split_images = [all_images[i] for i in sorted(idx_set)]
        img_ids      = {img["id"] for img in split_images}
        split_anns   = [a for a in all_annotations if a["image_id"] in img_ids]
        split_dir    = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Remap IDs to start at 1 within each split
        id_map = {img["id"]: new_id for new_id, img in enumerate(split_images, 1)}
        ann_remap = []
        new_ann_id = 1
        for ann in split_anns:
            a = dict(ann)
            a["image_id"] = id_map[ann["image_id"]]
            a["id"]       = new_ann_id
            ann_remap.append(a)
            new_ann_id += 1
        for img in split_images:
            img["id"] = id_map[img["id"]]

        # Copy image files
        # Build a lookup: filename → original source path
        fname_to_src = {img_path.name: img_path for img_path, _ in gather_rid2_pairs(rid2_root)}
        copied = 0
        for img in split_images:
            src = fname_to_src.get(img["file_name"])
            dst = split_dir / img["file_name"]
            if src and src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        print(f"  {split_name}: {len(split_images)} images, {len(ann_remap)} annotations, {copied} files copied")

        coco = {"images": split_images, "annotations": ann_remap, "categories": categories}
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(coco, f)
        return coco

    train_coco = build_split(train_idx, "train")
    valid_coco = build_split(val_idx,   "valid")
    return {"train": train_coco, "valid": valid_coco}


# ── Merge into existing dataset ───────────────────────────────────────────────

def merge_into(rid2_dataset_dir: Path, target_dir: Path):
    """
    Append rid2_dataset/{train,valid} into target_dir/{train,valid}.

    Image IDs and annotation IDs are offset to avoid collisions.
    Images are symlinked (not copied) to save disk space.
    """
    for split in ["train", "valid"]:
        target_split = target_dir / split
        rid2_split   = rid2_dataset_dir / split

        if not rid2_split.exists():
            print(f"  Skipping {split} — RID2 split not found at {rid2_split}")
            continue

        target_json = target_split / "_annotations.coco.json"
        rid2_json   = rid2_split   / "_annotations.coco.json"

        with open(target_json) as f:
            target_coco = json.load(f)
        with open(rid2_json) as f:
            rid2_coco = json.load(f)

        # Find offsets to avoid ID collisions
        img_id_offset = max(img["id"] for img in target_coco["images"]) + 1000
        ann_id_offset = max(ann["id"] for ann in target_coco["annotations"]) + 1000

        new_images = []
        for img in rid2_coco["images"]:
            new_img = dict(img)
            new_img["id"] = img["id"] + img_id_offset
            new_img["rid2"] = True   # tag for traceability
            new_images.append(new_img)

        new_anns = []
        for ann in rid2_coco["annotations"]:
            new_ann = dict(ann)
            new_ann["id"]       = ann["id"] + ann_id_offset
            new_ann["image_id"] = ann["image_id"] + img_id_offset
            new_anns.append(new_ann)

        # Symlink image files into target split dir (avoids duplicate disk use)
        linked = 0
        for rid2_img in rid2_coco["images"]:
            src = rid2_split / rid2_img["file_name"]
            dst = target_split / rid2_img["file_name"]
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())
                linked += 1

        # Write merged JSON
        merged = {
            "images":      target_coco["images"] + new_images,
            "annotations": target_coco["annotations"] + new_anns,
            "categories":  target_coco["categories"],
        }
        # Backup original
        backup = target_json.with_suffix(".pre_rid2.json")
        if not backup.exists():
            shutil.copy2(target_json, backup)
        with open(target_json, "w") as f:
            json.dump(merged, f)

        # Append RID2 chip filenames to chips_needed.txt so fetch_chips.py
        # (with --prefix rid2) can download them in Colab.
        # We append only new names — running twice is safe.
        chips_txt = target_split / "chips_needed.txt"
        existing_names: set[str] = set()
        if chips_txt.exists():
            existing_names = {l.strip() for l in chips_txt.read_text().splitlines() if l.strip()}
        rid2_names = [img["file_name"] for img in rid2_coco["images"]]
        new_names  = [n for n in rid2_names if n not in existing_names]
        if new_names:
            with open(chips_txt, "a") as f:
                for name in new_names:
                    f.write(name + "\n")
            print(f"    appended {len(new_names)} names to {chips_txt}")

        print(
            f"  {split}: +{len(new_images)} images, +{len(new_anns)} annotations "
            f"({linked} files symlinked). "
            f"New total: {len(merged['images'])} images, {len(merged['annotations'])} annotations."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RID2 → COCO converter + merger")
    ap.add_argument("--rid2-root",   required=True,
                    help="Path to extracted roof_information_dataset_2/ folder")
    ap.add_argument("--output",      default="training/rid2_dataset",
                    help="Where to write the converted COCO dataset")
    ap.add_argument("--val-frac",    type=float, default=0.10,
                    help="Fraction of accepted chips held out for validation (default 0.10)")
    ap.add_argument("--merge-into",  default=None,
                    help="If set, merge rid2_dataset into this existing COCO dataset directory "
                         "(e.g. training/roof_dataset_clean)")
    ap.add_argument("--s3-bucket",   default=None,
                    help="S3 bucket name. If set, upload converted chips after build.")
    ap.add_argument("--s3-prefix",   default="rid2",
                    help="S3 key prefix for RID2 chips (default: 'rid2')")
    ap.add_argument("--s3-workers",  type=int, default=16,
                    help="Parallel upload threads (default 16)")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    rid2_root  = Path(args.rid2_root)
    output_dir = Path(args.output)

    if not rid2_root.exists():
        sys.exit(f"ERROR: --rid2-root '{rid2_root}' does not exist.")

    # Step 1: convert
    print(f"\n{'='*60}")
    print(f"Step 1 — Converting RID2 → COCO")
    print(f"  Source : {rid2_root}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}")
    build_dataset(rid2_root, output_dir, val_frac=args.val_frac, seed=args.seed)

    # Step 2: upload to S3 (optional)
    if args.s3_bucket:
        print(f"\n{'='*60}")
        print(f"Step 2 — Uploading chips to s3://{args.s3_bucket}/{args.s3_prefix}/")
        print(f"{'='*60}")
        s3_upload_chips(output_dir, args.s3_bucket, args.s3_prefix, args.s3_workers)

    # Step 3: merge (optional)
    if args.merge_into:
        target = Path(args.merge_into)
        if not target.exists():
            sys.exit(f"ERROR: --merge-into '{target}' does not exist.")
        step = 3 if args.s3_bucket else 2
        print(f"\n{'='*60}")
        print(f"Step {step} — Merging into {target}")
        print(f"{'='*60}")
        merge_into(output_dir, target)
        print("\nDone. Original JSONs backed up as *_pre_rid2.json.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
