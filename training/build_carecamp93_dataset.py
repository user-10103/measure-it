#!/usr/bin/env python3
"""
Convert carecamp93 Automatic Roof Plane Extraction dataset → COCO format.

Source: https://github.com/carecamp93/Automatic_Roof_Plane_Extraction
Cities:  Las Vegas NV + Atlanta GA + Paris (use --cities lasvegas atlanta to skip Paris)
Format:  Per-building image chips + roof plane annotations as polyline shapefiles
         (planar graph edges) which we assemble into polygon rings via Shapely.

Download the data from the Google Drive links in the repo README, then run:

    python training/build_carecamp93_dataset.py \\
        --input  /path/to/carecamp93_data \\
        --output training/carecamp93_dataset \\
        --s3-bucket florida-roofs-v4 \\
        --s3-prefix carecamp93

Input directory structure expected (from Google Drive download):
    carecamp93_data/
        lasvegas/
            images/     *.png  (one per building)
            labels/     *.shp  or  *.json  (one per building, matching filename stem)
        atlanta/
            images/     *.png
            labels/     *.shp  or  *.json
        paris/
            images/     *.png
            labels/     *.shp  or  *.json

If the label format differs from what is expected, run with --inspect
to print a sample label file and adjust the parsing accordingly.

Dependencies:
    pip install geopandas shapely pyproj Pillow tqdm boto3
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

try:
    import geopandas as gpd
    from shapely.ops import polygonize, unary_union
    from shapely.geometry import Polygon, MultiPolygon, mapping
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    from PIL import Image
except ImportError:
    sys.exit("pip install Pillow")

from tqdm import tqdm

# ─── COCO categories (same as the rest of the pipeline) ──────────────────────
CATEGORIES = [
    {"id": 1, "name": "roof_polygon"},
    {"id": 2, "name": "facet"},
]
CAT_ROOF = 1
CAT_FACET = 2

CHIP_PX = 512   # resize all chips to 512×512 to match training set


# ─── Label parsers ────────────────────────────────────────────────────────────

def load_labels_shp(label_path: Path):
    """
    Load a shapefile of roof plane polylines/polygons.
    carecamp93 stores planar graph edges as polylines;
    polygonize() assembles them into closed rings.
    Returns list of Shapely Polygons in pixel coordinates.
    """
    gdf = gpd.read_file(str(label_path))
    polys = []

    # Case 1: already Polygon geometry
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            polys.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polys.extend(geom.geoms)

    if polys:
        return polys

    # Case 2: LineString / planar graph edges → polygonize
    lines = [g for g in gdf.geometry if g is not None]
    assembled = list(polygonize(lines))
    return assembled


def load_labels_json(label_path: Path, img_w: int, img_h: int):
    """
    Load JSON label file.  Tries common formats:
      - {"planes": [[x,y,...], ...]}           flat poly coords per plane
      - {"annotations": [{"segmentation": ...}]}  COCO-style
      - [[x,y,...], ...]                        bare list of planes
    Returns list of Shapely Polygons in pixel coordinates.
    """
    with open(label_path) as f:
        data = json.load(f)

    polys = []

    def _flat_to_poly(flat):
        pts = [(flat[i], flat[i+1]) for i in range(0, len(flat)-1, 2)]
        if len(pts) >= 3:
            p = Polygon(pts)
            return p if p.is_valid else p.buffer(0)
        return None

    if isinstance(data, list):
        # bare list of planes
        for item in data:
            if isinstance(item, list):
                p = _flat_to_poly(item)
                if p and not p.is_empty:
                    polys.append(p)

    elif isinstance(data, dict):
        # {"planes": [...]}
        for plane in data.get("planes", []):
            if isinstance(plane, list):
                p = _flat_to_poly(plane)
                if p and not p.is_empty:
                    polys.append(p)

        # COCO-style annotations
        for ann in data.get("annotations", []):
            for seg in ann.get("segmentation", []):
                p = _flat_to_poly(seg)
                if p and not p.is_empty:
                    polys.append(p)

    return polys


def load_labels(label_path: Path, img_w: int, img_h: int):
    """Dispatch to correct parser based on file extension."""
    suffix = label_path.suffix.lower()
    if suffix == ".shp":
        if not HAS_GPD:
            sys.exit("pip install geopandas  (needed for .shp labels)")
        return load_labels_shp(label_path)
    elif suffix in (".json", ".geojson"):
        return load_labels_json(label_path, img_w, img_h)
    else:
        print(f"  Unknown label format: {suffix} — skipping {label_path.name}")
        return []


def poly_to_coco_flat(poly: Polygon, img_w: int, img_h: int,
                       orig_w: int, orig_h: int):
    """
    Convert a Shapely Polygon in original image pixel space to
    COCO flat [x1,y1,x2,y2,...] scaled to CHIP_PX×CHIP_PX.
    """
    sx = CHIP_PX / orig_w
    sy = CHIP_PX / orig_h
    try:
        coords = [(x * sx, y * sy) for x, y in poly.exterior.coords]
    except Exception:
        return None
    flat = [v for c in coords for v in (float(c[0]), float(c[1]))]
    if len(flat) < 6:
        return None
    xs = flat[0::2]; ys = flat[1::2]
    if min(xs) > CHIP_PX or max(xs) < 0 or min(ys) > CHIP_PX or max(ys) < 0:
        return None
    return flat


def flat_bbox(flat):
    xs = flat[0::2]; ys = flat[1::2]
    return [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]


def flat_area(flat):
    xs = flat[0::2]; ys = flat[1::2]
    n = len(xs)
    return abs(sum(xs[i]*ys[(i+1)%n] - xs[(i+1)%n]*ys[i] for i in range(n))) / 2.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="carecamp93 roof planes → COCO instance segmentation"
    )
    ap.add_argument("--input",      required=True,
                    help="Root directory of downloaded carecamp93 data")
    ap.add_argument("--output",     default="training/carecamp93_dataset")
    ap.add_argument("--val-frac",   type=float, default=0.10)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--cities",     nargs="*",
                    default=["lasvegas", "atlanta"],
                    help="Which city subdirectories to include (default: lasvegas atlanta)")
    ap.add_argument("--s3-bucket",  default=None)
    ap.add_argument("--s3-prefix",  default="carecamp93")
    ap.add_argument("--workers",    type=int, default=8)
    ap.add_argument("--inspect",    action="store_true",
                    help="Print sample files and exit — use to diagnose format")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    # ── Inspection mode ───────────────────────────────────────────────────────
    if args.inspect:
        print(f"\nInspecting {inp} …\n")
        for city_dir in sorted(inp.iterdir()):
            if not city_dir.is_dir():
                continue
            print(f"── {city_dir.name} ──")
            img_dir = city_dir / "images"
            lbl_dir = city_dir / "labels"
            if img_dir.exists():
                imgs = sorted(img_dir.glob("*"))[:3]
                print(f"  images/  ({len(list(img_dir.glob('*')))} files)")
                for f in imgs: print(f"    {f.name}")
            if lbl_dir.exists():
                lbls = sorted(lbl_dir.glob("*"))[:3]
                print(f"  labels/  ({len(list(lbl_dir.glob('*')))} files)")
                for f in lbls:
                    print(f"    {f.name}")
                    if f.suffix.lower() in (".json", ".geojson"):
                        content = f.read_text()[:500]
                        print(f"    content preview: {content}")
            print()
        return

    # ── Collect all (image, label) pairs ─────────────────────────────────────
    import random
    rng = random.Random(args.seed)

    pairs = []
    for city in args.cities:
        city_dir = inp / city
        if not city_dir.exists():
            # try case-insensitive match
            matches = [d for d in inp.iterdir() if d.name.lower() == city.lower()]
            if matches:
                city_dir = matches[0]
            else:
                print(f"  WARNING: city directory '{city}' not found in {inp}")
                continue

        img_dir = city_dir / "images"
        lbl_dir = city_dir / "labels"
        if not img_dir.exists():
            print(f"  WARNING: no images/ in {city_dir}")
            continue
        if not lbl_dir.exists():
            print(f"  WARNING: no labels/ in {city_dir}")
            continue

        for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            # find matching label (try .shp, .json, .geojson)
            lbl_path = None
            for ext in (".shp", ".json", ".geojson"):
                candidate = lbl_dir / (stem + ext)
                if candidate.exists():
                    lbl_path = candidate
                    break
            if lbl_path:
                pairs.append((img_path, lbl_path, city))

    print(f"\nFound {len(pairs)} image/label pairs across cities: {args.cities}")
    if not pairs:
        sys.exit("No pairs found — run with --inspect to diagnose the directory structure")

    rng.shuffle(pairs)

    # ── Setup output dirs and S3 ───────────────────────────────────────────────
    for split in ("train", "valid"):
        (out / split).mkdir(parents=True, exist_ok=True)

    s3 = None
    if args.s3_bucket:
        import boto3
        s3 = boto3.client("s3")

    # ── Process pairs ─────────────────────────────────────────────────────────
    images_train, images_valid = [], []
    anns_train,   anns_valid   = [], []
    image_id = 1; ann_id = 1
    ok = skipped = 0

    for img_path, lbl_path, city in tqdm(pairs, desc="Processing", unit="building"):
        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
        except Exception as e:
            skipped += 1
            continue

        facet_polys = load_labels(lbl_path, orig_w, orig_h)
        if len(facet_polys) < 2:
            skipped += 1
            continue

        # Filter tiny or invalid polygons
        facet_polys = [p for p in facet_polys
                       if p is not None and p.is_valid
                       and not p.is_empty and p.area >= 4.0]
        if len(facet_polys) < 2:
            skipped += 1
            continue

        # Convert to COCO flat coords
        facet_flats = []
        for poly in facet_polys:
            flat = poly_to_coco_flat(poly, CHIP_PX, CHIP_PX, orig_w, orig_h)
            if flat and flat_area(flat) >= 1.0:
                facet_flats.append(flat)
        if len(facet_flats) < 2:
            skipped += 1
            continue

        # Roof polygon = union of all facets
        roof_flat = None
        try:
            union = unary_union(facet_polys)
            if union.geom_type == "MultiPolygon":
                union = max(union.geoms, key=lambda g: g.area)
            if union.geom_type == "Polygon":
                roof_flat = poly_to_coco_flat(union, CHIP_PX, CHIP_PX, orig_w, orig_h)
        except Exception:
            pass

        # Resize chip to CHIP_PX×CHIP_PX
        if img.size != (CHIP_PX, CHIP_PX):
            img = img.resize((CHIP_PX, CHIP_PX), Image.LANCZOS)

        # Stable filename based on source path hash
        fn  = hashlib.sha1(str(img_path).encode()).hexdigest()[:16] + ".png"
        is_val = rng.random() < args.val_frac
        split  = "valid" if is_val else "train"

        img.save(out / split / fn, format="PNG")
        if s3:
            key = f"{args.s3_prefix}/{fn}" if args.s3_prefix else fn
            try:
                s3.head_object(Bucket=args.s3_bucket, Key=key)
            except Exception:
                try:
                    s3.upload_file(str(out / split / fn), args.s3_bucket, key)
                except Exception as e:
                    print(f"\n  S3 fail {fn}: {e}")

        img_entry = {"id": image_id, "file_name": fn,
                     "width": CHIP_PX, "height": CHIP_PX,
                     "city": city}
        anns = []
        for flat in facet_flats:
            anns.append({
                "id": ann_id, "image_id": image_id,
                "category_id": CAT_FACET,
                "segmentation": [flat], "area": flat_area(flat),
                "bbox": flat_bbox(flat), "iscrowd": 0,
            })
            ann_id += 1
        if roof_flat:
            anns.append({
                "id": ann_id, "image_id": image_id,
                "category_id": CAT_ROOF,
                "segmentation": [roof_flat], "area": flat_area(roof_flat),
                "bbox": flat_bbox(roof_flat), "iscrowd": 0,
            })
            ann_id += 1
        image_id += 1

        if is_val:
            images_valid.append(img_entry); anns_valid.extend(anns)
        else:
            images_train.append(img_entry); anns_train.extend(anns)
        ok += 1

    # ── Write COCO JSON ───────────────────────────────────────────────────────
    for split, imgs, anns in [("train", images_train, anns_train),
                               ("valid", images_valid, anns_valid)]:
        coco = {"images": imgs, "annotations": anns, "categories": CATEGORIES}
        with open(out / split / "_annotations.coco.json", "w") as f:
            json.dump(coco, f)
        with open(out / split / "chips_needed_carecamp93.txt", "w") as f:
            f.write("".join(i["file_name"] + "\n" for i in imgs))

    total_facets = sum(1 for a in anns_train + anns_valid if a["category_id"] == CAT_FACET)
    print(f"\n{'='*55}")
    print(f"Done: {ok} chips  |  {skipped} skipped")
    print(f"  train: {len(images_train)} images")
    print(f"  valid: {len(images_valid)} images")
    print(f"  facet instances: {total_facets}")
    print(f"\nMerge into roof_dataset_v4:")
    print(f"  python training/merge_datasets.py \\")
    print(f"      --source training/roof_dataset_clean    phase1 \\")
    print(f"      --source training/carecamp93_dataset    carecamp93 \\")
    print(f"      --source training/switzerland_dataset   switzerland \\")
    print(f"      --output training/roof_dataset_v4")


if __name__ == "__main__":
    main()
