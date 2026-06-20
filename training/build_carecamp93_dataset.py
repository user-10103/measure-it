#!/usr/bin/env python3
"""
Convert carecamp93 Automatic Roof Plane Extraction dataset → COCO format.

Source: https://github.com/carecamp93/Automatic_Roof_Plane_Extraction
Cities:  Lozenets (Sofia BG), OudeMarkt (NL), Stadsveld (Enschede NL)
         (README claims Las Vegas + Atlanta + Paris but Drive only contains European cities)

Annotation format: .npy files — Python dicts mapping (x,y) corner tuples to
                   lists of connected (x,y) neighbor tuples (planar graph adjacency).
                   We reconstruct polygon rings via Shapely polygonize().
Image format:      256×256 RGB .jpg chips in rgb/ or clipped/ subdirs.

Download the Stage 1 data from:
  https://drive.google.com/drive/folders/12AmomRCLc28QwAtFo-9YXQpJq4Q_FTAk

Run:
    python training/build_carecamp93_dataset.py \\
        --input  /content/carecamp93/stage1 \\
        --output training/carecamp93_dataset \\
        --s3-bucket florida-roofs-v4 \\
        --s3-prefix carecamp93

Run --inspect first to see the actual folder/file structure:
    python training/build_carecamp93_dataset.py --input /content/carecamp93/stage1 --inspect

Dependencies:
    pip install numpy geopandas shapely Pillow tqdm boto3
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    from shapely.ops import polygonize, unary_union
    from shapely.geometry import Polygon, MultiPolygon, LineString
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
    from shapely.ops import polygonize, unary_union
    from shapely.geometry import Polygon, MultiPolygon, LineString

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
CAT_ROOF  = 1
CAT_FACET = 2

CHIP_PX = 512  # resize all chips to 512×512 to match training set

# Folder name candidates tried in order
IMG_FOLDER_NAMES = ["rgb", "clipped", "chips", "images", "img", "RGB", "Clipped"]
LBL_FOLDER_NAMES = ["npy", "labels", "annotations", "gt", "NPY"]
IMG_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


# ─── Label parsers ────────────────────────────────────────────────────────────

def load_labels_npy(label_path: Path):
    """
    Load a .npy planar-graph adjacency dict from carecamp93.

    Format: dict mapping (x, y) corner tuple → list of (x, y) neighbor tuples.
    We build LineStrings from each edge, then run polygonize() to get closed
    polygon rings representing individual roof planes.

    Coordinates are in pixel space (0–255 for 256×256 chips).
    """
    try:
        data = np.load(str(label_path), allow_pickle=True).item()
    except Exception as e:
        print(f"  [npy] failed to load {label_path.name}: {e}")
        return []

    if not isinstance(data, dict):
        print(f"  [npy] unexpected format in {label_path.name}: {type(data)}")
        return []

    lines = []
    for corner, neighbors in data.items():
        if not (isinstance(corner, (tuple, list)) and len(corner) == 2):
            continue
        for nb in (neighbors or []):
            if isinstance(nb, (tuple, list)) and len(nb) == 2:
                lines.append(LineString([corner, nb]))

    if not lines:
        return []

    polys = list(polygonize(lines))
    valid = []
    for p in polys:
        if p is None or p.is_empty:
            continue
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_valid and not p.is_empty:
            valid.append(p)
    return valid


def load_labels_shp(label_path: Path):
    """Load shapefile — polygons directly or polyline edges → polygonize."""
    if not HAS_GPD:
        sys.exit("pip install geopandas  (needed for .shp labels)")
    gdf = gpd.read_file(str(label_path))
    polys = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            polys.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polys.extend(geom.geoms)
    if polys:
        return polys
    lines = [g for g in gdf.geometry if g is not None]
    return list(polygonize(lines))


def load_labels_json(label_path: Path, img_w: int, img_h: int):
    """Load JSON label — tries planes/annotations/bare-list formats."""
    with open(label_path) as f:
        data = json.load(f)

    def _flat_to_poly(flat):
        pts = [(flat[i], flat[i+1]) for i in range(0, len(flat)-1, 2)]
        if len(pts) >= 3:
            p = Polygon(pts)
            return p if p.is_valid else p.buffer(0)
        return None

    polys = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                p = _flat_to_poly(item)
                if p and not p.is_empty:
                    polys.append(p)
    elif isinstance(data, dict):
        for plane in data.get("planes", []):
            if isinstance(plane, list):
                p = _flat_to_poly(plane)
                if p and not p.is_empty:
                    polys.append(p)
        for ann in data.get("annotations", []):
            for seg in ann.get("segmentation", []):
                p = _flat_to_poly(seg)
                if p and not p.is_empty:
                    polys.append(p)
    return polys


def load_labels(label_path: Path, img_w: int, img_h: int):
    """Dispatch to correct parser based on file extension."""
    suffix = label_path.suffix.lower()
    if suffix == ".npy":
        return load_labels_npy(label_path)
    elif suffix == ".shp":
        return load_labels_shp(label_path)
    elif suffix in (".json", ".geojson"):
        return load_labels_json(label_path, img_w, img_h)
    else:
        print(f"  Unknown label format: {suffix} — skipping {label_path.name}")
        return []


# ─── Coordinate helpers ───────────────────────────────────────────────────────

def poly_to_coco_flat(poly: Polygon, orig_w: int, orig_h: int):
    """Convert Shapely Polygon in original pixel space → COCO flat list scaled to CHIP_PX."""
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


# ─── Folder discovery ─────────────────────────────────────────────────────────

def find_subdir(city_dir: Path, candidates: list) -> Path | None:
    """Return first existing subdirectory from a list of name candidates."""
    for name in candidates:
        p = city_dir / name
        if p.is_dir():
            return p
    return None


def find_label_exts(lbl_dir: Path):
    """Return the dominant label extension in a label directory."""
    counts = {}
    for f in lbl_dir.iterdir():
        ext = f.suffix.lower()
        if ext in (".npy", ".shp", ".json", ".geojson"):
            counts[ext] = counts.get(ext, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="carecamp93 roof planes (European cities) → COCO instance segmentation"
    )
    ap.add_argument("--input",     required=True,
                    help="Root directory of downloaded Stage 1 carecamp93 data")
    ap.add_argument("--output",    default="training/carecamp93_dataset")
    ap.add_argument("--val-frac",  type=float, default=0.10)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--cities",    nargs="*",  default=None,
                    help="Which city subdirectories to include (default: all found)")
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--s3-prefix", default="carecamp93")
    ap.add_argument("--inspect",   action="store_true",
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
            # show top-level files/dirs
            for item in sorted(city_dir.iterdir())[:8]:
                tag = "DIR" if item.is_dir() else f"{item.stat().st_size//1024}KB"
                print(f"  [{tag}] {item.name}")
                if item.is_dir():
                    sub = sorted(item.iterdir())[:4]
                    for s in sub:
                        print(f"        {s.name}")
                    if len(list(item.iterdir())) > 4:
                        print(f"        … ({len(list(item.iterdir()))} total)")

            # try to find image + label dirs
            img_dir = find_subdir(city_dir, IMG_FOLDER_NAMES)
            lbl_dir = find_subdir(city_dir, LBL_FOLDER_NAMES)
            print(f"  → images at: {img_dir.name if img_dir else 'NOT FOUND'}")
            print(f"  → labels at: {lbl_dir.name if lbl_dir else 'NOT FOUND'}")
            if lbl_dir:
                ext = find_label_exts(lbl_dir)
                print(f"  → label ext: {ext if ext else 'UNKNOWN'}")
                if ext == ".npy":
                    sample = next((f for f in lbl_dir.iterdir() if f.suffix == ".npy"), None)
                    if sample:
                        try:
                            data = np.load(str(sample), allow_pickle=True).item()
                            n_corners = len(data)
                            n_edges   = sum(len(v) for v in data.values())
                            print(f"  → sample .npy: {sample.name} — {n_corners} corners, {n_edges} edges")
                            polys = load_labels_npy(sample)
                            print(f"  → polygonize → {len(polys)} roof planes")
                        except Exception as e:
                            print(f"  → .npy load error: {e}")
            print()
        return

    # ── Collect city directories ──────────────────────────────────────────────
    import random
    rng = random.Random(args.seed)

    if args.cities:
        city_dirs = []
        for c in args.cities:
            cd = inp / c
            if not cd.exists():
                matches = [d for d in inp.iterdir() if d.name.lower() == c.lower()]
                cd = matches[0] if matches else None
            if cd:
                city_dirs.append(cd)
            else:
                print(f"  WARNING: city '{c}' not found in {inp}")
    else:
        city_dirs = [d for d in sorted(inp.iterdir()) if d.is_dir()]

    print(f"\nProcessing cities: {[d.name for d in city_dirs]}")

    # ── Collect all (image, label) pairs ─────────────────────────────────────
    pairs = []
    for city_dir in city_dirs:
        img_dir = find_subdir(city_dir, IMG_FOLDER_NAMES)
        lbl_dir = find_subdir(city_dir, LBL_FOLDER_NAMES)

        if not img_dir:
            print(f"  WARNING: no image dir found in {city_dir.name} (tried: {IMG_FOLDER_NAMES})")
            continue
        if not lbl_dir:
            print(f"  WARNING: no label dir found in {city_dir.name} (tried: {LBL_FOLDER_NAMES})")
            continue

        lbl_ext = find_label_exts(lbl_dir)
        if not lbl_ext:
            print(f"  WARNING: no .npy/.shp/.json labels in {lbl_dir}")
            continue

        print(f"  {city_dir.name}: images={img_dir.name}/ labels={lbl_dir.name}/ ext={lbl_ext}")

        for img_ext in IMG_EXTS:
            for img_path in sorted(img_dir.glob(f"*{img_ext}")):
                lbl_path = lbl_dir / (img_path.stem + lbl_ext)
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path, city_dir.name))

    print(f"\nFound {len(pairs)} image/label pairs")
    if not pairs:
        sys.exit("No pairs found — run with --inspect to diagnose the directory structure")

    rng.shuffle(pairs)

    # ── Setup output dirs ─────────────────────────────────────────────────────
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
        except Exception:
            skipped += 1
            continue

        facet_polys = load_labels(lbl_path, orig_w, orig_h)

        # Filter tiny / invalid
        facet_polys = [
            p for p in facet_polys
            if p is not None and p.is_valid and not p.is_empty and p.area >= 4.0
        ]
        if len(facet_polys) < 2:
            skipped += 1
            continue

        # Convert to COCO flat coords
        facet_flats = []
        for poly in facet_polys:
            flat = poly_to_coco_flat(poly, orig_w, orig_h)
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
                roof_flat = poly_to_coco_flat(union, orig_w, orig_h)
        except Exception:
            pass

        # Resize to CHIP_PX×CHIP_PX
        if img.size != (CHIP_PX, CHIP_PX):
            img = img.resize((CHIP_PX, CHIP_PX), Image.LANCZOS)

        fn     = hashlib.sha1(str(img_path).encode()).hexdigest()[:16] + ".png"
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
                     "width": CHIP_PX, "height": CHIP_PX, "city": city}
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

    total_facets = sum(1 for a in anns_train + anns_valid if a["category_id"] == CAT_FACET)
    print(f"\n{'='*55}")
    print(f"Done: {ok} chips  |  {skipped} skipped")
    print(f"  train: {len(images_train)} images")
    print(f"  valid: {len(images_valid)} images")
    print(f"  facet instances: {total_facets}")
    print(f"\nNext: merge into roof_dataset_v2:")
    print(f"  python training/merge_datasets.py \\")
    print(f"      --source training/roof_dataset_clean   phase1 \\")
    print(f"      --source training/carecamp93_dataset   carecamp93 \\")
    print(f"      --source training/switzerland_dataset  switzerland \\")
    print(f"      --output training/roof_dataset_v2")


if __name__ == "__main__":
    main()
