#!/usr/bin/env python3
"""
Convert carecamp93 Automatic Roof Plane Extraction dataset → COCO format.

Source: https://github.com/carecamp93/Automatic_Roof_Plane_Extraction
Cities:  Lozenets (Sofia BG), OudeMarkt (NL), Stadsveld (Enschede NL)

Two input modes:

  --clip-from-tif  [DEFAULT for this dataset]
      Large GeoTIFF orthophotos + building footprint shapefiles + roof plane shapefiles.
      Clips per-building chips, converts geo→pixel coordinates, polygonize() → COCO.
      File layout per city dir:
        Outlines_Buildings_Buffer2m.shp  ← building footprints (polygon)
        Inner_Buildings_Planes.shp       ← roof plane boundaries (linestrings)
        *.tif                            ← orthophoto
      Run:
        python training/build_carecamp93_dataset.py \\
            --input /content/carecamp93 \\
            --output training/carecamp93_dataset \\
            --clip-from-tif \\
            --s3-bucket florida-roofs-v4 --s3-prefix carecamp93

  --pre-clipped  (if you already have 256×256 chips + .npy/.shp labels)
      Expects per-city: rgb/ or clipped/ folder + npy/ or labels/ folder.
      Run:
        python training/build_carecamp93_dataset.py \\
            --input /content/carecamp93 --output training/carecamp93_dataset

Always run --inspect first to see what's in the input directory:
    python training/build_carecamp93_dataset.py --input /content/carecamp93 --inspect

Dependencies:
    pip install rasterio geopandas shapely numpy Pillow tqdm boto3
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    from shapely.ops import polygonize, unary_union
    from shapely.geometry import Polygon, MultiPolygon, LineString, box
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    from PIL import Image
except ImportError:
    sys.exit("pip install Pillow")

from tqdm import tqdm

# ─── COCO categories ──────────────────────────────────────────────────────────
CATEGORIES = [{"id": 1, "name": "roof_polygon"}, {"id": 2, "name": "facet"}]
CAT_ROOF  = 1
CAT_FACET = 2
CHIP_PX   = 512

# Folder name candidates for pre-clipped mode
IMG_FOLDER_NAMES = ["rgb", "clipped", "chips", "images", "img", "RGB"]
LBL_FOLDER_NAMES = ["npy", "labels", "annotations", "gt"]
IMG_EXTS = [".jpg", ".jpeg", ".png"]


# ─── File discovery ───────────────────────────────────────────────────────────

def find_tif(city_dir: Path) -> Path | None:
    """Return the largest .tif in the city dir (the main orthophoto)."""
    tifs = sorted(city_dir.rglob("*.tif"), key=lambda f: f.stat().st_size, reverse=True)
    # skip .ovr (overview files) and tiny thumbnails
    for t in tifs:
        if t.suffix.lower() in (".ovr",):
            continue
        if t.stat().st_size > 1_000_000:  # >1 MB = real orthophoto
            return t
    return tifs[0] if tifs else None


def find_shp(city_dir: Path, patterns: list[str]) -> Path | None:
    """Return first shapefile whose name contains any of the given substrings."""
    for shp in city_dir.rglob("*.shp"):
        name = shp.name.lower()
        for pat in patterns:
            if pat.lower() in name:
                return shp
    return None


def discover_city_files(city_dir: Path):
    """
    Returns (tif, footprints_shp, planes_shp) or None if anything is missing.
    """
    tif = find_tif(city_dir)
    footprints = find_shp(city_dir, ["buffer", "outline", "footprint", "building"])
    planes = find_shp(city_dir, ["plane", "roof", "inner"])

    # prefer Inner_Buildings_Planes over a generic match
    specific = find_shp(city_dir, ["inner_buildings_plane", "inner_roof"])
    if specific:
        planes = specific

    return tif, footprints, planes


# ─── Coordinate helpers ───────────────────────────────────────────────────────

def geo_ring_to_pixel(coords, inv_transform):
    """Convert list of (x, y) geo coords to pixel (col, row) pairs."""
    return [inv_transform * (x, y) for x, y in coords]


def geom_to_pixel_lines(geom, inv_transform):
    """
    Convert a Shapely geometry (LineString, Multi*, Polygon) from
    geo coordinates to pixel-space LineStrings, ready for polygonize().
    """
    lines = []
    if geom is None or geom.is_empty:
        return lines

    gtype = geom.geom_type
    if gtype == "LineString":
        pts = geo_ring_to_pixel(geom.coords, inv_transform)
        if len(pts) >= 2:
            lines.append(LineString(pts))
    elif gtype == "MultiLineString":
        for ls in geom.geoms:
            pts = geo_ring_to_pixel(ls.coords, inv_transform)
            if len(pts) >= 2:
                lines.append(LineString(pts))
    elif gtype == "Polygon":
        pts = geo_ring_to_pixel(geom.exterior.coords, inv_transform)
        if len(pts) >= 3:
            lines.append(LineString(pts))
    elif gtype == "MultiPolygon":
        for p in geom.geoms:
            pts = geo_ring_to_pixel(p.exterior.coords, inv_transform)
            if len(pts) >= 3:
                lines.append(LineString(pts))
    return lines


def lines_to_facet_polys(lines, chip_w: int, chip_h: int, min_area: float = 4.0):
    """Run polygonize on lines, filter tiny/invalid polygons."""
    polys = list(polygonize(lines))
    valid = []
    for p in polys:
        if p is None or p.is_empty:
            continue
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_valid and not p.is_empty and p.area >= min_area:
            valid.append(p)
    return valid


# ─── COCO format helpers ──────────────────────────────────────────────────────

def poly_to_coco_flat(poly: Polygon, orig_w: int, orig_h: int):
    """Scale polygon from original chip size to CHIP_PX × CHIP_PX."""
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


# ─── Pre-clipped mode helpers ─────────────────────────────────────────────────

def find_subdir(city_dir: Path, candidates: list) -> Path | None:
    for name in candidates:
        p = city_dir / name
        if p.is_dir():
            return p
    return None


def find_label_ext(lbl_dir: Path) -> str | None:
    counts = {}
    for f in lbl_dir.iterdir():
        ext = f.suffix.lower()
        if ext in (".npy", ".shp", ".json", ".geojson"):
            counts[ext] = counts.get(ext, 0) + 1
    return max(counts, key=counts.get) if counts else None


def load_labels_npy(label_path: Path):
    """Load .npy planar-graph adjacency dict → polygonize → Shapely Polygons."""
    try:
        data = np.load(str(label_path), allow_pickle=True).item()
    except Exception as e:
        return []
    if not isinstance(data, dict):
        return []
    lines = []
    for corner, neighbors in data.items():
        if not (isinstance(corner, (tuple, list)) and len(corner) == 2):
            continue
        for nb in (neighbors or []):
            if isinstance(nb, (tuple, list)) and len(nb) == 2:
                lines.append(LineString([corner, nb]))
    return lines_to_facet_polys(lines, 256, 256)


def load_labels_shp(label_path: Path):
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
    return lines_to_facet_polys(lines, 512, 512)


def load_labels_json(label_path: Path):
    with open(label_path) as f:
        data = json.load(f)
    def _flat_to_poly(flat):
        pts = [(flat[i], flat[i+1]) for i in range(0, len(flat)-1, 2)]
        if len(pts) >= 3:
            p = Polygon(pts)
            return p if p.is_valid else p.buffer(0)
        return None
    polys = []
    items = data if isinstance(data, list) else data.get("planes", [])
    for item in items:
        if isinstance(item, list):
            p = _flat_to_poly(item)
            if p and not p.is_empty:
                polys.append(p)
    return polys


def load_labels(label_path: Path, img_w: int, img_h: int):
    suffix = label_path.suffix.lower()
    if suffix == ".npy":
        return load_labels_npy(label_path)
    elif suffix == ".shp":
        return load_labels_shp(label_path)
    elif suffix in (".json", ".geojson"):
        return load_labels_json(label_path)
    return []


# ─── Core: process one building from a GeoTIFF ───────────────────────────────

def process_building_from_tif(tif_ds, building_geom, planes_in_building,
                               city_name: str, out_dir: Path,
                               rng, val_frac: float, s3_client, s3_bucket, s3_prefix,
                               state: dict):
    """
    Clip one building chip from an open rasterio dataset, convert roof plane
    geometries to pixel space, generate COCO annotations.

    state dict holds: image_id, ann_id, images_train, images_valid,
                      anns_train, anns_valid, ok, skipped
    """
    try:
        minx, miny, maxx, maxy = building_geom.bounds
        window = window_from_bounds(minx, miny, maxx, maxy, transform=tif_ds.transform)
        window = window.round_lengths(op="ceil").round_offsets(op="floor")

        # guard against zero-size windows
        if window.width < 4 or window.height < 4:
            state["skipped"] += 1
            return

        data = tif_ds.read(window=window)          # (bands, H, W)
        chip_transform = tif_ds.window_transform(window)
        inv_transform  = ~chip_transform

        _, chip_h, chip_w = data.shape

        # Convert to uint8 RGB
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)
        if data.shape[0] >= 3:
            chip_rgb = np.transpose(data[:3], (1, 2, 0))
        elif data.shape[0] == 1:
            chip_rgb = np.repeat(np.transpose(data, (1, 2, 0)), 3, axis=2)
        else:
            state["skipped"] += 1
            return

        # Convert roof plane geometries → pixel-space line segments → polygonize
        all_lines = []
        for geom in planes_in_building:
            all_lines.extend(geom_to_pixel_lines(geom, inv_transform))

        facet_polys = lines_to_facet_polys(all_lines, chip_w, chip_h)
        if len(facet_polys) < 2:
            state["skipped"] += 1
            return

        # Convert to COCO flat coords (scaled to CHIP_PX)
        facet_flats = []
        for poly in facet_polys:
            flat = poly_to_coco_flat(poly, chip_w, chip_h)
            if flat and flat_area(flat) >= 1.0:
                facet_flats.append(flat)
        if len(facet_flats) < 2:
            state["skipped"] += 1
            return

        # Roof = union of all facets
        roof_flat = None
        try:
            union = unary_union(facet_polys)
            if union.geom_type == "MultiPolygon":
                union = max(union.geoms, key=lambda g: g.area)
            if union.geom_type == "Polygon":
                roof_flat = poly_to_coco_flat(union, chip_w, chip_h)
        except Exception:
            pass

        # Save chip (resized to CHIP_PX)
        img = Image.fromarray(chip_rgb, "RGB")
        img = img.resize((CHIP_PX, CHIP_PX), Image.LANCZOS)

        src_str = f"{city_name}_{minx:.2f}_{miny:.2f}"
        fn = hashlib.sha1(src_str.encode()).hexdigest()[:16] + ".png"
        is_val = rng.random() < val_frac
        split  = "valid" if is_val else "train"

        img.save(out_dir / split / fn, format="PNG")

        if s3_client:
            key = f"{s3_prefix}/{fn}" if s3_prefix else fn
            try:
                s3_client.head_object(Bucket=s3_bucket, Key=key)
            except Exception:
                try:
                    s3_client.upload_file(str(out_dir / split / fn), s3_bucket, key)
                except Exception as e:
                    print(f"\n  S3 fail {fn}: {e}")

        img_entry = {"id": state["image_id"], "file_name": fn,
                     "width": CHIP_PX, "height": CHIP_PX, "city": city_name}
        anns = []
        for flat in facet_flats:
            anns.append({
                "id": state["ann_id"], "image_id": state["image_id"],
                "category_id": CAT_FACET,
                "segmentation": [flat], "area": flat_area(flat),
                "bbox": flat_bbox(flat), "iscrowd": 0,
            })
            state["ann_id"] += 1
        if roof_flat:
            anns.append({
                "id": state["ann_id"], "image_id": state["image_id"],
                "category_id": CAT_ROOF,
                "segmentation": [roof_flat], "area": flat_area(roof_flat),
                "bbox": flat_bbox(roof_flat), "iscrowd": 0,
            })
            state["ann_id"] += 1
        state["image_id"] += 1

        if is_val:
            state["images_valid"].append(img_entry)
            state["anns_valid"].extend(anns)
        else:
            state["images_train"].append(img_entry)
            state["anns_train"].extend(anns)
        state["ok"] += 1

    except Exception as e:
        state["skipped"] += 1


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="carecamp93 → COCO instance segmentation"
    )
    ap.add_argument("--input",          required=True)
    ap.add_argument("--output",         default="training/carecamp93_dataset")
    ap.add_argument("--clip-from-tif",  action="store_true",
                    help="Clip chips from large GeoTIFFs (use for Stage 1 data)")
    ap.add_argument("--val-frac",       type=float, default=0.10)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--cities",         nargs="*",  default=None,
                    help="City subdirectories to process (default: all found)")
    ap.add_argument("--s3-bucket",      default=None)
    ap.add_argument("--s3-prefix",      default="carecamp93")
    ap.add_argument("--inspect",        action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    # ── Inspection ────────────────────────────────────────────────────────────
    if args.inspect:
        print(f"\nInspecting {inp}\n")
        for city_dir in sorted(inp.iterdir()):
            if not city_dir.is_dir():
                continue
            print(f"── {city_dir.name} ──")
            for item in sorted(city_dir.iterdir())[:10]:
                tag = "DIR" if item.is_dir() else f"{item.stat().st_size//1024}KB"
                print(f"  [{tag:>10}] {item.name}")
                if item.is_dir():
                    for s in sorted(item.iterdir())[:4]:
                        print(f"              {s.name}")

            tif, footprints, planes = discover_city_files(city_dir)
            print(f"  → tif:        {tif.name if tif else 'NOT FOUND'}")
            print(f"  → footprints: {footprints.name if footprints else 'NOT FOUND'}")
            print(f"  → planes:     {planes.name if planes else 'NOT FOUND'}")

            if footprints and HAS_GPD:
                gdf = gpd.read_file(str(footprints))
                print(f"  → {len(gdf)} buildings  CRS: {gdf.crs}")
            if planes and HAS_GPD:
                gdf = gpd.read_file(str(planes))
                geom_types = gdf.geometry.geom_type.value_counts().to_dict()
                print(f"  → {len(gdf)} plane features  types: {geom_types}")
            print()
        return

    import random
    rng = random.Random(args.seed)

    if not HAS_GPD:
        sys.exit("pip install geopandas shapely")

    city_dirs = (
        [inp / c for c in args.cities if (inp / c).is_dir()]
        if args.cities
        else [d for d in sorted(inp.iterdir()) if d.is_dir()]
    )
    print(f"Cities: {[d.name for d in city_dirs]}")

    for split in ("train", "valid"):
        (out / split).mkdir(parents=True, exist_ok=True)

    s3 = None
    if args.s3_bucket:
        import boto3
        s3 = boto3.client("s3")

    state = dict(
        image_id=1, ann_id=1,
        images_train=[], images_valid=[],
        anns_train=[], anns_valid=[],
        ok=0, skipped=0,
    )

    # ── GeoTIFF clipping mode ─────────────────────────────────────────────────
    if args.clip_from_tif:
        if not HAS_RASTERIO:
            sys.exit("pip install rasterio")

        for city_dir in city_dirs:
            tif, footprints_path, planes_path = discover_city_files(city_dir)
            city = city_dir.name

            if not tif:
                print(f"  {city}: no GeoTIFF found — skipping")
                continue
            if not footprints_path:
                print(f"  {city}: no building footprint shapefile found — skipping")
                continue
            if not planes_path:
                print(f"  {city}: no roof planes shapefile found — skipping")
                continue

            print(f"\n{city}: {tif.name}")

            with rasterio.open(str(tif)) as src:
                tif_crs = src.crs

                # Load and reproject shapefiles to match TIF CRS
                footprints_gdf = gpd.read_file(str(footprints_path))
                if footprints_gdf.crs and footprints_gdf.crs != tif_crs:
                    footprints_gdf = footprints_gdf.to_crs(tif_crs)

                planes_gdf = gpd.read_file(str(planes_path))
                if planes_gdf.crs and planes_gdf.crs != tif_crs:
                    planes_gdf = planes_gdf.to_crs(tif_crs)

                print(f"  {len(footprints_gdf)} buildings, {len(planes_gdf)} plane features")

                # Build spatial index on planes for fast per-building lookup
                planes_sindex = planes_gdf.sindex

                for _, building_row in tqdm(footprints_gdf.iterrows(),
                                            total=len(footprints_gdf),
                                            desc=f"  {city}"):
                    building_geom = building_row.geometry
                    if building_geom is None or building_geom.is_empty:
                        state["skipped"] += 1
                        continue

                    # Find plane features intersecting this building
                    candidate_idx = list(planes_sindex.intersection(building_geom.bounds))
                    if not candidate_idx:
                        state["skipped"] += 1
                        continue
                    candidates = planes_gdf.iloc[candidate_idx]
                    intersecting = candidates[candidates.intersects(building_geom)]
                    if len(intersecting) < 2:
                        state["skipped"] += 1
                        continue

                    process_building_from_tif(
                        src, building_geom,
                        list(intersecting.geometry),
                        city, out, rng, args.val_frac,
                        s3, args.s3_bucket, args.s3_prefix,
                        state,
                    )

    # ── Pre-clipped chip mode ─────────────────────────────────────────────────
    else:
        pairs = []
        for city_dir in city_dirs:
            img_dir = find_subdir(city_dir, IMG_FOLDER_NAMES)
            lbl_dir = find_subdir(city_dir, LBL_FOLDER_NAMES)
            if not img_dir or not lbl_dir:
                print(f"  {city_dir.name}: missing image or label dir — skipping")
                continue
            lbl_ext = find_label_ext(lbl_dir)
            if not lbl_ext:
                print(f"  {city_dir.name}: no .npy/.shp/.json labels — skipping")
                continue
            for ext in IMG_EXTS:
                for img_path in sorted(img_dir.glob(f"*{ext}")):
                    lbl_path = lbl_dir / (img_path.stem + lbl_ext)
                    if lbl_path.exists():
                        pairs.append((img_path, lbl_path, city_dir.name))

        print(f"Found {len(pairs)} image/label pairs")
        if not pairs:
            sys.exit("No pairs found — run --inspect or use --clip-from-tif")

        rng.shuffle(pairs)

        for img_path, lbl_path, city in tqdm(pairs, desc="Processing"):
            try:
                img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size
            except Exception:
                state["skipped"] += 1
                continue

            facet_polys = load_labels(lbl_path, orig_w, orig_h)
            facet_polys = [p for p in facet_polys
                           if p is not None and p.is_valid
                           and not p.is_empty and p.area >= 4.0]
            if len(facet_polys) < 2:
                state["skipped"] += 1
                continue

            facet_flats = [
                f for poly in facet_polys
                for f in [poly_to_coco_flat(poly, orig_w, orig_h)]
                if f and flat_area(f) >= 1.0
            ]
            if len(facet_flats) < 2:
                state["skipped"] += 1
                continue

            roof_flat = None
            try:
                union = unary_union(facet_polys)
                if union.geom_type == "MultiPolygon":
                    union = max(union.geoms, key=lambda g: g.area)
                if union.geom_type == "Polygon":
                    roof_flat = poly_to_coco_flat(union, orig_w, orig_h)
            except Exception:
                pass

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

            img_entry = {"id": state["image_id"], "file_name": fn,
                         "width": CHIP_PX, "height": CHIP_PX, "city": city}
            anns = []
            for flat in facet_flats:
                anns.append({
                    "id": state["ann_id"], "image_id": state["image_id"],
                    "category_id": CAT_FACET,
                    "segmentation": [flat], "area": flat_area(flat),
                    "bbox": flat_bbox(flat), "iscrowd": 0,
                })
                state["ann_id"] += 1
            if roof_flat:
                anns.append({
                    "id": state["ann_id"], "image_id": state["image_id"],
                    "category_id": CAT_ROOF,
                    "segmentation": [roof_flat], "area": flat_area(roof_flat),
                    "bbox": flat_bbox(roof_flat), "iscrowd": 0,
                })
                state["ann_id"] += 1
            state["image_id"] += 1

            if is_val:
                state["images_valid"].append(img_entry)
                state["anns_valid"].extend(anns)
            else:
                state["images_train"].append(img_entry)
                state["anns_train"].extend(anns)
            state["ok"] += 1

    # ── Write COCO JSON ───────────────────────────────────────────────────────
    for split, imgs, anns in [
        ("train", state["images_train"], state["anns_train"]),
        ("valid", state["images_valid"], state["anns_valid"]),
    ]:
        coco = {"images": imgs, "annotations": anns, "categories": CATEGORIES}
        with open(out / split / "_annotations.coco.json", "w") as f:
            json.dump(coco, f)

    total_facets = sum(
        1 for a in state["anns_train"] + state["anns_valid"]
        if a["category_id"] == CAT_FACET
    )
    print(f"\n{'='*55}")
    print(f"Done: {state['ok']} chips  |  {state['skipped']} skipped")
    print(f"  train: {len(state['images_train'])} images")
    print(f"  valid: {len(state['images_valid'])} images")
    print(f"  facet instances: {total_facets}")
    print(f"\nNext — merge:")
    print(f"  python training/merge_datasets.py \\")
    print(f"      --source training/roof_dataset_clean   phase1 \\")
    print(f"      --source training/carecamp93_dataset   carecamp93 \\")
    print(f"      --source training/switzerland_dataset  switzerland \\")
    print(f"      --output training/roof_dataset_v2")


if __name__ == "__main__":
    main()
