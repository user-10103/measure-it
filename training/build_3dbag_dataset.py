#!/usr/bin/env python3
"""
Build COCO instance segmentation dataset from 3DBAG LoD2 + PDOK aerial imagery.

Speed design
────────────
Two-phase pipeline — no spatial batching (which would blur small buildings):

  Phase A  Parallel-fetch all CityJSONs for candidate buildings (~300 ms each).
           Filter by reconstruction uncertainty and facet count.  No PDOK yet.
           Only buildings that pass quality gates proceed to Phase B.

  Phase B  Parallel-fetch one PDOK WMS chip per accepted building at the
           building's native scale (40–200 m chip, 25 cm/px source).
           Each chip is requested individually so resolution is fully preserved.

Fleet mode (--num-shards / --shard-idx)
───────────────────────────────────────
Split the 30-city list across N EC2 instances running simultaneously.
Each instance targets a different set of cities → different PDOK server load
distribution → effectively 4× the per-IP rate-limit headroom.
Chip filenames are sha1(building_id) so there are no S3 key collisions.
Each shard writes its own COCO JSON; merge_3dbag_shards.py combines them.

Timing with 4 × c5n.xlarge (32 workers each, targeting 5 000 chips each):
  • Building discovery:  ~2 min  per shard
  • Phase A (CityJSON):  ~3 min  (5 000 candidates × 300 ms ÷ 32 workers)
  • Phase B (PDOK WMS):  ~5 min  (3 000 accepted  × 1 s   ÷ 32 workers)
  • S3 upload + JSON:    ~2 min
  Total per shard: ~12–30 min  →  4 shards in parallel = same wall-clock time
  All 20 000 chips in ~30 min–2 hr depending on PDOK response time.

Dependencies:
    pip install requests shapely pyproj Pillow tqdm boto3 numpy

Usage:
    # Single instance — all 30 cities:
    python training/build_3dbag_dataset.py \\
        --output /data/3dbag_dataset --target 20000 --workers 32 \\
        --s3-bucket florida-roofs-v4 --s3-prefix 3dbag

    # Fleet mode — 4 parallel EC2 instances (shard 0–3):
    python training/build_3dbag_dataset.py \\
        --output /data/3dbag_shard_0 --target 5000 --workers 32 \\
        --num-shards 4 --shard-idx 0 \\
        --s3-bucket florida-roofs-v4 --s3-prefix 3dbag
"""

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import requests
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import pyproj
from tqdm import tqdm

# ─── API endpoints ────────────────────────────────────────────────────────────

THREDBAG_BASE = "https://api.3dbag.nl"
PDOK_WMS      = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
PDOK_LAYER    = "Actueel_ortho25"   # 25 cm/px, nationwide coverage

# ─── Chip geometry ───────────────────────────────────────────────────────────

MARGIN_FRAC  = 0.40
MIN_CHIP_M   = 40.0
MAX_CHIP_M   = 200.0

# ─── Quality gates ────────────────────────────────────────────────────────────

MIN_FACET_M2    = 4.0
MIN_FACETS      = 2
MAX_FACETS      = 50
MIN_ROOF_M2     = 15.0
MAX_BLANK_FRAC  = 0.05
MAX_UNCERTAINTY = 0.40
SIMPLIFY_M      = 0.20

# ─── COCO categories ─────────────────────────────────────────────────────────

CAT_ROOF_POLYGON = 1
CAT_FACET        = 2
CATEGORIES = [
    {"id": CAT_ROOF_POLYGON, "name": "roof_polygon"},
    {"id": CAT_FACET,        "name": "facet"},
]

# ─── Coordinate transform ─────────────────────────────────────────────────────

_wgs84_to_rd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)

# ─── 30-city coverage (WGS84 lon/lat for 3DBAG API) ──────────────────────────

CITY_BBOXES = [
    ("amsterdam_center",  4.855, 52.345, 4.935, 52.395),
    ("amsterdam_west",    4.820, 52.365, 4.880, 52.405),
    ("amsterdam_east",    4.930, 52.340, 4.990, 52.380),
    ("amsterdam_south",   4.870, 52.310, 4.940, 52.360),
    ("rotterdam_center",  4.420, 51.895, 4.500, 51.940),
    ("rotterdam_south",   4.460, 51.860, 4.540, 51.905),
    ("rotterdam_north",   4.430, 51.930, 4.510, 51.970),
    ("utrecht",           5.060, 52.065, 5.140, 52.115),
    ("den_haag_center",   4.270, 52.050, 4.360, 52.100),
    ("den_haag_west",     4.220, 52.060, 4.290, 52.100),
    ("eindhoven",         5.420, 51.420, 5.500, 51.470),
    ("groningen",         6.530, 53.190, 6.600, 53.230),
    ("tilburg",           5.060, 51.545, 5.120, 51.585),
    ("breda",             4.740, 51.575, 4.820, 51.615),
    ("nijmegen",          5.840, 51.825, 5.900, 51.860),
    ("haarlem",           4.620, 52.370, 4.680, 52.400),
    ("leiden",            4.470, 52.145, 4.520, 52.175),
    ("almere",            5.200, 52.355, 5.290, 52.405),
    ("arnhem",            5.880, 51.975, 5.940, 52.010),
    ("enschede",          6.870, 52.200, 6.940, 52.240),
    ("amersfoort",        5.365, 52.145, 5.420, 52.180),
    ("maastricht",        5.675, 50.840, 5.720, 50.870),
    ("zwolle",            6.085, 52.495, 6.140, 52.525),
    ("deventer",          6.150, 52.245, 6.210, 52.275),
    ("dordrecht",         4.650, 51.800, 4.720, 51.840),
    ("delft",             4.340, 51.990, 4.390, 52.020),
    ("apeldoorn",         5.950, 52.200, 6.020, 52.240),
    ("zaandam",           4.810, 52.425, 4.870, 52.460),
    ("hertogenbosch",     5.280, 51.680, 5.340, 51.720),
    ("alkmaar",           4.720, 52.620, 4.770, 52.650),
]


# ─── HTTP ─────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "measure-it-3dbag/1.0 (academic)"


def _get(url, params=None, retries=5, backoff=1.0):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))


# ─── 3DBAG helpers ────────────────────────────────────────────────────────────

def fetch_building_ids(lon_min, lat_min, lon_max, lat_max, page=500):
    """Return [(building_id, footprint_polygon_in_RD_New)] for bbox."""
    results, offset = [], 0
    while True:
        r = _get(f"{THREDBAG_BASE}/collections/pand/items", params={
            "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "limit": page, "offset": offset, "f": "json",
        })
        feats = r.json().get("features", [])
        if not feats:
            break
        for feat in feats:
            bid  = feat.get("id", "")
            geom = feat.get("geometry")
            if not geom or geom["type"] not in ("Polygon", "MultiPolygon"):
                continue
            ring = (geom["coordinates"][0] if geom["type"] == "Polygon"
                    else geom["coordinates"][0][0])
            xs, ys = _wgs84_to_rd.transform([p[0] for p in ring], [p[1] for p in ring])
            poly = ShapelyPolygon(zip(xs, ys))
            if poly.is_valid and poly.area > 0:
                results.append((bid, poly))
        if len(feats) < page:
            break
        offset += page
    return results


def fetch_cityjson(building_id):
    r = _get(f"{THREDBAG_BASE}/collections/pand/items/{building_id}",
             params={"f": "cityjson"})
    return r.json()


def extract_roof_surfaces(cjson):
    """Parse LoD2.2 RoofSurface polygons → (list[ShapelyPolygon_2D], uncertainty)."""
    tf   = cjson.get("transform", {})
    sc   = tf.get("scale",     [1, 1, 1])
    tr   = tf.get("translate", [0, 0, 0])
    vraw = cjson.get("vertices", [])

    def vxy(v):
        return (v[0] * sc[0] + tr[0], v[1] * sc[1] + tr[1])

    verts = [vxy(v) for v in vraw]
    polys, uncert = [], 0.0

    for obj in cjson.get("CityObjects", {}).values():
        if obj.get("type") not in ("Building", "BuildingPart"):
            continue
        u = obj.get("attributes", {}).get("b3_reconstructie_onzekerheid", 0.0)
        uncert = max(uncert, float(u))
        for geom in obj.get("geometry", []):
            if str(geom.get("lod", "")) != "2.2":
                continue
            sem   = geom.get("semantics", {})
            styps = sem.get("surfaces", [])
            svals = sem.get("values",   [])
            for i, boundary in enumerate(geom.get("boundaries", [])):
                if i >= len(svals) or svals[i] is None:
                    continue
                if styps[svals[i]].get("type") != "RoofSurface":
                    continue
                if not boundary or not boundary[0]:
                    continue
                try:
                    coords = [verts[vi] for vi in boundary[0]]
                    p = ShapelyPolygon(coords)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.is_empty or p.area < MIN_FACET_M2:
                        continue
                    p = p.simplify(SIMPLIFY_M, preserve_topology=True)
                    if p.geom_type == "Polygon" and p.area >= MIN_FACET_M2:
                        polys.append(p)
                except (IndexError, ValueError):
                    continue
    return polys, uncert


def is_blank(img, frac=MAX_BLANK_FRAC):
    return bool(np.all(np.array(img) > 248, axis=2).mean() > frac)


def fetch_building_chip(footprint):
    """
    Fetch a 512×512 PDOK WMS chip centred on the building at native resolution.
    Chip covers building bbox + MARGIN_FRAC padding, capped at MAX_CHIP_M.
    Returns (PIL.Image, chip_rd_bbox) — one WMS request per building, full quality.
    """
    bds  = footprint.bounds
    w    = bds[2] - bds[0]; h = bds[3] - bds[1]
    dim  = min(max(max(w, h) * (1.0 + MARGIN_FRAC), MIN_CHIP_M), MAX_CHIP_M)
    cx   = (bds[0] + bds[2]) / 2; cy = (bds[1] + bds[3]) / 2
    half = dim / 2
    xmin = cx - half; xmax = cx + half
    ymin = cy - half; ymax = cy + half

    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "CRS":  "EPSG:28992",
        "BBOX": f"{xmin:.3f},{ymin:.3f},{xmax:.3f},{ymax:.3f}",
        "WIDTH": CHIP_PX, "HEIGHT": CHIP_PX,
        "LAYERS": PDOK_LAYER, "FORMAT": "image/png", "STYLES": "",
    }
    r   = _get(PDOK_WMS, params=params)
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return img, (xmin, ymin, xmax, ymax)


# ─── Project RD-New polygon → pixel COCO flat list ───────────────────────────

def poly_to_pixels(poly, xmin, ymin, xmax, ymax, px=CHIP_PX):
    def pt(x, y):
        return ((x - xmin) / (xmax - xmin) * px,
                (ymax - y) / (ymax - ymin) * px)
    try:
        coords = [pt(x, y) for x, y in poly.exterior.coords]
    except Exception:
        return None
    flat = [v for c in coords for v in (float(c[0]), float(c[1]))]
    if len(flat) < 6:
        return None
    xs = flat[0::2]; ys = flat[1::2]
    if min(xs) > px or max(xs) < 0 or min(ys) > px or max(ys) < 0:
        return None
    return flat


def flat_bbox(flat):
    xs = flat[0::2]; ys = flat[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def flat_area(flat):
    xs = flat[0::2]; ys = flat[1::2]
    n = len(xs)
    return abs(sum(xs[i]*ys[(i+1)%n] - xs[(i+1)%n]*ys[i] for i in range(n))) / 2.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="3DBAG LoD2 + PDOK WMS → COCO dataset (two-phase, spatial batching)"
    )
    ap.add_argument("--output",     default="training/3dbag_dataset")
    ap.add_argument("--target",     type=int,   default=20_000,
                    help="Target accepted chips (default 20 000)")
    ap.add_argument("--val-frac",   type=float, default=0.10)
    ap.add_argument("--workers",    type=int,   default=32,
                    help="Parallel worker threads (default 32)")
    ap.add_argument("--s3-bucket",  default=None)
    ap.add_argument("--s3-prefix",  default="3dbag")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--resume",     action="store_true",
                    help="Skip building IDs already in _done_ids.txt")
    ap.add_argument("--num-shards", type=int,   default=1,
                    help="Total EC2 instances in fleet (default 1 = no sharding)")
    ap.add_argument("--shard-idx",  type=int,   default=0,
                    help="Which shard this instance handles (0-indexed)")
    ap.add_argument("--cities",     nargs="*",  default=None,
                    help="Process only these city names (space-separated)")
    args = ap.parse_args()

    if args.shard_idx >= args.num_shards:
        sys.exit(f"ERROR: shard-idx {args.shard_idx} >= num-shards {args.num_shards}")

    out = Path(args.output)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "valid").mkdir(parents=True, exist_ok=True)

    done_file = out / "_done_ids.txt"
    done_ids: set = set()
    if args.resume and done_file.exists():
        done_ids = {l.strip() for l in done_file.read_text().splitlines() if l.strip()}
        print(f"Resume: {len(done_ids)} already processed")

    # Select cities for this shard
    all_cities = [b for b in CITY_BBOXES
                  if not args.cities or b[0] in args.cities]
    my_cities  = [c for i, c in enumerate(all_cities) if i % args.num_shards == args.shard_idx]
    if not my_cities:
        sys.exit("No cities assigned to this shard")

    print(f"\n{'='*60}")
    print(f"Shard {args.shard_idx}/{args.num_shards} | cities: {[c[0] for c in my_cities]}")
    print(f"Target: {args.target} chips | Workers: {args.workers}")
    print(f"{'='*60}")

    # ── Step 1: discover building IDs ─────────────────────────────────────────
    t0 = time.time()
    print("\n[1/4] Discovering buildings from 3DBAG API …")
    all_buildings = []
    for name, lon_min, lat_min, lon_max, lat_max in my_cities:
        print(f"  {name:<28} … ", end="", flush=True)
        try:
            bldgs = fetch_building_ids(lon_min, lat_min, lon_max, lat_max)
            bldgs = [(b, fp) for b, fp in bldgs if b not in done_ids]
            all_buildings.extend(bldgs)
            print(f"{len(bldgs)} buildings")
        except Exception as e:
            print(f"FAILED ({e})")

    random.Random(args.seed).shuffle(all_buildings)
    # Generous over-sample to hit target after rejections (~40-50% rejection rate)
    candidates = all_buildings[:args.target * 4]
    print(f"  Total candidates: {len(candidates)} ({time.time()-t0:.0f}s)")

    # ── Step 2: Phase A — batch-fetch all CityJSONs ───────────────────────────
    t1 = time.time()
    print(f"\n[2/4] Phase A — fetching {len(candidates)} CityJSONs ({args.workers} workers) …")

    parsed: dict = {}   # bid → (footprint, [roof_surfaces])
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(fetch_cityjson, bid): (bid, fp)
                   for bid, fp in candidates}
        pbar = tqdm(as_completed(fut_map), total=len(fut_map), unit="bldg")
        for fut in pbar:
            bid, fp = fut_map[fut]
            try:
                cjson = fut.result()
                surfaces, uncert = extract_roof_surfaces(cjson)
                if uncert > MAX_UNCERTAINTY:
                    continue
                if not (MIN_FACETS <= len(surfaces) <= MAX_FACETS):
                    continue
                if sum(p.area for p in surfaces) < MIN_ROOF_M2:
                    continue
                parsed[bid] = (fp, surfaces)
            except Exception:
                pass
            pbar.set_postfix(accepted=len(parsed))

    accepted_buildings = list(parsed.items())  # [(bid, (fp, surfaces)), …]
    random.Random(args.seed).shuffle(accepted_buildings)
    accepted_buildings = accepted_buildings[:args.target]
    print(f"  Accepted: {len(accepted_buildings)} / {len(candidates)} "
          f"({time.time()-t1:.0f}s)")

    # ── Step 3: Phase B — fetch PDOK chip per accepted building ──────────────
    t2 = time.time()
    print(f"\n[3/4] Phase B — fetching {len(accepted_buildings)} PDOK chips "
          f"({args.workers} workers, 1 request/building, full resolution) …")

    s3 = None
    if args.s3_bucket:
        import boto3
        s3 = boto3.client("s3")

    rng      = random.Random(args.seed + 1)
    images_train, images_valid = [], []
    anns_train,   anns_valid   = [], []
    image_id = 1; ann_id = 1
    ok = rejected = 0
    done_log = open(done_file, "a")

    def process_one(item):
        bid, (fp, surfaces) = item
        try:
            chip, chip_bbox = fetch_building_chip(fp)
        except Exception:
            return None
        if is_blank(chip):
            return None
        cxmin, cymin, cxmax, cymax = chip_bbox
        facet_flats = []
        for poly in surfaces:
            flat = poly_to_pixels(poly, cxmin, cymin, cxmax, cymax)
            if flat and flat_area(flat) >= 1.0:
                facet_flats.append(flat)
        if len(facet_flats) < MIN_FACETS:
            return None
        roof_flat = None
        try:
            union = unary_union(surfaces)
            if union.geom_type == "MultiPolygon":
                union = max(union.geoms, key=lambda g: g.area)
            union = union.simplify(SIMPLIFY_M, preserve_topology=True)
            if union.geom_type == "Polygon":
                roof_flat = poly_to_pixels(union, cxmin, cymin, cxmax, cymax)
        except Exception:
            pass
        fn = hashlib.sha1(bid.encode()).hexdigest()[:16] + ".png"
        return bid, chip, fn, facet_flats, roof_flat

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(process_one, item): item[0]
                   for item in accepted_buildings}
        pbar = tqdm(as_completed(fut_map), total=len(fut_map),
                    unit="chip", desc="PDOK+crop")
        for fut in pbar:
            bid = fut_map[fut]
            try:
                result = fut.result()
            except Exception:
                result = None

            if result is None:
                rejected += 1
                done_log.write(bid + "\n")
                continue

            bid, chip, fn, facet_flats, roof_flat = result
            is_val = rng.random() < args.val_frac
            split  = "valid" if is_val else "train"

            chip.save(out / split / fn, format="PNG")
            if s3:
                key = f"{args.s3_prefix}/{fn}" if args.s3_prefix else fn
                try:
                    s3.head_object(Bucket=args.s3_bucket, Key=key)
                except Exception:
                    try:
                        s3.upload_file(str(out / split / fn), args.s3_bucket, key)
                    except Exception as e:
                        print(f"\nS3 fail {fn}: {e}")

            img_entry = {"id": image_id, "file_name": fn,
                         "width": CHIP_PX, "height": CHIP_PX}
            anns = []
            for flat in facet_flats:
                anns.append({"id": ann_id, "image_id": image_id,
                              "category_id": CAT_FACET,
                              "segmentation": [flat], "area": flat_area(flat),
                              "bbox": flat_bbox(flat), "iscrowd": 0})
                ann_id += 1
            if roof_flat:
                anns.append({"id": ann_id, "image_id": image_id,
                              "category_id": CAT_ROOF_POLYGON,
                              "segmentation": [roof_flat], "area": flat_area(roof_flat),
                              "bbox": flat_bbox(roof_flat), "iscrowd": 0})
                ann_id += 1
            image_id += 1

            if is_val:
                images_valid.append(img_entry); anns_valid.extend(anns)
            else:
                images_train.append(img_entry); anns_train.extend(anns)
            ok += 1
            done_log.write(bid + "\n"); done_log.flush()
            pbar.set_postfix(ok=ok, fail=rejected)

    done_log.close()

    # ── Step 4: write COCO JSONs + chips_needed ───────────────────────────────
    print(f"\n[4/4] Writing COCO JSON + chips_needed …")
    shard_tag = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
    for split, imgs, anns in [("train", images_train, anns_train),
                               ("valid", images_valid, anns_valid)]:
        coco = {"images": imgs, "annotations": anns, "categories": CATEGORIES}
        jpath = out / split / f"_annotations{shard_tag}.coco.json"
        with open(jpath, "w") as f:
            json.dump(coco, f)
        with open(out / split / f"chips_needed_3dbag{shard_tag}.txt", "w") as f:
            f.write("".join(i["file_name"] + "\n" for i in imgs))
        print(f"  {split}: {len(imgs)} chips → {jpath}")

    total_facets = sum(1 for a in anns_train + anns_valid if a["category_id"] == CAT_FACET)
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  Accepted: {ok} chips  |  Rejected: {rejected}")
    print(f"  Train: {len(images_train)} | Valid: {len(images_valid)}")
    print(f"  Total facet instances: {total_facets}")
    print(f"  Avg facets/chip: {total_facets/max(ok,1):.1f}")


if __name__ == "__main__":
    main()
