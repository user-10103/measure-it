#!/usr/bin/env python3
"""
Build COCO instance segmentation dataset from swissBUILDINGS3D 3.0 + SWISSIMAGE.

Data sources (both open / free):
  Buildings : swisstopo swissBUILDINGS3D 3.0  (CityGML LoD2, EPSG:2056 / LV95)
              https://www.swisstopo.admin.ch/en/landscape-model-buildings-3d-3
  Imagery   : swisstopo SWISSIMAGE 10cm WMS   (RGB, 10cm/px, EPSG:2056)
              https://wms.geo.admin.ch/

Why Switzerland is a good choice:
  - National LoD2 coverage (every building, not just cities)
  - Exceptionally accurate LiDAR source → tight roof-plane polygons
  - SWISSIMAGE is 10 cm vs PDOK's 25 cm → crisper chips
  - Different architecture styles than Florida/NL → better generalisation
  - All data is open government data (OGD), free to use commercially

Pipeline (single-machine, runs in Colab):
  1. Query swisstopo STAC API for CityGML tile ZIPs covering each city bbox
  2. Download ZIP → parse CityGML with lxml iterparse (memory-efficient)
  3. Filter buildings: MIN_FACETS=2, area thresholds, no uncertainty gate needed
  4. Fetch SWISSIMAGE WMS chip per accepted building (512×512, ~10–30 cm/px)
  5. Project LoD2 RoofSurface polygons → COCO flat polygon annotations
  6. Save chips as PNG; write COCO JSON + chips_needed_switzerland.txt
  7. Upload chips to S3 (optional)

Usage:
    python training/build_switzerland_dataset.py \\
        --output training/switzerland_dataset \\
        --target 5000 --workers 16 \\
        --s3-bucket florida-roofs-v4 --s3-prefix switzerland

    # Test connectivity + STAC discovery only:
    python training/build_switzerland_dataset.py --discover-only

Dependencies:
    pip install requests shapely pyproj Pillow tqdm boto3 lxml numpy
"""

import argparse
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
import requests
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from tqdm import tqdm

try:
    from lxml import etree as _ET
    _LXML = True
except ImportError:
    import xml.etree.ElementTree as _ET  # type: ignore
    _LXML = False
    print("[warn] lxml not found — falling back to stdlib ET (slower for large GML files)")
    print("       Install with: pip install lxml")

# ─── CityGML 2.0 namespace URIs ──────────────────────────────────────────────

_NS_BLDG = "http://www.opengis.net/citygml/building/2.0"
_NS_GML  = "http://www.opengis.net/gml"

# ─── API endpoints ────────────────────────────────────────────────────────────

STAC_BASE   = "https://data.geo.admin.ch/api/stac/v0.9"
COLLECTION  = "ch.swisstopo.swissbuildings3d_3-0"
STAC_ITEMS  = f"{STAC_BASE}/collections/{COLLECTION}/items"

SWISS_WMS   = "https://wms.geo.admin.ch/"
WMS_LAYER   = "ch.swisstopo.swissimage-product"  # best-available imagery, national
WMS_CRS     = "EPSG:2056"                         # LV95 Swiss national grid

# ─── Chip geometry ───────────────────────────────────────────────────────────

CHIP_PX      = 512
MARGIN_FRAC  = 0.40
MIN_CHIP_M   = 20.0    # metres — Swiss buildings can be smaller than Dutch
MAX_CHIP_M   = 150.0

# ─── Quality gates ────────────────────────────────────────────────────────────

MIN_FACET_M2  = 4.0    # m² — discard tiny slivers
MIN_FACETS    = 2      # must have at least 2 roof planes
MAX_FACETS    = 60
MIN_ROOF_M2   = 15.0
MAX_BLANK_FRAC = 0.05  # reject chips that are >5% pure white
SIMPLIFY_M    = 0.10   # polygon simplification (m) — Swiss data is precise

# ─── COCO categories ─────────────────────────────────────────────────────────

CAT_ROOF_POLYGON = 1
CAT_FACET        = 2
CATEGORIES = [
    {"id": CAT_ROOF_POLYGON, "name": "roof_polygon"},
    {"id": CAT_FACET,        "name": "facet"},
]

# ─── Swiss cities (WGS84 lon/lat for STAC bbox search) ───────────────────────
# Covers German-, French-, and Italian-speaking Switzerland for diversity.

CITY_BBOXES = [
    # German-speaking
    ("zurich_center",   8.490, 47.340, 8.590, 47.400),
    ("zurich_north",    8.495, 47.390, 8.585, 47.445),
    ("zurich_west",     8.440, 47.355, 8.520, 47.410),
    ("basel",           7.520, 47.520, 7.630, 47.590),
    ("bern",            7.380, 46.910, 7.500, 46.980),
    ("winterthur",      8.690, 47.465, 8.790, 47.525),
    ("st_gallen",       9.330, 47.390, 9.440, 47.455),
    ("lucerne",         8.255, 47.020, 8.360, 47.080),
    ("schaffhausen",    8.590, 47.680, 8.690, 47.730),
    ("thun",            7.565, 46.715, 7.680, 46.780),
    # French-speaking
    ("geneva",          6.090, 46.175, 6.205, 46.250),
    ("geneva_east",     6.160, 46.185, 6.250, 46.245),
    ("lausanne",        6.580, 46.490, 6.690, 46.560),
    ("fribourg",        7.110, 46.780, 7.195, 46.830),
    ("biel_bienne",     7.200, 47.110, 7.300, 47.175),
    # Italian-speaking
    ("lugano",          8.905, 45.980, 9.025, 46.050),
]


# ─── HTTP ─────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "measure-it-swiss/1.0 (academic)"


def _get(url, params=None, retries=5, backoff=1.0, timeout=60, stream=False):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout, stream=stream)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))


# ─── STAC tile discovery ───────────────────────────────────────────────────────

def fetch_tile_urls(lon_min, lat_min, lon_max, lat_max):
    """
    Return [(tile_id, download_url)] for CityGML ZIPs covering this WGS84 bbox.
    Handles STAC pagination automatically.
    """
    results  = []
    next_url = STAC_ITEMS
    params   = {
        "bbox":  f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "limit": 100,
        "f":     "application/json",
    }

    while next_url:
        try:
            r    = _get(next_url, params=params, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"    STAC query failed: {e}")
            break

        for feat in data.get("features", []):
            tile_id = feat.get("id", "unknown")
            assets  = feat.get("assets", {})
            url     = _pick_gml_url(assets)
            if url:
                results.append((tile_id, url))

        # Follow STAC "next" link (pagination)
        params   = None
        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break

    return results


def _pick_gml_url(assets: dict) -> str:
    """Extract the CityGML download URL from a STAC item's assets dict."""
    # Prefer explicit GML zip
    for key, asset in assets.items():
        href = asset.get("href", "")
        if href.endswith(".gml.zip"):
            return href
    # Fallback: any zip from swisstopo
    for key, asset in assets.items():
        href = asset.get("href", "")
        if href.endswith(".zip") and "swisstopo" in href.lower():
            return href
    # Last resort: any zip
    for key, asset in assets.items():
        href = asset.get("href", "")
        if href.endswith(".zip"):
            return href
    return ""


def test_stac_connection():
    """Print available tiles for one city bbox — useful for debugging."""
    print(f"Testing STAC API: {STAC_ITEMS}")
    try:
        tiles = fetch_tile_urls(8.490, 47.340, 8.590, 47.400)
        if tiles:
            print(f"  OK — found {len(tiles)} tile(s) for Zurich centre")
            for tid, url in tiles[:3]:
                print(f"    {tid}  →  {url}")
        else:
            print("  WARNING: 0 tiles returned — check collection name / bbox")
    except Exception as e:
        print(f"  ERROR: {e}")


def test_wms_connection():
    """Fetch a tiny WMS tile to verify SWISSIMAGE is reachable."""
    print(f"Testing SWISSIMAGE WMS: {SWISS_WMS}")
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "CRS": WMS_CRS,
        "BBOX": "2682000,1247000,2682512,1247512",
        "WIDTH": 64, "HEIGHT": 64,
        "LAYERS": WMS_LAYER,
        "FORMAT": "image/png",
        "STYLES": "",
    }
    try:
        r = _get(SWISS_WMS, params=params, timeout=15)
        img = Image.open(io.BytesIO(r.content))
        print(f"  OK — got {img.size} {img.mode} image")
    except Exception as e:
        print(f"  ERROR: {e}")


# ─── CityGML parsing ─────────────────────────────────────────────────────────

def _parse_pos_list(elem, dim=3):
    """Parse gml:posList / gml:pos text → [(x, y)] dropping Z."""
    text = (elem.text or "").strip()
    if not text:
        return []
    try:
        vals = list(map(float, text.split()))
    except ValueError:
        return []
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals) - (dim - 1), dim)]


def _parse_coordinates(elem):
    """Parse legacy gml:coordinates text 'x,y,z x,y,z …' → [(x, y)]."""
    text = (elem.text or "").strip()
    pts  = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts


def _build_polygon(coords):
    """Return a filtered, simplified Shapely Polygon or None."""
    if len(coords) < 3:
        return None
    try:
        p = ShapelyPolygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area < MIN_FACET_M2:
            return None
        p = p.simplify(SIMPLIFY_M, preserve_topology=True)
        if p.geom_type != "Polygon" or p.area < MIN_FACET_M2:
            return None
        return p
    except Exception:
        return None


def _extract_roof_polys_from_element(elem):
    """Find all gml:posList / gml:coordinates in a RoofSurface element → list of Polygon."""
    NS_PL   = f"{{{_NS_GML}}}posList"
    NS_POS  = f"{{{_NS_GML}}}pos"
    NS_CRDS = f"{{{_NS_GML}}}coordinates"

    polys = []
    # gml:posList (most common in CityGML 2.0)
    for pl in elem.iter(NS_PL):
        dim   = int(pl.get("srsDimension", "3"))
        coords = _parse_pos_list(pl, dim=dim)
        p = _build_polygon(coords)
        if p:
            polys.append(p)
    # gml:pos (one point per element — rare in LoD2)
    pos_pts = []
    for pos in elem.iter(NS_POS):
        dim  = int(pos.get("srsDimension", "3"))
        pts  = _parse_pos_list(pos, dim=dim)
        pos_pts.extend(pts)
    if pos_pts:
        p = _build_polygon(pos_pts)
        if p:
            polys.append(p)
    # gml:coordinates (legacy GML format)
    for crds in elem.iter(NS_CRDS):
        coords = _parse_coordinates(crds)
        p = _build_polygon(coords)
        if p:
            polys.append(p)
    return polys


def parse_citygml_buildings(path):
    """
    Yield (building_id, footprint_poly, [roof_surface_polys]) from a CityGML file.
    All geometries are Shapely Polygons in EPSG:2056 (LV95, metres).
    Uses iterparse + element clearing for O(1) memory regardless of file size.
    """
    NS_BLDG_B = f"{{{_NS_BLDG}}}Building"
    NS_BLDG_R = f"{{{_NS_BLDG}}}RoofSurface"
    NS_GML_ID = f"{{{_NS_GML}}}id"

    if _LXML:
        # lxml: filter on tag + sibling clearing
        context = _ET.iterparse(str(path), events=("end",), tag=NS_BLDG_B)
    else:
        context = _ET.iterparse(str(path), events=("end",))

    for event, elem in context:
        if not _LXML and elem.tag != NS_BLDG_B:
            continue

        bid = elem.get(NS_GML_ID, "") or elem.get("id", "")

        roof_polys = []
        for roof in elem.iter(NS_BLDG_R):
            roof_polys.extend(_extract_roof_polys_from_element(roof))

        # Memory: release element and its preceding siblings from the tree
        if _LXML:
            parent = elem.getparent()
            elem.clear()
            if parent is not None:
                while len(parent) and parent[0] is not elem:
                    del parent[0]
        else:
            elem.clear()

        if len(roof_polys) < MIN_FACETS:
            continue
        if len(roof_polys) > MAX_FACETS:
            continue

        total_area = sum(p.area for p in roof_polys)
        if total_area < MIN_ROOF_M2:
            continue

        try:
            footprint = unary_union(roof_polys).convex_hull
            if footprint.geom_type != "Polygon" or footprint.area < MIN_ROOF_M2:
                continue
        except Exception:
            continue

        yield bid, footprint, roof_polys


def download_parse_tile(tile_id, url, tmp_dir):
    """Download tile ZIP → unzip GML → parse buildings. Returns list."""
    try:
        r = _get(url, timeout=300)
        zip_data = r.content
    except Exception as e:
        print(f"\n    [{tile_id}] download failed: {e}")
        return []

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_data))
        gml_names = [n for n in zf.namelist() if n.endswith(".gml") or n.endswith(".xml")]
        if not gml_names:
            print(f"\n    [{tile_id}] no .gml in zip (contents: {zf.namelist()[:5]})")
            return []
        gml_path = Path(tmp_dir) / f"{tile_id}.gml"
        with zf.open(gml_names[0]) as src, open(gml_path, "wb") as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"\n    [{tile_id}] unzip failed: {e}")
        return []

    buildings = list(parse_citygml_buildings(gml_path))
    try:
        gml_path.unlink()
    except Exception:
        pass
    return buildings


# ─── SWISSIMAGE WMS chip fetch ────────────────────────────────────────────────

def is_blank(img, frac=MAX_BLANK_FRAC):
    return bool(np.all(np.array(img) > 248, axis=2).mean() > frac)


def fetch_swissimage_chip(footprint):
    """
    Fetch a CHIP_PX × CHIP_PX SWISSIMAGE chip centred on the building footprint.
    Returns (PIL.Image RGB, chip_bbox_lv95=(xmin,ymin,xmax,ymax)).
    Chip is in EPSG:2056 coordinates (metres).
    """
    bds  = footprint.bounds              # (xmin, ymin, xmax, ymax) in LV95
    w    = bds[2] - bds[0]
    h    = bds[3] - bds[1]
    dim  = min(max(max(w, h) * (1.0 + MARGIN_FRAC), MIN_CHIP_M), MAX_CHIP_M)
    cx   = (bds[0] + bds[2]) / 2
    cy   = (bds[1] + bds[3]) / 2
    half = dim / 2
    xmin = cx - half; xmax = cx + half
    ymin = cy - half; ymax = cy + half

    # WMS 1.3.0 with EPSG:2056: BBOX = xmin,ymin,xmax,ymax (Easting,Northing order)
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "CRS":    WMS_CRS,
        "BBOX":   f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}",
        "WIDTH":  CHIP_PX, "HEIGHT": CHIP_PX,
        "LAYERS": WMS_LAYER,
        "FORMAT": "image/png",
        "STYLES": "",
    }
    r   = _get(SWISS_WMS, params=params, timeout=30)
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img, (xmin, ymin, xmax, ymax)


# ─── Project LV95 polygon → pixel COCO flat list ─────────────────────────────

def poly_to_pixels(poly, xmin, ymin, xmax, ymax, px=CHIP_PX):
    """Project an LV95 Shapely Polygon to COCO flat pixel coords [x1,y1,x2,y2,…]."""
    def pt(x, y):
        return (
            (x - xmin) / (xmax - xmin) * px,
            (ymax - y) / (ymax - ymin) * px,   # flip Y (image top = max Northing)
        )
    try:
        coords = [pt(x, y) for x, y in poly.exterior.coords]
    except Exception:
        return None
    flat = [v for c in coords for v in (float(c[0]), float(c[1]))]
    if len(flat) < 6:
        return None
    xs = flat[0::2]; ys = flat[1::2]
    if min(xs) > px or max(xs) < 0 or min(ys) > px or max(ys) < 0:
        return None   # completely outside chip
    return flat


def flat_bbox(flat):
    xs = flat[0::2]; ys = flat[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def flat_area(flat):
    xs = flat[0::2]; ys = flat[1::2]
    n  = len(xs)
    return abs(sum(xs[i]*ys[(i+1)%n] - xs[(i+1)%n]*ys[i] for i in range(n))) / 2.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="swissBUILDINGS3D 3.0 + SWISSIMAGE → COCO roof-facet dataset"
    )
    ap.add_argument("--output",        default="training/switzerland_dataset")
    ap.add_argument("--target",        type=int,   default=5_000,
                    help="Target accepted chips (default 5000)")
    ap.add_argument("--val-frac",      type=float, default=0.10)
    ap.add_argument("--workers",       type=int,   default=16,
                    help="Parallel WMS fetch workers (default 16)")
    ap.add_argument("--s3-bucket",     default=None)
    ap.add_argument("--s3-prefix",     default="switzerland")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--cities",        nargs="*",  default=None,
                    help="Process only these city names (default: all)")
    ap.add_argument("--resume",        action="store_true",
                    help="Skip building IDs already in _done_ids.txt")
    ap.add_argument("--discover-only", action="store_true",
                    help="Test STAC + WMS connectivity then exit")
    args = ap.parse_args()

    if args.discover_only:
        print("=== Connection test ===")
        test_stac_connection()
        print()
        test_wms_connection()
        print("\nDone.")
        return

    out = Path(args.output)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "valid").mkdir(parents=True, exist_ok=True)

    done_file = out / "_done_ids.txt"
    done_ids: set = set()
    if args.resume and done_file.exists():
        done_ids = {l.strip() for l in done_file.read_text().splitlines() if l.strip()}
        print(f"Resume: {len(done_ids)} already processed")

    cities = [c for c in CITY_BBOXES if not args.cities or c[0] in args.cities]
    if not cities:
        sys.exit("No cities matched — check --cities argument")

    # ── Step 1: STAC tile discovery ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"swissBUILDINGS3D 3.0 → COCO  |  target {args.target} chips")
    print(f"Cities: {[c[0] for c in cities]}")
    print(f"{'='*60}")
    print(f"\n[1/4] Querying STAC API for {len(cities)} city bboxes …")
    print(f"  Collection: {COLLECTION}")

    seen_tiles = set()
    tile_queue = []
    for name, lon_min, lat_min, lon_max, lat_max in cities:
        print(f"  {name:<24} … ", end="", flush=True)
        try:
            tiles = fetch_tile_urls(lon_min, lat_min, lon_max, lat_max)
            new   = [(tid, url) for tid, url in tiles if tid not in seen_tiles]
            for tid, url in new:
                seen_tiles.add(tid)
                tile_queue.append((tid, url))
            print(f"{len(new)} new tiles")
        except Exception as e:
            print(f"FAILED ({e})")

    print(f"  Total unique tiles to process: {len(tile_queue)}")
    if not tile_queue:
        sys.exit(
            "\nERROR: No tiles found.\n"
            "Check connectivity and collection name:\n"
            f"  curl '{STAC_ITEMS}?bbox=8.49,47.34,8.59,47.40&limit=5'\n"
            "Run with --discover-only for a connection test."
        )

    # ── Step 2: Download & parse CityGML tiles ────────────────────────────────
    t0 = time.time()
    print(f"\n[2/4] Downloading & parsing CityGML tiles …")
    all_buildings = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (tile_id, url) in enumerate(tile_queue, 1):
            mb = 0
            print(f"  [{i}/{len(tile_queue)}] {tile_id[:40]:<42}", end="", flush=True)
            bldgs = download_parse_tile(tile_id, url, tmp_dir)
            bldgs = [(bid, fp, s) for bid, fp, s in bldgs if bid not in done_ids]
            all_buildings.extend(bldgs)
            print(f" {len(bldgs):>5} bldgs  (total: {len(all_buildings)})")
            if len(all_buildings) >= args.target * 3:
                print(f"  Sufficient candidates — stopping tile downloads early")
                break

    print(f"  Parsed {len(all_buildings)} candidate buildings  ({time.time()-t0:.0f}s)")
    if not all_buildings:
        sys.exit(
            "\nERROR: No buildings parsed.\n"
            "The CityGML files may use a different namespace or coordinate dimension.\n"
            "Try --discover-only to confirm tiles download correctly."
        )

    random.Random(args.seed).shuffle(all_buildings)
    # 2× oversample to absorb WMS failures / blank chips
    candidates = all_buildings[:max(args.target * 2, len(all_buildings))]

    # ── Step 3: Fetch SWISSIMAGE chips ────────────────────────────────────────
    t2 = time.time()
    print(f"\n[3/4] Fetching SWISSIMAGE chips ({args.workers} workers) …")
    print(f"  WMS layer : {WMS_LAYER}  ({WMS_CRS})")

    s3 = None
    if args.s3_bucket:
        import boto3
        s3 = boto3.client("s3")

    rng       = random.Random(args.seed + 1)
    images_train, images_valid = [], []
    anns_train,   anns_valid   = [], []
    image_id = 1; ann_id = 1
    ok = rejected = 0
    done_log = open(done_file, "a")

    def process_one(item):
        bid, footprint, surfaces = item
        try:
            chip, chip_bbox = fetch_swissimage_chip(footprint)
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
        fut_map = {ex.submit(process_one, item): item[0] for item in candidates}
        pbar    = tqdm(as_completed(fut_map), total=len(fut_map),
                       unit="chip", desc="SWISSIMAGE")
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
            if ok >= args.target:
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
                        print(f"\n  S3 fail {fn}: {e}")

            img_entry = {"id": image_id, "file_name": fn,
                         "width": CHIP_PX, "height": CHIP_PX}
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
                    "category_id": CAT_ROOF_POLYGON,
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
            done_log.write(bid + "\n"); done_log.flush()
            pbar.set_postfix(ok=ok, fail=rejected)

    done_log.close()

    # ── Step 4: Write COCO JSON + chips_needed ─────────────────────────────────
    print(f"\n[4/4] Writing COCO JSON + chips_needed …")
    for split, imgs, anns in [("train", images_train, anns_train),
                               ("valid", images_valid, anns_valid)]:
        coco  = {"images": imgs, "annotations": anns, "categories": CATEGORIES}
        jpath = out / split / "_annotations.coco.json"
        with open(jpath, "w") as f:
            json.dump(coco, f)
        with open(out / split / "chips_needed_switzerland.txt", "w") as f:
            f.write("".join(i["file_name"] + "\n" for i in imgs))
        print(f"  {split}: {len(imgs)} chips → {jpath}")

    total_facets = sum(1 for a in anns_train + anns_valid if a["category_id"] == CAT_FACET)
    elapsed      = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  Accepted: {ok}  |  Rejected: {rejected}")
    print(f"  Train: {len(images_train)} | Valid: {len(images_valid)}")
    print(f"  Facet instances: {total_facets}  "
          f"(avg {total_facets/max(ok,1):.1f}/chip)")
    print(f"\nNext — merge into roof_dataset_v4:")
    print(f"  python training/merge_datasets.py \\")
    print(f"      --source training/roof_dataset_clean    phase1 \\")
    print(f"      --source training/rid2_dataset          rid2 \\")
    print(f"      --source training/switzerland_dataset   switzerland \\")
    print(f"      --output training/roof_dataset_v4")
    print(f"\nThen fetch Switzerland chips in Colab:")
    print(f"  python training/fetch_chips.py \\")
    print(f"      --bucket {args.s3_bucket or 'florida-roofs-v4'} \\")
    print(f"      --prefix switzerland \\")
    print(f"      --dataset training/roof_dataset_v4 \\")
    print(f"      --listing chips_needed_switzerland.txt")


if __name__ == "__main__":
    main()
