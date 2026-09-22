#!/usr/bin/env python3
"""Generate an RF-DETR training dataset with PSEUDO-LABELS from the geometric
pipeline (no human annotation).

Why: facet F1 stayed at ~0 across training runs because facet labels were
misaligned (CRS bugs in three prior label-generation paths). Here the chip
raster AND the labels come out of one pipeline run, and the world->pixel
projection happens against the chip's own geotransform in this process —
there is no cross-system registration step to get wrong.

Per address:
  1. resolve the newest covering EPT (entwine index — no WESM.gpkg needed)
  2. run src.pipeline.process_address into a per-address output dir
  3. project facets.geojson (LiDAR UTM) onto naip_clipped.tif pixels
  4. QC gates; accepted chips + COCO entries written; reasons to manifest.jsonl
  5. split with make_splits(seed=42) into train/valid/test

Usage (pdal-env):
  PROJ_LIB=$CONDA_PREFIX/share/proj PYTHONPATH=. python training/build_pseudo_dataset.py \
      --addresses /home/salter/measure-it/adresses/addresses.json \
      --out training/roof_dataset_pseudo --limit 30 --gallery
"""
import argparse
import hashlib
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_pseudo_dataset")

# QC gates — reject the ADDRESS, not the facet; partial labels teach partial truth.
MIN_QC_CONFIDENCE = 0.6
MIN_FACETS, MAX_FACETS = 3, 40
MIN_UNION_IOU = 0.7          # facet union vs outline, pixel space
MIN_AREA_M2, MAX_AREA_M2 = 80.0, 600.0
# Canopy gate: tree crowns overhanging the roof corrupt the nDSM and produce
# fragmented, mushy facet labels even when registration is perfect. NDVI from
# the chip's own NIR band measures it directly.
MAX_CANOPY_FRACTION = 0.25   # of roof pixels with NDVI > 0.3
CANOPY_NDVI = 0.3

CATEGORIES = [
    {"id": 1, "name": "roof_polygon", "supercategory": "roof"},
    {"id": 2, "name": "facet", "supercategory": "roof"},
]


def address_id(addr: str) -> str:
    """Stable hex id for an address string (matches the project's hex-id style)."""
    return hashlib.sha1(addr.strip().lower().encode()).hexdigest()[:16]


def world_to_pixel_ring(coords, inv_transform, to_chip_crs=None):
    ring = []
    for x, y in coords:
        if to_chip_crs is not None:
            x, y = to_chip_crs(x, y)
        col, row = inv_transform * (x, y)
        ring.extend([round(float(col), 2), round(float(row), 2)])
    return ring


def process_one(addr: str, state: str, work_dir: Path):
    """Run the pipeline + label projection for one address.

    Returns (record dict for manifest, payload dict or None).
    payload = {chip_png, image_wh, outline_ring, facet_feats:[(ring, props)...]}
    """
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union, transform as shp_transform
    import rasterio
    from PIL import Image

    from src.lidar.dataset_discovery import discover_ept_from_entwine
    from src.utils.geocode import get_coordinates
    from src.pipeline import process_address

    aid = address_id(addr)
    rec = {"address": addr, "address_id": aid}

    coords = get_coordinates(address=addr)
    lat, lon = coords["lat"], coords["lon"]
    rec["lat"], rec["lon"] = lat, lon

    ept_url = discover_ept_from_entwine(lat, lon)
    if not ept_url:
        rec["status"] = "skip:no_ept_coverage"
        return rec, None
    rec["ept"] = ept_url.split("/")[-2]

    out_dir = work_dir / aid
    process_address(
        lat=lat, lon=lon, address=addr, state=state,
        ept_url=ept_url, output_dir=out_dir,
        export_pdf=False, export_dxf=False, export_csv=False,
    )

    fg_path = out_dir / "facets.geojson"
    chip_tif = out_dir / "naip_clipped.tif"
    if not fg_path.exists():
        rec["status"] = "skip:no_facets_geojson"
        return rec, None
    if not chip_tif.exists():
        rec["status"] = "skip:no_chip"
        return rec, None

    fg = json.load(open(fg_path))
    fg_props = fg.get("properties", {})
    qc_conf = fg_props.get("qc_confidence", 0.0)
    rec["qc_confidence"] = qc_conf
    if qc_conf < MIN_QC_CONFIDENCE:
        rec["status"] = f"reject:qc_confidence<{MIN_QC_CONFIDENCE}"
        return rec, None

    facet_feats, outline_geom = [], None
    for feat in fg["features"]:
        kind = feat["properties"].get("kind")
        geom = shp_shape(feat["geometry"])
        if kind == "outline":
            outline_geom = geom
        elif kind == "facet":
            facet_feats.append((geom, feat["properties"]))

    if outline_geom is None:
        rec["status"] = "skip:no_outline"
        return rec, None
    rec["n_facets"] = len(facet_feats)
    if not (MIN_FACETS <= len(facet_feats) <= MAX_FACETS):
        rec["status"] = f"reject:facet_count={len(facet_feats)}"
        return rec, None
    rec["outline_area_m2"] = round(outline_geom.area, 1)
    if not (MIN_AREA_M2 <= outline_geom.area <= MAX_AREA_M2):
        rec["status"] = f"reject:area={outline_geom.area:.0f}m2"
        return rec, None

    union = unary_union([g for g, _ in facet_feats])
    iou = union.intersection(outline_geom).area / max(union.union(outline_geom).area, 1e-9)
    rec["union_iou"] = round(iou, 3)
    if iou < MIN_UNION_IOU:
        rec["status"] = f"reject:union_iou={iou:.2f}"
        return rec, None

    with rasterio.open(chip_tif) as src:
        chip_crs = str(src.crs)
        inv = ~src.transform
        W, H = src.width, src.height
        rgb = src.read([1, 2, 3])
        nir = src.read(4).astype(np.float32) if src.count >= 4 else None
        chip_transform = src.transform

    to_chip = None
    facets_crs = fg_props.get("crs") or ""
    if facets_crs and facets_crs != chip_crs:
        from pyproj import CRS, Transformer
        if not CRS.from_user_input(facets_crs).equals(CRS.from_user_input(chip_crs)):
            to_chip = Transformer.from_crs(facets_crs, chip_crs, always_xy=True).transform
            logger.info(f"{aid}: reprojecting labels {facets_crs} -> {chip_crs}")

    # Canopy gate: NDVI over the roof outline. Overhanging crowns corrupt the
    # nDSM-derived facets (observed: 16 mushy fragments on a canopy-covered
    # house) — registration stays perfect but the label SHAPES are garbage.
    if nir is not None:
        from rasterio.features import geometry_mask as _geom_mask
        from shapely.ops import transform as _shp_tx2
        _outline_chip = outline_geom
        if to_chip is not None:
            _outline_chip = _shp_tx2(to_chip, outline_geom)
        try:
            roof_px = ~_geom_mask([_outline_chip.__geo_interface__], out_shape=(H, W),
                                  transform=chip_transform)
            if roof_px.any():
                red = rgb[0].astype(np.float32)
                ndvi = (nir - red) / (nir + red + 1e-6)
                canopy = float((ndvi[roof_px] > CANOPY_NDVI).mean())
                rec["canopy_fraction"] = round(canopy, 3)
                if canopy > MAX_CANOPY_FRACTION:
                    rec["status"] = f"reject:canopy={canopy:.2f}"
                    return rec, None
        except Exception as _ce:
            logger.debug(f"{aid}: canopy gate skipped ({_ce})")

    outline_ring = world_to_pixel_ring(outline_geom.exterior.coords, inv, to_chip)
    facet_rings = []
    for geom, props in facet_feats:
        ring = world_to_pixel_ring(geom.exterior.coords, inv, to_chip)
        if len(ring) >= 6:
            facet_rings.append((ring, props))

    # registration sanity: outline ring must land inside the chip frame
    xs, ys = outline_ring[0::2], outline_ring[1::2]
    if min(xs) < -5 or min(ys) < -5 or max(xs) > W + 5 or max(ys) > H + 5:
        rec["status"] = "reject:outline_outside_chip"
        return rec, None

    chip_png = out_dir / f"{aid}.png"
    Image.fromarray(np.transpose(rgb, (1, 2, 0)).astype(np.uint8)).save(chip_png)

    rec["status"] = "ok"
    return rec, {
        "chip_png": chip_png,
        "wh": (W, H),
        "outline_ring": outline_ring,
        "facet_rings": facet_rings,
    }


def ring_bbox_area(ring):
    xs, ys = ring[0::2], ring[1::2]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0, max(ys) - y0
    # shoelace for area
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return [x0, y0, w, h], abs(a) / 2.0


def render_gallery(png_path: Path, payload: dict, qa_dir: Path):
    from PIL import Image, ImageDraw
    img = Image.open(payload["chip_png"]).convert("RGB")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    rng = np.random.RandomState(7)
    for ring, _ in payload["facet_rings"]:
        pts = list(zip(ring[0::2], ring[1::2]))
        c = tuple(int(v) for v in rng.randint(40, 255, 3)) + (90,)
        d.polygon(pts, fill=c, outline=(255, 255, 0, 255))
    pts = list(zip(payload["outline_ring"][0::2], payload["outline_ring"][1::2]))
    d.line(pts + [pts[0]], fill=(0, 120, 255, 255), width=3)
    Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB").save(qa_dir / png_path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addresses", required=True, help="addresses.json")
    ap.add_argument("--out", default="training/roof_dataset_pseudo")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--state", default="fl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gallery", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S")

    out = Path(args.out)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    done = {}
    if manifest_path.exists():   # resumable: skip already-attempted addresses
        for line in open(manifest_path):
            r = json.loads(line)
            done[r["address_id"]] = r

    addrs = json.load(open(args.addresses))
    # addresses.json contains duplicates (149 observed) — dedup by stable id
    # BEFORE slicing so offset/limit windows never overlap on the same address
    seen_ids = set()
    unique = []
    for entry in addrs:
        a = entry["address"] if isinstance(entry, dict) else str(entry)
        k = address_id(a)
        if k not in seen_ids:
            seen_ids.add(k)
            unique.append(entry)
    if len(unique) < len(addrs):
        logger.info(f"Deduped address list: {len(addrs)} -> {len(unique)}")
    batch = unique[args.offset:args.offset + args.limit]

    accepted = {}
    mf = open(manifest_path, "a")
    for i, entry in enumerate(batch, 1):
        addr = entry["address"] if isinstance(entry, dict) else str(entry)
        aid = address_id(addr)
        prior = done.get(aid)
        if prior and prior.get("status") == "ok" and (work / aid / f"{aid}.png").exists():
            logger.info(f"[{i}/{len(batch)}] cached ok: {addr}")
            fg = json.load(open(work / aid / "labels.json"))
            accepted[aid] = {
                "chip_png": work / aid / f"{aid}.png",
                "wh": tuple(fg["wh"]), "outline_ring": fg["outline_ring"],
                "facet_rings": [tuple(x) for x in fg["facet_rings"]],
            }
            continue
        if prior and prior.get("status", "").startswith(("reject", "skip", "error")):
            logger.info(f"[{i}/{len(batch)}] previously {prior['status']}: {addr}")
            continue
        logger.info(f"[{i}/{len(batch)}] {addr}")
        try:
            rec, payload = process_one(addr, args.state, work)
        except Exception as e:
            rec = {"address": addr, "address_id": aid,
                   "status": f"error:{type(e).__name__}", "detail": str(e)[:300]}
            payload = None
            logger.warning(f"  failed: {e}\n{traceback.format_exc(limit=3)}")
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        logger.info(f"  -> {rec.get('status')}")
        if payload:
            accepted[aid] = payload
            json.dump({"wh": payload["wh"], "outline_ring": payload["outline_ring"],
                       "facet_rings": payload["facet_rings"]},
                      open(work / aid / "labels.json", "w"))
    mf.close()

    logger.info(f"Accepted {len(accepted)}/{len(batch)} addresses")
    if not accepted:
        logger.error("Nothing accepted — inspect manifest.jsonl")
        sys.exit(1)

    # ── assemble COCO + splits (reuses the project's deterministic splitter) ──
    from src.data.ls_to_coco import make_splits
    splits = make_splits(sorted(accepted.keys()), seed=args.seed)
    name_map = {"train": "train", "val": "valid", "test": "test"}

    for split, aids in splits.items():
        folder = out / name_map.get(split, split)
        folder.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        ann_id = 1
        for img_id, aid in enumerate(sorted(aids), 1):
            p = accepted[aid]
            W, H = p["wh"]
            images.append({"id": img_id, "address_id": aid,
                           "file_name": f"{aid}.png", "width": W, "height": H})
            bbox, area = ring_bbox_area(p["outline_ring"])
            anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                         "segmentation": [p["outline_ring"]],
                         "bbox": bbox, "area": area, "iscrowd": 0})
            ann_id += 1
            for ring, props in p["facet_rings"]:
                bbox, area = ring_bbox_area(ring)
                anns.append({"id": ann_id, "image_id": img_id, "category_id": 2,
                             "segmentation": [ring], "bbox": bbox, "area": area,
                             "iscrowd": 0,
                             "attributes": {
                                 "pitch_string": props.get("pitch_string"),
                                 "aspect_bin": props.get("aspect_bin"),
                                 "slope_deg": props.get("slope_deg"),
                                 "is_flat": props.get("is_flat"),
                             }})
                ann_id += 1
            shutil.copy2(p["chip_png"], folder / f"{aid}.png")
        json.dump({"images": images, "annotations": anns, "categories": CATEGORIES},
                  open(folder / "_annotations.coco.json", "w"))
        logger.info(f"  {folder.name}: {len(images)} images, {len(anns)} anns")

    if args.gallery:
        qa = out / "_qa"
        qa.mkdir(exist_ok=True)
        for aid, p in accepted.items():
            try:
                render_gallery(Path(f"{aid}.png"), p, qa)
            except Exception as e:
                logger.warning(f"gallery {aid}: {e}")
        logger.info(f"QA gallery: {qa} ({len(accepted)} overlays) — eyeball before training!")


if __name__ == "__main__":
    main()
