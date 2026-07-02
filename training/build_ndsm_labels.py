#!/usr/bin/env python3
"""Auto-generate roof-facet TRAINING LABELS from LiDAR — the "label factory".

For each address: run the pipeline to fetch the NAIP chip + nDSM, extract clean
facets from the height surface (src.roofs.ndsm_segment), map them into the NAIP
chip's pixel space, and emit COCO instance annotations. Labels are MEASURED from
LiDAR (not hand-drawn, not a model's guesses), FL-native, at scale.

Output: <out>/images/*.png + <out>/annotations.coco.json (+ QA overlays).

Usage (Colab, with AWS creds for NAIP):
  python training/build_ndsm_labels.py --out training/ndsm_dataset \
      --addresses "909 Spring Island Way, Orlando, FL 32828" \
                  "542 Waterscape Way, Orlando, FL 32828"
"""
import argparse
import glob
import json
import os
from typing import Optional, Tuple

import numpy as np

FACET_CATEGORY_ID = 2   # matches roof_dataset (1=roof_polygon, 2=facet)


def _naip_and_ndsm(out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    res_p = os.path.join(out_dir, "results.json")
    naip = None
    if os.path.exists(res_p):
        res = json.load(open(res_p))
        naip = (res.get("naip") or {}).get("tif_path")
    nd = sorted(glob.glob(os.path.join(out_dir, "*ndsm*.tif")))
    return naip, (nd[0] if nd else None)


def chip_to_coco(out_dir: str, image_id: int, images_dir: str,
                 min_facet_px: int = 30) -> Optional[dict]:
    """Build a COCO image + facet annotations for one processed roof.

    Returns {"image": {...}, "annotations": [...], "facets": [ndsm dicts]} or None.
    """
    import rasterio
    from rasterio.warp import transform_geom
    from PIL import Image
    from src.roofs.ndsm_segment import ndsm_facets

    naip, ndsm = _naip_and_ndsm(out_dir)
    if not (naip and ndsm and os.path.exists(naip) and os.path.exists(ndsm)):
        return None

    with rasterio.open(ndsm) as nd:
        z = nd.read(1).astype(float)
        ztf, zcrs = nd.transform, nd.crs
    z[~np.isfinite(z)] = 0.0

    # CONFINE TO THE ROOF: mask the nDSM to the building footprint so we don't
    # label trees / pool cages / driveways / yard (anything tall passes the
    # >0.3m above-ground test otherwise). Footprint = roof_polygon.geojson.
    footprint = None
    fp_path = os.path.join(out_dir, "roof_polygon.geojson")
    if os.path.exists(fp_path):
        try:
            import geopandas as gpd
            from rasterio.features import geometry_mask
            g = gpd.read_file(fp_path)
            if zcrs is not None:
                g = g.to_crs(zcrs)
            footprint = (g.union_all() if hasattr(g, "union_all")
                         else g.unary_union).buffer(1.0)   # +1 m for eave overhang
            keep = geometry_mask([footprint.__geo_interface__], out_shape=z.shape,
                                 transform=ztf, invert=True)
            z = np.where(keep, z, 0.0)                      # zero out non-roof pixels
        except Exception as _fe:
            footprint = None                               # fall back to unmasked
    else:
        return None   # no footprint -> can't trust the labels; skip this roof

    facets = ndsm_facets(z, ztf, clean=True)
    if footprint is not None:                              # belt: clip facets too
        clipped = []
        for f in facets:
            try:
                p = f["polygon"].intersection(footprint)
                if p.is_empty or p.area < 1.0:
                    continue
                if p.geom_type != "Polygon":
                    p = max((q for q in getattr(p, "geoms", []) if q.geom_type == "Polygon"),
                            key=lambda q: q.area, default=None)
                    if p is None:
                        continue
                f = dict(f); f["polygon"] = p
            except Exception:
                continue
            clipped.append(f)
        facets = clipped
    if not facets:
        return None

    with rasterio.open(naip) as im:
        arr = im.read([1, 2, 3])
        ntf, ncrs, W, H = im.transform, im.crs, im.width, im.height
    inv = ~ntf

    fn = f"{image_id:06d}.png"
    Image.fromarray(np.transpose(arr, (1, 2, 0)).astype("uint8")).save(
        os.path.join(images_dir, fn))

    anns = []
    for f in facets:
        geo = f["polygon"].__geo_interface__
        if zcrs is not None and ncrs is not None and str(zcrs) != str(ncrs):
            geo = transform_geom(zcrs, ncrs, geo)          # nDSM CRS -> NAIP CRS
        rings = geo.get("coordinates", [])
        if not rings:
            continue
        flat, xs, ys = [], [], []
        for x, y in rings[0]:                              # exterior ring -> pixels
            c, r = inv * (x, y)
            flat += [float(c), float(r)]
            xs.append(c); ys.append(r)
        if len(flat) < 6:
            continue
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if (x1 - x0) * (y1 - y0) < min_facet_px:
            continue
        anns.append({
            "image_id": image_id, "category_id": FACET_CATEGORY_ID,
            "segmentation": [flat],
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "area": float((x1 - x0) * (y1 - y0)), "iscrowd": 0,
        })
    if not anns:
        return None
    return {"image": {"id": image_id, "file_name": fn, "width": W, "height": H},
            "annotations": anns, "facets": facets}


def qa_overlay(out_dir: str, images_dir: str, image: dict, anns: list, dst: str):
    """Save a QA picture: the NAIP chip with the facet-label polygons drawn on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    from PIL import Image
    img = np.array(Image.open(os.path.join(images_dir, image["file_name"])))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img); ax.axis("off")
    for a in anns:
        pts = np.array(a["segmentation"][0]).reshape(-1, 2)
        ax.add_patch(MplPoly(pts, fill=True, alpha=0.30, edgecolor="yellow",
                             facecolor="cyan", linewidth=1.5))
    ax.set_title(f"{image['file_name']} — {len(anns)} facet labels")
    plt.savefig(dst, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--addresses", nargs="+", required=True)
    ap.add_argument("--state", default="fl")
    ap.add_argument("--qa", action="store_true", help="also write QA overlays")
    args = ap.parse_args()

    from src.pipeline import process_address, setup_logging
    setup_logging("WARNING")
    images_dir = os.path.join(args.out, "images")
    qa_dir = os.path.join(args.out, "qa")
    os.makedirs(images_dir, exist_ok=True)
    if args.qa:
        os.makedirs(qa_dir, exist_ok=True)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "roof_polygon"},
                           {"id": FACET_CATEGORY_ID, "name": "facet"}]}
    ann_id = 1
    for i, addr in enumerate(args.addresses, start=1):
        work = os.path.join(args.out, f"_work/{i:03d}")
        try:
            process_address(address=addr, state=args.state, fetch_naip=True,
                            fetch_lidar=True, output_dir=work, segment_method="ndsm")
            r = chip_to_coco(work, i, images_dir)
            if r is None:
                print(f"[{i}] {addr[:40]:40} -> no labels (skip)")
                continue
            coco["images"].append(r["image"])
            for a in r["annotations"]:
                a["id"] = ann_id; ann_id += 1
                coco["annotations"].append(a)
            if args.qa:
                qa_overlay(work, images_dir, r["image"], r["annotations"],
                           os.path.join(qa_dir, f"{i:06d}_qa.png"))
            print(f"[{i}] {addr[:40]:40} -> {len(r['annotations'])} facet labels")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] {addr[:40]:40} -> ERROR {str(e)[:60]}")

    out_json = os.path.join(args.out, "annotations.coco.json")
    json.dump(coco, open(out_json, "w"))
    print(f"\nDataset: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} facet labels -> {out_json}")


if __name__ == "__main__":
    main()
