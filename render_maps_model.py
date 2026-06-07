#!/usr/bin/env python3
"""Render real MODEL -> MAP reports: the end-to-end test the mAP can't give you.

mAP scores masks on the val split; it does NOT tell you whether those masks make
good ROOF MAPS (clean tiling, typed ridge/hip/valley edges, sensible areas).
This runs the trained RF-DETR-Seg checkpoint on chip images and pushes each
prediction through the full deliverable pipeline:

    RFDETRBackend.predict(chip) -> {outline, facets}
      -> process_chip_rgb (tiling -> geometric aspect -> pitch policy -> edges)
      -> report.pdf/json + facets.csv/edges.csv + map_{plain,length,area,pitch}.png

World scaling:
  - a GeoTIFF chip uses its OWN transform -> accurate sqft/ft.
  - a plain PNG uses --gsd (approximate area, but the map GEOMETRY/edges are
    still correct -- which is what we're validating first).

Run where rfdetr + the checkpoint live (i.e. the pod):
  python render_maps_model.py \
      --checkpoint output/checkpoint_best_ema.pth \
      --chips training/roof_dataset/valid --out model_maps --gsd 0.3 --limit 15

Then compare model_maps/ against the stand-in maps (client_maps3/) to see whether
the trained model holds up versus the hand-cleaned annotations.
"""
import argparse
import glob
import json
import os

import numpy as np

from src.roofs.rfdetr_backend import RFDETRBackend
from src.output.diagram import render_diagram
from src.rgb_pipeline import process_chip_rgb


def load_chip(path):
    """Return (rgb_uint8 HxWx3, pixel_to_world or None).

    A georeferenced GeoTIFF yields an exact pixel->world map; anything else
    returns None so the caller falls back to a nominal GSD.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            if src.crs is not None and src.count >= 3:
                arr = src.read([1, 2, 3])                 # (3,H,W)
                rgb = np.transpose(arr, (1, 2, 0)).astype("uint8")
                T = src.transform

                def p2w(x, y, T=T):                       # (col,row) -> (east,north)
                    wx, wy = T * (x, y)
                    return float(wx), float(wy)

                return rgb, p2w
    except Exception:
        pass
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"), dtype="uint8"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--chips", required=True, help="dir of chip images, or a glob")
    ap.add_argument("--out", default="model_maps")
    ap.add_argument("--gsd", type=float, default=0.3, help="m/px for non-geo PNG chips")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--mask-epsilon", type=float, default=0.005,
                    help="Douglas-Peucker epsilon_frac for mask->polygon (default: 0.005)")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()

    backend = RFDETRBackend(args.checkpoint, threshold=args.threshold, mask_epsilon=args.mask_epsilon)

    if os.path.isdir(args.chips):
        files = sorted(sum((glob.glob(os.path.join(args.chips, f"*.{e}"))
                            for e in ("png", "tif", "tiff", "jpg")), []))
    else:
        files = sorted(glob.glob(args.chips))
    files = files[: args.limit]
    if not files:
        print(f"No chip images found at {args.chips}")
        return

    os.makedirs(args.out, exist_ok=True)
    summary = []
    n_outline = 0
    for path in files:
        cid = os.path.splitext(os.path.basename(path))[0]
        rgb, p2w = load_chip(path)
        pred = backend.predict(rgb)
        nf = len(pred["facets"])
        n_outline += pred["outline"] is not None
        outdir = os.path.join(args.out, cid)
        kw = dict(output_dir=outdir, write_outputs=True, address=cid, building_id=cid)
        if p2w is not None:
            res = process_chip_rgb(pred, pixel_to_world=p2w, **kw)
        else:
            res = process_chip_rgb(pred, gsd_m_per_px=args.gsd, **kw)
        for mode in ("plain", "length", "area", "pitch"):
            render_diagram(res["report_input"], mode=mode).save(
                os.path.join(outdir, f"map_{mode}.png"))
        s = res["summary"]
        edge_n = {e["edge_type"]: 0 for e in res["report_input"]["edges"]}
        for e in res["report_input"]["edges"]:
            edge_n[e["edge_type"]] += 1
        summary.append({
            "chip": cid, "outline": pred["outline"] is not None, "facets_pred": nf,
            "total_sqft": s["total_area_sqft"], "pitch": s["predominant_pitch"],
            "needs_review": s["num_needs_review"], "edges": edge_n,
        })
        print(f"  {cid}: outline={'Y' if pred['outline'] else 'N'} "
              f"facets={nf:2d} area={s['total_area_sqft']:6.0f}sqft "
              f"pitch={s['predominant_pitch']:5s} edges={edge_n}")

    json.dump(summary, open(os.path.join(args.out, "_summary.json"), "w"), indent=2)
    interior = sum(sum(v for k, v in r["edges"].items()
                       if k in ("ridge", "hip", "valley")) for r in summary)
    print(f"\n{len(summary)} model->maps in {args.out}/  "
          f"({n_outline} with outline, {interior} interior edges total)")


if __name__ == "__main__":
    main()
