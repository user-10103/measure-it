"""
Label Studio export -> COCO instance-segmentation converter + dataset splits.

The annotation team cleans roof outlines + facets (with per-facet pitch/aspect)
in Label Studio. This module converts the LS JSON export into the same COCO
schema the rest of measure-it consumes, and produces a deterministic
train/val/test split keyed by address_id.

LS export schema (input): a list of tasks, each
  {"data": {"image": "s3://bucket/key/<address_id>.png"},
   "annotations": [{"result": [<region>, ...]}]}
where a <region> is a polygonlabels region (outline/facet) or a per-region
choices region (pitch/aspect) sharing the polygon region's "id". Polygon points
are PERCENT (0-100) of original_width/original_height.

CLI:
  python -m src.data.ls_to_coco --ls-export export.json \
      --out-coco coco.json --out-split splits.json [--seed 42] [--ratios 0.8 0.1 0.1]
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional

from shapely.geometry import Polygon

from src.roofs.categories import CAT_ROOF, CAT_FACET, PITCH_LABEL_TO_SLOPE

logger = logging.getLogger(__name__)

# polygonlabels label -> COCO category id
LABEL_TO_CAT = {"roof_outline": CAT_ROOF, "facet": CAT_FACET}


def pitch_to_slope_deg(pitch_label: Optional[str]) -> Optional[float]:
    """Map an LS pitch-class label to a representative slope (bin midpoint).

    Tolerant of degree-symbol / dash variation: matches on the leading word
    (flat/low/medium/steep). Returns None for unknown/missing labels.
    """
    if not pitch_label:
        return None
    head = pitch_label.strip().lower()
    for key, slope in PITCH_LABEL_TO_SLOPE.items():
        if head.startswith(key):
            return slope
    return None


def _address_id_from_image(image_ref: str) -> str:
    """s3://bucket/key/addr_001.png -> 'addr_001'."""
    base = os.path.basename(image_ref.split("?")[0])
    return os.path.splitext(base)[0]


def _points_to_pixels(points: List[List[float]], w: int, h: int) -> List[float]:
    """Percent points [[xpct,ypct],...] -> flat pixel ring [x0,y0,x1,y1,...]."""
    flat: List[float] = []
    for x_pct, y_pct in points:
        flat.append(round(x_pct / 100.0 * w, 3))
        flat.append(round(y_pct / 100.0 * h, 3))
    return flat


def _polygon_metrics(flat_ring: List[float]):
    """Return (area, bbox[x,y,w,h]) for a flat [x0,y0,...] ring, or (0, None)."""
    coords = [(flat_ring[i], flat_ring[i + 1]) for i in range(0, len(flat_ring) - 1, 2)]
    if len(coords) < 3:
        return 0.0, None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return 0.0, None
    minx, miny, maxx, maxy = poly.bounds
    return float(poly.area), [round(minx, 3), round(miny, 3),
                              round(maxx - minx, 3), round(maxy - miny, 3)]


def ls_export_to_coco(tasks: List[dict]) -> dict:
    """Convert a Label Studio task list into a COCO instance-seg dict."""
    images: List[dict] = []
    annotations: List[dict] = []
    img_id = 0
    ann_id = 0

    for task in tasks:
        regions = (task.get("annotations") or [{}])[0].get("result") or []
        if not regions:
            continue

        polys = [r for r in regions if r.get("type") == "polygonlabels"]
        if not polys:
            continue

        # collect per-region pitch/aspect choices keyed by region id
        choices: Dict[str, Dict[str, Optional[str]]] = {}
        for r in regions:
            if r.get("type") == "choices":
                vals = (r.get("value") or {}).get("choices") or []
                if vals:
                    choices.setdefault(r.get("id"), {})[r.get("from_name")] = vals[0]

        w = polys[0].get("original_width")
        h = polys[0].get("original_height")
        image_ref = (task.get("data") or {}).get("image", "")
        addr = _address_id_from_image(image_ref)
        img_id += 1
        images.append({"id": img_id, "address_id": addr,
                       "file_name": os.path.basename(image_ref.split("?")[0]),
                       "width": w, "height": h})

        for r in polys:
            value = r.get("value") or {}
            labels = value.get("polygonlabels") or []
            cat = LABEL_TO_CAT.get(labels[0]) if labels else None
            if cat is None:
                continue
            pts = value.get("points") or []
            if len(pts) < 3:
                continue
            ring = _points_to_pixels(pts, w, h)
            area, bbox = _polygon_metrics(ring)
            if bbox is None:
                continue
            ann_id += 1
            ann = {"id": ann_id, "image_id": img_id, "category_id": cat,
                   "segmentation": [ring], "area": round(area, 3),
                   "bbox": bbox, "iscrowd": 0}
            if cat == CAT_FACET:
                ch = choices.get(r.get("id"), {})
                pitch_label = ch.get("pitch")
                ann["attributes"] = {
                    "pitch_label": pitch_label,
                    "aspect_label": ch.get("aspect"),
                    "slope_deg": pitch_to_slope_deg(pitch_label),
                }
            annotations.append(ann)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CAT_ROOF, "name": "roof_polygon"},
                       {"id": CAT_FACET, "name": "facet"}],
    }
    logger.info("LS->COCO: %d images, %d annotations", len(images), len(annotations))
    return coco


def make_splits(image_ids: List[str], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> Dict[str, List[str]]:
    """Deterministic train/val/test split keyed by address_id.

    Dedups + sorts ids first, then shuffles with a seeded RNG so the partition
    is reproducible regardless of input order. Remainder goes to test.
    """
    ids = sorted(set(image_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Label Studio export -> COCO + splits")
    ap.add_argument("--ls-export", required=True, help="Label Studio JSON export")
    ap.add_argument("--out-coco", required=True, help="output COCO json path")
    ap.add_argument("--out-split", help="output split json path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tasks = json.load(open(args.ls_export))
    coco = ls_export_to_coco(tasks)
    json.dump(coco, open(args.out_coco, "w"))
    logger.info("wrote %s", args.out_coco)

    if args.out_split:
        ids = [im["address_id"] for im in coco["images"]]
        splits = make_splits(ids, ratios=tuple(args.ratios), seed=args.seed)
        json.dump(splits, open(args.out_split, "w"))
        logger.info("wrote %s (train=%d val=%d test=%d)", args.out_split,
                    len(splits["train"]), len(splits["val"]), len(splits["test"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
