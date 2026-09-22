"""
Rebuild a leakage-free, de-duplicated, single-convention dataset from contaminated splits.

Why this exists — the committed `roof_dataset` could not produce a trustworthy
accuracy number:
  * 2,392 train images but only 913 unique roofs (~2.6x duplicate image records),
  * 244 of 266 valid roofs ALSO appear in train (~92% leakage — the "held-out" set
    was not held out),
  * two facet category schemas across sets ('roof facet' vs 'facet').

This merges the given splits back into ONE pool, collapses duplicate roofs (keeping
the richest labelling), canonicalises the category schema to {1: roof_polygon,
2: facet}, and re-splits GROUPED BY ADDRESS so a roof can never land in two splits.
It asserts zero overlap before writing — a split that leaks is a bug, not a warning.

Emits `<out>/{train,test}/_annotations.coco.json` plus a `chips_needed.txt` per split
so `fetch_chips.py` can pull the images for each split independently.

Usage:
  python -m training.build_clean_eval \
      training/roof_dataset/train/_annotations.coco.json \
      training/roof_dataset/valid/_annotations.coco.json \
      training/roof_dataset/test/_annotations.coco.json \
      --out training/roof_dataset_clean_eval --test-frac 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

FACET_NAMES = {"facet", "roof_facet", "roof facet", "roof face"}
ROOF_NAMES = {"roof_polygon", "roof_outline", "roof", "outline"}
CANON = [{"id": 1, "name": "roof_polygon", "supercategory": "roof"},
         {"id": 2, "name": "facet", "supercategory": "roof"}]


def _group_key(im: dict) -> str:
    """The ROOF identity: address_id if present, else the file stem. All chips of one
    roof share it, so grouping on it is what keeps a roof out of two splits."""
    g = im.get("address_id")
    if g not in (None, ""):
        return str(g)
    return os.path.splitext(os.path.basename(im.get("file_name", "")))[0]


def _canon_map(coco: dict) -> Dict[int, int]:
    """Source category id -> canonical id (1 roof_polygon / 2 facet); unmapped dropped."""
    out = {}
    for c in coco.get("categories", []):
        nm = str(c.get("name", "")).lower()
        if nm in FACET_NAMES:
            out[c["id"]] = 2
        elif nm in ROOF_NAMES:
            out[c["id"]] = 1
    return out


def merge_and_dedupe(cocos: List[dict]):
    """Merge sources, canonicalise categories, collapse duplicate roofs.

    Returns (images, anns_by_group, dropped_stats). One image is kept per roof — the
    record carrying the MOST facet annotations (rescues the 0-vs-N case where the same
    chip was ingested twice and only one copy got labelled).
    """
    per_group: Dict[str, list] = defaultdict(list)   # group -> [(image, [anns])]
    unmapped = Counter()
    for coco in cocos:
        cmap = _canon_map(coco)
        by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            cid = cmap.get(a["category_id"])
            if cid is None:
                unmapped[a["category_id"]] += 1
                continue
            a = dict(a); a["category_id"] = cid
            by_img[a["image_id"]].append(a)
        for im in coco.get("images", []):
            per_group[_group_key(im)].append((im, by_img.get(im["id"], [])))

    kept_images, kept_anns = [], []
    dup_roofs = 0
    next_img, next_ann = 1, 1
    for group, records in sorted(per_group.items()):
        if len(records) > 1:
            dup_roofs += 1
        # richest labelling wins (most facet anns, then most anns overall)
        im, anns = max(records, key=lambda r: (sum(1 for a in r[1] if a["category_id"] == 2),
                                               len(r[1])))
        im = dict(im)
        im["id"] = next_img
        im["address_id"] = group
        im["file_name"] = os.path.basename(im.get("file_name", f"{group}.png"))
        kept_images.append(im)
        for a in anns:
            a = dict(a)
            a["id"] = next_ann
            a["image_id"] = next_img
            kept_anns.append(a)
            next_ann += 1
        next_img += 1
    return kept_images, kept_anns, {"duplicate_roofs": dup_roofs,
                                    "unmapped_categories": dict(unmapped),
                                    "input_records": sum(len(r) for r in per_group.values())}


def split_by_address(images, anns, test_frac: float, seed: int):
    """Deterministic hashed split GROUPED BY ADDRESS — the only thing that prevents
    a roof appearing in both splits."""
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    out = {"train": {"images": [], "annotations": []},
           "test": {"images": [], "annotations": []}}
    for im in images:
        g = im["address_id"]
        h = int(hashlib.md5(f"{seed}:{g}".encode()).hexdigest(), 16) % 1000
        split = "test" if h < test_frac * 1000 else "train"
        out[split]["images"].append(im)
        out[split]["annotations"].extend(anns_by_img.get(im["id"], []))
    return out


def _stats(images, anns) -> str:
    per = Counter()
    for a in anns:
        if a["category_id"] == 2:
            per[a["image_id"]] += 1
    counts = [per.get(im["id"], 0) for im in images]
    nz = [c for c in counts if c > 0]
    nz_sorted = sorted(nz)
    med = nz_sorted[len(nz_sorted) // 2] if nz_sorted else 0
    mean = (sum(nz) / len(nz)) if nz else 0.0
    return (f"{len(images)} roofs, {sum(counts)} facets, "
            f"facets/roof mean {mean:.2f} median {med}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco", nargs="+", help="source COCO json(s) to merge and re-split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args()

    cocos = [json.load(open(p)) for p in ns.coco]
    images, anns, info = merge_and_dedupe(cocos)
    print(f"input image records : {info['input_records']}")
    print(f"unique roofs kept   : {len(images)}   (collapsed {info['duplicate_roofs']} "
          f"duplicated roof(s))")
    if info["unmapped_categories"]:
        print(f"dropped anns from unmapped categories: {info['unmapped_categories']}")

    splits = split_by_address(images, anns, ns.test_frac, ns.seed)

    # HARD leakage check — this is the whole point of the script.
    tr = {im["address_id"] for im in splits["train"]["images"]}
    te = {im["address_id"] for im in splits["test"]["images"]}
    overlap = tr & te
    if overlap:
        raise SystemExit(f"LEAKAGE: {len(overlap)} roof(s) in both splits — aborting")
    print(f"leakage check       : OK (0 roofs shared between train and test)")

    for name in ("train", "test"):
        d = os.path.join(ns.out, name)
        os.makedirs(d, exist_ok=True)
        payload = {"images": splits[name]["images"],
                   "annotations": splits[name]["annotations"],
                   "categories": CANON}
        json.dump(payload, open(os.path.join(d, "_annotations.coco.json"), "w"))
        with open(os.path.join(d, "chips_needed.txt"), "w") as fh:
            for im in splits[name]["images"]:
                fh.write(im["file_name"] + "\n")
        print(f"{name:5s}: {_stats(splits[name]['images'], splits[name]['annotations'])}")

    print(f"\nwrote -> {ns.out}/{{train,test}}/_annotations.coco.json (+ chips_needed.txt)")
    print("categories canonicalised to {1: roof_polygon, 2: facet}")


if __name__ == "__main__":
    main()
