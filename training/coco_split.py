"""
Grouped train/valid split for a Label Studio COCO export.

The fresh LS export is ONE coco.json + one images dir; prep_sam3_facets needs
train/ and valid/ splits. Splitting per-image leaks the same roof across both
sets, so we group by ADDRESS (image['address_id'] if present, else the file
stem) and assign whole groups — the same no-leakage rule ls_to_coco used.
Writes <out>/{train,valid}/_annotations.coco.json + copies the images, keeping
file_name stable so a readiness KEEP manifest (by file_name) still matches.

Usage:
  python -m training.coco_split export.coco.json IMAGES_DIR OUT_DIR [--val-frac 0.1] [--seed 42]
"""
import argparse, json, os, shutil, hashlib
from collections import defaultdict


def _group_key(im):
    g = im.get("address_id")
    if g not in (None, ""):
        return str(g)
    return os.path.splitext(os.path.basename(im["file_name"]))[0]


def split(coco, images_dir, out_dir, val_frac=0.1, seed=42):
    imgs = coco["images"]
    groups = defaultdict(list)
    for im in imgs:
        groups[_group_key(im)].append(im)
    # deterministic hashed assignment (seed-stable, no RNG import needed)
    val_groups = set()
    for g in groups:
        h = int(hashlib.md5(f"{seed}:{g}".encode()).hexdigest(), 16) % 1000
        if h < val_frac * 1000:
            val_groups.add(g)

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    counts = {}
    for name, want_val in (("train", False), ("valid", True)):
        d = os.path.join(out_dir, name)
        img_out = os.path.join(d, "images") if False else d   # images beside the json
        os.makedirs(d, exist_ok=True)
        keep_imgs, keep_anns = [], []
        for g, glist in groups.items():
            if (g in val_groups) != want_val:
                continue
            for im in glist:
                keep_imgs.append(im)
                keep_anns.extend(anns_by_img.get(im["id"], []))
                src = os.path.join(images_dir, im["file_name"])
                dst = os.path.join(d, os.path.basename(im["file_name"]))
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
        out = {"images": keep_imgs, "annotations": keep_anns,
               "categories": coco["categories"]}
        json.dump(out, open(os.path.join(d, "_annotations.coco.json"), "w"))
        counts[name] = (len(keep_imgs), len(keep_anns))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco"); ap.add_argument("images_dir"); ap.add_argument("out_dir")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    c = json.load(open(a.coco))
    counts = split(c, a.images_dir, a.out_dir, a.val_frac, a.seed)
    for k, (ni, na) in counts.items():
        print(f"{k}: {ni} images, {na} annotations")
    print(f"-> {a.out_dir}/{{train,valid}}/_annotations.coco.json")


if __name__ == "__main__":
    main()
