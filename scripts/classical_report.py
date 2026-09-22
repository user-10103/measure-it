#!/usr/bin/env python3
"""Training-free roof report: 3-inch county imagery + classical geometry.

NO ML model, NO cv2 — pure PIL + numpy + shapely.
  outline (from annotation)  ->  fetch real 3-inch over the same bbox
  ->  numpy Hough line detection inside the outline  ->  extend lines +
  polygonize into facets  ->  measure-it process_chip_rgb (tiling, geometric
  aspect, pitch policy, typed edges)  ->  Roof Report PDF.

Outputs PDF + overlay PNG to ~/Downloads/classical_roof_reports/.
"""
import os, sys, glob, json, sqlite3, math, io
from pathlib import Path
import urllib.request, urllib.parse
from collections import defaultdict

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rgb_pipeline import process_chip_rgb

ADR = "/home/salter/Desktop/adresses"
DB = f"{ADR}/sam3_handoff/workspace/florida_roofs_v2/state.db"
COCO = f"{ADR}/sam3_handoff/workspace/florida_roofs_v2/deliverable/coco_v2_2.json"
TIF = lambda cid: f"{ADR}/chip_samples/{cid}.tif"
OUT = Path.home() / "Downloads" / "classical_roof_reports"
PINELLAS = "https://egis.pinellas.gov/gis/rest/services/Aerials/Aerials2024/ImageServer"
UA = {"User-Agent": "Mozilla/5.0"}


def fetch_3inch_utm(xmin, ymin, xmax, ymax, epsg, w, h, timeout=60):
    q = urllib.parse.urlencode({"bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": epsg,
                                "size": f"{w},{h}", "imageSR": epsg, "format": "png", "f": "image"})
    req = urllib.request.Request(f"{PINELLAS}/exportImage?{q}", headers=UA)
    png = urllib.request.urlopen(req, timeout=timeout).read()
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"non-PNG: {png[:120]!r}")
    return Image.open(io.BytesIO(png)).convert("RGB")


def rasterize_mask(poly, W, H):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in poly.exterior.coords], fill=255)
    return np.asarray(m) > 0


def hough_lines(gray, mask, topk=8, mag_pct=86, min_frac=0.09):
    """Detect up to topk dominant straight lines as (rho, theta) within mask."""
    H, W = gray.shape
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.hypot(gx, gy)
    mag[~mask] = 0.0
    inside = mag[mask]
    if inside.size == 0:
        return []
    thr = np.percentile(inside, mag_pct)
    ys, xs = np.where(mag >= max(thr, 1e-3))
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(0)
    if len(xs) > 5000:
        idx = rng.choice(len(xs), 5000, replace=False); xs, ys = xs[idx], ys[idx]
    thetas = np.deg2rad(np.arange(0, 180, 1.0))
    cos, sin = np.cos(thetas), np.sin(thetas)
    rhos = xs[:, None] * cos[None, :] + ys[:, None] * sin[None, :]   # N x 180
    diag = int(math.hypot(H, W))
    rq = np.clip(np.round(rhos).astype(int) + diag, 0, 2 * diag)
    acc = np.zeros((2 * diag + 1, 180), np.int32)
    for t in range(180):
        np.add.at(acc[:, t], rq[:, t], 1)
    peak = acc.max()
    lines = []
    for _ in range(topk):
        idx = int(np.argmax(acc)); r, t = np.unravel_index(idx, acc.shape)
        if acc[r, t] < max(0.11 * peak, min_frac * math.hypot(H, W)):
            break
        lines.append((r - diag, thetas[t]))
        acc[max(0, r - 18):r + 18, max(0, t - 6):t + 6] = 0   # NMS
    return lines


def line_to_seg(rho, theta, H, W):
    a, b = math.cos(theta), math.sin(theta)
    x0, y0 = a * rho, b * rho
    diag = math.hypot(H, W)
    return LineString([(x0 - b * diag, y0 + a * diag), (x0 + b * diag, y0 - a * diag)])


def detect_facets(rgb_img, outline, min_area_frac=0.04):
    """Line-detection facets (Hough) — good on high-contrast ridges."""
    W, H = rgb_img.size
    gray = np.asarray(rgb_img.convert("L").filter(ImageFilter.GaussianBlur(1)))
    mask = rasterize_mask(outline, W, H)
    lines = hough_lines(gray, mask)
    cuts = [line_to_seg(r, t, H, W) for r, t in lines]
    merged = unary_union([outline.exterior] + cuts)
    faces = [p for p in polygonize(merged)
             if outline.contains(p.representative_point()) and p.area > min_area_frac * outline.area]
    return faces, lines


def _kmeans1d(x, k, iters=30):
    c = np.percentile(x, np.linspace(6, 94, k))
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean()
    return lab, c


def detect_facets_shading(rgb_img, outline, k=6, min_area_frac=0.03):
    """Facets as constant-shading regions: each roof plane has a distinct sun-tone.

    Robust where ridges are low-contrast creases (uniform shingle). Median+blur
    kills shingle texture and small objects (vents); k-means on brightness labels
    each plane; rasterio vectorizes the label regions into facet polygons.
    """
    from rasterio.features import shapes
    from shapely.geometry import shape as shp_shape
    W, H = rgb_img.size
    mask = rasterize_mask(outline, W, H)
    sm = np.asarray(rgb_img.convert("L")
                    .filter(ImageFilter.MedianFilter(7))
                    .filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    vals = sm[mask]
    if vals.size < 50:
        return []
    lab_flat, _ = _kmeans1d(vals, k)
    lab = np.zeros((H, W), np.int32)
    lab[mask] = lab_flat + 1                       # 0 = outside
    faces = []
    for geom, val in shapes(lab, mask=mask, connectivity=8):
        if val == 0:
            continue
        p = shp_shape(geom).buffer(0)
        for poly in (p.geoms if p.geom_type == "MultiPolygon" else [p]):
            if poly.area > min_area_frac * outline.area:
                faces.append(poly.simplify(2.0).intersection(outline))
    faces = [f for f in faces if f.geom_type == "Polygon" and f.area > min_area_frac * outline.area]
    return faces


def run(cid, addr, clean_outlines):
    o = clean_outlines.get(cid)
    if not o:
        return f"{cid}: no clean human outline"
    src_poly = Polygon(np.array(o["poly"]).reshape(-1, 2)).buffer(0)
    if src_poly.geom_type != "Polygon":
        src_poly = max(src_poly.geoms, key=lambda p: p.area)
    cw, ch = o["w"], o["h"]

    with rasterio.open(TIF(cid)) as r:
        b = r.bounds; epsg = int(r.crs.to_epsg())
    bw, bh = b.right - b.left, b.top - b.bottom
    W = int(round(bw / 0.0762)); H = int(round(bh / 0.0762))
    rgb = fetch_3inch_utm(b.left, b.bottom, b.right, b.top, epsg, W, H)

    sx, sy = W / cw, H / ch
    outline = Polygon([(x * sx, y * sy) for x, y in src_poly.exterior.coords]).buffer(0)
    if outline.geom_type != "Polygon":
        outline = max(outline.geoms, key=lambda p: p.area)

    # primary: shading segmentation (robust on uniform shingle); fallback: line-Hough
    facets = detect_facets_shading(rgb, outline)
    method = "shading"
    lines = []
    if len(facets) < 2:
        facets, lines = detect_facets(rgb, outline)
        method = "hough"
    if not facets:
        facets, method = [outline], "flat"

    # overlay (chip + outline + detected facets) — embedded as the report aerial
    OUT.mkdir(parents=True, exist_ok=True)
    ov = rgb.copy(); d = ImageDraw.Draw(ov)
    d.line(list(outline.exterior.coords) + [outline.exterior.coords[0]], fill=(0, 255, 255), width=4)
    rng = np.random.default_rng(7)
    for f in facets:
        c = tuple(int(v) for v in rng.integers(70, 255, 3))
        d.line([tuple(p) for p in f.exterior.coords], fill=c, width=3)
    ov_path = OUT / f"{cid}_overlay.png"
    ov.save(ov_path)

    pred = {"outline": list(outline.exterior.coords),
            "facets": [{"polygon": list(f.exterior.coords)} for f in facets]}
    px_w, px_h = bw / W, bh / H
    p2w = lambda x, y: (x * px_w, y * px_h)
    res = process_chip_rgb(pred, address=addr, building_id=cid, pixel_to_world=p2w,
                           aerial_image_path=str(ov_path), output_dir=str(OUT / cid),
                           write_outputs=True)

    s = res["summary"]; pdf = res["output_files"].get("pdf", "")
    return (f"OK  {cid}  {addr[:30]:30s} | {W}x{H}px | facets={len(facets)} lines={len(lines)} | "
            f"{s['total_area_sqft']:.0f} sqft | {s['predominant_pitch']} | {Path(pdf).name}")


def load_clean_outlines():
    """cid -> {poly, w, h} from the human-verified roof_dataset_clean (cat 1 = roof_outline)."""
    out = {}
    for f in glob.glob(f"{Path(__file__).resolve().parent.parent}/training/roof_dataset_clean/*/_annotations.coco.json"):
        d = json.load(open(f))
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            cid = os.path.splitext(im["file_name"])[0]
            polys = [a["segmentation"][0] for a in anns[im["id"]] if a["category_id"] == 1]
            if polys:
                best = max(polys, key=lambda s: Polygon(np.array(s).reshape(-1, 2)).area)
                out[cid] = {"poly": best, "w": im["width"], "h": im["height"]}
    return out


def main():
    cands = sys.argv[1:] or None
    con = sqlite3.connect(DB)
    addr_of = dict(con.execute("select id, raw_address from addresses").fetchall())
    clean = load_clean_outlines()

    local = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{ADR}/chip_samples/*.tif")}
    PIN = ("petersburg", "clearwater", "largo", "seminole", "pinellas park", "dunedin",
           "tarpon springs", "palm harbor", "safety harbor", "oldsmar", "gulfport", "st pete")
    if not cands:
        cands = [i for i in sorted(local)
                 if i in clean and i in addr_of and any(c in addr_of[i].lower() for c in PIN)]
    print(f"OUTPUT DIR: {OUT}   ({len(cands)} roofs, clean outlines)")
    for cid in cands:
        try:
            print(run(cid, addr_of.get(cid, cid), clean), flush=True)
        except Exception as e:
            import traceback; print(f"ERR {cid}: {type(e).__name__}: {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
