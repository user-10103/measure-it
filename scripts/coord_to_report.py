#!/usr/bin/env python3
"""coord_to_report.py — coordinates/address -> Roof Report, ALGORITHMS ONLY.

No trained model, no pre-existing annotations, no labeled data of ours. Generalizes
to any Florida address:

  address | "lat,lon"
   -> geocode + county            (US Census, free)
   -> building outline            (OpenStreetMap building footprint, public)
   -> fetch real 3-inch aerial    (county GIS, public-record, over the footprint bbox)
   -> facets                      (shading-segmentation k-means + line fallback)
   -> measure-it report           (tiling, geometric aspect, pitch policy, edges) -> PDF

Outputs to ~/Downloads/coord_reports/<slug>/report.pdf + overlay.
"""
import os, sys, json, math, io, re
from pathlib import Path
import urllib.request, urllib.parse

import numpy as np
import shapely
from rasterio.features import shapes as rio_shapes
from rasterio import Affine
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import Polygon, Point, shape as shp_shape, mapping as shp_mapping
from shapely.ops import unary_union, polygonize

FP_CACHE = Path.home() / ".cache" / "measureit_footprints.json"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rgb_pipeline import process_chip_rgb
from scripts.classical_report import (rasterize_mask, hough_lines, line_to_seg,
                                       detect_facets_shading)

UA = {"User-Agent": "Mozilla/5.0 (measure-it coord->report)"}
OUT = Path.home() / "Downloads" / "coord_reports"
CENSUS_ADDR = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
CENSUS_COORD = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
COUNTY_EP = {
    "Pinellas":     ("https://egis.pinellas.gov/gis/rest/services/Aerials/Aerials2024/ImageServer", 0.0762),
    "Hillsborough": ("https://maps.hillsboroughcounty.org/arcgis/rest/services/AerialsNew/Aerials2025_3_inch_MrSid/ImageServer", 0.0762),
    "Pasco":        ("https://pascogis.pascocountyfl.net/giswebi/rest/services/Aerials/Aerials2023/ImageServer", 0.0762),
}
FCDOP = ("https://ca.dep.state.fl.us/arcgis/rest/services/Imagery/Aerial_Imagery_2019/ImageServer", 0.15)


def _get(url, timeout=40):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def geocode(address):
    q = urllib.parse.urlencode({"address": address, "benchmark": "Public_AR_Current",
                                "vintage": "Current_Current", "format": "json"})
    d = json.loads(_get(f"{CENSUS_ADDR}?{q}"))
    m = d["result"]["addressMatches"]
    if not m:
        raise ValueError(f"no geocode match for {address!r}")
    a = m[0]; c = a["geographies"].get("Counties", [{}])[0].get("BASENAME")
    return a["coordinates"]["x"], a["coordinates"]["y"], c, a["matchedAddress"]


def county_of(lon, lat):
    q = urllib.parse.urlencode({"x": lon, "y": lat, "benchmark": "Public_AR_Current",
                                "vintage": "Current_Current", "format": "json"})
    d = json.loads(_get(f"{CENSUS_COORD}?{q}"))
    cs = d["result"]["geographies"].get("Counties", [])
    return cs[0]["BASENAME"] if cs else None


def osm_footprint(lon, lat, radius=45):
    """Nearest OSM building polygon to the point (lon/lat ring). Cached on disk."""
    key = f"{round(lon, 5)},{round(lat, 5)}"
    cache = {}
    if FP_CACHE.exists():
        cache = json.loads(FP_CACHE.read_text())
        if key in cache:
            return shp_shape(cache[key]) if cache[key] else None
    ql = (f"[out:json][timeout:25];(way[\"building\"](around:{radius},{lat},{lon});"
          f"relation[\"building\"](around:{radius},{lat},{lon}););out geom;")
    data = urllib.parse.urlencode({"data": ql}).encode()
    d = None; last = None
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(2):
            try:
                d = json.loads(urllib.request.urlopen(
                    urllib.request.Request(mirror, data=data, headers=UA), timeout=60).read())
                break
            except Exception as e:
                last = e
        if d is not None:
            break
    if d is None:
        raise RuntimeError(f"all Overpass mirrors failed: {last}")
    pt = Point(lon, lat); best = None; bestd = 1e9
    for el in d.get("elements", []):
        g = el.get("geometry")
        if not g or len(g) < 4:
            continue
        try:
            poly = Polygon([(n["lon"], n["lat"]) for n in g]).buffer(0)
        except Exception:
            continue
        if poly.geom_type != "Polygon" or poly.area == 0:
            continue
        dist = 0.0 if poly.contains(pt) else poly.distance(pt)
        if dist < bestd:
            bestd, best = dist, poly
    FP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache[key] = shp_mapping(best) if best is not None else None
    FP_CACHE.write_text(json.dumps(cache))
    return best


def export_bbox(base, xmin, ymin, xmax, ymax, w, h, timeout=70):
    q = urllib.parse.urlencode({"bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": 4326,
                                "size": f"{w},{h}", "imageSR": 3857, "format": "png", "f": "image"})
    png = _get(f"{base}/exportImage?{q}", timeout=timeout)
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"non-PNG: {png[:120]!r}")
    return Image.open(io.BytesIO(png)).convert("RGB")


def detect_facets_lines(rgb_img, outline, min_area_frac=0.04):
    W, H = rgb_img.size
    gray = np.asarray(rgb_img.convert("L").filter(ImageFilter.GaussianBlur(1)))
    lines = hough_lines(gray, rasterize_mask(outline, W, H))
    cuts = [line_to_seg(r, t, H, W) for r, t in lines]
    merged = unary_union([outline.exterior] + cuts)
    return [p for p in polygonize(merged)
            if outline.contains(p.representative_point()) and p.area > min_area_frac * outline.area]


def hip_facets_from_footprint(outline, min_area_frac=0.02):
    """Clean hip-roof facets straight from footprint GEOMETRY (no imagery).

    For a uniform-pitch hipped roof the ridge runs along the footprint's long
    axis; facets are 2 trapezoids (long sides) + 2 end triangles. Computed from
    the minimum rotated rectangle -> straight edges, exact tiling. Approximate
    on L-shapes/gables, but geometrically clean where most homes are rectangular.
    """
    obb = outline.minimum_rotated_rectangle
    cs = [np.array(c) for c in list(obb.exterior.coords)[:4]]
    e1, e2 = cs[1] - cs[0], cs[2] - cs[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if l1 >= l2:
        u, a, v, b = e1 / l1, l1 / 2, e2 / l2, l2 / 2
    else:
        u, a, v, b = e2 / l2, l2 / 2, e1 / l1, l1 / 2
    C = np.mean(cs, axis=0)
    R1, R2 = C - (a - b) * u, C + (a - b) * u           # ridge endpoints
    A, B = C + a * u + b * v, C + a * u - b * v          # +u end corners
    D, E = C - a * u - b * v, C - a * u + b * v          # -u end corners
    quads = [Polygon([A, B, R2]), Polygon([D, E, R1]),
             Polygon([E, A, R2, R1]), Polygon([B, D, R1, R2])]
    facets = []
    for q in quads:
        f = q.buffer(0).intersection(outline).buffer(0)
        for p in (f.geoms if f.geom_type == "MultiPolygon" else [f]):
            if p.geom_type == "Polygon" and p.area > min_area_frac * outline.area:
                facets.append(p)
    return facets


def _seg_dist(P, a, b):
    ab = b - a; L2 = float(ab @ ab)
    if L2 == 0:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _edges_of(poly):
    cs = [np.array(c) for c in list(poly.exterior.coords)[:-1]]
    n = len(cs)
    return [(cs[i], cs[(i + 1) % n]) for i in range(n)]


def straight_skeleton_facets(outline, edge_idx=None, max_grid=300, min_area_frac=0.012):
    """Uniform-pitch straight-skeleton facets via the nearest-eave-edge partition.

    Each interior point is assigned to the footprint edge it is closest to; that
    region is the roof facet sloping down to that eave. Exact straight skeleton
    for convex footprints; produces correct ridges (between parallel edges), hips
    (45° between perpendicular edges) and valleys (at reflex corners) for L/T/U
    shapes too. `edge_idx` restricts which edges generate facets (drop gable edges).
    """
    edges = _edges_of(outline)
    n = len(edges)
    use = list(range(n)) if edge_idx is None else list(edge_idx)
    minx, miny, maxx, maxy = outline.bounds
    scale = max(maxx - minx, maxy - miny) / max_grid
    W = max(8, int(round((maxx - minx) / scale))); H = max(8, int(round((maxy - miny) / scale)))
    xs = minx + (np.arange(W) + 0.5) * scale
    ys = miny + (np.arange(H) + 0.5) * scale
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    inside = shapely.contains_xy(outline, P[:, 0], P[:, 1])
    best = np.full(P.shape[0], 1e18); lab = np.full(P.shape[0], -1, np.int32)
    for ei in use:
        d = _seg_dist(P, *edges[ei])
        m = d < best; best[m] = d[m]; lab[m] = ei
    lab[~inside] = -1
    labimg = (lab.reshape(H, W) + 1).astype(np.int32)
    transform = Affine(scale, 0, minx, 0, scale, miny)
    by_edge = {}
    for geom, val in rio_shapes(labimg, mask=(labimg > 0), transform=transform, connectivity=8):
        ei = int(val) - 1
        p = shp_shape(geom).buffer(0).intersection(outline)
        for q in (p.geoms if p.geom_type == "MultiPolygon" else [p]):
            if q.geom_type == "Polygon" and q.area > min_area_frac * outline.area:
                by_edge.setdefault(ei, []).append(q.simplify(scale * 0.8))
    return {ei: max(ps, key=lambda z: z.area) for ei, ps in by_edge.items()}, edges


def roof_is_flat(rgb, facets, spread_thresh=0.06):
    """Flat roof = one plane = uniform tone across all skeleton regions.

    A pitched roof has facets facing different ways -> a clear spread in per-facet
    mean brightness (sunlit vs shaded slopes). A flat roof's 'facets' are all the
    same plane -> nearly identical means. Returns (is_flat, spread).
    """
    gray = np.asarray(rgb.convert("L").filter(ImageFilter.MedianFilter(7)), float)
    H, W = gray.shape
    means = []
    for f in facets:
        m = rasterize_mask(f, W, H)
        if m.sum() >= 30:
            means.append(float(gray[m].mean()))
    if len(means) < 2:
        return False, 0.0
    spread = (max(means) - min(means)) / (np.mean(means) + 1e-6)
    return spread < spread_thresh, spread


def _ridge_aligned_line(rgb, facet, eave):
    """True if an end facet contains a straight line that is (1) aligned with the
    ridge axis (perpendicular to its eave) AND (2) passes near the eave MIDPOINT —
    the geometric signature of a GABLE ridge reaching the eave peak. A hip has no
    such line (hips run diagonally to the corners; stray eave/shingle lines are
    parallel but offset from the midpoint, so they're rejected)."""
    a, b = eave; d = b - a; L = np.linalg.norm(d) or 1.0; d = d / L
    n = np.array([-d[1], d[0]])                       # ridge-axis direction
    mid = (a + b) / 2
    W, H = rgb.size
    gray = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(1)))
    for rho, theta in hough_lines(gray, rasterize_mask(facet, W, H), topk=5):
        ldir = np.array([-math.sin(theta), math.cos(theta)])
        if abs(float(ldir @ n)) <= 0.82:              # not parallel to ridge axis
            continue
        dist = abs(mid[0] * math.cos(theta) + mid[1] * math.sin(theta) - rho)
        if dist < 0.22 * L:                           # ridge passes through the eave midpoint
            return True
    return False


def gable_edges(rgb, facets_by_edge, edges, outline, contrast_thresh=0.15):
    """Classify each END facet hip vs gable from imagery.

    An end facet is a short-edge facet meeting the ridge near a point. A HIP end
    is one sloped plane (uniform tone across it); a GABLE end is split by the ridge
    into two opposing slopes -> a brightness step across the eave-parallel axis.
    Returns the set of edge indices that look like gables (to drop from the skeleton).
    """
    gray = np.asarray(rgb.convert("L").filter(ImageFilter.MedianFilter(5)), float)
    Himg, Wimg = gray.shape
    elen = {ei: np.linalg.norm(b - a) for ei, (a, b) in enumerate(edges)}
    med = np.median(list(elen.values()))
    gables = set()
    for ei, f in facets_by_edge.items():
        if elen[ei] > 0.9 * med:          # only short (end) edges are hip/gable candidates
            continue
        a, b = edges[ei]; d = b - a; L = np.linalg.norm(d) or 1; d = d / L
        m = rasterize_mask(f, Wimg, Himg)
        ys, xs = np.where(m)
        if len(xs) < 40:
            continue
        mid = (a + b) / 2
        proj = (xs - mid[0]) * d[0] + (ys - mid[1]) * d[1]   # along the eave direction
        g = gray[ys, xs]
        lo, hi = g[proj < 0], g[proj >= 0]
        if len(lo) < 15 or len(hi) < 15:
            continue
        contrast = abs(lo.mean() - hi.mean()) / (g.mean() + 1e-6)
        # GABLE requires BOTH: a brightness step across the end AND a ridge-axis
        # line reaching the eave. Either alone over-fires (shadows / texture).
        if contrast > contrast_thresh and _ridge_aligned_line(rgb, f, edges[ei]):
            gables.add(ei)
    return gables


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def report_for(coord):
    # 1) resolve to lon/lat + county
    if re.match(r"^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$", coord):
        lat, lon = [float(v) for v in coord.split(",")]
        county = county_of(lon, lat); label = coord
    else:
        lon, lat, county, label = geocode(coord)
    base, gsd = COUNTY_EP.get(county, FCDOP)
    src = county if county in COUNTY_EP else f"FCDOP(6in) [{county}]"

    # 2) building outline from OSM (public footprint)
    fp = osm_footprint(lon, lat)
    if fp is None:
        return f"{label}: no OSM building footprint found (skip — needs image-seg fallback)"

    # 3) chip bbox = footprint bounds + 18% margin; fetch 3-inch over it
    x0, y0, x1, y1 = fp.bounds
    mx, my = (x1 - x0) * 0.18, (y1 - y0) * 0.18
    bx0, by0, bx1, by1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    bw_m = (bx1 - bx0) * 111320 * math.cos(math.radians(lat))
    bh_m = (by1 - by0) * 111320
    W = max(64, int(round(bw_m / gsd))); H = max(64, int(round(bh_m / gsd)))
    rgb = export_bbox(base, bx0, by0, bx1, by1, W, H)

    # 4) project footprint lon/lat -> chip pixels (y flips: north=up)
    def to_px(lo, la):
        return ((lo - bx0) / (bx1 - bx0) * W, (by1 - la) / (by1 - by0) * H)
    outline = Polygon([to_px(lo, la) for lo, la in fp.exterior.coords]).buffer(0)
    if outline.geom_type != "Polygon":
        outline = max(outline.geoms, key=lambda p: p.area)

    # 5) facets — ALGORITHMS ONLY.
    # primary: true straight skeleton of the footprint (handles L/T/complex shapes),
    # with a hip-vs-gable image check that drops gable edges so the ridge runs to
    # the eave there. Imagery shading/line only as a degenerate fallback.
    method = "straight-skeleton"; flat = False
    try:
        reg = outline.simplify(max(outline.length * 0.004, 1.0))   # regularize noisy OSM
        if reg.geom_type != "Polygon" or reg.area < 0.5 * outline.area:
            reg = outline
        f0, edges = straight_skeleton_facets(reg)
        # (a) flat-roof check: uniform tone across all skeleton regions -> one flat plane
        is_flat, spread = roof_is_flat(rgb, list(f0.values()))
        if is_flat:
            facets = [reg]; flat = True; method = f"flat (spread={spread:.2f})"
        else:
            # (b) hip-vs-gable: OFF by default. Even the corroborated test misfires
            # because a hip's central ridge is collinear with a gable's ridge (both
            # on the centerline through the end-eave midpoint), so nadir line-geometry
            # can't separate them. Reliable hip/gable needs the trained model or 3D.
            # Opt in with DETECT_GABLE=1.
            gabs = gable_edges(rgb, f0, edges, reg) if os.environ.get("DETECT_GABLE") == "1" else set()
            if gabs:
                f0, _ = straight_skeleton_facets(reg, edge_idx=[ei for ei in range(len(edges)) if ei not in gabs])
                method += f" (+{len(gabs)} gable)"
            facets = list(f0.values())
    except Exception as e:
        facets = []; method = f"skeleton-failed:{type(e).__name__}"
    if len(facets) < 2 and not flat:
        facets = detect_facets_shading(rgb, outline); method = "shading"
    if len(facets) < 2 and not flat:
        facets = detect_facets_lines(rgb, outline); method = "hough"
    if not facets:
        facets = [outline]; method = "flat"

    # 6) overlay + report
    OUT.mkdir(parents=True, exist_ok=True)
    odir = OUT / slug(label); odir.mkdir(parents=True, exist_ok=True)
    ov = rgb.copy(); d = ImageDraw.Draw(ov)
    d.line(list(outline.exterior.coords) + [outline.exterior.coords[0]], fill=(0, 255, 255), width=4)
    rng = np.random.default_rng(7)
    for f in facets:
        c = tuple(int(v) for v in rng.integers(70, 255, 3))
        d.line([tuple(p) for p in f.exterior.coords], fill=c, width=3)
    ov_path = odir / "overlay.png"; ov.save(ov_path)

    fdict = (lambda f: {"polygon": list(f.exterior.coords),
                        **({"slope_deg": 0.0, "aspect_bin": "flat"} if flat else {})})
    pred = {"outline": list(outline.exterior.coords),
            "facets": [fdict(f) for f in facets]}
    px_w, px_h = bw_m / W, bh_m / H
    res = process_chip_rgb(pred, address=label, building_id=slug(label),
                           pixel_to_world=lambda x, y: (x * px_w, y * px_h),
                           aerial_image_path=str(ov_path), output_dir=str(odir), write_outputs=True)
    s = res["summary"]
    return (f"OK  {label[:38]:38s} | {src} | {W}x{H}px | facets={len(facets)}({method}) | "
            f"{s['total_area_sqft']:.0f} sqft | {s['predominant_pitch']} | {odir/'report.pdf'}")


def main():
    coords = sys.argv[1:] or [
        "4500 Indian Rocks Rd, Largo, FL 33774",
        "1701 Sunset Point Rd, Clearwater, FL 33765",
        "6000 62nd Ave N, Pinellas Park, FL 33781",
    ]
    print(f"OUTPUT DIR: {OUT}")
    for c in coords:
        try:
            print(report_for(c), flush=True)
        except Exception as e:
            import traceback; print(f"ERR {c}: {type(e).__name__}: {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
