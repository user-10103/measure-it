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
MSFT_LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
MSFT_LINKS_CACHE = Path.home() / ".cache" / "msft_links.csv"

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


def _quadkey(lat, lon, z=9):
    sinlat = math.sin(math.radians(lat))
    x = (lon + 180) / 360
    y = 0.5 - math.log((1 + sinlat) / (1 - sinlat)) / (4 * math.pi)
    tx, ty = int(x * 2 ** z), int(y * 2 ** z)
    qk = ""
    for i in range(z, 0, -1):
        digit, mask = 0, 1 << (i - 1)
        if tx & mask: digit += 1
        if ty & mask: digit += 2
        qk += str(digit)
    return qk


def msft_footprint(lon, lat):
    """Nearest Microsoft (Global ML) building footprint to the point. Cached."""
    key = f"ms:{round(lon, 5)},{round(lat, 5)}"
    cache = json.loads(FP_CACHE.read_text()) if FP_CACHE.exists() else {}
    if key in cache:
        return shp_shape(cache[key]) if cache[key] else None
    try:
        if not MSFT_LINKS_CACHE.exists():
            MSFT_LINKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            MSFT_LINKS_CACHE.write_bytes(_get(MSFT_LINKS, timeout=60))
        import csv, gzip
        qk = _quadkey(lat, lon, 9)
        url = None
        for row in csv.DictReader(open(MSFT_LINKS_CACHE)):
            if row.get("QuadKey") == qk:
                url = row["Url"]; break
        best = None
        if url:
            raw = _get(url, timeout=120)
            txt = gzip.decompress(raw).decode() if url.endswith(".gz") or raw[:2] == b"\x1f\x8b" else raw.decode()
            pt = Point(lon, lat); bd = 1e9
            for line in txt.splitlines():
                if not line.strip():
                    continue
                feat = json.loads(line)
                g = feat.get("geometry")
                if not g:
                    continue
                poly = shp_shape(g).buffer(0)
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                d = 0.0 if poly.contains(pt) else poly.distance(pt)
                if d < bd:
                    bd, best = d, poly
                    if d == 0.0:
                        break
    except Exception:
        best = None
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


def _xform(poly, s, dx, dy, cx, cy):
    return Polygon([((x - cx) * s + cx + dx, (y - cy) * s + cy + dy)
                    for x, y in poly.exterior.coords])


def _boundary_edge_score(poly, E):
    """Mean image-edge strength sampled along a polygon's boundary."""
    H, W = E.shape
    per = poly.exterior.length
    n = max(48, int(per / 3))
    tot = cnt = 0.0
    for i in range(n):
        p = poly.exterior.interpolate(i / n, normalized=True)
        x, y = int(round(p.x)), int(round(p.y))
        if 0 <= x < W and 0 <= y < H:
            tot += E[y, x]; cnt += 1
    return tot / max(cnt, 1)


def refine_outline(rgb, outline, max_shift=10, buffers=(0, 3, 6, 9, 12), E=None):
    """Register + expand the footprint onto the real roof edges in the image.

    OSM/MS give the WALL footprint; the roof overhangs the walls by a roughly
    CONSTANT eave distance. Search a constant outward offset (mitre buffer = sharp
    corners) plus a small translation, maximizing the boundary's overlap with image
    edges. Constant offset is more correct than a uniform scale, which over-expands
    long walls relative to short ones. buffers are in pixels (~0.076 m/px: 12px≈3ft).
    """
    if E is None:
        gray = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(1)), float)
        gy, gx = np.gradient(gray)
        E = np.hypot(gx, gy); E = E / (E.max() + 1e-6)
    cx, cy = outline.centroid.x, outline.centroid.y
    best, bestsc = outline, _boundary_edge_score(outline, E)
    for bd in buffers:
        exp = outline if bd == 0 else outline.buffer(bd, join_style=2, mitre_limit=4.0)
        if exp.geom_type != "Polygon":
            exp = max(exp.geoms, key=lambda p: p.area) if hasattr(exp, "geoms") else outline
        for dx in range(-max_shift, max_shift + 1, 2):
            for dy in range(-max_shift, max_shift + 1, 2):
                cand = _xform(exp, 1.0, dx, dy, cx, cy)
                sc = _boundary_edge_score(cand, E)
                if sc > bestsc:
                    bestsc, best = sc, cand
    return best.buffer(0), bestsc   # absolute edge score (comparable across sources)


def _rotate(pts, ang, cx, cy):
    c, s = math.cos(ang), math.sin(ang)
    return [[(x - cx) * c - (y - cy) * s + cx, (x - cx) * s + (y - cy) * c + cy] for x, y in pts]


def regularize_footprint(poly, angle_tol=18.0, min_rectilinear=0.65):
    """Snap a noisy footprint to a clean rectilinear (Manhattan) polygon.

    Most homes are rectilinear; raw OSM/MS outlines have jittery, off-square edges
    that give the straight skeleton ragged facets. Find the dominant orientation,
    rotate to it, force each edge axis-aligned, rotate back. Skips genuinely
    angled/curved roofs (octagons, bays) so it never distorts them.
    """
    poly = poly.buffer(0).simplify(max(poly.length * 0.008, 1.0))
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n < 4:
        return poly
    cx, cy = poly.centroid.x, poly.centroid.y
    sin4 = cos4 = 0.0                                   # dominant orientation mod 90°
    for i in range(n):
        a, b = coords[i], coords[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        sin4 += L * math.sin(4 * ang); cos4 += L * math.cos(4 * ang)
    th = math.atan2(sin4, cos4) / 4
    rot = _rotate(coords, -th, cx, cy)
    axis_len = tot = 0.0; cls = []
    for i in range(n):
        a, b = rot[i], rot[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy); tot += L
        off = min(abs(math.degrees(math.atan2(dy, dx))) % 90, 90 - abs(math.degrees(math.atan2(dy, dx))) % 90)
        if off < angle_tol:
            axis_len += L
        cls.append("H" if abs(dx) >= abs(dy) else "V")
    if tot == 0 or axis_len / tot < min_rectilinear:    # not rectilinear -> leave alone
        return poly
    pts = [list(p) for p in rot]
    for _ in range(15):                                 # relax edges to axis-aligned
        for i in range(n):
            j = (i + 1) % n
            if cls[i] == "H":
                m = (pts[i][1] + pts[j][1]) / 2; pts[i][1] = pts[j][1] = m
            else:
                m = (pts[i][0] + pts[j][0]) / 2; pts[i][0] = pts[j][0] = m
    out = Polygon(_rotate(pts, th, cx, cy)).buffer(0)
    if out.geom_type != "Polygon":
        out = max(out.geoms, key=lambda p: p.area)
    return out if 0.6 * poly.area < out.area < 1.5 * poly.area else poly


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

    # 2) building outline — BEST OF BOTH: OpenStreetMap + Microsoft footprints
    cands = [(n, f) for n, f in (("osm", osm_footprint(lon, lat)),
                                 ("msft", msft_footprint(lon, lat))) if f is not None]
    if not cands:
        return f"{label}: no footprint (OSM or Microsoft)"

    # 3) chip bbox over the union of candidates + 18% margin; fetch 3-inch
    xs0 = min(f.bounds[0] for _, f in cands); ys0 = min(f.bounds[1] for _, f in cands)
    xs1 = max(f.bounds[2] for _, f in cands); ys1 = max(f.bounds[3] for _, f in cands)
    mx, my = (xs1 - xs0) * 0.18, (ys1 - ys0) * 0.18
    bx0, by0, bx1, by1 = xs0 - mx, ys0 - my, xs1 + mx, ys1 + my
    bw_m = (bx1 - bx0) * 111320 * math.cos(math.radians(lat))
    bh_m = (by1 - by0) * 111320
    W = max(64, int(round(bw_m / gsd))); H = max(64, int(round(bh_m / gsd)))
    rgb = export_bbox(base, bx0, by0, bx1, by1, W, H)

    def to_px(lo, la):
        return ((lo - bx0) / (bx1 - bx0) * W, (by1 - la) / (by1 - by0) * H)

    # 4) refine BOTH footprints onto the real roof edges; keep the better fit
    gray = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(1)), float)
    gyy, gxx = np.gradient(gray); E = np.hypot(gxx, gyy); E = E / (E.max() + 1e-6)
    refine = os.environ.get("NO_REFINE") != "1"
    scored = []
    for name, f in cands:
        o = Polygon([to_px(lo, la) for lo, la in f.exterior.coords]).buffer(0)
        if o.geom_type != "Polygon":
            o = max(o.geoms, key=lambda p: p.area)
        ro, sc = refine_outline(rgb, o, E=E) if refine else (o, _boundary_edge_score(o, E))
        scored.append((sc, name, o, ro))
    scored.sort(reverse=True, key=lambda t: t[0])
    bestsc, fp_src, outline_raw, outline = scored[0]
    src += f" | {fp_src}"
    if len(scored) > 1:
        src += f">{scored[1][1]}"
    # regularize the refined outline -> single canonical roof outline (facets + report agree)
    if os.environ.get("NO_REGULARIZE") != "1":
        rg = regularize_footprint(outline)
        if rg.geom_type == "Polygon" and 0.6 * outline.area < rg.area < 1.5 * outline.area:
            outline = rg

    # 5) facets — ALGORITHMS ONLY.
    # primary: true straight skeleton of the footprint (handles L/T/complex shapes),
    # with a hip-vs-gable image check that drops gable edges so the ridge runs to
    # the eave there. Imagery shading/line only as a degenerate fallback.
    method = "straight-skeleton"; flat = False
    try:
        # outline is already refined + regularized (single canonical roof outline)
        f0, edges = straight_skeleton_facets(outline)
        # (a) flat-roof check: uniform tone across all skeleton regions -> one flat plane
        is_flat, spread = roof_is_flat(rgb, list(f0.values()))
        if is_flat:
            facets = [outline]; flat = True; method = f"flat (spread={spread:.2f})"
        else:
            # (b) hip-vs-gable: OFF by default. Even the corroborated test misfires
            # because a hip's central ridge is collinear with a gable's ridge (both
            # on the centerline through the end-eave midpoint), so nadir line-geometry
            # can't separate them. Reliable hip/gable needs the trained model or 3D.
            # Opt in with DETECT_GABLE=1.
            gabs = gable_edges(rgb, f0, edges, outline) if os.environ.get("DETECT_GABLE") == "1" else set()
            if gabs:
                f0, _ = straight_skeleton_facets(outline, edge_idx=[ei for ei in range(len(edges)) if ei not in gabs])
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
    d.line(list(outline_raw.exterior.coords) + [outline_raw.exterior.coords[0]],
           fill=(255, 90, 0), width=2)                 # raw OSM footprint (orange)
    d.line(list(outline.exterior.coords) + [outline.exterior.coords[0]],
           fill=(0, 255, 255), width=4)                # refined roof outline (cyan)
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
    return (f"OK  {label[:30]:30s} | {src} | facets={len(facets)}({method}) | "
            f"{s['total_area_sqft']:.0f} sqft | fit={bestsc:.3f} | {odir.name}/report.pdf")


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
