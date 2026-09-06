"""
Top-down vector roof diagram (PIL) for the report pages — EagleView clean-line
style: white/lightly-shaded facets, thin grey internal seams, a bold outline, and
colour-coded typed edges. Per-mode labels:
  'plain'  — clean outline only
  'length' — every edge coloured by type (valleys dashed) + its length in feet
  'area'   — each facet labelled with its square footage
  'pitch'  — each facet shaded (blue = pitched, grey = flat) + pitch + downslope arrow
  'notes'  — each facet lettered A..Z, smallest to largest
North is up (y flipped).
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.output.units import m_to_ft, m2_to_sqft

logger = logging.getLogger(__name__)

# EagleView-style palette: clean line drawing, not multicolour blobs.
WHITE_FILL = (255, 255, 255)
PITCHED_FILL = (214, 228, 246)      # light blue = pitched (>=3/12), pitch diagram
FLAT_FILL = (224, 224, 224)         # grey = flat, both here and the pitch diagram
SEAM = (165, 165, 165)              # thin grey internal facet seams
ROOF_OUTLINE = (25, 25, 25)         # bold near-black roof boundary
REVIEW_OUTLINE = (230, 140, 30)     # steep/uncertain facet -> verify in oblique
LABEL_RGB = (35, 35, 35)

# edge styling: (RGB, dashed) — matches EagleView's length-diagram legend
EDGE_STYLE: Dict[str, Tuple[Tuple[int, int, int], bool]] = {
    "ridge": ((198, 32, 32), False),          # red
    "hip": ((198, 32, 32), False),            # red
    "valley": ((32, 64, 200), True),          # blue, dashed
    "rake": ((30, 120, 40), False),           # green
    "eave": ((25, 25, 25), False),            # black
    "step_flashing": ((205, 140, 0), False),  # gold
    "wall_flashing": ((205, 140, 0), False),  # gold
    "transition": ((130, 130, 130), False),   # grey
    "parapet": ((130, 130, 130), False),
    "unspecified": ((130, 130, 130), False),
}

_DIRV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0),
         "NE": (0.71, -0.71), "NW": (-0.71, -0.71),
         "SE": (0.71, 0.71), "SW": (-0.71, 0.71)}


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_arrow(draw, x, y, dx, dy, length, color, width=2):
    ex, ey = x + dx * length, y + dy * length
    draw.line([(x, y), (ex, ey)], fill=color, width=width)
    ang = math.atan2(dy, dx)
    for da in (2.5, -2.5):
        draw.line([(ex, ey),
                   (ex + 5 * math.cos(ang + da), ey + 5 * math.sin(ang + da))],
                  fill=color, width=width)


def _dashed_line(draw, p0, p1, color, width=3, dash=7, gap=5):
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return
    ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
    d = 0.0
    while d < dist:
        a = (x0 + ux * d, y0 + uy * d)
        b = (x0 + ux * min(d + dash, dist), y0 + uy * min(d + dash, dist))
        draw.line([a, b], fill=color, width=width)
        d += dash + gap


def _bounds(report_input: dict) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for f in report_input.get("facets", []):
        for x, y in f.get("polygon_xy", []):
            xs.append(x); ys.append(y)
    for e in report_input.get("edges", []):
        for x, y in e.get("geometry_xy", []):
            xs.append(x); ys.append(y)
    for x, y in report_input.get("outline_xy", []):
        xs.append(x); ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _make_transform(bounds, size, margin):
    minx, miny, maxx, maxy = bounds
    w, h = size
    span_x = max(maxx - minx, 1e-6)
    span_y = max(maxy - miny, 1e-6)
    scale = min((w - 2 * margin) / span_x, (h - 2 * margin) / span_y)
    off_x = (w - span_x * scale) / 2
    off_y = (h - span_y * scale) / 2

    def tx(x, y):
        sx = off_x + (x - minx) * scale
        sy = h - (off_y + (y - miny) * scale)   # flip y so north is up
        return sx, sy
    return tx


def _facet_letters(facets: List[dict]) -> Dict[int, str]:
    """A..Z (then AA, AB, ...) by ascending area — the EagleView Notes convention."""
    def area(f):
        return f.get("surface_area_m2") or f.get("plan_area_m2", 0.0)
    order = sorted(range(len(facets)), key=lambda i: area(facets[i]))
    out = {}
    for rank, i in enumerate(order):
        out[i] = chr(65 + rank) if rank < 26 else "A" + chr(65 + rank - 26)
    return out


def _centroid(poly):
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def render_diagram(report_input: dict, mode: str = "plain",
                   size: Tuple[int, int] = (760, 560), margin: int = 50) -> Image.Image:
    """Render the roof diagram in the given label mode -> PIL RGB image."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    bounds = _bounds(report_input)
    if bounds is None:
        draw.text((size[0] // 2 - 40, size[1] // 2), "no geometry", fill=LABEL_RGB,
                  font=_font(14))
        return img
    tx = _make_transform(bounds, size, margin)
    f_small, f_lab = _font(12), _font(14)
    facets = report_input.get("facets", [])

    # 1. facets — clean fills + thin grey seams (no multicolour blobs)
    for f in facets:
        poly = f.get("polygon_xy", [])
        if len(poly) < 3:
            continue
        pts = [tx(x, y) for x, y in poly]
        if f.get("is_flat"):
            fill = FLAT_FILL
        elif mode == "pitch":
            fill = PITCHED_FILL
        else:
            fill = WHITE_FILL
        edge = REVIEW_OUTLINE if f.get("needs_review") else SEAM
        draw.polygon(pts, fill=fill, outline=edge)

    # 2. roof outline — bold boundary over the facet fills
    outline_xy = report_input.get("outline_xy", [])
    if len(outline_xy) >= 3:
        opts = [tx(x, y) for x, y in outline_xy]
        draw.line(opts + [opts[0]], fill=ROOF_OUTLINE, width=3)

    # 3. edges — coloured by type in length mode (valleys dashed) + length labels
    if mode == "length":
        for e in report_input.get("edges", []):
            g = e.get("geometry_xy", [])
            if len(g) < 2:
                continue
            p0, p1 = tx(*g[0]), tx(*g[1])
            color, dashed = EDGE_STYLE.get(e.get("edge_type"), ((90, 90, 90), False))
            if dashed:
                _dashed_line(draw, p0, p1, color, width=3)
            else:
                draw.line([p0, p1], fill=color, width=3)
            ft = m_to_ft(e.get("length_m", 0.0))
            if ft >= 1:                          # skip sub-foot clutter like EagleView
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                draw.text((mx - 6, my - 7), str(int(round(ft))), fill=color, font=f_small)

    # 4. per-facet labels
    if mode in ("area", "pitch", "notes"):
        letters = _facet_letters(facets) if mode == "notes" else {}
        for i, f in enumerate(facets):
            poly = f.get("polygon_xy", [])
            if len(poly) < 3:
                continue
            sx, sy = tx(*_centroid(poly))
            if mode == "area":
                label = str(int(round(m2_to_sqft(
                    f.get("surface_area_m2") or f.get("plan_area_m2", 0.0)))))
                draw.text((sx - 9, sy - 7), label, fill=LABEL_RGB, font=f_small)
            elif mode == "notes":
                draw.text((sx - 4, sy - 8), letters.get(i, "?"), fill=LABEL_RGB, font=f_lab)
            else:  # pitch
                review = f.get("needs_review")
                color = REVIEW_OUTLINE if review else LABEL_RGB
                if f.get("is_flat"):
                    draw.text((sx - 10, sy - 7), "Flat", fill=color, font=f_small)
                else:
                    pitch = f.get("pitch_string") or "–"
                    draw.text((sx - 12, sy - 15), f"{pitch}{'?' if review else ''}",
                              fill=color, font=f_small)
                    d = _DIRV.get(f.get("aspect_bin", ""))
                    if d:
                        _draw_arrow(draw, sx, sy + 6, d[0], d[1], 13, color)

    _draw_compass(draw, size, _font(12))
    return img


def _draw_compass(draw, size, font):
    cx, cy = size[0] - 32, size[1] - 32
    draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(120, 120, 120), width=1)
    draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(120, 120, 120), width=1)
    draw.text((cx - 3, cy - 26), "N", fill=LABEL_RGB, font=font)
