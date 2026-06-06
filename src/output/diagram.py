"""
Top-down vector roof diagram (PIL) for the report pages.

Renders facet polygons (pitched shaded vs flat light) and typed edges, with
per-mode labels: 'plain', 'length' (edge feet), 'area' (facet sqft),
'pitch' (facet pitch + downslope arrow). North is up (y flipped).
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.output.units import m_to_ft, m2_to_sqft
from src.roofs.categories import EDGE_COLORS

logger = logging.getLogger(__name__)

PITCHED_FILL = (212, 226, 245)
FLAT_FILL = (242, 242, 242)
OUTLINE = (70, 110, 165)
REVIEW_OUTLINE = (230, 140, 30)   # steep/uncertain facet -> verify in oblique
LABEL_RGB = (40, 40, 40)

ARROWS = {"N": "^", "S": "v", "E": ">", "W": "<",
          "NE": "/", "SW": "/", "NW": "\\", "SE": "\\"}


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _bounds(report_input: dict) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for f in report_input.get("facets", []):
        for x, y in f.get("polygon_xy", []):
            xs.append(x); ys.append(y)
    for e in report_input.get("edges", []):
        for x, y in e.get("geometry_xy", []):
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
    f_small, f_lab = _font(11), _font(13)

    # facets
    for f in report_input.get("facets", []):
        poly = f.get("polygon_xy", [])
        if len(poly) < 3:
            continue
        pts = [tx(x, y) for x, y in poly]
        fill = FLAT_FILL if f.get("is_flat") else PITCHED_FILL
        edge = REVIEW_OUTLINE if f.get("needs_review") else OUTLINE
        draw.polygon(pts, fill=fill, outline=edge)

    # edges (colored in length mode, otherwise the facet outlines carry the shape)
    for e in report_input.get("edges", []):
        g = e.get("geometry_xy", [])
        if len(g) < 2:
            continue
        p0, p1 = tx(*g[0]), tx(*g[1])
        color = EDGE_COLORS.get(e.get("edge_type"), (90, 90, 90)) if mode == "length" \
            else OUTLINE
        draw.line([p0, p1], fill=color, width=3 if mode == "length" else 2)
        if mode == "length":
            ft = m_to_ft(e.get("length_m", 0.0))
            mxy = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            draw.text((mxy[0] - 6, mxy[1] - 6), str(int(round(ft))), fill=LABEL_RGB,
                      font=f_small)

    # per-facet labels
    if mode in ("area", "pitch"):
        for f in report_input.get("facets", []):
            poly = f.get("polygon_xy", [])
            if len(poly) < 3:
                continue
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            sx, sy = tx(cx, cy)
            if mode == "area":
                label = str(int(round(m2_to_sqft(f.get("plan_area_m2", 0.0)))))
                if f.get("is_flat"):
                    label = "Flat " + label
            else:  # pitch
                pitch = (f.get("pitch_string") or "0:12").split(":")[0]
                arrow = "" if f.get("is_flat") else ARROWS.get(f.get("aspect_bin", ""), "")
                flag = "?" if f.get("needs_review") else ""
                label = f"{pitch}{flag} {arrow}".strip()
            color = REVIEW_OUTLINE if (mode == "pitch" and f.get("needs_review")) else LABEL_RGB
            draw.text((sx - 8, sy - 7), label, fill=color, font=f_small)

    _draw_compass(draw, size, _font(12))
    return img


def _draw_compass(draw, size, font):
    cx, cy = size[0] - 32, size[1] - 32
    draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(120, 120, 120), width=1)
    draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(120, 120, 120), width=1)
    draw.text((cx - 3, cy - 26), "N", fill=LABEL_RGB, font=font)
