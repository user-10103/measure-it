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
# distinct per-facet fills (pastel tab20 — light enough to keep dark labels legible)
FACET_PALETTE = [
    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150),
    (197, 176, 213), (247, 182, 210), (219, 219, 141), (158, 218, 229),
    (196, 176, 148), (206, 219, 156), (240, 190, 130), (180, 205, 230),
]
OUTLINE = (70, 110, 165)
ROOF_OUTLINE = (25, 55, 110)      # zero-shot roof outline -> bold boundary over the facets
REVIEW_OUTLINE = (230, 140, 30)   # steep/uncertain facet -> verify in oblique
LABEL_RGB = (40, 40, 40)

ARROWS = {"N": "^", "S": "v", "E": ">", "W": "<",
          "NE": "/", "SW": "/", "NW": "\\", "SE": "\\"}

# screen-space (north-up) unit vectors per aspect bin = the downslope direction
import math as _math
_DIRV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0),
         "NE": (0.71, -0.71), "NW": (-0.71, -0.71),
         "SE": (0.71, 0.71), "SW": (-0.71, 0.71)}


def _draw_arrow(draw, x, y, dx, dy, length, color, width=2):
    """Short downslope arrow with a real drawn arrowhead (not an ASCII glyph)."""
    ex, ey = x + dx * length, y + dy * length
    draw.line([(x, y), (ex, ey)], fill=color, width=width)
    ang = _math.atan2(dy, dx)
    for da in (2.5, -2.5):
        draw.line([(ex, ey),
                   (ex + 5 * _math.cos(ang + da), ey + 5 * _math.sin(ang + da))],
                  fill=color, width=width)


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

    # facets — each one its own colour (like the model overlay); flat stays grey
    for i, f in enumerate(report_input.get("facets", [])):
        poly = f.get("polygon_xy", [])
        if len(poly) < 3:
            continue
        pts = [tx(x, y) for x, y in poly]
        fill = FLAT_FILL if f.get("is_flat") else FACET_PALETTE[i % len(FACET_PALETTE)]
        edge = REVIEW_OUTLINE if f.get("needs_review") else OUTLINE
        draw.polygon(pts, fill=fill, outline=edge)

    # zero-shot roof outline — bold boundary drawn over the facet fills
    outline_xy = report_input.get("outline_xy", [])
    if len(outline_xy) >= 3:
        opts = [tx(x, y) for x, y in outline_xy]
        draw.line(opts + [opts[0]], fill=ROOF_OUTLINE, width=4)

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
                label = str(int(round(m2_to_sqft(f.get("surface_area_m2") or f.get("plan_area_m2", 0.0)))))
                if f.get("is_flat"):
                    label = "Flat " + label
                draw.text((sx - 8, sy - 7), label, fill=LABEL_RGB, font=f_small)
            else:  # pitch — full "5:12"; "–" when unmeasured (never a fake "0")
                review = f.get("needs_review")
                color = REVIEW_OUTLINE if review else LABEL_RGB
                if f.get("is_flat"):
                    draw.text((sx - 10, sy - 7), "Flat", fill=color, font=f_small)
                else:
                    pitch = f.get("pitch_string") or "–"
                    label = f"{pitch}{'?' if review else ''}"
                    draw.text((sx - 10, sy - 14), label, fill=color, font=f_small)
                    d = _DIRV.get(f.get("aspect_bin", ""))
                    if d:                       # drawn downslope arrowhead
                        _draw_arrow(draw, sx, sy + 5, d[0], d[1], 13, color)

    _draw_compass(draw, size, _font(12))
    return img


def _draw_compass(draw, size, font):
    cx, cy = size[0] - 32, size[1] - 32
    draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(120, 120, 120), width=1)
    draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(120, 120, 120), width=1)
    draw.text((cx - 3, cy - 26), "N", fill=LABEL_RGB, font=font)
