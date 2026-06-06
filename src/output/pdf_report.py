"""
Assemble the 6-page Roofr-style Roof Report PDF (reportlab).

Pages: cover, diagram, length report, area report, pitch & direction, summary.
generate_report(report_input, out_pdf) is the single entry point.
"""

import logging
import os
from typing import List, Optional, Tuple

from src.output.diagram import render_diagram
from src.output.report_data import ReportModel, build_report_model
from src.output.units import m_to_ftin, sqft_to_squares

logger = logging.getLogger(__name__)

BLUE = (0.18, 0.49, 0.77)
DARK = (0.15, 0.15, 0.15)
GREY = (0.45, 0.45, 0.45)


def _pitch_slash(pitch_string: str) -> str:
    return pitch_string.replace(":", "/")


def _require_reportlab():
    try:
        from reportlab.pdfgen import canvas  # noqa: F401
        from reportlab.lib.pagesizes import letter  # noqa: F401
        from reportlab.lib.utils import ImageReader  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("reportlab is required for PDF export: pip install reportlab") from e


def _draw_diagram(c, report_input, mode, x, y, w, h):
    from reportlab.lib.utils import ImageReader
    img = render_diagram(report_input, mode=mode, size=(int(w * 2), int(h * 2)))
    c.drawImage(ImageReader(img), x, y, width=w, height=h, preserveAspectRatio=True,
                anchor="n")


def _header(c, W, H, title, address):
    c.setFillColorRGB(*BLUE)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(54, H - 70, title)
    c.setFillColorRGB(*GREY)
    c.setFont("Helvetica", 10)
    c.drawString(54, H - 86, address)


def _footer(c, W, page_no):
    c.setFillColorRGB(*GREY)
    c.setFont("Helvetica", 8)
    c.drawString(54, 36, "Prepared by measure-it")
    c.drawRightString(W - 54, 36, str(page_no))


def generate_report(report_input: dict, out_pdf: str,
                    model: Optional[ReportModel] = None) -> str:
    """Render the 6-page report to out_pdf. Returns the path."""
    _require_reportlab()
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader

    if model is None:
        model = build_report_model(report_input)
    W, H = letter
    address = report_input.get("address", "")
    c = canvas.Canvas(out_pdf, pagesize=letter)

    # ---- Page 1: cover ----
    c.setFillColorRGB(*BLUE)
    c.setFont("Helvetica-Bold", 30)
    c.drawString(54, H - 110, "Roof Report")
    c.setFillColorRGB(*GREY)
    c.setFont("Helvetica", 11)
    c.drawString(54, H - 128, "Prepared by measure-it")
    c.drawString(54, H - 158, address)
    c.setFillColorRGB(*DARK)
    c.setFont("Helvetica", 11)
    stats = [f"{int(round(model.total_area_sqft))} sqft",
             f"{model.num_facets} facets",
             f"Predominant pitch {_pitch_slash(model.predominant_pitch)}"]
    for i, s in enumerate(stats):
        c.drawRightString(W - 54, H - 110 - i * 16, s)
    # Occluded/unsegmentable roof: stamp a loud full-width strip across the top
    # so a blank-roof report can never be mistaken for a finished one. Drawn at
    # the page top where nothing else lives -> no overpaint.
    if report_input.get("occluded_roof"):
        c.setFillColorRGB(0.85, 0.20, 0.15)
        c.rect(0, H - 30, W, 30, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(W / 2, H - 20,
                            "INCOMPLETE - MANUAL REVIEW REQUIRED  (roof not segmentable from imagery)")
        c.setFillColorRGB(*DARK)
    aerial = report_input.get("aerial_image_path")
    box = (90, H - 560, W - 180, 380)
    if aerial and os.path.exists(aerial):
        try:
            c.drawImage(ImageReader(aerial), box[0], box[1], width=box[2], height=box[3],
                        preserveAspectRatio=True, anchor="n")
        except Exception:
            _placeholder(c, box)
    else:
        _placeholder(c, box)
    _footer(c, W, 1)
    c.showPage()

    # ---- Page 2: diagram ----
    _header(c, W, H, "Diagram", address)
    _draw_diagram(c, report_input, "plain", 80, H - 600, W - 160, 460)
    _footer(c, W, 2)
    c.showPage()

    # ---- Page 3: length measurement report ----
    _header(c, W, H, "Length measurement report", address)
    _draw_diagram(c, report_input, "length", 90, H - 640, W - 180, 340)
    _length_legend(c, W, H, model)        # legend after diagram so text isn't overpainted
    _footer(c, W, 3)
    c.showPage()

    # ---- Page 4: area measurement report ----
    _header(c, W, H, "Area measurement report", address)
    _area_totals(c, W, H, model)
    _draw_diagram(c, report_input, "area", 90, H - 620, W - 180, 360)
    _footer(c, W, 4)
    c.showPage()

    # ---- Page 5: pitch & direction ----
    _header(c, W, H, "Pitch & direction measurement report", address)
    _draw_diagram(c, report_input, "pitch", 90, H - 600, W - 180, 420)
    _footer(c, W, 5)
    c.showPage()

    # ---- Page 6: summary ----
    _header(c, W, H, "Report summary", address)
    _summary_tables(c, W, H, model)
    _footer(c, W, 6)
    c.showPage()

    c.save()
    logger.info("Wrote report PDF: %s", out_pdf)
    return out_pdf


def _placeholder(c, box):
    x, y, w, h = box
    c.setStrokeColorRGB(*GREY)
    c.setFillColorRGB(0.93, 0.93, 0.93)
    c.rect(x, y, w, h, fill=1, stroke=1)
    c.setFillColorRGB(*GREY)
    c.setFont("Helvetica", 11)
    c.drawCentredString(x + w / 2, y + h / 2, "aerial image")


def _kv_lines(c, x, y, pairs: List[Tuple[str, str]], dy=16):
    c.setFont("Helvetica", 10)
    for i, (k, v) in enumerate(pairs):
        c.setFillColorRGB(*DARK)
        c.drawString(x, y - i * dy, k)
        c.drawString(x + 180, y - i * dy, v)


def _length_legend(c, W, H, model: ReportModel):
    order = ["eave", "ridge", "hip", "valley", "rake", "step_flashing",
             "wall_flashing", "transition", "parapet", "unspecified"]
    names = {"eave": "Eaves", "ridge": "Ridges", "hip": "Hips", "valley": "Valleys",
             "rake": "Rakes", "step_flashing": "Step flashing", "wall_flashing":
             "Wall flashing", "transition": "Transitions", "parapet": "Parapet wall",
             "unspecified": "Unspecified"}
    pairs = [(names[t], model.edge_totals_ftin.get(t, "0ft 0in")) for t in order]
    _kv_lines(c, 54, H - 120, pairs)


def _area_totals(c, W, H, model: ReportModel):
    left = [("Total roof area", f"{int(round(model.total_area_sqft))} sqft"),
            ("Pitched roof area", f"{int(round(model.pitched_area_sqft))} sqft"),
            ("Flat roof area", f"{int(round(model.flat_area_sqft))} sqft"),
            ("Two story area", "0 sqft"),
            ("Two layer area", "0 sqft")]
    _kv_lines(c, 54, H - 120, left)
    right = [("Predominant pitch", _pitch_slash(model.predominant_pitch)),
             ("Predominant pitch area", f"{int(round(model.predominant_pitch_area_sqft))} sqft"),
             ("Unspecified pitch area", f"{int(round(model.unspecified_pitch_area_sqft))} sqft")]
    _kv_lines(c, W / 2, H - 120, right)


def _summary_tables(c, W, H, model: ReportModel):
    e = model.edge_totals_ftin
    rows = [("Total roof area", f"{int(round(model.total_area_sqft))} sqft"),
            ("Total pitched area", f"{int(round(model.pitched_area_sqft))} sqft"),
            ("Total flat area", f"{int(round(model.flat_area_sqft))} sqft"),
            ("Total roof facets", f"{model.num_facets} facets"),
            ("Predominant pitch", _pitch_slash(model.predominant_pitch)),
            ("Total eaves", e.get("eave", "0ft 0in")),
            ("Total valleys", e.get("valley", "0ft 0in")),
            ("Total hips", e.get("hip", "0ft 0in")),
            ("Total ridges", e.get("ridge", "0ft 0in")),
            ("Total rakes", e.get("rake", "0ft 0in")),
            ("Total wall flashing", e.get("wall_flashing", "0ft 0in")),
            ("Total step flashing", e.get("step_flashing", "0ft 0in")),
            ("Total transitions", e.get("transition", "0ft 0in")),
            ("Total parapet wall", e.get("parapet", "0ft 0in")),
            ("Total unspecified", e.get("unspecified", "0ft 0in"))]
    c.setFillColorRGB(*BLUE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2, H - 110, "Measurements")
    _kv_lines(c, W / 2, H - 130, rows, dy=15)

    # waste table (lower-left)
    c.setFillColorRGB(*BLUE)
    c.setFont("Helvetica-Bold", 11)
    y0 = 230
    c.drawString(54, y0, "Waste %")
    c.setFillColorRGB(*DARK)
    c.setFont("Helvetica", 9)
    xs = [54, 130, 200, 270, 340, 410, 470, 540]
    headers = [w["pct"] for w in model.waste_table]
    for i, p in enumerate(headers):
        c.drawString(xs[1] + i * 60, y0, f"{p}%")
    c.drawString(54, y0 - 16, "Area (sqft)")
    c.drawString(54, y0 - 32, "Squares")
    for i, w in enumerate(model.waste_table):
        c.drawString(xs[1] + i * 60, y0 - 16, str(int(round(w["area_sqft"]))))
        c.drawString(xs[1] + i * 60, y0 - 32, str(w["squares"]))
