"""Bridge the SAM imagery engine to the 6-page roof report.

Turns a ``SamRoof`` (from ``roofs.sam_segment.segment_roof_sam`` — the roof
outline + clean facet polygons in a planar/metric CRS) into the pipeline's
``report_input`` dict and, from there, the finished ``roof_report.pdf`` via the
existing ``output.pdf_report.generate_report``.

This is the last brick in the address-to-report flow for the SAM engine:
    address -> NAIP -> segment_roof_sam -> sam_roof_to_pdf -> roof_report.pdf

No LiDAR here, so pitch is "unspecified" (surface area = plan area) and edges
reduce to the roof outline (eaves). Typed edges (ridge/hip/valley) and real
pitch arrive when the LiDAR plane fit is wired in — they slot into the same
``report_input`` with no change to the report layer.
"""
from __future__ import annotations

import hashlib
from datetime import date
from typing import List, Optional

from src.output.pdf_report import generate_report
from src.roofs.geom_edges import edges_from_outline_and_facets


def _report_id(address: str) -> str:
    """Stable, human-readable report id: MI-<YYYYMMDD>-<addr hash>. Same address
    on the same day is idempotent; different addresses never collide."""
    h = hashlib.sha1(address.encode("utf-8")).hexdigest()[:6].upper()
    return f"MI-{date.today():%Y%m%d}-{h}"


def _largest(poly):
    if poly is None or getattr(poly, "is_empty", True):
        return None
    if poly.geom_type == "MultiPolygon":
        return max(poly.geoms, key=lambda g: g.area)
    return poly if poly.geom_type == "Polygon" else None


def facets_to_report_input(sam_roof, address: str,
                           aerial_image_path: Optional[str] = None) -> dict:
    """SamRoof -> the pipeline's report_input dict (same shape process_address builds).

    Coordinates are taken straight from the SAM polygons, so if
    ``segment_roof_sam`` was given the NAIP tile's transform, areas/lengths are
    metric; in pixel space they're pixel units (fine for a visual demo).
    """
    facets: List[dict] = []
    facet_polys = []
    for f in getattr(sam_roof, "facets", []):
        poly = _largest(getattr(f, "polygon", None))
        if poly is None or poly.is_empty:
            continue
        facet_polys.append(poly)
        facets.append({
            "facet_id": f.facet_id,
            "polygon_xy": [list(pt) for pt in poly.exterior.coords],
            "plan_area_m2": float(poly.area),
            "surface_area_m2": None,      # needs pitch (LiDAR) -> report uses plan area
            "slope_deg": None,
            "pitch_string": None,         # renders as "unspecified"
            "aspect_bin": None,
            "is_flat": False,
            "needs_review": bool(getattr(f, "needs_review", False)),
        })

    # Edges: build the typed edge graph geometrically from the outline + facet
    # partition (perimeter -> eaves; internal seams -> ridge/hip/valley by corner
    # convexity). Exact eave-vs-rake split + pitch arrive with the LiDAR path.
    outline = _largest(getattr(sam_roof, "outline", None))
    outline_xy = [list(pt) for pt in outline.exterior.coords] if outline is not None else []
    edges = edges_from_outline_and_facets(outline, facet_polys)

    return {
        "address": address,
        "report_id": _report_id(address),   # report_qc.metadata wants a stable id
        "facets": facets,
        "edges": edges,
        "outline_xy": outline_xy,      # zero-shot roof outline -> bold boundary in the diagram
        "aerial_image_path": aerial_image_path,
    }


def sam_roof_to_pdf(sam_roof, address: str, out_pdf: str,
                    aerial_image_path: Optional[str] = None) -> str:
    """SamRoof -> a 6-page roof_report.pdf. Returns the output path."""
    report_input = facets_to_report_input(sam_roof, address, aerial_image_path)
    generate_report(report_input, out_pdf)
    return out_pdf
