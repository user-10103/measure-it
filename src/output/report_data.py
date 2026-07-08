"""
Build the aggregated Roof Report model from a neutral (metric) input dict.

Input contract (all metric units):
  report_input = {
    "address": str, "building_id": str,
    "aerial_image_path": Optional[str], "north_angle_deg": float,
    "facets": [{"facet_id", "polygon_xy" [[x,y]..], "plan_area_m2", "slope_deg",
                "pitch_string", "aspect_bin", "is_flat"}],
    "edges":  [{"edge_type", "length_m", "geometry_xy" [[x0,y0],[x1,y1]]}],
  }
Output: ReportModel with imperial totals matching the Roofr report pages.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from src.output.units import m2_to_sqft, m_to_ft, m_to_ftin, sqft_to_squares
from src.roofs.categories import REPORT_EDGE_TYPES

logger = logging.getLogger(__name__)

WASTE_PCTS = [0, 10, 12, 15, 17, 20, 22]


@dataclass
class ReportModel:
    address: str
    building_id: str
    total_area_sqft: float
    pitched_area_sqft: float
    flat_area_sqft: float
    num_facets: int
    predominant_pitch: str
    predominant_pitch_area_sqft: float
    unspecified_pitch_area_sqft: float
    num_needs_review: int
    edge_totals_ft: Dict[str, float]            # edge_type -> total feet
    edge_totals_ftin: Dict[str, str]            # edge_type -> "Xft Yin"
    pitch_breakdown: Dict[str, Dict[str, float]]  # pitch -> {area_sqft, squares}
    waste_table: List[Dict[str, float]]         # [{pct, area_sqft, squares}]
    facet_rows: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "building_id": self.building_id,
            "total_area_sqft": round(self.total_area_sqft, 1),
            "pitched_area_sqft": round(self.pitched_area_sqft, 1),
            "flat_area_sqft": round(self.flat_area_sqft, 1),
            "num_facets": self.num_facets,
            "predominant_pitch": self.predominant_pitch,
            "predominant_pitch_area_sqft": round(self.predominant_pitch_area_sqft, 1),
            "unspecified_pitch_area_sqft": round(self.unspecified_pitch_area_sqft, 1),
            "num_needs_review": self.num_needs_review,
            "edge_totals_ftin": self.edge_totals_ftin,
            "pitch_breakdown": self.pitch_breakdown,
            "waste_table": self.waste_table,
        }


def build_report_model(report_input: dict) -> ReportModel:
    facets = report_input.get("facets", [])
    edges = report_input.get("edges", [])

    # facet areas -> imperial + pitched/flat split + per-pitch aggregation
    total = pitched = flat = unspecified = 0.0
    pitch_area: Dict[str, float] = defaultdict(float)
    facet_rows = []
    for f in facets:
        sqft = m2_to_sqft(f.get("surface_area_m2") or f.get("plan_area_m2", 0.0))
        total += sqft
        is_flat = bool(f.get("is_flat"))
        pitch = f.get("pitch_string") or "unspecified"
        if is_flat:
            flat += sqft
        else:
            pitched += sqft
        if pitch == "unspecified":
            unspecified += sqft
        pitch_area[pitch] += sqft
        facet_rows.append({"facet_id": f.get("facet_id"), "area_sqft": round(sqft, 1),
                           "pitch_string": pitch, "aspect_bin": f.get("aspect_bin"),
                           "slope_deg": f.get("slope_deg"), "is_flat": is_flat,
                           "needs_review": bool(f.get("needs_review")),
                           "pitch_source": f.get("pitch_source")})

    num_needs_review = sum(1 for f in facets if f.get("needs_review"))

    predominant_pitch, predominant_area = "0:12", 0.0
    if pitch_area:
        predominant_pitch = max(pitch_area, key=pitch_area.get)
        predominant_area = pitch_area[predominant_pitch]

    # edge lengths by type (feet)
    edge_ft: Dict[str, float] = {t: 0.0 for t in REPORT_EDGE_TYPES}
    for e in edges:
        t = e.get("edge_type", "unspecified")
        edge_ft[t] = edge_ft.get(t, 0.0) + m_to_ft(e.get("length_m", 0.0))
    edge_ftin = {t: m_to_ftin(ft / m_to_ft(1.0)) for t, ft in edge_ft.items()}

    pitch_breakdown = {p: {"area_sqft": round(a, 1), "squares": sqft_to_squares(a)}
                       for p, a in sorted(pitch_area.items())}

    waste = [{"pct": p, "area_sqft": round(total * (1 + p / 100.0), 1),
              "squares": sqft_to_squares(total * (1 + p / 100.0))} for p in WASTE_PCTS]

    model = ReportModel(
        address=report_input.get("address", ""),
        building_id=report_input.get("building_id", ""),
        total_area_sqft=total, pitched_area_sqft=pitched, flat_area_sqft=flat,
        num_facets=len(facets), predominant_pitch=predominant_pitch,
        predominant_pitch_area_sqft=predominant_area,
        unspecified_pitch_area_sqft=unspecified, num_needs_review=num_needs_review,
        edge_totals_ft=edge_ft, edge_totals_ftin=edge_ftin,
        pitch_breakdown=pitch_breakdown, waste_table=waste, facet_rows=facet_rows,
    )
    logger.info("Report model: %.0f sqft, %d facets, predominant %s",
                total, len(facets), predominant_pitch)
    return model
