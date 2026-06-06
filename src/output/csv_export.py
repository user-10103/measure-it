"""CSV exporters for facets and edges."""

import csv
import logging
from typing import List

from src.output.units import m2_to_sqft, m_to_ft

logger = logging.getLogger(__name__)


def export_facets_csv(facets: List[dict], out_path: str) -> str:
    """Write one row per facet (id, area sqft, pitch, aspect, flat)."""
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["facet_id", "plan_area_sqft", "pitch_string", "aspect_bin", "is_flat"])
        for f in facets:
            w.writerow([f.get("facet_id"), round(m2_to_sqft(f.get("plan_area_m2", 0.0)), 1),
                        f.get("pitch_string"), f.get("aspect_bin"), bool(f.get("is_flat"))])
    logger.info("Wrote facets CSV: %s (%d rows)", out_path, len(facets))
    return out_path


def export_edges_csv(edges: List[dict], out_path: str) -> str:
    """Write one row per edge (type, length ft)."""
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edge_type", "length_ft"])
        for e in edges:
            w.writerow([e.get("edge_type"), round(m_to_ft(e.get("length_m", 0.0)), 2)])
    logger.info("Wrote edges CSV: %s (%d rows)", out_path, len(edges))
    return out_path
