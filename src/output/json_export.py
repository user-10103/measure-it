"""JSON export of the roof report (machine-readable companion to the PDF)."""

import json
import logging
from typing import Optional

from src.output.report_data import ReportModel, build_report_model

logger = logging.getLogger(__name__)


def export_report_json(report_input: dict, out_path: str,
                       model: Optional[ReportModel] = None) -> str:
    """Write the report input + aggregated model to JSON. Returns the path."""
    if model is None:
        model = build_report_model(report_input)
    payload = {
        "address": report_input.get("address", ""),
        "building_id": report_input.get("building_id", ""),
        "summary": model.to_dict(),
        "facets": model.facet_rows,
        "edges": report_input.get("edges", []),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote report JSON: %s", out_path)
    return out_path
