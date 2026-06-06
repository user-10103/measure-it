"""
RGB-only roof measurement pipeline.

Runs the imagery-only path that the trained facet model enables, bypassing the
LiDAR + fusion stages of src/pipeline.py:

    chip + model prediction
      -> segment_facets(method="ml")            (polygon facets + synthesized planes)
      -> classify_edges_from_facets             (typed ridge/hip/valley/eave/rake)
      -> per-facet metrics (slope/pitch/aspect/area)
      -> Roofr-style Roof Report (PDF + JSON + CSV)

The model backend is pluggable (RoofModelBackend); use CocoStandinBackend to run
end-to-end on the cleaned v2_4 annotations before the trained weights exist.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.roofs.segment import segment_facets
from src.roofs.edges import classify_edges_from_facets
from src.roofs.tiling import tile_facets
from src.roofs.metrics import compute_aspect_bin, compute_aspect_deg
from src.roofs.plane_fit import plane_from_pitch_aspect
from src.roofs.geometric_aspect import facet_aspect_deg
from src.roofs.pitch_policy import classify_pitch, roof_pitch_prior
from src.output.report_data import build_report_model
from src.output.pdf_report import generate_report
from src.output.json_export import export_report_json
from src.output.csv_export import export_facets_csv, export_edges_csv

logger = logging.getLogger(__name__)


def _identity(x, y):
    return (x, y)


def build_report_input(
    address: str, building_id: str, facets: List, edges: List,
    aerial_image_path: Optional[str] = None,
) -> dict:
    """Assemble the neutral (metric) report-input dict from Facet + RoofEdge objects.

    Pitch is annotated under the settled policy (src/roofs/pitch_policy.py): flat
    vs pitched is the robust signal; pitched facets take the roof's common-pitch
    prior (per-facet slope magnitude is noise); steep outliers are flagged
    needs_review. Aspect stays geometric (from the plane / ridge direction).
    """
    roof_x12 = roof_pitch_prior([f.slope_deg for f in facets])
    facet_dicts = []
    for f in facets:
        slope = f.slope_deg if f.slope_deg is not None else 0.0
        ann = classify_pitch(slope_deg=f.slope_deg, roof_pitch_x12=roof_x12)
        aspect_bin = f.aspect_bin
        if aspect_bin is None and f.plane is not None and not ann.is_flat:
            aspect_bin = compute_aspect_bin(compute_aspect_deg(f.plane))
        facet_dicts.append({
            "facet_id": f.facet_id,
            "polygon_xy": [list(pt) for pt in f.polygon.exterior.coords],
            "plan_area_m2": float(f.polygon.area),
            "slope_deg": slope,
            "pitch_string": ann.pitch_string,
            "aspect_bin": aspect_bin if aspect_bin else "flat",
            "is_flat": ann.is_flat,
            "needs_review": ann.needs_review,
            "pitch_source": ann.source,
        })
    edge_dicts = [{
        "edge_type": e.edge_type.value,
        "length_m": float(e.length_m),
        "geometry_xy": [list(pt) for pt in e.geometry.coords][:2],
    } for e in edges]
    return {"address": address, "building_id": building_id,
            "aerial_image_path": aerial_image_path, "north_angle_deg": 0.0,
            "facets": facet_dicts, "edges": edge_dicts}


def process_chip_rgb(
    prediction: Dict,
    address: str = "",
    building_id: str = "",
    gsd_m_per_px: float = 0.3,
    pixel_to_world: Optional[Callable] = None,
    aerial_image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    write_outputs: bool = True,
) -> Dict:
    """Run the RGB-only pipeline on one chip's model prediction.

    Args:
        prediction: {"outline": [...]|None, "facets": [{polygon, slope_deg, aspect_bin}]}
            in PIXEL coords (e.g. CocoStandinBackend.predict_for(address_id)).
        gsd_m_per_px: ground sample distance; used to scale pixels -> metres when
            `pixel_to_world` is not supplied.
        pixel_to_world: optional explicit (x,y)->(X,Y) metric mapping (e.g. a NAIP
            affine transform). Overrides gsd scaling.
        output_dir: where to write report.pdf / .json / facets.csv / edges.csv.

    Returns:
        dict with report_input, the aggregated summary, and any output file paths.
    """
    if pixel_to_world is None:
        g = gsd_m_per_px
        pixel_to_world = lambda x, y: (x * g, y * g)  # noqa: E731

    facets = segment_facets(method="ml", prediction=prediction, pixel_to_world=pixel_to_world)
    logger.info("RGB pipeline: %d facets for %s", len(facets), address or building_id)

    outline = None
    if prediction.get("outline"):
        from shapely.geometry import Polygon
        pts = [tuple(pixel_to_world(x, y)) for x, y in prediction["outline"]]
        if len(pts) >= 3:
            outline = Polygon(pts)
            if not outline.is_valid:
                outline = outline.buffer(0)

    # Guard: outline detected but zero facets → occluded/unrecognised roof.
    # The model correctly abstained; flag it so callers can route to manual review
    # rather than delivering a blank report.
    occluded_roof = (outline is not None and len(facets) == 0)
    if occluded_roof:
        import logging as _log
        _log.getLogger(__name__).warning(
            "process_chip_rgb: outline present but 0 facets for %s — "
            "flagging occluded_roof=True (needs_review)",
            address or building_id,
        )

    # Snap facets into a shared-boundary partition: closes gaps, completes areas,
    # and lets the classifier read interior ridge/hip/valley edges (which need
    # adjacent facets to share boundaries). See src/roofs/tiling.py.
    snapped = tile_facets([(f.facet_id, f.polygon) for f in facets], outline)
    snapped_by_id = {fid: poly for fid, poly in snapped}
    for f in facets:
        sp = snapped_by_id.get(f.facet_id)
        if sp is not None:
            f.polygon = sp

    # Derive aspect from GEOMETRY (downslope = facet -> its eave on the outline)
    # rather than trusting the noisy LS aspect labels, and rebuild each facet's
    # plane from (slope, geometric aspect) so the edge classifier sees facets
    # that genuinely oppose across ridges. Falls back to the labelled aspect when
    # the eave can't be located. See src/roofs/geometric_aspect.py.
    for f in facets:
        asp_deg = facet_aspect_deg(f.polygon, outline)
        if asp_deg is not None:
            f.aspect_bin = compute_aspect_bin(asp_deg)
            slope = f.slope_deg if f.slope_deg is not None else 0.0
            f.plane = plane_from_pitch_aspect(slope, aspect=asp_deg)

    facet_polys = snapped
    planes = [f.plane for f in facets]
    edges = classify_edges_from_facets(facet_polys, planes, outline)

    report_input = build_report_input(address, building_id, facets, edges, aerial_image_path)
    model = build_report_model(report_input)

    output_files: Dict[str, str] = {}
    if write_outputs and output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        output_files["pdf"] = generate_report(report_input, str(out / "report.pdf"), model)
        output_files["json"] = export_report_json(report_input, str(out / "report.json"), model)
        output_files["facets_csv"] = export_facets_csv(report_input["facets"],
                                                       str(out / "facets.csv"))
        output_files["edges_csv"] = export_edges_csv(report_input["edges"],
                                                     str(out / "edges.csv"))

    return {"report_input": report_input, "summary": model.to_dict(),
            "output_files": output_files,
            "occluded_roof": occluded_roof}


def process_address_id(
    address_id: str, backend, address: str = "", **kwargs
) -> Dict:
    """Convenience: run the pipeline for one address_id via a COCO stand-in backend."""
    prediction = backend.predict_for(address_id)
    return process_chip_rgb(prediction, address=address or address_id,
                            building_id=address_id, **kwargs)
