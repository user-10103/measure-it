"""One call: user input (address / coords / maps link) -> the 6-page roof report.

This is the deployment code path — the SAME one the Colab demo converged on,
folded into the repo so AWS, Colab, and tests all run identical logic:

    resolve_location -> building footprint -> NAIP tile (S3, requester-pays)
    -> LOOSE roof chip (~18 m buffer; tight crops starve the facet model)
    -> segment_roof_sam (zero-shot outline + fine-tuned facets,
       score_thr=0.15, whole-roof-mask drop)
    -> outline fallback (facet union) so eaves always exist
    -> sam_roof_to_pdf -> the 6-page report (Roofr-style: cover / diagram /
       lengths / areas / pitch / summary)

The GPU predictors and the chip fetcher are injected, so everything below the
model is unit-testable without a GPU or network.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from src.output.sam_report import facets_to_report_input
from src.output.pdf_report import generate_report
from src.roofs.sam_segment import segment_roof_sam
from src.utils.resolve_location import resolve_location

logger = logging.getLogger(__name__)

# Colab-validated defaults (2026-07-04)
CHIP_BUFFER_M = 18.0     # loose crop — the facet model needs context
FOOTPRINT_BUFFER_M = 60  # building-selection search radius
SCORE_THR = 0.15         # in-training checkpoint scores low; NMS dedupes


def fetch_chip(lat: float, lon: float, state: str, out_dir: Path,
               chip_buffer_m: float = CHIP_BUFFER_M):
    """NAIP roof chip for a location.

    Returns (chip HxWx3 uint8, Affine, png_path, anchor_mask) — the anchor is
    the TARGET building footprint rasterized to chip pixels, so segmentation
    can pick the right roof when neighbors are visible in the loose crop.
    """
    import geopandas as gpd
    import rasterio
    from PIL import Image
    from rasterio import features as rio_features
    from rasterio.mask import mask as rio_mask

    from src.ingestion.naip import get_naip_for_location
    from src.roofs.select_candidates import select_building

    sel = select_building(lat, lon, buffer_meters=FOOTPRINT_BUFFER_M)
    footprint = gpd.GeoDataFrame(geometry=[sel["selected"].geometry],
                                 crs="EPSG:4326")
    naip_tif, _, _ = get_naip_for_location(lat, lon, state.lower(), footprint,
                                           output_dir=out_dir)
    if not naip_tif:
        raise RuntimeError(f"No NAIP imagery available near ({lat}, {lon})")
    with rasterio.open(naip_tif) as src:
        chip_crs = str(src.crs)
        fp_geom = footprint.to_crs(src.crs).geometry.iloc[0]
        geom = fp_geom.buffer(chip_buffer_m)
        arr, transform = rio_mask(src, [geom.__geo_interface__], crop=True,
                                  filled=True)
    chip = np.transpose(arr[:3], (1, 2, 0)).astype("uint8")
    anchor = rio_features.rasterize(
        [(fp_geom, 1)], out_shape=chip.shape[:2], transform=transform,
        dtype="uint8").astype(bool)
    png = out_dir / "chip.png"
    Image.fromarray(chip).save(png)
    meta = {"crs": chip_crs, "footprint_wgs84": footprint.geometry.iloc[0]}
    return chip, transform, str(png), anchor, meta


def _fallback_outline(roof):
    """Outline from the facet union when the zero-shot prompt missed — the
    report then still gets a perimeter (eaves) and the facets tile something."""
    from shapely.ops import unary_union
    polys = [f.polygon for f in roof.facets
             if f.polygon is not None and not f.polygon.is_empty]
    if not polys:
        return None
    u = unary_union(polys)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    return u if u.geom_type == "Polygon" and not u.is_empty else None


@dataclass
class ReportResult:
    pdf_path: Optional[str]      # None for scale-less image uploads (no honest sqft)
    chip_path: str
    lat: float
    lon: float
    location_source: str
    num_facets: int
    outline_found: bool          # zero-shot outline (False = facet-union fallback)
    plan_area_m2: float
    edge_totals_m: Dict[str, float] = field(default_factory=dict)
    num_pitched: int = 0         # facets with LiDAR pitch (0 = imagery-only report)

    def to_dict(self) -> dict:
        return {**self.__dict__}


def generate_roof_report(
    location: Union[str, Tuple[float, float]],
    state: str,
    predict_facets,
    predict_outline=None,
    out_dir: Union[str, Path] = "output/report",
    label: Optional[str] = None,
    chip_fetcher: Callable = fetch_chip,
    score_thr: float = SCORE_THR,
    chip_buffer_m: float = CHIP_BUFFER_M,
    lidar_points=None,
    use_lidar: Optional[bool] = None,
) -> ReportResult:
    """User input -> roof_report.pdf. The whole pipeline, one entry point.

    Args:
        location: anything a user types (address / "lat, lon" / maps link),
            or an explicit (lat, lon) tuple.
        state: 2-letter state for the NAIP archive (e.g. "FL").
        predict_facets / predict_outline: from
            ``sam3_predictors.load_sam3_predictors`` (or fakes in tests).
        chip_fetcher: injected imagery step (network-free in tests).
        lidar_points: optional roof point cloud (x,y,z struct or (N,3)) in the
            chip's world CRS — enables per-facet pitch + true sloped area via
            read-only fusion (facet shapes are never modified). Without it the
            report shows pitch "unspecified" (plan areas).
        use_lidar: fetch the points automatically via the pure-python EPT
            reader (USGS usgs-lidar-public, no credentials). Default: the
            MEASURE_IT_LIDAR env var ("1" = on). Coverage miss or fetch error
            degrades gracefully to the imagery-only report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(location, str):
        loc = resolve_location(location)
        lat, lon, source = loc["lat"], loc["lon"], loc["source"]
        label = label or location
    else:
        lat, lon = float(location[0]), float(location[1])
        source, label = "coordinates", label or f"{lat:.5f}, {lon:.5f}"

    fetched = chip_fetcher(lat, lon, state, out_dir,
                           chip_buffer_m=chip_buffer_m)
    chip, transform, chip_png = fetched[0], fetched[1], fetched[2]
    anchor = fetched[3] if len(fetched) > 3 else None   # footprint mask
    meta = fetched[4] if len(fetched) > 4 else {}       # crs + footprint_wgs84

    roof = segment_roof_sam(
        predict_facets, chip, transform=transform,
        predict_outline=predict_outline, anchor_mask=anchor,
        roof_concept="roof", facet_concept="roof facet",
        score_thr=score_thr, iou_thr=0.5,
    )
    outline_found = roof.outline is not None
    if not outline_found:
        roof.outline = _fallback_outline(roof)
        if roof.outline is not None:
            logger.warning("zero-shot outline missed — using facet-union fallback")

    report_input = facets_to_report_input(roof, label, aerial_image_path=chip_png)

    # auto-fetch LiDAR when enabled (env MEASURE_IT_LIDAR=1 or use_lidar=True)
    import os as _os
    if use_lidar is None:
        use_lidar = _os.getenv("MEASURE_IT_LIDAR", "0") == "1"
    ground_z = None
    if (lidar_points is None and use_lidar and roof.facets
            and meta.get("crs") and meta.get("footprint_wgs84") is not None):
        try:
            from src.lidar.ept_fetch import fetch_roof_points
            lidar_points, ground_z = fetch_roof_points(
                lat, lon, meta["footprint_wgs84"], meta["crs"],
                with_ground=True)
        except Exception as e:  # noqa: BLE001 — LiDAR is additive, never fatal
            logger.warning("LiDAR fetch failed (%s) — imagery-only report", e)
        if lidar_points is not None and ground_z is None:
            # no class-2 points in the tiles -> OpenTopography DTM fallback
            try:
                from src.lidar.opentopo import get_ground_elevation
                ground_z = get_ground_elevation(lat, lon)
            except Exception as e:  # noqa: BLE001
                logger.warning("OT ground fallback failed (%s)", e)

    # LiDAR fusion: annotate -> evidence-based coplanar merge -> pitch fields
    # -> edge refinements (rakes, flashing, true 3D lengths)
    if lidar_points is not None and roof.facets:
        from src.roofs.fuse_sam_lidar import (
            annotate_facets_with_lidar, fuse_into_report_input,
            merge_coplanar_facets)
        from src.roofs.geom_edges import (
            apply_3d_edge_lengths, relabel_flat_junction_flashing,
            relabel_rakes)
        annotations = annotate_facets_with_lidar(roof.facets, lidar_points,
                                          ground_z=ground_z)
        # merge facets LiDAR proves are one plane (fixes model over-
        # segmentation; area-conserving; off via MEASURE_IT_PLANE_MERGE=0)
        if _os.getenv("MEASURE_IT_PLANE_MERGE", "1") == "1":
            merged, changed = merge_coplanar_facets(
                roof.facets, lidar_points, annotations)
            if changed:
                roof.facets = merged
                annotations = annotate_facets_with_lidar(merged, lidar_points,
                                              ground_z=ground_z)
                report_input = facets_to_report_input(
                    roof, label, aerial_image_path=chip_png)
        fuse_into_report_input(report_input, annotations)
        # eave vs rake: pure relabel from each facet's downslope direction
        aspects = {fid: a["aspect_deg"] for fid, a in annotations.items()
                   if not a.get("is_flat")}
        relabel_rakes(report_input["edges"], roof.facets, aspects)
        # pitched<->flat junctions are flashing, not valleys (Holland criterion)
        relabel_flat_junction_flashing(report_input["edges"], roof.facets,
                                       annotations)
        # plan -> true sloped lengths (hips/valleys/rakes lengthen; level
        # eaves/ridges stay put)
        grads = {fid: a["grad"] for fid, a in annotations.items()}
        apply_3d_edge_lengths(report_input["edges"], roof.facets, grads)

    pdf_path = out_dir / "roof_report.pdf"
    generate_report(report_input, str(pdf_path))

    edge_totals: Dict[str, float] = {}
    for e in report_input["edges"]:
        edge_totals[e["edge_type"]] = (
            edge_totals.get(e["edge_type"], 0.0) + e["length_m"])
    plan_area = sum(f["plan_area_m2"] for f in report_input["facets"])

    logger.info("report: %s -> %d facet(s), outline=%s, %.1f m2 -> %s",
                label, len(report_input["facets"]), outline_found,
                plan_area, pdf_path)
    return ReportResult(
        pdf_path=str(pdf_path), chip_path=chip_png, lat=lat, lon=lon,
        location_source=source, num_facets=len(report_input["facets"]),
        outline_found=outline_found, plan_area_m2=plan_area,
        edge_totals_m=edge_totals,
        num_pitched=sum(1 for f in report_input["facets"]
                        if f.get("pitch_string")),
    )


def generate_report_from_image(
    chip: np.ndarray,
    predict_facets,
    predict_outline=None,
    out_dir: Union[str, Path] = "output/report",
    label: str = "uploaded roof",
    scale_m_per_px: Optional[float] = None,
    score_thr: float = SCORE_THR,
) -> ReportResult:
    """User-uploaded aerial image -> report. Honest-units rule:

    * ``scale_m_per_px`` given (e.g. a GeoTIFF's resolution) -> real metric
      areas -> the full 6-page PDF.
    * no scale -> a plain JPG has no ground truth for square feet, so we return
      the facet overlay + counts (pdf_path=None) instead of fabricating units.
      The address path is the measured product.
    """
    from affine import Affine
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chip_png = out_dir / "chip.png"
    Image.fromarray(chip).save(chip_png)

    transform = (Affine(scale_m_per_px, 0, 0, 0, -scale_m_per_px, 0)
                 if scale_m_per_px else None)
    roof = segment_roof_sam(
        predict_facets, chip, transform=transform,
        predict_outline=predict_outline,
        roof_concept="roof", facet_concept="roof facet",
        score_thr=score_thr, iou_thr=0.5,
    )
    outline_found = roof.outline is not None
    if not outline_found:
        roof.outline = _fallback_outline(roof)

    # facet overlay preview (pixel space) for the result card
    overlay_png = str(out_dir / "facets.png")
    if roof.label_map is not None:
        from src.roofs.mask_facets import render_facets
        px_facets = roof.facets if transform is None else []
        render_facets(chip, px_facets, roof.label_map, out_path=overlay_png,
                      title=label)
    else:
        overlay_png = str(chip_png)

    report_input = facets_to_report_input(roof, label,
                                          aerial_image_path=str(chip_png))
    pdf_path: Optional[str] = None
    if scale_m_per_px:                      # metric -> honest sqft -> full PDF
        pdf_path = str(out_dir / "roof_report.pdf")
        generate_report(report_input, pdf_path)

    edge_totals: Dict[str, float] = {}
    for e in report_input["edges"]:
        edge_totals[e["edge_type"]] = (
            edge_totals.get(e["edge_type"], 0.0) + e["length_m"])
    return ReportResult(
        pdf_path=pdf_path, chip_path=overlay_png, lat=0.0, lon=0.0,
        location_source="image", num_facets=len(report_input["facets"]),
        outline_found=outline_found,
        plan_area_m2=sum(f["plan_area_m2"] for f in report_input["facets"]),
        edge_totals_m=edge_totals,
    )
