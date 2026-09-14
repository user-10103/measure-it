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


# plain-English wording for a client-facing cover; the check ids stay in the log
_WHY = {"edges_typed": "roof edge structure not resolved",
        "pitch_resolved": "roof pitch could not be measured",
        "slope_applied": "sloped area not verified",
        "facets_partition": "overlapping roof faces",
        "facets_coverage": "roof faces do not cover the roof",
        "area_sane": "implausible roof area",
        "area_positive": "no measurable roof area",
        "facets_present": "no roof faces detected",
        "facet_table": "per-face detail missing",
        "facets_vs_lidar_planes": "roof appears under-segmented — elevation data "
                                  "shows more roof faces than were detected",
        "facet_plane_fit": "roof surface is not a single plane — pitch and area "
                           "would be averages across sections that disagree"}


def incomplete_reason(qc: dict) -> Optional[str]:
    """Why a report must be stamped INCOMPLETE, in words a client understands.
    None when the world-class gate passed."""
    if qc.get("passed"):
        return None
    bad = [c["id"] for c in qc.get("checks", [])
           if c["severity"] == "FAIL" and not c["ok"]]
    return "; ".join(_WHY.get(b, b) for b in bad) or "failed quality gate"


def _lidar_clip_geometry(meta: dict, roof):
    """Geometry to clip the LiDAR fetch to: the building footprint UNIONED with the
    roof outline we actually segmented.

    The points were clipped to the MS Buildings footprint while the FACETS come
    from the SAM outline, and nothing reconciled the two. Wherever they diverge —
    a geocode pin tens of metres off the building, an anchor override picking a
    different mask, a footprint-healing expansion — facets land in a region no
    point was ever fetched for, and every one comes back pitch-less. That is
    1600 Sarno Rd: 550 points fetched, 0 of 6 facets annotated, and with no pitch
    the edges cannot be classified either (0 ridges/hips, 188 ft "unspecified").

    Clipping to the union guarantees every facet we are about to ask about is
    covered. Falls back to the footprint when the outline is unusable.
    """
    fp = meta["footprint_wgs84"]
    outline = getattr(roof, "outline", None)
    if outline is None or outline.is_empty or not getattr(roof, "georeferenced", False):
        return fp                      # pixel-space outline can't be reprojected
    try:
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform, unary_union
        to_wgs = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True).transform
        merged = unary_union([fp, shp_transform(to_wgs, outline)])
        if merged.is_empty:
            return fp
        grew = merged.area > fp.area * 1.01
        logger.info("LiDAR clip = footprint %s detected outline",
                    "UNION" if grew else "(outline adds nothing beyond)")
        return merged
    except Exception as e:  # noqa: BLE001 — clip widening is additive, never fatal
        logger.warning("could not union the outline into the LiDAR clip (%s) — "
                       "using footprint alone", e)
        return fp


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
    qc: Dict = field(default_factory=dict)   # world-class gate result (report_qc.score_report)
    # Which imagery this roof was measured from, and what was tried first. The
    # resolver recorded both on `meta` and NOTHING read them: not the return
    # value, not the PDF, not a visible log line -- so a report measured off a
    # 30 cm NAIP chip was indistinguishable from one measured off a 15 cm
    # county ortho, and the reason the better source was declined was
    # unrecoverable. Area and pitch accuracy depend materially on which it was.
    imagery_source: Optional[str] = None      # "county-3in" / "fl-statewide" / "naip"
    imagery_gsd_m: Optional[float] = None
    imagery_year: Optional[int] = None
    imagery_attempts: list = field(default_factory=list)  # sources declined, with why

    def to_dict(self) -> dict:
        return {**self.__dict__}


def generate_roof_report(
    location: Union[str, Tuple[float, float]],
    state: Optional[str],
    predict_facets,
    predict_outline=None,
    out_dir: Union[str, Path] = "output/report",
    label: Optional[str] = None,
    chip_fetcher: Optional[Callable] = None,
    score_thr: float = SCORE_THR,
    chip_buffer_m: float = CHIP_BUFFER_M,
    lidar_points=None,
    use_lidar: Optional[bool] = None,
) -> ReportResult:
    """User input -> roof_report.pdf. The whole pipeline, one entry point.

    Args:
        location: anything a user types (address / "lat, lon" / maps link),
            or an explicit (lat, lon) tuple.
        state: 2-letter state for the NAIP archive (e.g. "FL"). Optional —
            derived from the coordinates when omitted. Every caller used to
            default this to "FL", so the serving path had never been asked for
            imagery outside one state; a national run needs it from the
            geocode, not from a literal.
        predict_facets / predict_outline: from
            ``sam3_predictors.load_sam3_predictors`` (or fakes in tests).
        chip_fetcher: injected imagery step (network-free in tests). Defaults
            to imagery_select.fetch_chip_best — county GIS orthophoto where one
            is registered for the location, NAIP everywhere else. The facet
            model is fine-tuned on GIS imagery, so serving NAIP unconditionally
            (the previous default) meant inferring at 4x the training GSD.
            Pass fetch_chip explicitly to force the NAIP path.
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

    if state is None:
        from src.ingestion.imagery_select import state_county_for
        state, _county = state_county_for(lat, lon)
        if state is None:
            raise ValueError(
                f"Could not determine the state for ({lat}, {lon}); pass "
                "state= explicitly. NAIP is archived per state, so guessing "
                "one would fetch imagery for the wrong part of the country.")
        logger.info("state resolved from coordinates: %s", state)
    if chip_fetcher is None:
        from src.ingestion.imagery_select import fetch_chip_best
        chip_fetcher = fetch_chip_best
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
                lat, lon, _lidar_clip_geometry(meta, roof), meta["crs"],
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
            apply_3d_edge_lengths, classify_internal_edges, relabel_rakes)
        # Why each unannotated facet is unannotated. Absent from `annotations`
        # covers five different causes and the report could not tell them apart,
        # so "unspecified" had to be diagnosed by re-running the address. Cleared
        # before each re-annotation so it describes the FINAL facet set.
        lidar_declines: dict = {}
        annotations = annotate_facets_with_lidar(roof.facets, lidar_points,
                                          ground_z=ground_z,
                                          declines=lidar_declines)
        # split facets that LiDAR proves span TWO planes (fixes model UNDER-
        # segmentation — a hip wing returned as one blob; area-conserving; off
        # via MEASURE_IT_PLANE_SPLIT=0). Runs before the merge so a freshly split
        # facet can still be re-merged if it turns out coplanar with a neighbor.
        if _os.getenv("MEASURE_IT_PLANE_SPLIT", "1") == "1":
            from src.roofs.fuse_sam_lidar import (split_level_facets,
                                                  split_multiplane_facets)
            split, did_split = split_multiplane_facets(roof.facets, lidar_points)
            # ...then the FLAT case: sections at different heights, which the
            # angle split cannot see because parallel planes never intersect.
            split, did_level = split_level_facets(split, lidar_points)
            did_split = did_split or did_level
            if did_split:
                roof.facets = split
                lidar_declines.clear()
                annotations = annotate_facets_with_lidar(roof.facets, lidar_points,
                                                  ground_z=ground_z,
                                                  declines=lidar_declines)
                report_input = facets_to_report_input(
                    roof, label, aerial_image_path=chip_png)
        # merge facets LiDAR proves are one plane (fixes model over-
        # segmentation; area-conserving; off via MEASURE_IT_PLANE_MERGE=0)
        if _os.getenv("MEASURE_IT_PLANE_MERGE", "1") == "1":
            from src.roofs.fuse_sam_lidar import absorb_unannotated_orphans
            merged, changed = merge_coplanar_facets(
                roof.facets, lidar_points, annotations)
            # small facets with no LiDAR evidence join their measured neighbor
            merged, absorbed = absorb_unannotated_orphans(merged, annotations)
            if changed or absorbed:
                roof.facets = merged
                lidar_declines.clear()
                annotations = annotate_facets_with_lidar(merged, lidar_points,
                                              ground_z=ground_z,
                                              declines=lidar_declines)
                report_input = facets_to_report_input(
                    roof, label, aerial_image_path=chip_png)
        # Independent check on FACET COUNT, which the gate cannot see for itself:
        # a roof returned as too few planes passes partition/coverage/edges_typed
        # trivially while under-reporting surface area. Runs on the FINAL facets.
        from src.roofs.fuse_sam_lidar import detect_multiplane_facets
        report_input["multiplane_facets"] = detect_multiplane_facets(
            roof.facets, lidar_points)
        if lidar_declines:
            report_input["lidar_declines"] = {str(k): v
                                              for k, v in lidar_declines.items()}
        fuse_into_report_input(report_input, annotations)
        # eave vs rake: pure relabel from each facet's downslope direction
        aspects = {fid: a["aspect_deg"] for fid, a in annotations.items()
                   if not a.get("is_flat")}
        relabel_rakes(report_input["edges"], roof.facets, aspects)
        # internal seams: physics-based classification from the plane aspects
        # (ridge/hip/valley/flashing/transition — the Holland criterion)
        report_input["edges"] = classify_internal_edges(
            report_input["edges"], roof.facets, annotations)
        # plan -> true sloped lengths (hips/valleys/rakes lengthen; level
        # eaves/ridges stay put)
        grads = {fid: a["grad"] for fid, a in annotations.items()}
        apply_3d_edge_lengths(report_input["edges"], roof.facets, grads)

    # World-class gate runs BEFORE the PDF is written: a report that fails must be
    # STAMPED INCOMPLETE, never shipped looking finished. Scoring after
    # generate_report() meant a garbage roof (e.g. 11 facets with zero ridges or
    # hips - geometrically impossible) printed a polished, confident PDF and only
    # logged a warning nobody reads.
    from src.output.report_qc import score_report, format_report_qc
    # The footprint we SELECTED, so the gate can ask whether the roof we
    # measured is that building. Fetched for the LiDAR clip and then never
    # compared against the result: the Don CeSar (a large hotel) produced a
    # 1,232 sqft two-facet roof and PASSED, because every gate check is about
    # internal geometric self-consistency and none is about identity. A small,
    # plausible, self-consistent roof is exactly what passes.
    try:
        import geopandas as _gpd
        _fp = meta.get("footprint_wgs84")
        if _fp is not None and meta.get("crs"):
            _fpa = float(_gpd.GeoSeries([_fp], crs="EPSG:4326")
                         .to_crs(meta["crs"]).area.iloc[0])
            if _fpa > 0:
                report_input["footprint_plan_area_m2"] = _fpa
    except Exception as _fe:  # noqa: BLE001 - advisory
        logger.info("footprint area unavailable (%s)", _fe)

    # Which imagery this roof was actually measured from. The facet model is
    # fine-tuned on GIS orthophotos, so a NAIP-sourced report is an out-of-domain
    # inference and must say so rather than look identical to an in-domain one.
    # Survives the report_input rebuilds above by being set last.
    for _k in ("imagery_source", "imagery_gsd_m", "imagery_year",
               "imagery_county"):
        if _k in meta:
            report_input[_k] = meta[_k]

    qc = score_report(report_input)
    if not qc["passed"]:
        report_input["incomplete_reason"] = incomplete_reason(qc)
        logger.warning("report below world-class bar (score %.0f%%):\n%s",
                        qc["score"] * 100, format_report_qc(qc))

    pdf_path = out_dir / "roof_report.pdf"
    generate_report(report_input, str(pdf_path))

    edge_totals: Dict[str, float] = {}
    for e in report_input["edges"]:
        edge_totals[e["edge_type"]] = (
            edge_totals.get(e["edge_type"], 0.0) + e["length_m"])
    plan_area = sum(f["plan_area_m2"] for f in report_input["facets"])

    logger.info("report: %s -> %d facet(s), outline=%s, %.1f m2, gate=%s -> %s",
                label, len(report_input["facets"]), outline_found,
                plan_area, "PASS" if qc["passed"] else "FAIL", pdf_path)
    return ReportResult(
        imagery_source=meta.get("imagery_source"),
        imagery_gsd_m=meta.get("imagery_gsd_m"),
        imagery_year=meta.get("imagery_year"),
        imagery_attempts=list(meta.get("imagery_attempts") or []),
        pdf_path=str(pdf_path), chip_path=chip_png, lat=lat, lon=lon,
        location_source=source, num_facets=len(report_input["facets"]),
        outline_found=outline_found, plan_area_m2=plan_area,
        edge_totals_m=edge_totals,
        num_pitched=sum(1 for f in report_input["facets"]
                        if f.get("pitch_string")),
        qc=qc,
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
