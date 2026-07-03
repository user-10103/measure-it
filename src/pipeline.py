"""
Main roof measurement pipeline.
Orchestrates the complete workflow from address to roof metrics.
"""
import logging
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import geopandas as gpd

# Ensure PROJ data is found when running inside a conda env whose PROJ_LIB
# wasn't inherited from the shell (e.g. invoked with PYTHONNOUSERSITE=1).
if not os.environ.get("PROJ_LIB"):
    import pyproj
    _proj_dir = Path(pyproj.datadir.get_data_dir())
    if _proj_dir.exists():
        os.environ["PROJ_LIB"] = str(_proj_dir)
from src.config import OUTPUT_DIR, validate_config
from src.utils.geocode import get_coordinates
from src.roofs.select_candidates import select_building, export_candidates
from src.ingestion.naip import get_naip_for_location
from src.roofs.extract_polygon import get_roof_polygon
from src.roofs.measurements import get_all_measurements

# New modules for advanced roof metrics
from src.lidar.dsm import generate_height_models, get_ndsm_stats
from src.fusion.align import align_raster_to_target
from src.fusion.stack import stack_naip_with_ndsm
from src.fusion.clip import clip_to_footprint
from src.roofs.extract_points import raster_to_points, compute_vegetation_percentage
from src.roofs.segment import segment_facets
from src.roofs.naip_segment import naip_guided_segmentation
from src.roofs.plane_fit import fit_plane_ransac, extract_inlier_boundary, fit_plane_to_raw_points
from src.roofs.metrics import compute_all_facet_metrics, compute_building_metrics
from src.roofs.edges import (
    classify_edges_from_facets, merge_collinear_edges, compute_edge_lengths_by_type,
)
from src.qc.validators import validate_metrics
from src.output.json_export import export_report_json
from src.output.csv_export import export_facets_csv, export_edges_csv
from src.output.pdf_report import generate_report as export_pdf_plan_sheet

def check_pdf_available():
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False

def export_dxf(*a, **kw):
    raise NotImplementedError("DXF export not implemented")

def check_dxf_available():
    return False

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """
    Configure logging for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def process_address(
    address: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    state: Optional[str] = None,
    buffer_meters: float = 150,
    fetch_naip: bool = True,
    fetch_lidar: bool = True,
    naip_year: Optional[str] = None,
    lidar_datasets: Optional[list] = None,
    ept_url: Optional[str] = None,
    output_dir: Optional[Path] = None,
    segment_method: str = "kmeans",
    export_pdf: bool = True,
    export_dxf: bool = True,
    export_csv: bool = True,
    use_facet_reconstruct: bool = False,
    model_checkpoint: Optional[str] = None,
    model_threshold: float = 0.40
) -> Dict:
    """
    Process a single address/location through the complete pipeline.

    Workflow:
    1. Geocode address to coordinates
    2. Select building footprint from MS Buildings
    3. Fetch NAIP imagery (optional)
    4. Extract/refine roof polygon with LiDAR (optional)
    5. Generate height models (DSM/DTM/nDSM)
    6. Fuse NAIP + nDSM into 5-band raster
    7. Extract points and segment facets
    8. Fit planes and compute metrics
    9. Run QC and export outputs

    Args:
        address: Street address (alternative to lat/lon)
        lat: Explicit latitude
        lon: Explicit longitude
        state: State abbreviation (e.g., "fl") - required for NAIP/LiDAR
        buffer_meters: Search radius for buildings
        fetch_naip: Whether to fetch NAIP imagery
        fetch_lidar: Whether to fetch LiDAR data
        naip_year: Preferred NAIP year
        lidar_datasets: List of LiDAR datasets to try
        output_dir: Output directory (defaults to config OUTPUT_DIR)
        segment_method: Facet segmentation method ("single", "kmeans", "regiongrow")
        export_pdf: Whether to generate PDF roof plan sheet
        export_dxf: Whether to generate DXF CAD file
        export_csv: Whether to generate CSV files

    Returns:
        Dict with complete results including:
        - input: Original input parameters
        - building: Building selection info
        - polygon: Roof polygon info
        - naip: NAIP imagery info
        - lidar: LiDAR data info
        - measurements: Basic area/perimeter measurements
        - roof_metrics: Advanced facet/edge metrics (if LiDAR available)
        - qc: Quality control flags and confidence
        - output_files: Paths to generated files

    Example:
        >>> result = process_address(
        ...     address="16347 Heathrow Dr, Tampa, FL 33647",
        ...     state="fl"
        ... )
        >>> print(result["measurements"]["plan_area_m2"])
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ROOF MEASUREMENT PIPELINE")
    logger.info("=" * 80)

    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Step 1: Geocode
    logger.info("\n[1/9] GEOCODING")
    logger.info("-" * 80)

    try:
        coords = get_coordinates(address=address, lat=lat, lon=lon)
        lat = coords["lat"]
        lon = coords["lon"]
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        raise

    # Step 2: Select building
    logger.info("\n[2/9] BUILDING SELECTION")
    logger.info("-" * 80)

    try:
        selection = select_building(lat, lon, buffer_meters=buffer_meters)
        building_polygon = gpd.GeoDataFrame(
            geometry=[selection["selected"].geometry],
            crs="EPSG:4326"
        )

        # Export candidates
        candidates_path = output_dir / "candidates.geojson"
        export_candidates(selection["candidates"], str(candidates_path))

    except Exception as e:
        logger.error(f"Building selection failed: {e}")
        raise

    # Step 3: Fetch NAIP imagery (optional)
    logger.info("\n[3/9] NAIP IMAGERY ACQUISITION")
    logger.info("-" * 80)

    naip_data = {}
    if fetch_naip and state:
        try:
            naip_tif, naip_png, naip_meta = get_naip_for_location(
                lat, lon, state.lower(), building_polygon, naip_year,
                output_dir=output_dir,
            )
            if naip_tif:
                naip_data = {
                    "tif_path": str(naip_tif),
                    "png_path": str(naip_png),
                    "metadata": naip_meta
                }
                logger.info("NAIP imagery acquired successfully")
            else:
                logger.warning("No NAIP imagery available")
        except Exception as e:
            logger.warning(f"NAIP acquisition failed: {e}")
    else:
        logger.info("Skipping NAIP acquisition (disabled or missing state)")

    # Step 4: Extract/refine roof polygon (EPT LiDAR integrated)
    logger.info("\n[4/9] ROOF POLYGON EXTRACTION & LIDAR REFINEMENT")
    logger.info("-" * 80)

    try:
        if fetch_lidar:
            if ept_url:
                logger.info(f"EPT LiDAR refinement enabled - using hardcoded URL: {ept_url}")
            else:
                logger.info("EPT LiDAR refinement enabled - discovering dataset...")
            polygon_result = get_roof_polygon(building_polygon, lat, lon, ept_url=ept_url)
        else:
            logger.info("EPT LiDAR refinement disabled - using MS Buildings footprint")
            # Directly use MS Buildings footprint without LiDAR
            building_geom = building_polygon.union_all() if hasattr(building_polygon, 'union_all') else building_polygon.unary_union
            polygon_result = {
                "polygon_wgs84": building_geom,
                "source": "ms_buildings",
                "metadata": {}
            }

        final_polygon = polygon_result["polygon_wgs84"]
        polygon_source = polygon_result["source"]

        # Extract LiDAR data for advanced metrics (if available)
        lidar_points = polygon_result.get("metadata", {}).get("points")
        lidar_grid_x = polygon_result.get("metadata", {}).get("grid_x")
        lidar_grid_y = polygon_result.get("metadata", {}).get("grid_y")
        lidar_grid_z = polygon_result.get("metadata", {}).get("grid_z")

        # Export polygon
        polygon_gdf = gpd.GeoDataFrame(
            {"source": [polygon_source]},
            geometry=[final_polygon],
            crs="EPSG:4326"
        )
        polygon_path = output_dir / "roof_polygon.geojson"
        polygon_gdf.to_file(polygon_path, driver="GeoJSON")
        logger.info(f"Exported polygon: {polygon_path}")

    except Exception as e:
        logger.error(f"Polygon extraction failed: {e}")
        raise

    # Step 5: Basic measurements
    logger.info("\n[5/9] BASIC MEASUREMENTS")
    logger.info("-" * 80)

    try:
        measurements = get_all_measurements(
            final_polygon, lat, lon, source=polygon_source
        )
    except Exception as e:
        logger.error(f"Measurement calculation failed: {e}")
        raise

    # Generate building ID from coordinates
    building_id = f"bldg_{lat:.4f}_{lon:.4f}".replace(".", "_").replace("-", "n")
    footprint_area = measurements.get("plan_area_m2", 0.0)

    # Initialize advanced metrics containers
    roof_metrics = None
    qc_result = None
    output_files = {"results": None}

    # Steps 6-9: Advanced roof analysis (requires LiDAR data)
    if lidar_points is not None and len(lidar_points) > 0:
        logger.info("\n[6/9] HEIGHT MODEL GENERATION")
        logger.info("-" * 80)

        try:
            # Use the CRS PDAL reported for the output points rather than guessing.
            lidar_crs = polygon_result.get("metadata", {}).get("points_crs") or ""
            if not lidar_crs:
                # Fallback for paths that didn't go through the EPT client
                x_vals = lidar_points['X']
                if np.abs(x_vals).max() > 180:
                    from pyproj import CRS
                    lidar_crs = CRS.from_proj4(
                        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
                    ).to_string()
                    logger.warning(
                        f"points_crs missing — falling back to AEQD heuristic "
                        f"(X max: {x_vals.max():.0f})"
                    )
                else:
                    lidar_crs = "EPSG:4326"
            else:
                logger.info(f"Using EPT-reported CRS: {lidar_crs[:60]}")

            height_result = generate_height_models(
                lidar_points,
                str(output_dir),
                building_id,
                crs=lidar_crs
            )
            ndsm_path = height_result["ndsm_path"]
            ndsm = height_result["ndsm"]
            ndsm_grid_x = height_result["grid_x"]
            ndsm_grid_y = height_result["grid_y"]
            logger.info(f"Height models saved: DSM, DTM, nDSM")
            output_files["dsm"] = height_result["dsm_path"]
            output_files["dtm"] = height_result["dtm_path"]
            output_files["ndsm"] = ndsm_path

        except Exception as e:
            logger.error(f"Height model generation failed: {e}")
            ndsm_path = None
            ndsm = None
            ndsm_grid_x = None
            ndsm_grid_y = None

        # Step 7: Raster fusion (optional - requires NAIP)
        logger.info("\n[7/9] RASTER FUSION")
        logger.info("-" * 80)

        fused_path = None
        clipped_path = None

        if naip_data.get("tif_path") and ndsm_path:
            try:
                naip_path = naip_data["tif_path"]
                aligned_naip_path = str(output_dir / "naip_aligned.tif")
                fused_path_str = str(output_dir / "fused_5band.tif")
                clipped_path_str = str(output_dir / f"{building_id}_clipped.tif")

                # Align NAIP to nDSM grid
                aligned_naip_path = align_raster_to_target(
                    naip_path, ndsm_path, aligned_naip_path
                )
                logger.info("NAIP aligned to nDSM grid")

                # Stack bands
                fused_path = stack_naip_with_ndsm(
                    aligned_naip_path, ndsm_path, fused_path_str
                )
                logger.info("5-band fusion complete (R,G,B,NIR,Height)")
                output_files["fused"] = fused_path

                # Clip to footprint
                clipped_path = clip_to_footprint(
                    fused_path, final_polygon, clipped_path_str
                )
                logger.info("Fused raster clipped to footprint")
                output_files["clipped"] = clipped_path

            except Exception as e:
                logger.warning(f"Raster fusion failed: {e}. Continuing with nDSM only.")
                clipped_path = None
        else:
            logger.info("Skipping fusion (missing NAIP or nDSM)")

        # Step 8: Point extraction, segmentation, and plane fitting
        logger.info("\n[8/9] ROOF ANALYSIS (SEGMENTATION & METRICS)")
        logger.info("-" * 80)

        try:
            # Vegetation classification from fused raster (when available).
            # Segmentation and plane fitting ALWAYS use raw LiDAR points with
            # absolute elevation z — the fused raster's nDSM band (0-18m relative)
            # would collapse RANSAC coefficients to ~0 against EPSG:3857 x/y coords.
            if clipped_path:
                _, point_meta = raster_to_points(clipped_path)
                vegetation_pct = point_meta.get("vegetation_percentage", 0.0)
                logger.info(f"Vegetation pct from NAIP fusion: {vegetation_pct:.1f}%")
            else:
                vegetation_pct = 0.0

            # Always segment with raw LiDAR points (absolute z, required by RANSAC).
            # Normalize field names from uppercase (PDAL) to lowercase (segment.py).
            if hasattr(lidar_points.dtype, 'names') and 'Z' in lidar_points.dtype.names:
                new_dtype = [(name.lower(), lidar_points.dtype[name]) for name in lidar_points.dtype.names]
                points = np.empty(len(lidar_points), dtype=new_dtype)
                for name in lidar_points.dtype.names:
                    points[name.lower()] = lidar_points[name]
            else:
                points = lidar_points
            logger.info(f"Using raw LiDAR points for segmentation ({len(points):,} pts)")

            logger.info(f"Extracted {len(points):,} points for analysis")

            # Pre-compute ground_z before filtering so the threshold is consistent.
            # 10th-percentile Z of all LiDAR returns is a robust DTM proxy.
            ground_z_raw = (
                float(np.percentile(lidar_points["Z"], 10))
                if lidar_points is not None and "Z" in (lidar_points.dtype.names or ())
                else None
            )

            # Filter to roof-level points only before segmentation.
            # Without this, ground returns (class 2) and low vegetation get
            # partitioned into "flat facets" that inflate the segment count and
            # produce 0:12 pitches.
            if ground_z_raw is not None and 'z' in (points.dtype.names or ()):
                roof_mask = points['z'] > ground_z_raw + 2.0
                n_before = len(points)
                points = points[roof_mask]
                logger.info(
                    f"Roof-level filter: {len(points):,}/{n_before:,} points "
                    f"above ground+2m (ground≈{ground_z_raw:.1f}m)"
                )

            # Persist roof-level points so bench.py can re-run segmentation
            # from cache without re-fetching EPT.
            try:
                _pts_npy = output_dir / "lidar_points_roof.npy"
                np.save(str(_pts_npy), points)
                logger.debug(f"Saved {len(points):,} roof-level points → {_pts_npy}")
            except Exception as _e:
                logger.debug(f"Could not save points cache: {_e}")

            # Segment facets — RF-DETR model (if a checkpoint is given), then
            # NAIP-guided, then k-means. The model supplies 2D facet shapes;
            # LiDAR points are attached and real planes fit below (fusion).
            naip_seg_used = False
            naip_clipped_tif = naip_data.get("tif_path")
            _pts_crs_seg = polygon_result.get("metadata", {}).get("points_crs")
            if model_checkpoint and naip_clipped_tif and _pts_crs_seg:
                logger.info("Attempting RF-DETR model segmentation…")
                try:
                    from src.roofs.model_segment import (
                        segment_facets_model, gap_fill_facets)
                    _model_facets = segment_facets_model(
                        naip_clipped_tif, _pts_crs_seg, model_checkpoint,
                        lidar_points=points, threshold=model_threshold,
                    )
                    if _model_facets:
                        # Phase C fusion: fill the outline the (sparse) model
                        # missed with LiDAR facets, so the set tiles the roof and
                        # the perimeter (eave/rake) types correctly.
                        _outline_seg = None
                        try:
                            from pyproj import (CRS as _CRS,
                                                Transformer as _Tr)
                            from shapely.ops import transform as _shp_tx
                            _tf = _Tr.from_crs(
                                "EPSG:4326", _CRS.from_user_input(_pts_crs_seg),
                                always_xy=True).transform
                            _outline_seg = _shp_tx(_tf, final_polygon)
                        except Exception as _oe:
                            logger.warning(f"gap-fill outline reproject failed ({_oe})")
                        facets = gap_fill_facets(_model_facets, _outline_seg, points)
                        naip_seg_used = True
                        logger.info(
                            "Model (RF-DETR) + LiDAR gap-fill: %d facet(s) — "
                            "planes fit below", len(facets))
                    else:
                        logger.info("Model segmentation returned no facets — "
                                    "falling back")
                except Exception as _me:
                    logger.warning(f"Model segmentation failed ({_me}) — fallback")

            # nDSM (height-surface) segmentation: cut facets on the real ridge/
            # hip creases (clean boundaries), then attach LiDAR points for the
            # per-facet plane. Opt-in via segment_method="ndsm".
            if not naip_seg_used and segment_method == "ndsm" and ndsm_path and \
                    _pts_crs_seg:
                logger.info("Attempting nDSM (height-surface) segmentation…")
                try:
                    from src.roofs.ndsm_segment import segment_facets_ndsm
                    _nd_facets = segment_facets_ndsm(
                        ndsm_path, _pts_crs_seg, lidar_points=points)
                    if _nd_facets:
                        facets = _nd_facets
                        naip_seg_used = True
                        logger.info("nDSM segmentation: %d facet(s) with LiDAR "
                                    "support", len(facets))
                    else:
                        logger.info("nDSM segmentation returned no facets — fallback")
                except Exception as _nde:
                    logger.warning(f"nDSM segmentation failed ({_nde}) — fallback")

            if not naip_seg_used and naip_clipped_tif and ndsm_path and \
                    polygon_result.get("metadata", {}).get("points_crs"):
                _pts_crs = polygon_result["metadata"]["points_crs"]
                logger.info("Attempting NAIP-guided segmentation…")
                try:
                    _naip_facets = naip_guided_segmentation(
                        naip_clipped_path=Path(naip_clipped_tif),
                        ndsm_path=Path(ndsm_path),
                        footprint_wgs84=final_polygon,
                        lidar_points=points,
                        points_crs=_pts_crs,
                    )
                    if _naip_facets:
                        facets = _naip_facets
                        naip_seg_used = True
                        logger.info(
                            f"NAIP segmentation succeeded: {len(facets)} facets "
                            f"(sizes: {[f.count for f in facets]})"
                        )
                    else:
                        logger.info("NAIP segmentation returned no facets — using k-means fallback")
                except Exception as _nse:
                    logger.warning(f"NAIP segmentation failed ({_nse}) — using k-means fallback")

            if not naip_seg_used:
                _fm = "kmeans" if segment_method == "ndsm" else segment_method
                facets = segment_facets(points, method=_fm)
                logger.info(f"Segmented into {len(facets)} facet(s) using {segment_method}")

            # Fit planes to each facet
            planes = []
            facet_polygons = []
            planes_for_poly = []   # parallel to facet_polygons; used for edge classification
            for facet in facets:
                plane = fit_plane_ransac(facet.points)
                planes.append(plane)
                if plane.success:
                    # NAIP facets carry an accurate polygon boundary derived from
                    # imagery; use it directly instead of the inlier convex hull
                    # (which overcounts and can stray outside the roof outline).
                    if naip_seg_used and facet.polygon is not None:
                        boundary = facet.polygon
                    else:
                        boundary = extract_inlier_boundary(facet.points, plane.inlier_mask)
                    facet_polygons.append((facet.facet_id, boundary))
                    planes_for_poly.append(plane)

                    # Fix 1: re-fit using raw EPT points clipped to this facet's
                    # 2D boundary.  Dense raw points allow tighter RANSAC thresholds
                    # (0.10 m / 65% inliers) for slope estimates accurate to ~1.5°.
                    if ground_z_raw is not None and boundary is not None:
                        refined = fit_plane_to_raw_points(
                            lidar_points, boundary, ground_z_raw,
                            dist_thresh=0.10, min_inlier_ratio=0.65,
                        )
                        if refined is not None and refined.success:
                            from src.roofs.plane_fit import PlaneModel as _PM
                            old_slope = math.degrees(math.atan(plane.gradient_magnitude))
                            new_slope = math.degrees(math.atan(refined.gradient_magnitude))
                            logger.info(
                                "Facet %d: raw-EPT slope %.1f° (raster was %.1f°)",
                                facet.facet_id, new_slope, old_slope,
                            )
                            # Adopt slope from raw-EPT refit but keep the original
                            # inlier_mask so it stays sized for facet.points (not
                            # the raw-EPT subset used for refitting).
                            slope_updated = _PM(
                                a=refined.a,
                                b=refined.b,
                                c=refined.c,
                                inlier_mask=plane.inlier_mask,
                                inlier_count=plane.inlier_count,
                                residual_median=refined.residual_median,
                                success=refined.success,
                            )
                            planes[-1] = slope_updated
                            planes_for_poly[-1] = slope_updated

            # Drop facets whose plane fit failed outright — their plane params
            # are garbage (the RANSAC best-effort coefficients without enough
            # inliers produce slopes like 87°) and they have no boundary, so
            # they'd flow into metrics with bogus pitch and hull-overlap areas.
            _ok_ids = {fp[0] for fp in facet_polygons}
            _n_failed = len(facets) - len(_ok_ids)
            if _n_failed:
                logger.warning(
                    "Dropping %d/%d facet(s) with failed plane fits (insufficient "
                    "inliers) — their points stay unmodeled", _n_failed, len(facets),
                )
                planes = [p for f, p in zip(facets, planes) if f.facet_id in _ok_ids]
                facets = [f for f in facets if f.facet_id in _ok_ids]

            # Screened-enclosure filter: drop facets with slope < 5° AND mean nDSM
            # height < 2.5 m.  These are lanais / screened enclosures; their shallow
            # slope and low elevation above grade distinguish them from real roof planes.
            if ndsm is not None and ndsm_grid_x is not None and facet_polygons:
                import shapely as _shp
                gx_flat = ndsm_grid_x.ravel()
                gy_flat = ndsm_grid_y.ravel()
                ndsm_flat = ndsm.ravel()
                keep_mask = []
                for (facet_id, boundary), plane in zip(facet_polygons, planes_for_poly):
                    slope_deg = math.degrees(math.atan(plane.gradient_magnitude))
                    if slope_deg >= 5.0:
                        keep_mask.append(True)
                        continue
                    in_poly = _shp.contains_xy(boundary, gx_flat, gy_flat)
                    mean_ndsm = float(ndsm_flat[in_poly].mean()) if in_poly.any() else 0.0
                    is_enclosure = mean_ndsm < 2.5
                    if is_enclosure:
                        logger.info(
                            "Facet %d: dropped (slope=%.1f° < 5°, mean_ndsm=%.2f m < 2.5 m)",
                            facet_id, slope_deg, mean_ndsm,
                        )
                    keep_mask.append(not is_enclosure)

                n_dropped = keep_mask.count(False)
                if n_dropped:
                    logger.warning(
                        "Enclosure filter: dropped %d/%d facet(s)", n_dropped, len(facet_polygons)
                    )
                    keep_ids = {fp[0] for fp, k in zip(facet_polygons, keep_mask) if k}
                    facet_polygons = [fp for fp, k in zip(facet_polygons, keep_mask) if k]
                    planes_for_poly = [p for p, k in zip(planes_for_poly, keep_mask) if k]
                    planes = [p for f, p in zip(facets, planes) if f.facet_id in keep_ids]
                    facets = [f for f in facets if f.facet_id in keep_ids]

            # Steep-facet gate: planes steeper than 60° are walls or vegetation
            # caught by the clustering, not roof (steepest standard pitch 24:12
            # is 63°, but residential FL tops out near 12:12; the bogus facets
            # we see are 80°+ wall slivers with tiny plan area).
            if facet_polygons:
                keep_mask = []
                for (facet_id, boundary), plane in zip(facet_polygons, planes_for_poly):
                    slope_deg = math.degrees(math.atan(plane.gradient_magnitude))
                    too_steep = slope_deg > 60.0
                    if too_steep:
                        logger.info(
                            "Facet %d: dropped (slope=%.1f° > 60° — wall/vegetation noise)",
                            facet_id, slope_deg,
                        )
                    keep_mask.append(not too_steep)
                if not all(keep_mask):
                    keep_ids = {fp[0] for fp, k in zip(facet_polygons, keep_mask) if k}
                    facet_polygons = [fp for fp, k in zip(facet_polygons, keep_mask) if k]
                    planes_for_poly = [p for p, k in zip(planes_for_poly, keep_mask) if k]
                    planes = [p for f, p in zip(facets, planes) if f.facet_id in keep_ids]
                    facets = [f for f in facets if f.facet_id in keep_ids]

            # Partition facets: inlier-hull boundaries overlap heavily and
            # extend past the roof edge, double-counting area and never meeting
            # cleanly for edge classification. Clip each boundary to the roof
            # outline, then subtract already-claimed area (highest-inlier facet
            # claims first) so the set tiles the roof.
            facet_boundaries_by_id = {}
            outline_native = None
            _did_reconstruct = False

            # Phase 2: clean-facet reconstruction — regularize candidate facets
            # into straight-edged, footprint-tiling polygons. Gated behind
            # use_facet_reconstruct; on any failure or empty result we fall
            # through to the legacy partition block below (fully reversible).
            if use_facet_reconstruct and facet_polygons:
                try:
                    from src.roofs.facet_reconstruct import reconstruct_facets
                    _lidar_crs = polygon_result.get("metadata", {}).get("points_crs")
                    _outline_native = None
                    if _lidar_crs and final_polygon is not None:
                        from pyproj import CRS as _CRS, Transformer as _Tr
                        from shapely.ops import transform as _shp_tx
                        _to_native = _Tr.from_crs(
                            "EPSG:4326", _CRS.from_user_input(_lidar_crs), always_xy=True
                        ).transform
                        _outline_native = _shp_tx(_to_native, final_polygon)
                    _rc_fp, _rc_fb, _rc_pfp = reconstruct_facets(
                        _outline_native, facet_polygons, planes_for_poly
                    )
                    if _rc_fp:
                        outline_native = _outline_native
                        facet_polygons = _rc_fp
                        facet_boundaries_by_id = _rc_fb
                        planes_for_poly = _rc_pfp
                        _keep_ids = {fid for fid, _ in facet_polygons}
                        planes = [p for f, p in zip(facets, planes)
                                  if f.facet_id in _keep_ids]
                        facets = [f for f in facets if f.facet_id in _keep_ids]
                        _did_reconstruct = True
                        logger.info(
                            "Facet reconstruction: %d clean facet(s) via "
                            "reconstruct_facets", len(facet_polygons)
                        )
                except Exception as _re:
                    logger.warning(
                        f"facet_reconstruct failed ({_re}) — legacy partition block"
                    )

            if facet_polygons and not _did_reconstruct:
                from shapely.geometry import Polygon as _Poly

                def _valid(g):
                    # NAIP-derived polygons can self-intersect (especially after
                    # reprojection); invalid inputs make GEOS set-ops raise
                    # TopologyException. buffer(0) is the standard repair.
                    return g if (g is None or g.is_valid) else g.buffer(0)

                try:
                    _lidar_crs = polygon_result.get("metadata", {}).get("points_crs")
                    if _lidar_crs and final_polygon is not None:
                        from pyproj import CRS as _CRS, Transformer as _Tr
                        from shapely.ops import transform as _shp_tx
                        _to_native = _Tr.from_crs(
                            "EPSG:4326", _CRS.from_user_input(_lidar_crs), always_xy=True
                        ).transform
                        outline_native = _valid(_shp_tx(_to_native, final_polygon))
                except Exception as _oe:
                    logger.warning(
                        f"Outline reprojection failed ({_oe}) — partitioning facets "
                        f"against each other only"
                    )

                # NAIP segmentation legitimately extends past the MS Buildings
                # footprint (8 m porch/lanai expansion). Clipping those cells
                # to the MS outline would amputate the porch — use the union
                # of the NAIP cells themselves as the roof extent instead.
                # CAP: a porch adds tens of percent; on treed suburban lots the
                # expansion captures vegetation and the union can balloon to 4x
                # the footprint (observed: 244 m2 house -> 1012 m2 "extent").
                # Beyond 1.4x we keep the MS outline as the trustworthy extent.
                if naip_seg_used and outline_native is not None:
                    try:
                        from shapely.ops import unary_union as _uun
                        _cells_union = _valid(_uun(
                            [_valid(b) for _, b in facet_polygons if b is not None]
                        ))
                        if _cells_union.geom_type == "Polygon" and _cells_union.area > 0:
                            # nDSM edge smear: the Gaussian smoothing (sigma
                            # 2.5 px at 0.5 m grid) bleeds roof height into
                            # ground pixels, dilating the height-threshold
                            # boundary outward by ~0.6 m uniformly along the
                            # perimeter (verified visually vs the Roofr truth:
                            # a 0.9 m band where real overhang is ~0.2-0.3 m).
                            _cells_union = _valid(_cells_union.buffer(-0.6))
                            _ext = _cells_union.union(outline_native)
                            # Spatial collar: legitimate roof beyond the MS
                            # footprint (eave overhang, attached porch) lives
                            # within a few meters of it; yard vegetation does
                            # not. Observed: an unclipped union carried +10.5%
                            # plan area of canopy fringe under the ratio cap.
                            _ext = _valid(_ext.intersection(outline_native.buffer(3.0)))
                            if _ext.geom_type == "Polygon":
                                _ratio = _ext.area / max(outline_native.area, 1e-9)
                                if _ratio <= 1.4:
                                    logger.info(
                                        "Roof extent: NAIP cells + footprint (3 m collar) "
                                        "= %.1f sq m (MS footprint alone %.1f)",
                                        _ext.area, outline_native.area,
                                    )
                                    outline_native = _ext
                                else:
                                    logger.warning(
                                        "NAIP extent %.1f sq m is %.1fx the MS footprint "
                                        "(%.1f sq m) — vegetation capture suspected; "
                                        "keeping MS outline", _ext.area, _ratio,
                                        outline_native.area,
                                    )
                    except Exception as _ue:
                        logger.warning(f"NAIP roof-extent union failed ({_ue}) — using MS outline")

                order = sorted(
                    range(len(facet_polygons)),
                    key=lambda i: -getattr(planes_for_poly[i], "inlier_count", 0),
                )
                claimed = None
                cells = {}
                for i in order:
                    fid, boundary = facet_polygons[i]
                    cell = _valid(boundary)
                    if outline_native is not None:
                        cell = cell.intersection(outline_native)
                    if claimed is not None:
                        cell = cell.difference(claimed)
                    if cell.geom_type != "Polygon":
                        polys = [g for g in getattr(cell, "geoms", []) if isinstance(g, _Poly)]
                        cell = max(polys, key=lambda g: g.area) if polys else None
                    if cell is None or cell.is_empty or cell.area < 2.0:
                        logger.info(
                            "Facet %d: dropped by partition (claimed by overlapping "
                            "neighbours or outside roof outline)", fid,
                        )
                        continue
                    cells[i] = cell
                    claimed = cell if claimed is None else claimed.union(cell)

                # Absorb the unclaimed remainder of the outline into adjacent
                # cells so the facet set tiles the roof COMPLETELY. Perimeter
                # edges (eave/rake/parapet) only exist where a cell boundary
                # lies on the outline; gaps left by unmodeled points would
                # otherwise erase most of the perimeter (observed: 2 ft of
                # eave on a building Roofr measures at 508 ft).
                if cells and outline_native is not None and claimed is not None:
                    def _poly_pieces(geom, min_area=0.01):
                        if geom.is_empty:
                            return []
                        if geom.geom_type == "Polygon":
                            return [geom] if geom.area > min_area else []
                        return [g for g in getattr(geom, "geoms", [])
                                if g.geom_type == "Polygon" and g.area > min_area]

                    # nDSM samples let us assign each unmodeled piece to the
                    # TOUCHING facet whose plane best explains its heights —
                    # longest-shared-boundary alone routes flat-roof edge
                    # strips into adjacent pitched mega-facets.
                    _gx = _gy = _gz = None
                    if ndsm is not None and ndsm_grid_x is not None:
                        import shapely as _shp_mod
                        _gx = ndsm_grid_x.ravel()
                        _gy = ndsm_grid_y.ravel()
                        _gz = ndsm.ravel()

                    def _plane_residual(plane, piece):
                        if _gx is None:
                            return None
                        sel = _shp_mod.contains_xy(piece, _gx, _gy)
                        if not sel.any():
                            return None
                        zp = plane.predict(_gx[sel], _gy[sel])
                        return float(np.nanmean(np.abs(zp - _gz[sel])))

                    pieces = _poly_pieces(outline_native.difference(claimed))
                    absorbed = 0.0
                    for _pass in range(4):
                        if not pieces:
                            break
                        deferred = []
                        progressed = False
                        for piece in pieces:
                            # candidates: cells sharing boundary with the piece
                            touching = []
                            for ci, cell in cells.items():
                                shared_len = piece.boundary.intersection(cell.boundary).length
                                if shared_len > 1e-9:
                                    touching.append((ci, shared_len))
                            best_i = None
                            if touching:
                                # best height-explained candidate, else longest shared
                                scored = []
                                for ci, shared_len in touching:
                                    res = _plane_residual(planes_for_poly[ci], piece)
                                    if res is not None:
                                        scored.append((res, ci))
                                if scored:
                                    best_i = min(scored)[1]
                                else:
                                    best_i = max(touching, key=lambda t: t[1])[0]
                            else:
                                best_i = min(cells, key=lambda ci: cells[ci].distance(piece))
                            merged = cells[best_i].union(piece)
                            if merged.geom_type == "Polygon":
                                cells[best_i] = merged
                                absorbed += piece.area
                                progressed = True
                            else:
                                deferred.append(piece)   # disjoint — retry after neighbours grow
                        pieces = deferred
                        if not progressed:
                            break
                    if absorbed:
                        logger.info(
                            "Remainder absorption: %.1f sq m of unmodeled roof "
                            "merged into adjacent facets%s", absorbed,
                            f" ({sum(p.area for p in pieces):.1f} sq m unplaced)" if pieces else "",
                        )

                # Shared-topology simplification: absorbed cell boundaries are
                # jagged at the 0.5 m DSM grid, which fragments ridge lines and
                # manufactures short spurious valleys. Simplifying polygons
                # independently would break neighbour coincidence; instead,
                # node the boundary network (unary_union), merge degree-2
                # chains into arcs (junction nodes become arc ENDPOINTS, which
                # Douglas-Peucker pins), simplify each arc, re-polygonize.
                # Neighbours stay exactly coincident by construction.
                if len(cells) > 1:
                    try:
                        from shapely.ops import (
                            unary_union as _uu2,
                            polygonize as _pz,
                            linemerge as _lm,
                        )
                        # Snap cells to 1mm grid to fix floating-point vertex
                        # misalignment introduced by the difference/intersection
                        # partition ops; shared edges must be exactly coincident
                        # for linemerge to produce closed rings.
                        try:
                            from shapely import set_precision as _sp
                            _grid_cells = {i: _sp(c, 0.001) for i, c in cells.items()}
                        except Exception:
                            _grid_cells = cells
                        _arcs = _lm(_uu2([c.boundary for c in _grid_cells.values()]))
                        _arc_list = list(getattr(_arcs, "geoms", [_arcs]))
                        logger.debug(
                            "Boundary arcs: %d total, lengths min=%.3f max=%.3f",
                            len(_arc_list),
                            min(a.length for a in _arc_list) if _arc_list else 0,
                            max(a.length for a in _arc_list) if _arc_list else 0,
                        )
                        _simp = [a.simplify(0.4, preserve_topology=False) for a in _arc_list]
                        _raw_polys = list(_pz(_simp))
                        logger.debug("Polygonize: %d raw polygons", len(_raw_polys))
                        _new_polys = [p for p in _raw_polys if p.area > 0.5]

                        _new_cells = {}
                        _clean = bool(_new_polys)
                        for _np in _new_polys:
                            # Assign by max-overlap with original cell, not
                            # representative_point — simplified boundaries can
                            # shift enough that rep_pt falls in a neighbour.
                            _best = None
                            _best_ov = -1.0
                            for _ci, _cc in cells.items():
                                try:
                                    _ov = _cc.intersection(_np).area
                                except Exception:
                                    _ov = 0.0
                                if _ov > _best_ov:
                                    _best_ov = _ov
                                    _best = _ci
                            if _best is None:
                                _best = min(cells, key=lambda i: cells[i].distance(
                                    _np.representative_point()))
                            if _best in _new_cells:
                                _u = _new_cells[_best].union(_np)
                                if _u.geom_type != "Polygon":
                                    _clean = False
                                    break
                                _new_cells[_best] = _u
                            else:
                                _new_cells[_best] = _np
                        _old_area = sum(c.area for c in cells.values())
                        _new_area = sum(c.area for c in _new_cells.values()) if _clean else 0.0
                        _drift_pct = 100.0 * abs(_new_area - _old_area) / max(_old_area, 1e-9)
                        _arc_ok = (
                            _clean and _new_cells
                            and _drift_pct <= 5.0
                            and len(_new_cells) >= len(cells) * 0.8
                        )
                        if _arc_ok:
                            cells = _new_cells
                            _rim = _uu2(list(cells.values()))
                            if _rim.geom_type == "Polygon":
                                outline_native = _rim
                            logger.info(
                                "Boundary simplification: %d arcs -> %d cells "
                                "(area drift %.2f%%)", len(_arc_list), len(cells),
                                _drift_pct,
                            )
                        else:
                            # Arc approach failed or lost cells — simplify each
                            # cell independently (no coincident-boundary guarantee
                            # but all cells preserved).
                            _fb = {
                                i: c.simplify(0.4, preserve_topology=True)
                                for i, c in cells.items()
                            }
                            _fb_area = sum(c.area for c in _fb.values())
                            _fb_drift = 100.0 * abs(_fb_area - _old_area) / max(_old_area, 1e-9)
                            if _fb_drift <= 5.0 and all(
                                c.geom_type == "Polygon" for c in _fb.values()
                            ):
                                cells = _fb
                                _rim = _uu2(list(cells.values()))
                                if _rim.geom_type == "Polygon":
                                    outline_native = _rim
                                logger.info(
                                    "Boundary per-cell simplification: %d cells "
                                    "(area drift %.2f%%)", len(cells), _fb_drift,
                                )
                            else:
                                logger.warning(
                                    "Boundary simplification rejected (clean=%s "
                                    "arc_cells=%d arc_drift=%.2f%% "
                                    "percell_drift=%.2f%%) — keeping jagged cells",
                                    _clean, len(_new_cells), _drift_pct, _fb_drift,
                                )
                    except Exception as _se:
                        logger.warning(
                            f"Boundary simplification failed ({_se}) — keeping jagged cells"
                        )

                kept_idx = sorted(cells.keys())
                if len(kept_idx) < len(facet_polygons):
                    keep_ids = {facet_polygons[i][0] for i in kept_idx}
                    planes = [p for f, p in zip(facets, planes) if f.facet_id in keep_ids]
                    facets = [f for f in facets if f.facet_id in keep_ids]
                facet_polygons = [(facet_polygons[i][0], cells[i]) for i in kept_idx]
                planes_for_poly = [planes_for_poly[i] for i in kept_idx]
                facet_boundaries_by_id = dict(facet_polygons)
                raw_sum = sum(c.area for c in cells.values())
                logger.info(
                    "Facet partition: %d facet(s) tile %.1f sq m of the %.1f sq m footprint",
                    len(facet_polygons), raw_sum, footprint_area,
                )

            # Compute facet metrics (plan areas from the partitioned cells, not
            # inlier convex hulls)
            facet_metrics_list = compute_all_facet_metrics(
                facets, planes, boundaries=facet_boundaries_by_id
            )
            logger.info(f"Computed metrics for {len(facet_metrics_list)} facets")

            # Drop bogus steep facets: slopes > 70° are gable-end walls or
            # vegetation slivers captured by clustering, not roof surfaces.
            # (Steepest real residential pitch 24:12 ≈ 63.4°; we allow 70° to
            # cover the occasional steep dormer while still dropping 80°+ noise.)
            _steep_drop = [i for i, fm in enumerate(facet_metrics_list) if fm.slope_deg > 70.0]
            if _steep_drop:
                for _si in _steep_drop:
                    logger.info(
                        f"Dropped facet {facet_metrics_list[_si].facet_id}: "
                        f"slope {facet_metrics_list[_si].slope_deg:.1f}° > 70° (wall/noise)"
                    )
                facet_metrics_list = [
                    fm for i, fm in enumerate(facet_metrics_list) if i not in set(_steep_drop)
                ]
                logger.info(f"After steep-facet drop: {len(facet_metrics_list)} facet(s)")

            # Merge k-means fragments: facets with the same aspect bin AND pitch
            # string are the same physical plane split by spatial clustering.
            # Skip for NAIP/nDSM segmentation — aspect regions are already
            # geometrically distinct facets, not k-means artifacts.
            if not naip_seg_used:
                from collections import defaultdict as _dd
                from src.roofs.metrics import FacetMetrics as _FM
                _groups = _dd(list)
                for _i, _fm in enumerate(facet_metrics_list):
                    _groups[(_fm.aspect_bin, _fm.pitch_string)].append(_i)
                _merged = []
                for (_abin, _pitch), _idxs in _groups.items():
                    if len(_idxs) == 1:
                        _merged.append(facet_metrics_list[_idxs[0]])
                    else:
                        _base = facet_metrics_list[_idxs[0]]
                        _merged.append(_FM(
                            facet_id=_base.facet_id,
                            plane=_base.plane,
                            slope_deg=_base.slope_deg,
                            pitch_string=_pitch,
                            aspect_deg=_base.aspect_deg,
                            aspect_bin=_abin,
                            plan_area_m2=sum(facet_metrics_list[_i].plan_area_m2 for _i in _idxs),
                            surface_area_m2=sum(facet_metrics_list[_i].surface_area_m2 for _i in _idxs),
                            inliers_count=sum(facet_metrics_list[_i].inliers_count for _i in _idxs),
                            residual_median_m=min(facet_metrics_list[_i].residual_median_m for _i in _idxs),
                            is_flat=_base.is_flat,
                        ))
                        logger.info(f"Merged {len(_idxs)} {_pitch}/{_abin} k-means fragments into one facet")
                        # Merge the fragments' partition cells too, so the diagram and
                        # edge classification see one polygon per physical plane. Only
                        # adopt the union when it stays a single Polygon (adjacent
                        # fragments); disjoint pieces keep their own geometry.
                        _frag_ids = [facet_metrics_list[_i].facet_id for _i in _idxs]
                        _frag_polys = [facet_boundaries_by_id[fid] for fid in _frag_ids
                                       if fid in facet_boundaries_by_id]
                        if len(_frag_polys) > 1:
                            from shapely.ops import unary_union as _uu
                            _u = _uu(_frag_polys)
                            if _u.geom_type == "Polygon":
                                _base_id = _base.facet_id
                                _plane_by_fid = {
                                    fid: p for (fid, _), p in zip(facet_polygons, planes_for_poly)
                                }
                                facet_polygons = [
                                    (fid, (_u if fid == _base_id else poly))
                                    for fid, poly in facet_polygons
                                    if fid == _base_id or fid not in _frag_ids
                                ]
                                planes_for_poly = [_plane_by_fid[fid] for fid, _ in facet_polygons]
                                facet_boundaries_by_id = dict(facet_polygons)
                if len(_merged) < len(facet_metrics_list):
                    logger.info(f"Facet merge: {len(facet_metrics_list)} → {len(_merged)} facets")
                facet_metrics_list = _merged

            # Normalize plan areas so they sum to the building footprint.
            # Convex hulls of 3D-cluster projections overlap in 2D (a N-facing
            # and a S-facing cluster both extend to the ridge in plan view), so
            # raw convex-hull areas double-count the ridge zone.
            # SKIP for NAIP segmentation: its cells legitimately cover MORE
            # than the MS footprint (8 m porch/lanai expansion) and already
            # tile the roof extent exactly — rescaling to the smaller MS
            # polygon would compress every facet by the porch fraction.
            _raw_total = sum(fm.plan_area_m2 for fm in facet_metrics_list)
            if not naip_seg_used and \
                    _raw_total > 0 and footprint_area > 0 and _raw_total != footprint_area:
                _scale = footprint_area / _raw_total
                for _fm in facet_metrics_list:
                    _fm.plan_area_m2 = round(_fm.plan_area_m2 * _scale, 2)
                    _cos = math.cos(math.radians(min(_fm.slope_deg, 89.0)))
                    _fm.surface_area_m2 = round(_fm.plan_area_m2 / max(_cos, 0.01), 2)
                logger.info(
                    f"Normalized plan areas (raw total {_raw_total:.1f} → "
                    f"footprint {footprint_area:.1f} sq m, scale={_scale:.3f})"
                )

            # Classify edges — use classify_edges_from_facets (the modern path that
            # handles parapet/step_flashing and produces facet-boundary-exact edges)
            # followed by merge_collinear_edges to collapse the tiling artefact
            # (LiDAR facet boundaries produce ~10× too many collinear sub-segments).
            if len(facet_polygons) > 1:
                if use_facet_reconstruct:
                    # Phase 4: grid-snap for shared-boundary coincidence, then
                    # classify + collinear-merge (all inside reconstruct_edges).
                    from src.roofs.facet_reconstruct import reconstruct_edges
                    edges = reconstruct_edges(
                        facet_polygons, planes_for_poly,
                        outline=outline_native, step_height_m=1.0,
                    )
                else:
                    edges = classify_edges_from_facets(
                        facet_polygons, planes_for_poly, outline=outline_native,
                        # LiDAR planes carry real intercepts, so a height gap at a
                        # junction marks a story step -> wall/step flashing. 1.0 m
                        # separates true steps (2.5-3 m here) from ridge junctions
                        # where two fits disagree by a few decimeters at 1 m LiDAR.
                        step_height_m=1.0,
                    )
                    edges = merge_collinear_edges(edges)
                edge_summary = compute_edge_lengths_by_type(edges)
                logger.info(
                    f"Edge classification: {len(edges)} edges after merge "
                    f"(was up to ~10× more before collinear merge)"
                )
            else:
                edges = []
                edge_summary = {}

            # Build complete metrics
            facet_metrics_dicts = [fm.to_dict() for fm in facet_metrics_list]
            edge_dicts = [e.to_dict() for e in edges]

            roof_metrics = compute_building_metrics(
                building_id,
                facet_metrics_list,  # Pass FacetMetrics objects, not dicts
                footprint_area
            )
            roof_metrics["edges"] = edge_summary
            roof_metrics["n_edges"] = len(edges)
            roof_metrics["n_facets"] = len(facet_polygons)
            # Standardized headline areas (always present under these names so
            # downstream consumers/reports don't get None from a key mismatch).
            # plan_area_m2   = flat footprint area (the ~86%-accurate deliverable)
            # total_area_m2  = pitch-corrected roof SURFACE area (roofing headline)
            roof_metrics["plan_area_m2"] = round(footprint_area, 2)
            roof_metrics["total_area_m2"] = roof_metrics.get(
                "total_surface_area_m2", 0.0)

            logger.info(
                f"Roof analysis complete: {len(facets)} facets, "
                f"{len(edges)} edges, dominant pitch: {roof_metrics.get('dominant_pitch', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Roof analysis failed: {e}")
            facet_metrics_dicts = []
            edge_dicts = []
            vegetation_pct = 0.0
            # Preserve the plan (footprint) area even when facet analysis fails —
            # it's the core deliverable and doesn't depend on facets/model.
            roof_metrics = {
                "plan_area_m2": round(footprint_area, 2),
                "total_area_m2": None,     # needs facets (pitch) — unavailable here
                "n_facets": 0, "n_edges": 0,
            }

        # Step 9: QC and Export
        logger.info("\n[9/9] QUALITY CONTROL & EXPORT")
        logger.info("-" * 80)

        try:
            # Run QC validation
            qc_result = validate_metrics(
                roof_metrics if roof_metrics else {},
                ndsm,
                vegetation_pct
            )
            logger.info(f"QC complete: confidence={qc_result.confidence_score:.2f}, flags={len(qc_result.flags)}")

            # Export JSON metrics
            metrics_path = output_dir / "metrics.json"
            export_report_json(
                {
                    "address": address or f"{lat},{lon}",
                    "building_id": building_id,
                    "facets": facet_metrics_dicts,
                    "edges": edge_dicts if 'edge_dicts' in dir() else [],
                },
                str(metrics_path),
            )
            output_files["metrics_json"] = str(metrics_path)
            logger.info(f"Exported: {metrics_path}")

            # Export facet geometry in the LiDAR CRS for downstream consumers
            # (pseudo-label dataset generation projects these onto the chip
            # raster in pixel space — see training/build_pseudo_dataset.py).
            if facet_boundaries_by_id:
                from shapely.geometry import mapping as _shp_mapping
                _fm_by_id = {fm.get("facet_id"): fm for fm in facet_metrics_dicts}
                _features = []
                for _fid, _cell in sorted(facet_boundaries_by_id.items()):
                    _fm = _fm_by_id.get(_fid, {})
                    _features.append({
                        "type": "Feature",
                        "geometry": _shp_mapping(_cell),
                        "properties": {
                            "kind": "facet",
                            "facet_id": _fid,
                            "pitch_string": _fm.get("pitch_string"),
                            "aspect_bin": _fm.get("aspect_bin"),
                            "slope_deg": _fm.get("slope_deg"),
                            "is_flat": _fm.get("is_flat"),
                        },
                    })
                if outline_native is not None:
                    _features.append({
                        "type": "Feature",
                        "geometry": _shp_mapping(outline_native),
                        "properties": {"kind": "outline"},
                    })
                _fg_path = output_dir / "facets.geojson"
                _pcrs = polygon_result.get("metadata", {}).get("points_crs")
                _fg = {
                    "type": "FeatureCollection",
                    "features": _features,
                    "properties": {
                        "crs": _pcrs,      # legacy custom field (kept for back-compat)
                        "qc_confidence": float(qc_result.confidence_score),
                        "building_id": building_id,
                    },
                }
                # Standard GeoJSON named-CRS member so readers (geopandas/GDAL)
                # don't default to WGS84 — coords are in the LiDAR UTM CRS, not
                # lat/lon. Without this, overlays land off-frame (wrong SRID tag).
                if _pcrs:
                    try:
                        from pyproj import CRS as _CRS
                        _auth = _CRS.from_user_input(_pcrs).to_authority()
                        if _auth:
                            _fg["crs"] = {"type": "name", "properties": {
                                "name": f"urn:ogc:def:crs:{_auth[0]}::{_auth[1]}"}}
                    except Exception as _ce:
                        logger.warning(f"facets.geojson CRS tag failed: {_ce}")
                json.dump(_fg, open(_fg_path, "w"))
                output_files["facets_geojson"] = str(_fg_path)
                logger.info(f"Exported: {_fg_path}")

            # Export CSV files
            if export_csv and facet_metrics_dicts:
                facets_csv_path = output_dir / "facets.csv"
                export_facets_csv(facet_metrics_dicts, str(facets_csv_path))
                output_files["facets_csv"] = str(facets_csv_path)
                logger.info(f"Exported: {facets_csv_path}")

                if edge_dicts:
                    edges_csv_path = output_dir / "edges.csv"
                    export_edges_csv(edge_dicts, str(edges_csv_path))
                    output_files["edges_csv"] = str(edges_csv_path)
                    logger.info(f"Exported: {edges_csv_path}")

            # Export PDF (optional)
            if export_pdf and check_pdf_available() and roof_metrics:
                pdf_path = output_dir / "roof_report.pdf"
                # generate_report() consumes the same dict shape as the JSON
                # export, plus diagram geometry (polygon_xy / geometry_xy in a
                # shared planar CRS) and the aerial preview for the cover page.
                boundary_by_id = {fid: b for fid, b in facet_polygons}
                pdf_facets = []
                for fm in facet_metrics_dicts:
                    b = boundary_by_id.get(fm.get("facet_id"))
                    pdf_facets.append({
                        **fm,
                        "polygon_xy": list(b.exterior.coords) if b is not None else [],
                    })
                pdf_edges = [
                    {**ed, "geometry_xy": list(e.geometry.coords)}
                    for ed, e in zip(edge_dicts, edges)
                ]
                report_input = {
                    "address": address or f"{lat},{lon}",
                    "building_id": building_id,
                    "facets": pdf_facets,
                    "edges": pdf_edges,
                    "aerial_image_path": naip_data.get("png_path"),
                }
                export_pdf_plan_sheet(report_input, str(pdf_path))
                output_files["pdf"] = str(pdf_path)
                logger.info(f"Exported: {pdf_path}")
            elif export_pdf and not check_pdf_available():
                logger.warning("PDF export skipped (reportlab not installed)")

            # Export DXF (optional)
            if export_dxf and check_dxf_available() and roof_metrics:
                dxf_path = output_dir / "roof.dxf"
                polygon_coords = list(final_polygon.exterior.coords)
                export_dxf(
                    building_id,
                    polygon_coords,
                    edge_dicts,
                    facet_metrics_dicts,
                    str(dxf_path)
                )
                output_files["dxf"] = str(dxf_path)
                logger.info(f"Exported: {dxf_path}")
            elif export_dxf and not check_dxf_available():
                logger.warning("DXF export skipped (ezdxf not installed)")

        except Exception as e:
            logger.error(f"QC/Export failed: {e}")

    else:
        logger.info("\n[6-9/9] ADVANCED ANALYSIS SKIPPED")
        logger.info("-" * 80)
        logger.info("LiDAR data not available - skipping advanced roof metrics")
        logger.info("Basic measurements only (plan area, perimeter)")

    # Compile results
    results = {
        "input": {
            "address": address,
            "lat": lat,
            "lon": lon,
            "state": state
        },
        "building": {
            "building_id": building_id,
            "distance_from_pin_m": selection["dist_m"],
            "n_candidates": len(selection["candidates"]),
            "candidates_file": str(candidates_path)
        },
        "polygon": {
            "source": polygon_source,
            "file": str(polygon_path),
            "metadata": {
                k: v for k, v in polygon_result.get("metadata", {}).items()
                if k not in ("points", "grid_x", "grid_y", "grid_z")  # Exclude large arrays
            }
        },
        "naip": naip_data,
        "lidar": {
            "enabled": fetch_lidar,
            "refined": polygon_source == "ept_lidar_refined",
            "dataset_name": polygon_result.get("metadata", {}).get("dataset_name"),
            "point_count": polygon_result.get("metadata", {}).get("point_count")
        },
        "measurements": measurements,
        "roof_metrics": roof_metrics,
        "qc": qc_result.to_dict() if qc_result else None,
        "output_files": output_files
    }

    # Save results
    results_path = output_dir / "results.json"
    output_files["results"] = str(results_path)
    with open(results_path, "w") as f:
        # Convert non-serializable types
        results_json = json.loads(json.dumps(results, default=str))
        json.dump(results_json, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 80)

    return results


def main():
    """
    Command-line entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Roof measurement pipeline")
    parser.add_argument("--address", type=str, help="Street address")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--state", type=str, required=True, help="State abbreviation (e.g., fl)")
    parser.add_argument("--buffer", type=float, default=150, help="Search radius in meters")
    parser.add_argument("--no-naip", action="store_true", help="Skip NAIP imagery")
    parser.add_argument("--no-lidar", action="store_true", help="Skip LiDAR data")
    parser.add_argument("--naip-year", type=str, help="Preferred NAIP year")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--segment-method", type=str, default="kmeans",
                        choices=["single", "kmeans", "regiongrow"],
                        help="Facet segmentation method (default: kmeans)")
    parser.add_argument("--ept-url", type=str, help="Hardcode EPT URL (bypasses WESM discovery)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF export")
    parser.add_argument("--no-dxf", action="store_true", help="Skip DXF export")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Run pipeline
    try:
        results = process_address(
            address=args.address,
            lat=args.lat,
            lon=args.lon,
            state=args.state,
            buffer_meters=args.buffer,
            fetch_naip=not args.no_naip,
            fetch_lidar=not args.no_lidar,
            naip_year=args.naip_year,
            ept_url=args.ept_url,
            output_dir=args.output,
            segment_method=args.segment_method,
            export_pdf=not args.no_pdf,
            export_dxf=not args.no_dxf,
            export_csv=not args.no_csv
        )

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Building ID: {results['building']['building_id']}")
        print(f"Plan area: {results['measurements']['plan_area_m2']:.2f} sq m")
        print(f"Perimeter: {results['measurements']['perimeter_m']:.2f} m")
        print(f"Source: {results['polygon']['source']}")

        # Print advanced metrics if available
        if results.get("roof_metrics"):
            rm = results["roof_metrics"]
            print(f"Dominant pitch: {rm.get('dominant_pitch', 'N/A')}")
            print(f"Total surface area: {rm.get('total_surface_area_m2', 0):.2f} sq m")
            print(f"Facets: {rm.get('num_facets', 0)}")

        if results.get("qc"):
            qc = results["qc"]
            print(f"QC confidence: {qc.get('confidence_score', 0):.2f}")

        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
