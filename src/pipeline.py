"""
Main roof measurement pipeline.
Orchestrates the complete workflow from address to roof metrics.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import geopandas as gpd
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
from src.roofs.plane_fit import fit_plane_ransac, extract_inlier_boundary
from src.roofs.metrics import compute_all_facet_metrics, compute_building_metrics
from src.roofs.edges import classify_all_edges, compute_edge_lengths_by_type
from src.qc.validators import validate_metrics
from src.output.json_export import export_building_metrics
from src.output.csv_export import export_facets_csv, export_edges_csv
from src.output.pdf_export import export_pdf_plan_sheet, check_pdf_available
from src.output.dxf_export import export_dxf, check_dxf_available

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
    output_dir: Optional[Path] = None,
    segment_method: str = "kmeans",
    export_pdf: bool = True,
    export_dxf: bool = True,
    export_csv: bool = True
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
                lat, lon, state, building_polygon, naip_year
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
            logger.info("EPT LiDAR refinement enabled - discovering dataset...")
            polygon_result = get_roof_polygon(building_polygon, lat, lon)
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
            # Detect CRS from point coordinates (LiDAR points may be in UTM, not WGS84)
            x_vals = lidar_points['X']
            if np.abs(x_vals).max() > 180:  # Likely projected (UTM)
                # Use AEQD projection centered on lat/lon for consistency
                from pyproj import CRS
                lidar_crs = CRS.from_proj4(
                    f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
                ).to_string()
                logger.info(f"Detected projected CRS for LiDAR points (X max: {x_vals.max():.0f})")
            else:
                lidar_crs = "EPSG:4326"

            height_result = generate_height_models(
                lidar_points,
                str(output_dir),
                building_id,
                crs=lidar_crs
            )
            ndsm_path = height_result["ndsm_path"]
            ndsm = height_result["ndsm"]
            logger.info(f"Height models saved: DSM, DTM, nDSM")
            output_files["dsm"] = height_result["dsm_path"]
            output_files["dtm"] = height_result["dtm_path"]
            output_files["ndsm"] = ndsm_path

        except Exception as e:
            logger.error(f"Height model generation failed: {e}")
            ndsm_path = None
            ndsm = None

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
            # Use fused raster if available, otherwise fall back to nDSM with LiDAR points
            if clipped_path:
                points, point_meta = raster_to_points(clipped_path)
                vegetation_pct = point_meta.get("vegetation_percentage", 0.0)
            else:
                # Fall back to using raw LiDAR points directly
                # Normalize field names from uppercase (PDAL) to lowercase (segment.py expects lowercase)
                if hasattr(lidar_points.dtype, 'names') and 'Z' in lidar_points.dtype.names:
                    new_dtype = [(name.lower(), lidar_points.dtype[name]) for name in lidar_points.dtype.names]
                    points = np.empty(len(lidar_points), dtype=new_dtype)
                    for name in lidar_points.dtype.names:
                        points[name.lower()] = lidar_points[name]
                    logger.debug(f"Normalized field names: {list(points.dtype.names)[:5]}...")
                else:
                    points = lidar_points
                vegetation_pct = 0.0
                logger.info("Using raw LiDAR points (no fusion available)")

            logger.info(f"Extracted {len(points):,} points for analysis")

            # Segment facets
            facets = segment_facets(points, method=segment_method)
            logger.info(f"Segmented into {len(facets)} facet(s) using {segment_method}")

            # Fit planes to each facet
            planes = []
            facet_polygons = []
            for facet in facets:
                plane = fit_plane_ransac(facet.points)
                planes.append(plane)
                if plane.success:
                    boundary = extract_inlier_boundary(facet.points, plane.inlier_mask)
                    facet_polygons.append((facet.facet_id, boundary))

            # Compute facet metrics
            facet_metrics_list = compute_all_facet_metrics(facets, planes)
            logger.info(f"Computed metrics for {len(facet_metrics_list)} facets")

            # Classify edges
            if len(facet_polygons) > 1:
                x_min = min(p['X'].min() for p in [f.points for f in facets])
                x_max = max(p['X'].max() for p in [f.points for f in facets])
                y_min = min(p['Y'].min() for p in [f.points for f in facets])
                y_max = max(p['Y'].max() for p in [f.points for f in facets])
                bounds = (x_min, y_min, x_max, y_max)
                edges = classify_all_edges(facet_polygons, planes, bounds)
                edge_summary = compute_edge_lengths_by_type(edges)
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

            logger.info(
                f"Roof analysis complete: {len(facets)} facets, "
                f"{len(edges)} edges, dominant pitch: {roof_metrics.get('dominant_pitch', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Roof analysis failed: {e}")
            facet_metrics_dicts = []
            edge_dicts = []
            vegetation_pct = 0.0

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
            export_building_metrics(
                building_id,
                facet_metrics_dicts,
                footprint_area,
                edge_summary if 'edge_summary' in dir() else {},
                qc_result.to_dict(),
                str(metrics_path)
            )
            output_files["metrics_json"] = str(metrics_path)
            logger.info(f"Exported: {metrics_path}")

            # Export CSV files
            if export_csv and facet_metrics_dicts:
                facets_csv_path = output_dir / "facets.csv"
                export_facets_csv(building_id, facet_metrics_dicts, str(facets_csv_path))
                output_files["facets_csv"] = str(facets_csv_path)
                logger.info(f"Exported: {facets_csv_path}")

                if edge_dicts:
                    edges_csv_path = output_dir / "edges.csv"
                    export_edges_csv(building_id, edge_dicts, str(edges_csv_path))
                    output_files["edges_csv"] = str(edges_csv_path)
                    logger.info(f"Exported: {edges_csv_path}")

            # Export PDF (optional)
            if export_pdf and check_pdf_available() and roof_metrics:
                pdf_path = output_dir / "roof_plan.pdf"
                polygon_coords = list(final_polygon.exterior.coords)
                export_pdf_plan_sheet(
                    roof_metrics,
                    polygon_coords,
                    str(pdf_path)
                )
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
