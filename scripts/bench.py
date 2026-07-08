#!/usr/bin/env python3
"""
Benchmark harness: re-run segmentation → metrics → edges from cached inputs.

Avoids re-fetching NAIP + EPT (saves ~4 min per run). Loads:
  lidar_points_roof.npy   (saved by pipeline.py post roof-level filter)
  naip_aligned.tif        (NAIP aligned to nDSM resolution)
  *_ndsm.tif              (nDSM in native CRS)
  roof_polygon.geojson    (MS Buildings footprint)
  results.json            (points_crs, address for truth lookup)

Usage:
  python scripts/bench.py output/holland_final
  python scripts/bench.py output/holland_final --height 1.3
  python scripts/bench.py output/holland_final --sweep

  # Sweep porch threshold too:
  python scripts/bench.py output/holland_final --sweep --sweep-porch
"""
import argparse
import json
import math
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TRUTH_FILE = Path(__file__).parent / "bench_truth.json"
SQM_TO_SQFT = 10.7639
M_TO_FT = 3.28084


# ── Truth loading ─────────────────────────────────────────────────────────────

def load_truth(address: str):
    if not TRUTH_FILE.exists():
        return None
    truth = json.loads(TRUTH_FILE.read_text())
    # Case-insensitive prefix match
    addr_lo = address.lower()
    for k, v in truth.items():
        if k.lower().startswith(addr_lo[:20]) or addr_lo.startswith(k.lower()[:20]):
            return v
    return None


# ── Input loading ─────────────────────────────────────────────────────────────

def load_inputs(out_dir: Path):
    results_path = out_dir / "results.json"
    if not results_path.exists():
        sys.exit(f"ERROR: {results_path} not found")

    results = json.loads(results_path.read_text())
    address = results.get("input", {}).get("address", "unknown")

    pts_crs = (
        results.get("polygon", {})
        .get("metadata", {})
        .get("points_crs", "EPSG:26917")
    )

    # Footprint WGS84
    from shapely import wkt as _wkt
    fp_wkt = (
        results.get("polygon", {})
        .get("metadata", {})
        .get("refined_polygon_wgs84")
    )
    if fp_wkt:
        footprint_wgs84 = _wkt.loads(fp_wkt)
    else:
        import geopandas as _gpd
        gdf = _gpd.read_file(out_dir / "roof_polygon.geojson")
        footprint_wgs84 = gdf.to_crs("EPSG:4326").union_all()

    # nDSM TIF
    ndsm_candidates = sorted(out_dir.glob("*_ndsm.tif"))
    if not ndsm_candidates:
        sys.exit(f"ERROR: no *_ndsm.tif found in {out_dir}")
    ndsm_path = ndsm_candidates[0]

    # NAIP aligned TIF
    naip_path = out_dir / "naip_aligned.tif"
    if not naip_path.exists():
        sys.exit(f"ERROR: naip_aligned.tif not found in {out_dir}")

    # LiDAR points cache
    pts_npy = out_dir / "lidar_points_roof.npy"
    if not pts_npy.exists():
        sys.exit(
            f"ERROR: {pts_npy} not found.\n"
            "Re-run the full pipeline once to create this cache:\n"
            "  python -m src.pipeline <address> --state FL --ept-url <url>"
        )

    import numpy as np
    lidar_points = np.load(str(pts_npy), allow_pickle=False)

    return dict(
        address=address,
        pts_crs=pts_crs,
        footprint_wgs84=footprint_wgs84,
        ndsm_path=ndsm_path,
        naip_path=naip_path,
        lidar_points=lidar_points,
        footprint_area_m2=results.get("polygon", {}).get("metadata", {}).get(
            "refined_area_m2",
            results.get("measurements", {}).get("plan_area_m2", 0.0),
        ),
    )


# ── Single benchmark run ──────────────────────────────────────────────────────

def run_one(inputs, height_m: float, porch_m: float, verbose: bool = False):
    import src.roofs.naip_segment as _ns
    from src.roofs.naip_segment import naip_guided_segmentation
    from src.roofs.plane_fit import fit_plane_ransac
    from src.roofs.metrics import compute_all_facet_metrics, compute_building_metrics
    from src.roofs.edges import (
        classify_edges_from_facets,
        merge_collinear_edges,
        compute_edge_lengths_by_type,
    )

    # Patch module-level constants before calling segmentation
    _ns.MIN_NDSM_HEIGHT_M = height_m
    _ns.MIN_NDSM_HEIGHT_PORCH_M = porch_m

    facets = naip_guided_segmentation(
        naip_clipped_path=inputs["naip_path"],
        ndsm_path=inputs["ndsm_path"],
        footprint_wgs84=inputs["footprint_wgs84"],
        lidar_points=inputs["lidar_points"],
        points_crs=inputs["pts_crs"],
    )

    if not facets:
        return None

    planes = []
    facet_polygons = []    # (facet_id, polygon) for RANSAC-successful facets only
    planes_for_poly = []   # parallel to facet_polygons
    for f in facets:
        plane = fit_plane_ransac(f.points)
        planes.append(plane)
        if plane.success and f.polygon is not None:
            facet_polygons.append((f.facet_id, f.polygon))
            planes_for_poly.append(plane)

    # The pipeline's partition step clips each cell to the union of
    # SUCCESSFUL-RANSAC cells (not the full 8m expansion from _fill_footprint).
    # Replicate that here: build the outline from successful cells and clip.
    from shapely.ops import unary_union as _uu
    from shapely.geometry import Polygon as _Poly

    def _valid(g):
        return g if (g is None or g.is_valid) else g.buffer(0)

    boundaries = {}
    if facet_polygons:
        cells_union = _valid(_uu([_valid(p) for _, p in facet_polygons]))

        # Replicate pipeline partition: highest-inlier cell claims first
        order = sorted(
            range(len(facet_polygons)),
            key=lambda i: -getattr(planes_for_poly[i], "inlier_count", 0),
        )
        claimed = None
        cells = {}
        for idx in order:
            fid, poly = facet_polygons[idx]
            cell = _valid(poly)
            if cells_union is not None:
                cell = _valid(cell.intersection(cells_union))
            if claimed is not None:
                cell = _valid(cell.difference(claimed))
            if cell is None or cell.is_empty or cell.area < 2.0:
                continue
            if cell.geom_type != "Polygon":
                polys = [g for g in getattr(cell, "geoms", []) if isinstance(g, _Poly)]
                cell = max(polys, key=lambda g: g.area) if polys else None
            if cell is None or cell.is_empty:
                continue
            cells[fid] = cell
            claimed = cell if claimed is None else _valid(claimed.union(cell))

        boundaries = cells

    # Reorder facets/planes to match the surviving cells, then compute metrics
    surviving_ids = set(boundaries.keys())
    facets_ok = [f for f in facets if f.facet_id in surviving_ids]
    planes_ok = [p for f, p in zip(facets, planes) if f.facet_id in surviving_ids]

    if not facets_ok:
        return None

    metrics_list = compute_all_facet_metrics(facets_ok, planes_ok, boundaries=boundaries)

    # Drop extreme slopes (walls/noise)
    metrics_list = [fm for fm in metrics_list if fm.slope_deg <= 70.0]

    # Edge classification (use surviving partition cells)
    surviving_facet_polys = [(fid, boundaries[fid]) for fid in boundaries]
    surviving_planes = [
        p for f, p in zip(facets, planes)
        if f.facet_id in boundaries
    ]
    if len(surviving_facet_polys) > 1:
        edges = classify_edges_from_facets(surviving_facet_polys, surviving_planes, step_height_m=1.0)
        edges = merge_collinear_edges(edges)
        edge_summary_m = compute_edge_lengths_by_type(edges)
    else:
        edge_summary_m = {}

    total_m2 = sum(fm.surface_area_m2 for fm in metrics_list)
    flat_m2 = sum(fm.plan_area_m2 for fm in metrics_list if fm.slope_deg < 5.0)

    edge_ft = {k: v * M_TO_FT for k, v in edge_summary_m.items()}

    return dict(
        n_facets=len(metrics_list),
        total_sqft=round(total_m2 * SQM_TO_SQFT),
        flat_sqft=round(flat_m2 * SQM_TO_SQFT),
        edge_ft=edge_ft,
    )


# ── Pretty printing ───────────────────────────────────────────────────────────

def pct(v, ref):
    if ref == 0:
        return "n/a"
    return f"{100*(v-ref)/ref:+.1f}%"


def print_table(result, truth, label=""):
    hdr = f"  {'metric':<20}  {'pipeline':>10}  "
    if truth:
        hdr += f"{'roofr':>10}  {'diff':>8}  {'pct':>8}"
    print(f"\n{'─'*70}")
    if label:
        print(f"  {label}")
    print(hdr)
    print(f"{'─'*70}")

    rows = [
        ("total_sqft",  result["total_sqft"],  truth["total_sqft"] if truth else None),
        ("flat_sqft",   result["flat_sqft"],   truth["flat_sqft"]  if truth else None),
        ("n_facets",    result["n_facets"],     truth["n_facets"]   if truth else None),
    ]
    for name, v, t in rows:
        if t is not None:
            diff = v - t
            print(f"  {name:<20}  {v:>10}  {t:>10}  {diff:>+8}  {pct(v,t):>8}")
        else:
            print(f"  {name:<20}  {v:>10}")

    if truth and truth.get("edges_ft"):
        print(f"  {'─'*66}")
        print(f"  {'edge':<20}  {'pipeline':>10}  {'roofr':>10}  {'diff':>8}")
        print(f"  {'─'*66}")
        all_edge_types = sorted(
            set(list(result["edge_ft"].keys()) + list(truth["edges_ft"].keys()))
        )
        for et in all_edge_types:
            pv = result["edge_ft"].get(et, 0.0)
            tv = truth["edges_ft"].get(et, 0.0)
            diff = pv - tv
            print(f"  {et:<20}  {pv:>10.0f}  {tv:>10.0f}  {diff:>+8.0f}")
    elif result["edge_ft"]:
        print(f"  {'─'*30}")
        for et, v in sorted(result["edge_ft"].items()):
            print(f"  {et:<20}  {v:>10.0f}")

    print(f"{'─'*70}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Bench re-run from cached inputs")
    ap.add_argument("output_dir", help="Pipeline output directory (e.g. output/holland_final)")
    ap.add_argument("--height", type=float, default=None,
                    help="Override MIN_NDSM_HEIGHT_M (core footprint threshold)")
    ap.add_argument("--porch-height", type=float, default=None,
                    help="Override MIN_NDSM_HEIGHT_PORCH_M (expansion ring threshold)")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep MIN_NDSM_HEIGHT_M from 1.0 to 2.0 in 0.1 steps")
    ap.add_argument("--sweep-porch", action="store_true",
                    help="Also sweep porch height (0.5–1.5m) in the sweep mode")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    inputs = load_inputs(out_dir)
    truth = load_truth(inputs["address"])

    print(f"\nBenchmark: {inputs['address']}")
    if truth:
        print(f"Truth source: {truth.get('source','?')}")
    else:
        print("No truth entry found in bench_truth.json — showing pipeline output only")

    # Read current defaults from module
    import src.roofs.naip_segment as _ns
    default_h = _ns.MIN_NDSM_HEIGHT_M
    default_p = _ns.MIN_NDSM_HEIGHT_PORCH_M

    if args.sweep:
        heights = [round(h * 0.1, 1) for h in range(10, 21)]  # 1.0 → 2.0
        porch_vals = [round(p * 0.1, 1) for p in range(5, 16)] if args.sweep_porch else [default_p]

        print(f"\nSweep mode: {len(heights)} × {len(porch_vals)} combinations\n")

        # Print header
        print(f"  {'height':>7}  {'porch':>6}  {'total_sqft':>11}  {'flat_sqft':>10}  {'n_facets':>9}", end="")
        if truth:
            print(f"  {'area_diff':>9}  {'flat_diff':>9}", end="")
        print()
        print(f"  {'─'*85}")

        for h in heights:
            for p in porch_vals:
                r = run_one(inputs, height_m=h, porch_m=p, verbose=args.verbose)
                if r is None:
                    print(f"  {h:>7.1f}  {p:>6.1f}  {'FAILED':>11}")
                    continue
                line = f"  {h:>7.1f}  {p:>6.1f}  {r['total_sqft']:>11}  {r['flat_sqft']:>10}  {r['n_facets']:>9}"
                if truth:
                    ta = r['total_sqft'] - truth['total_sqft']
                    tf = r['flat_sqft'] - truth['flat_sqft']
                    line += f"  {ta:>+9}  {tf:>+9}"
                print(line)

        print(f"\n  Current defaults: height={default_h}m  porch={default_p}m")
    else:
        h = args.height if args.height is not None else default_h
        p = args.porch_height if args.porch_height is not None else default_p
        label = f"height={h}m  porch={p}m"
        if args.height is not None or args.porch_height is not None:
            label += "  (overridden)"
        else:
            label += "  (current defaults)"

        r = run_one(inputs, height_m=h, porch_m=p, verbose=args.verbose)
        if r is None:
            print("ERROR: segmentation returned no facets")
            sys.exit(1)
        print_table(r, truth, label=label)

    # Restore defaults so any subsequent imports see the right values
    _ns.MIN_NDSM_HEIGHT_M = default_h
    _ns.MIN_NDSM_HEIGHT_PORCH_M = default_p


if __name__ == "__main__":
    main()
