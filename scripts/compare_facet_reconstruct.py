#!/usr/bin/env python3
"""Validate the clean-facet reconstruction (Phase 2 + 4) on a real address.

Runs the full LiDAR pipeline TWICE on the same address — once with the legacy
partition/edge path and once with ``use_facet_reconstruct=True`` — then prints a
side-by-side comparison and checks the acceptance gates:

  * area drift between the two runs is small,
  * (optional) new area is within +/-4% of a reference (e.g. a Roofr report),
  * edge-segment count drops toward the real 8-20 (from the ~60 jagged baseline).

Meant to run where the LiDAR stack + network exist (Colab / the pod), since
``pdal`` and NAIP/EPT access are required. Example (Colab cell):

    !pip -q install buildingregulariser
    !python scripts/compare_facet_reconstruct.py \
        --address "110 Holland Lane, Lutz, FL 33548" --state fl \
        --outdir output/_compare --roofr-area-sqft 8844
"""
import argparse
import json
import os

from src.pipeline import process_address, setup_logging

M2_TO_SQFT = 10.7639


def _facet_count(out_dir: str) -> int:
    """Count facets from the written facets.geojson (robust to return-shape)."""
    p = os.path.join(out_dir, "facets.geojson")
    if not os.path.exists(p):
        return -1
    try:
        gj = json.load(open(p))
        return len(gj.get("features", []))
    except Exception:
        return -1


def _edge_count(out_dir: str) -> int:
    """Count edge segments from edges.geojson if the pipeline wrote one."""
    for name in ("edges.geojson", "roof_edges.geojson"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            try:
                return len(json.load(open(p)).get("features", []))
            except Exception:
                pass
    return -1


def _metrics(results: dict, out_dir: str) -> dict:
    meas = results.get("measurements", {}) or {}
    roof = results.get("roof_metrics", {}) or {}
    area_m2 = (roof.get("total_area_m2")
               or roof.get("plan_area_m2")
               or meas.get("plan_area_m2") or 0.0)
    edge_summary = roof.get("edges", {}) or {}
    return {
        "area_m2": float(area_m2),
        "area_sqft": float(area_m2) * M2_TO_SQFT,
        "n_facets": _facet_count(out_dir),
        "n_edges": _edge_count(out_dir),
        "edge_lengths_m": {k: round(v, 1) for k, v in edge_summary.items()},
    }


def _run(address, lat, lon, state, out_dir, use_reconstruct):
    return process_address(
        address=address, lat=lat, lon=lon, state=state,
        fetch_naip=True, fetch_lidar=True, output_dir=out_dir,
        use_facet_reconstruct=use_reconstruct,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", type=str, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--state", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="output/_compare")
    ap.add_argument("--roofr-area-sqft", type=float, default=None,
                    help="reference area (e.g. from a Roofr report) for the ±4% gate")
    args = ap.parse_args()
    setup_logging("INFO")

    old_dir = os.path.join(args.outdir, "old")
    new_dir = os.path.join(args.outdir, "new")

    print("\n### RUN 1/2 — legacy path (use_facet_reconstruct=False)")
    old = _metrics(_run(args.address, args.lat, args.lon, args.state, old_dir, False), old_dir)
    print("### RUN 2/2 — clean-facet path (use_facet_reconstruct=True)")
    new = _metrics(_run(args.address, args.lat, args.lon, args.state, new_dir, True), new_dir)

    def _row(label, o, n):
        print(f"  {label:<16} {str(o):>14}   {str(n):>14}")

    print("\n" + "=" * 60)
    print("  COMPARISON                 LEGACY          RECONSTRUCT")
    print("=" * 60)
    _row("area (sqft)", round(old["area_sqft"], 0), round(new["area_sqft"], 0))
    _row("facets", old["n_facets"], new["n_facets"])
    _row("edge segments", old["n_edges"], new["n_edges"])
    print("\n  edge lengths by type (m):")
    print("    legacy    :", old["edge_lengths_m"])
    print("    reconstruct:", new["edge_lengths_m"])

    # --- gates ---
    print("\n" + "-" * 60)
    print("  GATES")
    print("-" * 60)
    if old["area_m2"] > 0:
        drift = abs(new["area_m2"] - old["area_m2"]) / old["area_m2"] * 100
        print(f"  area drift new-vs-legacy : {drift:5.1f}%   "
              f"{'PASS' if drift <= 10 else 'FAIL (>10%)'}")
    if args.roofr_area_sqft:
        dev = abs(new["area_sqft"] - args.roofr_area_sqft) / args.roofr_area_sqft * 100
        print(f"  area vs reference (±4%)  : {dev:5.1f}%   "
              f"{'PASS' if dev <= 4 else 'REVIEW (>4%)'}")
    if new["n_edges"] >= 0:
        ok = 4 <= new["n_edges"] <= 24
        print(f"  edge count in ~8-20      : {new['n_edges']:>4}    "
              f"{'PASS' if ok else 'REVIEW'}   (legacy {old['n_edges']})")
    print("=" * 60)
    print(f"\nPDFs: {old_dir}/roof_report.pdf   vs   {new_dir}/roof_report.pdf\n")


if __name__ == "__main__":
    main()
