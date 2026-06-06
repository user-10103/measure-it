#!/usr/bin/env python3
"""
Simple example demonstrating the roof-metrics pipeline.
"""
import logging
from pathlib import Path
from src.pipeline import process_address, setup_logging


def main():
    """Run a simple example."""
    # Setup logging
    setup_logging("INFO")

    # Example address from successful experiment
    address = "16347 Heathrow Dr, Tampa, FL 33647"

    print("\n" + "=" * 80)
    print("ROOF-METRICS PIPELINE EXAMPLE")
    print("=" * 80)
    print(f"Address: {address}")
    print("=" * 80 + "\n")

    # Run pipeline
    try:
        results = process_address(
            address=address,
            state="fl",  # Required for NAIP/LiDAR
            buffer_meters=150,  # Search radius
            fetch_naip=True,  # Download aerial imagery
            fetch_lidar=True,  # Download elevation data
            output_dir="output/example"
        )

        # Display results
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        measurements = results["measurements"]
        print(f"\n📏 Measurements:")
        print(f"   Plan Area:  {measurements['plan_area_m2']:.2f} sq m ({measurements['plan_area_m2'] * 10.764:.2f} sq ft)")
        print(f"   Perimeter:  {measurements['perimeter_m']:.2f} m ({measurements['perimeter_m'] * 3.281:.2f} ft)")
        print(f"   Vertices:   {measurements['n_vertices']}")
        print(f"   Source:     {measurements['source']}")

        polygon_info = results["polygon"]
        print(f"\n🏠 Polygon:")
        print(f"   Method:     {polygon_info['source']}")
        print(f"   File:       {polygon_info['file']}")

        building_info = results["building"]
        print(f"\n🎯 Building Selection:")
        print(f"   Distance:   {building_info['distance_from_pin_m']:.2f} m from pin")
        print(f"   Candidates: {building_info['n_candidates']} evaluated")

        naip_info = results["naip"]
        if naip_info:
            print(f"\n🛰️  NAIP Imagery:")
            print(f"   Clipped:    {naip_info['tif_path']}")
            print(f"   Preview:    {naip_info['png_path']}")

        lidar_info = results["lidar"]
        print(f"\n⛰️  LiDAR Data:")
        print(f"   Available:  {lidar_info['available']}")
        if lidar_info['dataset']:
            print(f"   Dataset:    {lidar_info['dataset']}")

        print(f"\n📁 Output Directory:")
        print(f"   {Path(results['polygon']['file']).parent}")

        print("\n" + "=" * 80)
        print("✅ EXAMPLE COMPLETE")
        print("=" * 80 + "\n")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
