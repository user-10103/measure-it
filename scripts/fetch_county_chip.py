#!/usr/bin/env python3
"""Fetch a 3-inch (7.6 cm) county aerial chip for an address and save it.

Pinellas works from anywhere. Hillsborough + Pasco block some datacenter IPs
(HTTP 000) -- run those from a residential IP or Colab (Google egress reaches
them). This is the per-address imagery front-door for the Roof Report pipeline.

Usage:
    python scripts/fetch_county_chip.py "1466 Dartmouth Dr, Clearwater, FL 33756"
    python scripts/fetch_county_chip.py "<addr>" --size-m 50 --out chip.png
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingestion.county_imagery import fetch_for_address  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("address")
    ap.add_argument("--size-m", type=float, default=50.0, help="chip side in metres")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    png, p2w, meta = fetch_for_address(args.address, size_m=args.size_m)
    out = args.out or f"chip_{meta['county']}_{meta['year']}.png"
    Path(out).write_bytes(png)
    print(f"address : {meta['address']}")
    print(f"county  : {meta['county']} {meta['year']}  ({meta['source']})")
    print(f"gsd     : {meta['gsd_m']} m/px  ({meta['gsd_m']*39.37:.1f} in)")
    print(f"chip    : {meta['width']}x{meta['height']} px  ->  {out}")
    print(f"feed to pipeline: process_chip_rgb(pred, gsd_m_per_px={meta['gsd_m']}, ...)")


if __name__ == "__main__":
    main()
