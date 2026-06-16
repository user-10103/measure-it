#!/usr/bin/env python3
"""
Download the Florida Roofs training dataset from S3.

Prerequisites
─────────────
  pip install boto3 tqdm

AWS credentials must already be configured:
  aws configure   (or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars)

Usage
─────
  # Download everything (chips + annotations)
  python download_dataset.py --output ./florida_roofs_dataset

  # Download only annotations (no chips, fast)
  python download_dataset.py --output ./florida_roofs_dataset --annotations-only

  # Download only one source
  python download_dataset.py --output ./florida_roofs_dataset --source phase1
  python download_dataset.py --output ./florida_roofs_dataset --source rid2

What you get
────────────
  florida_roofs_dataset/
    annotations/
      train/_annotations.coco.json   ← COCO instance segmentation format
      valid/_annotations.coco.json      categories: roof_polygon (1), facet (2)
      test/_annotations.coco.json
    chips/
      phase1/   ← 2,237 NAIP Florida aerial chips  (512×512 RGB PNG, ~1m/px)
      rid2/     ← 1,692 RID2 Dutch aerial chips    (512×512 RGB PNG, ~8cm/px)

Annotation format
─────────────────
  Standard COCO instance segmentation JSON.
  Each image has two annotation types:
    category_id=1  roof_polygon  — outline of the entire roof
    category_id=2  facet         — individual roof plane (hip, gable, valley, etc.)
  segmentation field is a flat polygon [x1,y1,x2,y2,...] in pixel coordinates.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    sys.exit("Missing dependency: pip install boto3")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing dependency: pip install tqdm")

BUCKET   = "florida-roofs-v4"
SOURCES  = ["phase1", "rid2"]
SPLITS   = ["train", "valid", "test"]


def list_prefix(s3, prefix):
    """Return all object keys under a prefix."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_file(s3, key, dst_path):
    """Download one S3 key to dst_path, skip if already present."""
    if dst_path.exists():
        return "skip"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(BUCKET, key, str(dst_path))
        return "ok"
    except ClientError as e:
        return f"fail:{e}"


def main():
    ap = argparse.ArgumentParser(
        description="Download Florida Roofs dataset from S3"
    )
    ap.add_argument(
        "--output", default="./florida_roofs_dataset",
        help="Destination directory (default: ./florida_roofs_dataset)"
    )
    ap.add_argument(
        "--source", choices=SOURCES + ["all"], default="all",
        help="Which chip source to download (default: all)"
    )
    ap.add_argument(
        "--annotations-only", action="store_true",
        help="Download annotation JSONs only, skip chips"
    )
    ap.add_argument(
        "--workers", type=int, default=16,
        help="Parallel download threads (default: 16)"
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Connect to S3 ─────────────────────────────────────────────────────────
    try:
        s3 = boto3.client("s3")
        s3.head_bucket(Bucket=BUCKET)
    except NoCredentialsError:
        sys.exit(
            "No AWS credentials found.\n"
            "Run:  aws configure\n"
            "  or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars."
        )
    except ClientError as e:
        sys.exit(f"Cannot access bucket '{BUCKET}': {e}")

    print(f"Connected to s3://{BUCKET}")
    print(f"Output directory: {out.resolve()}\n")

    # ── Annotations ───────────────────────────────────────────────────────────
    print("── Annotations ──────────────────────────────────────────")
    ann_keys = list_prefix(s3, "annotations/merged_naip_rid2/")
    if not ann_keys:
        print("  WARNING: no annotations found under annotations/merged_naip_rid2/")
        print("  (ask the data team to upload them)")
    else:
        for key in ann_keys:
            rel  = key.replace("annotations/merged_naip_rid2/", "")
            dst  = out / "annotations" / rel
            stat = download_file(s3, key, dst)
            flag = "✓" if stat in ("ok", "skip") else "✗"
            note = "(already present)" if stat == "skip" else ""
            print(f"  {flag} {rel} {note}")

    if args.annotations_only:
        print("\nAnnotations-only mode — done.")
        return

    # ── Chips ─────────────────────────────────────────────────────────────────
    sources = SOURCES if args.source == "all" else [args.source]

    for src in sources:
        print(f"\n── {src} chips ({'NAIP Florida' if src == 'phase1' else 'RID2 Dutch'}) ──")
        keys = list_prefix(s3, f"{src}/")
        png_keys = [k for k in keys if k.endswith(".png")]
        print(f"  {len(png_keys)} chips in S3")

        jobs = []
        for key in png_keys:
            fname = key.split("/")[-1]
            dst   = out / "chips" / src / fname
            if not dst.exists():
                jobs.append((key, dst))

        skipped = len(png_keys) - len(jobs)
        if skipped:
            print(f"  {skipped} already present — skipping")
        if not jobs:
            print("  Nothing to download.")
            continue

        print(f"  Downloading {len(jobs)} chips ({args.workers} threads) …")
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = {ex.submit(download_file, s3, k, d): k for k, d in jobs}
            with tqdm(as_completed(fut), total=len(fut), unit="chip", desc=f"  {src}") as bar:
                for f in bar:
                    result = f.result()
                    if result == "ok":
                        ok += 1
                    else:
                        fail += 1
                        if result.startswith("fail:"):
                            tqdm.write(f"  WARN: {fut[f]} → {result}")
                    bar.set_postfix(ok=ok, fail=fail)
        print(f"  Done: {ok} downloaded, {fail} failed, {skipped} skipped")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(1 for _ in out.rglob("*.png"))
    ann_count = sum(1 for _ in out.rglob("*.json"))
    print(f"\n{'='*50}")
    print(f"Dataset ready at: {out.resolve()}")
    print(f"  Chips       : {total}")
    print(f"  Annotation JSONs: {ann_count}")
    print(f"\nTo verify annotations:")
    print(f"  python -c \"import json; d=json.load(open('{out}/annotations/train/_annotations.coco.json')); print(len(d['images']), 'images,', len(d['annotations']), 'annotations')\"")


if __name__ == "__main__":
    main()
