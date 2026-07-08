#!/usr/bin/env python3
"""Download chip PNGs from S3 into each split dir, per chips_needed.txt.

Uses the RunPod/EC2 instance role or standard AWS env credentials -- NO keys are
hardcoded or passed on the CLI (enter creds via the environment at runtime).

Usage:
  python fetch_chips.py --bucket florida-roofs-v2-chips --prefix phase1 \
      --dataset roof_dataset
"""
import argparse
import concurrent.futures
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="phase1")
    ap.add_argument("--dataset", default="roof_dataset")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--listing", default="chips_needed.txt",
                    help="Filename of the chips list inside each split dir "
                         "(default: chips_needed.txt). Use chips_needed_{prefix}.txt "
                         "when working with merged datasets.")
    args = ap.parse_args()

    import boto3
    s3 = boto3.client("s3")
    prefix = args.prefix.strip("/")

    jobs = []
    for split in ("train", "valid", "test"):
        d = os.path.join(args.dataset, split)
        listing = os.path.join(d, args.listing)
        if not os.path.exists(listing):
            continue
        for name in (l.strip() for l in open(listing) if l.strip()):
            dst = os.path.join(d, name)
            if not os.path.exists(dst):
                key = f"{prefix}/{name}" if prefix else name
                jobs.append((key, dst))

    print(f"{len(jobs)} chips to download from s3://{args.bucket}/{prefix}/ ...")
    ok = fail = 0

    def pull(job):
        key, dst = job
        try:
            s3.download_file(args.bucket, key, dst)
            return True
        except Exception as e:
            print(f"  FAIL {key}: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(pull, jobs):
            ok += r
            fail += (not r)
    print(f"done: {ok} downloaded, {fail} failed")


if __name__ == "__main__":
    main()
