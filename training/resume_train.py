#!/usr/bin/env python3
"""Resume RF-DETR training from last.ckpt (PTL checkpoint).

Uses rfdetr's native 'resume' kwarg which maps to:
  trainer.fit(module, datamodule, ckpt_path=config.resume or None)

This properly restores epoch, optimizer state, LR scheduler, EMA weights.

Pidfile guard: only one instance allowed at a time.
"""

import os
import sys
import signal
import torch; torch.backends.cudnn.enabled = False
import argparse
from rfdetr import RFDETRSegPreview

PIDFILE = "/tmp/resume_train.pid"

def acquire_pidfile():
    """Exit immediately if another instance is running."""
    if os.path.exists(PIDFILE):
        with open(PIDFILE) as f:
            old_pid = f.read().strip()
        try:
            old_pid = int(old_pid)
            os.kill(old_pid, 0)  # signal 0 = check existence
            print(f"ERROR: another resume_train is already running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # stale pidfile — safe to overwrite
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))

def release_pidfile():
    try:
        os.remove(PIDFILE)
    except FileNotFoundError:
        pass

def main():
    acquire_pidfile()
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dataset", default="roof_dataset")
        ap.add_argument("--output", default="output")
        ap.add_argument("--epochs", type=int, default=50)
        ap.add_argument("--batch-size", type=int, default=1)
        ap.add_argument("--grad-accum", type=int, default=4)
        ap.add_argument("--lr", type=float, default=1e-4)
        ap.add_argument("--resolution", type=int, default=512)
        ap.add_argument("--resume", default="output/last.ckpt",
                        help="PTL checkpoint to resume from")
        args = ap.parse_args()

        model = RFDETRSegPreview(resolution=args.resolution)
        model.train(
            dataset_dir=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            lr=args.lr,
            output_dir=args.output,
            resume=args.resume,   # <-- rfdetr native: maps to ckpt_path in trainer.fit()
        )

        print(f"Done. Best weights in {args.output}/")
    finally:
        release_pidfile()

if __name__ == "__main__":
    main()
