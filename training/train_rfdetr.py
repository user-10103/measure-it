#!/usr/bin/env python3
"""Fine-tune RF-DETR-Seg on the roof facet+outline dataset (RunPod / Colab GPU).

RF-DETR-Seg is Apache-2.0 across all segmentation tiers (DINOv2 backbone, built
for small-data fine-tuning) and ingests the Roboflow/COCO layout that
build_dataset.py produced -- see the deep-research finding. The model predicts
ONLY facet + roof outline instance masks; pitch (policy) and aspect (geometric)
are applied downstream in measure-it, so this is a plain 2-class instance-seg
fine-tune.

Usage:
    # Fresh start
    python train_rfdetr.py --dataset roof_dataset_v2 --output output --epochs 50

    # Resume after Colab disconnect
    python train_rfdetr.py --dataset roof_dataset_v2 --output output --epochs 50 \
        --resume /content/drive/MyDrive/roof_training/run1/last.ckpt

Known-good workarounds baked in
--------------------------------
1. float8 shim  -- torch < 2.5 lacks float8_* dtypes that transformers/rfdetr
   import at module load time; we stub them to float32 before any rfdetr import.
2. fused_optimizer=False  -- rfdetr default TrainConfig sets fused_optimizer=True
   which triggers an AdamW dtype mismatch under bfloat16 AMP on torch 2.x.
   We patch rfdetr.config before instantiating the model so the fix survives
   pod restarts and pip reinstalls without touching site-packages.
3. batch_size=1 default  -- rfdetr seg matcher does torch.cat([v["masks"] ...])
   across the batch; different spatial sizes crash the cat. Single-image batches
   avoid the collation problem; grad_accum compensates for effective batch size.
4. CSV logger extrasaction patch -- Lightning's CSV logger raises ValueError on
   resume when the checkpoint restores metric history with more columns than the
   current CSV header. We patch DictWriter to use extrasaction='ignore'.
"""

import argparse
import torch
torch.backends.cudnn.enabled = False

# -- 1. float8 shim (must run before any rfdetr/transformers import) ----------
for _n in ["float8_e4m3fn", "float8_e4m3fnuz",
           "float8_e5m2",   "float8_e5m2fnuz", "float8_e8m0fnu"]:
    if not hasattr(torch, _n):
        setattr(torch, _n, torch.float32)

# -- 2. fused_optimizer patch (before rfdetr instantiation) -------------------
try:
    import rfdetr.config as _rfc
    import dataclasses as _dc
    for _cls in vars(_rfc).values():
        if _dc.is_dataclass(_cls):
            for _f in _dc.fields(_cls):
                if _f.name == "fused_optimizer":
                    _f.default = False  # type: ignore[misc]
except Exception as _e:
    import warnings
    warnings.warn(f"fused_optimizer patch failed ({_e}); AdamW dtype errors may occur")

# -- 3. CSV logger patch -------------------------------------------------------
# Two failure modes fixed here:
#   (a) Extra string keys in old rows that aren't in the current header
#       → extrasaction='ignore' silently drops them.
#   (b) None keys in rows (from malformed CSV or a class with no predictions
#       returning None as its metric key) → explicit filter before writerow.
# Also patches log_metrics to strip None keys before they reach the CSV at all.
try:
    import csv as _csv
    import lightning_fabric.loggers.csv_logs as _csv_logs
    from pathlib import Path as _Path

    def _patched_rewrite(self, fieldnames):
        # Strip None from fieldnames — can appear when a class has no predictions
        fieldnames = [f for f in fieldnames if f is not None]
        tmp = _Path(str(self.metrics_file_path) + ".tmp")
        try:
            with open(self.metrics_file_path, "r") as rf, open(tmp, "w", newline="") as wf:
                reader = _csv.DictReader(rf)
                writer = _csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    # Drop None keys from every row (trailing-comma artefact)
                    writer.writerow({k: v for k, v in row.items() if k is not None})
            tmp.replace(self.metrics_file_path)
        except Exception:
            _Path(self.metrics_file_path).unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)

    _csv_logs.ExperimentWriter._rewrite_with_new_header = _patched_rewrite

    # Also intercept log_metrics so None keys never enter the metrics dict
    _orig_log_metrics = _csv_logs.ExperimentWriter.log_metrics

    def _patched_log_metrics(self, metrics_dict, step):
        clean = {k: v for k, v in metrics_dict.items() if k is not None}
        return _orig_log_metrics(self, clean, step)

    _csv_logs.ExperimentWriter.log_metrics = _patched_log_metrics

except Exception as _csv_e:
    import warnings
    warnings.warn(f"CSV logger patch failed ({_csv_e}); resume may raise ValueError on metrics.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roof_dataset",
                    help="dir with train/valid/test + _annotations.coco.json")
    ap.add_argument("--output", default="output")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 avoids mask-spatial-dim collation crash (use grad-accum)")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="effective batch = bs*grad_accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--resume", default=None,
                    help="path to last.ckpt for resuming a previous run")
    ap.add_argument("--pretrain-weights", default=None,
                    help="path to a .pth to use as starting weights (not full resume)")
    ap.add_argument("--model", default="large",
                    help="rfdetr seg variant: preview | large | xlarge (default: large)")
    args = ap.parse_args()

    _model_map = {
        "preview": "RFDETRSegPreview",
        "large":   "RFDETRSegLarge",
        "xlarge":  "RFDETRSegXLarge",
    }
    _cls_name = _model_map.get(args.model.lower(), "RFDETRSegLarge")
    import rfdetr as _rfdetr
    if not hasattr(_rfdetr, _cls_name):
        raise ValueError(f"rfdetr has no class {_cls_name}. "
                         f"Run: pip install -U rfdetr  (need v1.4+)")
    ModelCls = getattr(_rfdetr, _cls_name)
    print(f"Using model class: {_cls_name}")
    model = ModelCls(
        resolution=args.resolution,
        pretrain_weights=args.pretrain_weights,
    )
    model.train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.output,
        resume=args.resume,
    )
    print(f"Done. Weights + checkpoints in {args.output}/")
    print("Export the best checkpoint and wire it into measure-it as a "
          "RoofModelBackend (replaces CocoStandinBackend).")


if __name__ == "__main__":
    main()
