#!/usr/bin/env python3
"""Fine-tune RF-DETR-Seg on the roof facet+outline dataset (RunPod GPU).

RF-DETR-Seg is Apache-2.0 across all segmentation tiers (DINOv2 backbone, built
for small-data fine-tuning) and ingests the Roboflow/COCO layout that
build_dataset.py produced -- see the deep-research finding. The model predicts
ONLY facet + roof outline instance masks; pitch (policy) and aspect (geometric)
are applied downstream in measure-it, so this is a plain 2-class instance-seg
fine-tune.

Usage (on the RunPod pod, after fetch_chips):
    python train_rfdetr.py --dataset roof_dataset --output output --epochs 50

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
    ap.add_argument("--model", default="preview",
                    help="rfdetr seg variant; 'preview' = RFDETRSegPreview")
    args = ap.parse_args()

    from rfdetr import RFDETRSegPreview
    model = RFDETRSegPreview(resolution=args.resolution)
    model.train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.output,
    )
    print(f"Done. Weights + checkpoints in {args.output}/")
    print("Export the best checkpoint and wire it into measure-it as a "
          "RoofModelBackend (replaces CocoStandinBackend).")


if __name__ == "__main__":
    main()
