#!/usr/bin/env python3
"""Fine-tune RF-DETR-Seg on the roof facet+outline dataset (RunPod GPU).

RF-DETR-Seg is Apache-2.0 across all segmentation tiers (DINOv2 backbone, built
for small-data fine-tuning) and ingests the Roboflow/COCO layout that
build_dataset.py produced -- see the deep-research finding. The model predicts
ONLY facet + roof_outline instance masks; pitch (policy) and aspect (geometric)
are applied downstream in measure-it, so this is a plain 2-class instance-seg
fine-tune.

Usage (on the RunPod pod, after fetch_chips):
  python train_rfdetr.py --dataset roof_dataset --output output --epochs 50
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roof_dataset",
                    help="dir with train/valid/test + _annotations.coco.json")
    ap.add_argument("--output", default="output")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="MUST stay 1: chips are variable-sized and DETR can't collate "
                         "a batch of different-sized images (matcher tensor mismatch). "
                         "batch>1 crashes; use --grad-accum for effective batch.")
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = bs*grad_accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--model", default="preview",
                    help="rfdetr seg variant; 'preview' = RFDETRSegPreview")
    args = ap.parse_args()

    # imported here so --help works without torch/rfdetr installed
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
