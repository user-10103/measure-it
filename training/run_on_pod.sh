#!/usr/bin/env bash
# One-shot RunPod driver: install deps -> fetch chips from S3 -> train.
# Run from runpod/train/ on the pod, AFTER setting AWS creds for the chip bucket:
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=...
#   bash run_on_pod.sh
set -euo pipefail

CHIP_BUCKET="${CHIP_BUCKET:-florida-roofs-v2-chips}"
CHIP_PREFIX="${CHIP_PREFIX:-phase1}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-4}"

echo "==> installing deps"
pip install -r requirements.txt

echo "==> fetching chip images from s3://${CHIP_BUCKET}/${CHIP_PREFIX}/"
python fetch_chips.py --bucket "$CHIP_BUCKET" --prefix "$CHIP_PREFIX" --dataset roof_dataset

echo "==> training RF-DETR-Seg (${EPOCHS} epochs, batch ${BATCH})"
python train_rfdetr.py --dataset roof_dataset --output output --epochs "$EPOCHS" --batch-size "$BATCH"

echo "==> done. weights are in ./output/  -- download them off the pod."
