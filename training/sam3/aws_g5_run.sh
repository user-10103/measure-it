#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run ON the AWS g5.xlarge `measure-it-gpu` box (DLAMI PyTorch 2.7 = torch+CUDA
# present). Two modes, fastest-answer first:
#   MODE=infer  (default) — run the EXISTING fine-tuned checkpoint through the
#                world-class gate on a smoke address. First GATE result, no retrain.
#   MODE=train  — readiness-gate the fresh labels, prep clean data, retrain SAM3,
#                then infer + gate.
#
# Account 017341176694 / us-west-2. SSH in from your IP (SG sg-0e67301a80339fe2b
# allows 22 from your address; update the SG if your IP rotated). Then:
#   git -C code pull && bash code/measure-it-main/training/sam3/aws_g5_run.sh
# ---------------------------------------------------------------------------
set -euo pipefail
MODE=${MODE:-infer}
CODE=${CODE:-$HOME/code}
MEASURE=${MEASURE:-$CODE/measure-it-main}
BUCKET=s3://measure-it-prod-017341176694
CKPT=${CKPT:-$HOME/sam3_roof_ft_boxonly_ep9.pt}     # the existing fine-tuned model
SMOKE=${SMOKE:-"28.0303, -80.69809"}                 # FL address w/ LiDAR coverage
export MEASURE_IT_LIDAR=${MEASURE_IT_LIDAR:-1}       # pitch fusion on

cd "$MEASURE"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('CUDA OK', torch.cuda.get_device_name(0))"

# fetch the existing checkpoint from S3 if not local
if [ ! -f "$CKPT" ]; then
  echo "pulling existing checkpoint from S3..."
  aws s3 cp "$BUCKET/models/sam3_roof_ft_boxonly_ep9.pt" "$CKPT"
fi

if [ "$MODE" = "train" ]; then
  echo "== retrain on readiness-gated fresh labels =="
  : "${EXPORT_COCO:?set EXPORT_COCO to a FRESH Label Studio COCO export}"
  : "${ROOF_DATASET:?set ROOF_DATASET to the train/valid split dir}"
  python -m training.label_readiness "$EXPORT_COCO" /tmp/keep.json
  python -m training.sam3.prep_sam3_facets \
    --train-coco "$ROOF_DATASET/train/_annotations.coco.json" --train-images "$ROOF_DATASET/train" \
    --val-coco   "$ROOF_DATASET/valid/_annotations.coco.json" --val-images   "$ROOF_DATASET/valid" \
    --out /tmp/sam3_data --keep /tmp/keep.json
  ( cd "${SAM3_REPO:?set SAM3_REPO}" && \
    sed -e "s#<DATA_ROOT>#/tmp/sam3_data#" -e "s#<LOG_DIR>#$HOME/sam3_ft#" \
        -e "s#<BPE_PATH>#$HOME/bpe_simple_vocab_16e6.txt.gz#" \
        "$MEASURE/training/sam3/roof_facet_ft.yaml" > sam3/train/configs/roof_facet_ft.yaml && \
    python sam3/train/train.py -c configs/roof_facet_ft.yaml --use-cluster 0 --num-gpus 1 )
  CKPT=$(ls -t "$HOME"/sam3_ft/checkpoints/*.pt* | head -1)
  echo "retrained checkpoint: $CKPT"
fi

echo "== inference + WORLD-CLASS gate on $SMOKE (checkpoint: $CKPT) =="
python - "$CKPT" "$SMOKE" <<'PY'
import sys, json
from src.roofs.sam3_predictors import load_sam3_predictors
from src.serve.report_service import generate_roof_report
from src.output.report_qc import format_report_qc
ckpt, smoke = sys.argv[1], sys.argv[2]
predict_facets, predict_outline = load_sam3_predictors(ckpt)   # ckpt_path (positional)
res = generate_roof_report(smoke, "FL", predict_facets, predict_outline,
                           out_dir="/tmp/report_out", use_lidar=True)
print("PDF:", res.pdf_path, "| facets:", res.num_facets, "| pitched:", res.num_pitched)
print(format_report_qc(res.qc))
print("\nGATE:", "PASS — world-class" if res.qc["passed"] else "FAIL — see checks above")
PY
