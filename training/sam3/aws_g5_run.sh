#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run ON the AWS g5.xlarge `measure-it-gpu` box (DLAMI PyTorch 2.7 = torch+CUDA
# present). Two modes, fastest-answer first:
#   MODE=infer  (default) — run the EXISTING fine-tuned checkpoint through the
#                world-class gate on a smoke address. First GATE result, no retrain.
#   MODE=train  — DISABLED (credit footgun). Retrain via the credit-safe
#                RunPod runbook: adresses/runpod/RUNPOD_ONETIME_TRAIN.md
#
# Account 017341176694 / us-west-2. SSH in from your IP (SG sg-0e67301a80339fe2b
# allows 22 from your address; update the SG if your IP rotated). Then:
#   git -C code pull && bash code/measure-it-main/training/sam3/aws_g5_run.sh
# ---------------------------------------------------------------------------
set -euo pipefail
MODE=${MODE:-infer}
CODE=${CODE:-$HOME/code}
MEASURE=${MEASURE:-$CODE}                            # repo root: training/ src/ live here (no measure-it-main subdir)
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
  # REMOVED — this branch was a credit footgun: it ran the FULL 40-epoch config
  # (no epoch cap / LR-schedule scaling -> ~13h ~$10+), WITHOUT patch_sam3.py (so
  # the mid-run matcher NaN crash is live) and WITHOUT incremental checkpoint sync
  # (a crash at hour 9 loses all 9). Use the credit-safe RunPod runbook instead.
  echo "ERROR: MODE=train is disabled here — it lacked the NaN patch, epoch cap," >&2
  echo "       and checkpoint sync. Use adresses/runpod/RUNPOD_ONETIME_TRAIN.md" >&2
  echo "       (Stage 0 pre-flight + bounded, resumable, patched pod run)." >&2
  exit 2
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
