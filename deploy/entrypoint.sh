#!/usr/bin/env bash
# Container startup: pull models from S3 (via the instance IAM role — no keys,
# no Hugging Face logins), then serve.
set -euo pipefail

BUCKET="${MODEL_BUCKET:?set MODEL_BUCKET, e.g. measure-it-prod-017341176694}"
CKPT_KEY="${MODEL_CKPT_KEY:-models/sam3_roof_ft_boxonly_ep9.pt}"

mkdir -p /opt/models /data/jobs

if [ ! -f "$MEASURE_IT_CKPT" ]; then
  echo "==> fetching fine-tuned checkpoint s3://$BUCKET/$CKPT_KEY"
  aws s3 cp "s3://$BUCKET/$CKPT_KEY" "$MEASURE_IT_CKPT" --only-show-errors
fi

# Base SAM 3 weights: prefer the S3-cached HF cache (uploaded once at first
# boot per docs/DEPLOY_AWS.md 6b). If absent AND HF_TOKEN is set, the first
# model build will download from HF and we push the cache back for next time.
if aws s3 ls "s3://$BUCKET/models/hf/" >/dev/null 2>&1; then
  echo "==> syncing base-model cache from s3://$BUCKET/models/hf/"
  aws s3 sync "s3://$BUCKET/models/hf/" "$HF_HOME" --only-show-errors
else
  echo "==> no models/hf/ cache in S3 yet."
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "    HF_TOKEN present — first run will download from Hugging Face;"
    echo "    afterwards run: aws s3 sync $HF_HOME s3://$BUCKET/models/hf/"
  else
    echo "    WARNING: no cache and no HF_TOKEN — base model load will fail."
  fi
fi

echo "==> starting API"
exec uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --workers 1
