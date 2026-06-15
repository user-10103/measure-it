#!/usr/bin/env bash
# EC2 fleet launcher for the 3DBAG pipeline.
#
# HOW TO USE
# ──────────
# Fill in the 6 variables below, then run from your laptop:
#
#   bash training/ec2_3dbag.sh
#
# This launches 4 × c5n.xlarge instances in parallel.
# Each processes 7–8 Dutch cities (30 cities ÷ 4 shards).
# All shards upload chips to the SAME S3 bucket (no collisions — filenames
# are sha1(building_id)).  Each shard writes its own _annotations_shard{N}.coco.json.
#
# When all 4 finish, pull the COCO JSONs:
#   aws s3 cp s3://$S3_BUCKET/3dbag/ training/3dbag_dataset/ --recursive
#
# Then merge shards:
#   python training/build_3dbag_dataset_merge_shards.py \
#       --input training/3dbag_dataset --output training/3dbag_dataset
#
# Cost: 4 × c5n.xlarge × $0.22/hr × ~1 hr ≈ $0.88
# Timing: ~1–3 hours for 20 000 chips total (5 000 per shard)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Fill these in ──────────────────────────────────────────────────────────────
GH_TOKEN=""                     # GitHub personal access token (repo:read)
AWS_ACCESS_KEY_ID=""            # New AWS account key
AWS_SECRET_ACCESS_KEY=""        # New AWS account secret
AWS_REGION="us-east-1"
S3_BUCKET="florida-roofs-v4"    # Your new S3 bucket (will be created if missing)
KEY_PAIR_NAME=""                # EC2 key pair name for SSH access
# ──────────────────────────────────────────────────────────────────────────────

NUM_SHARDS=4
TARGET_PER_SHARD=5000           # 4 × 5000 = 20 000 total chips
WORKERS=32                      # Matches c5n.xlarge vCPU × 8 (I/O bound)
INSTANCE_TYPE="c5n.xlarge"      # 4 vCPU, 10.5 Gbps network — good for API-heavy work

# Amazon Linux 2023 in us-east-1 (fast startup, has python3 + pip)
AMI_ID="ami-0c02fb55956c7d316"

# Security group must allow outbound HTTPS (443) for API calls.
# Replace with your SG ID, or use the default SG.
SG_ID="default"

echo "Launching $NUM_SHARDS × $INSTANCE_TYPE for 3DBAG pipeline …"
echo "  Target: $((NUM_SHARDS * TARGET_PER_SHARD)) total chips"
echo "  Bucket: s3://$S3_BUCKET"
echo ""

# Create the bucket if it doesn't exist
aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null || {
    echo "Creating bucket s3://$S3_BUCKET …"
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION"
    else
        aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION" \
            --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
}

INSTANCE_IDS=()

for SHARD_IDX in $(seq 0 $((NUM_SHARDS - 1))); do
    # Build the user-data script for this shard
    USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -euo pipefail
LOG=/home/ec2-user/3dbag_shard_${SHARD_IDX}.log
exec >"\$LOG" 2>&1

echo "=== Shard ${SHARD_IDX}/${NUM_SHARDS} start \$(date) ==="

# Install deps
yum install -y git python3 python3-pip 2>/dev/null || \
    (apt-get update -qq && apt-get install -y git python3 python3-pip)
pip3 install -q requests shapely pyproj Pillow tqdm boto3 numpy

# AWS creds
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_REGION}"

# Clone repo
cd /home/ec2-user
rm -rf measure-it
git clone --branch classical-coord-pipeline \
    "https://${GH_TOKEN}@github.com/user-10103/measure-it.git" measure-it
cd measure-it
export PYTHONPATH=/home/ec2-user/measure-it

# Run pipeline
python3 training/build_3dbag_dataset.py \
    --output   /home/ec2-user/3dbag_shard_${SHARD_IDX} \
    --target   ${TARGET_PER_SHARD} \
    --workers  ${WORKERS} \
    --s3-bucket "${S3_BUCKET}" \
    --s3-prefix "3dbag" \
    --num-shards ${NUM_SHARDS} \
    --shard-idx  ${SHARD_IDX} \
    --resume

# Push COCO metadata to S3
aws s3 cp /home/ec2-user/3dbag_shard_${SHARD_IDX}/train/ \
    "s3://${S3_BUCKET}/3dbag/train/" \
    --exclude "*.png" --recursive
aws s3 cp /home/ec2-user/3dbag_shard_${SHARD_IDX}/valid/ \
    "s3://${S3_BUCKET}/3dbag/valid/" \
    --exclude "*.png" --recursive

echo "=== Shard ${SHARD_IDX} done \$(date) ==="
# Auto-terminate to stop billing
aws ec2 terminate-instances \
    --region "${AWS_REGION}" \
    --instance-ids "\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
USERDATA
)

    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$AWS_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_PAIR_NAME" \
        --security-groups "$SG_ID" \
        --user-data "$USER_DATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=3dbag-shard-${SHARD_IDX}},{Key=Project,Value=measure-it}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    INSTANCE_IDS+=("$INSTANCE_ID")
    echo "  Shard $SHARD_IDX → $INSTANCE_ID launched"
done

echo ""
echo "All $NUM_SHARDS shards launched: ${INSTANCE_IDS[*]}"
echo ""
echo "Monitor progress:"
echo "  watch -n 30 'aws ec2 describe-instances --instance-ids ${INSTANCE_IDS[*]} \\
      --query \"Instances[].{ID:InstanceId,State:State.Name}\" --output table'"
echo ""
echo "When all instances terminate (auto self-terminate on completion),"
echo "pull COCO metadata and merge shards:"
echo ""
echo "  mkdir -p training/3dbag_dataset/train training/3dbag_dataset/valid"
echo "  aws s3 cp s3://$S3_BUCKET/3dbag/ training/3dbag_dataset/ --recursive --exclude '*.png'"
echo "  python training/merge_3dbag_shards.py \\"
echo "      --input training/3dbag_dataset \\"
echo "      --num-shards $NUM_SHARDS"
