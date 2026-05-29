#!/usr/bin/env bash
# scripts/launch.sh
# Elastic launch for moe-engine. Designed for a Kubernetes/Kubeflow
# Training-Operator PyTorchJob or a bare-metal Slurm run.
#
# Required environment:
#   NUM_NODES        e.g. 128
#   GPUS_PER_NODE    e.g. 8
#   RDZV_ENDPOINT    e.g. etcd-headless.train.svc.cluster.local:2379
#   S3_ENDPOINT_URL  (optional, MinIO) e.g. http://minio.train.svc:9000
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for the object store

set -euo pipefail

NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-moe-engine-$(date +%Y%m%d%H%M%S)}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29400}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
CONFIG="${CONFIG:-configs/default.yaml}"

cd "$(dirname "$0")/.."

exec torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --max_restarts="${MAX_RESTARTS}" \
  --rdzv_id="${RDZV_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  --rdzv_conf=timeout=900 \
  train.py --config "${CONFIG}"
