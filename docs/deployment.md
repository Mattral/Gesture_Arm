# Deployment

This document describes how to deploy moe-engine for distributed training.
It is based on the actual cluster launch helper and the supported runtime
configuration in the repository.

## Supported deployment modes

### Local / development

- Single-process CPU or GPU using `torchrun --standalone`.
- Useful for regression, debugging, and smoke verification.

### Multi-node cluster

- Multi-node training is launched via `torchrun` with a rendezvous endpoint.
- The project provides `moe-engine/scripts/launch.sh` as a simple startup script.

### Kubernetes / bare-metal

- `scripts/launch.sh` is compatible with both Kubernetes and bare-metal cluster
  deployments. It expects the environment variables listed below.
- `scripts/simulate_node_failure.sh` can be used for failure-injection during
  Kubernetes or local rank-based testing.

## Deployment requirements

### Runtime dependencies

- Python 3.11+ (matching `moe-engine/pyproject.toml`)
- `torch>=2.5.0`
- `triton>=3.0.0`
- `numpy>=1.26.0`
- `boto3` and `botocore` when using S3/MinIO remote checkpointing

### Infrastructure dependencies

- A shared rendezvous endpoint for distributed training.
  `RDZV_ENDPOINT` should be reachable from all participating nodes.
- Optional object storage for remote checkpoint mirrors.
  `S3_ENDPOINT_URL` is supported by the code and tests.

## Recommended deployment environment variables

The `scripts/launch.sh` helper expects the following variables:

- `NUM_NODES`
- `GPUS_PER_NODE`
- `RDZV_ID`
- `RDZV_ENDPOINT`
- `MAX_RESTARTS`
- `CONFIG`

For object storage:

- `S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

For chaos or failure injection tests:

- `CHAOS_LATENCY_STEP`
- `CHAOS_LATENCY_SECONDS`

## Launch example

```bash
NUM_NODES=32 \
GPUS_PER_NODE=8 \
RDZV_ENDPOINT=head-node:29500 \
RDZV_ID=moe-engine-run-001 \
CONFIG=configs/default.yaml \
S3_ENDPOINT_URL=http://minio.train.svc:9000 \
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
bash scripts/launch.sh
```

This wraps a `torchrun` invocation and applies the cluster launch parameters.

## Rendezvous backends

- `c10d` is the default backend used by `scripts/launch.sh`.
- The code also supports `etcd` for larger-scale runs when configured.
  `moe-engine/pkg/elastic/fault_monitor.py` initializes etcd rendezvous when
  the backend is available.

## Checkpoint deployment

- `checkpoint.local_dir` is the local NVMe staging location.
- `checkpoint.remote_uri` can be `s3://...` or `file://...`.
- The elastic harness mirrors local checkpoints to remote storage for durability.

## Operational guidance

- Deploy with private networking and restricted access to rendezvous and
  object storage endpoints.
- Keep object storage credentials out of source control.
- Use the smoke configuration (`configs/smoke.yaml`) for deployment validation
  before moving to full-scale runs.

## Failure injection

- `scripts/simulate_node_failure.sh` can kill ranks or Kubernetes pods.
- This script is intended to validate the elastic recovery behavior implemented
  in the repository.

## Limitations

- There is no container image provided by the repo.
- The deployment model assumes a Python runtime with appropriate hardware
  drivers and PyTorch/CUDA support already installed.
