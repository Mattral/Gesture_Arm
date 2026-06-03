# Setup and Operations

This document describes how to install, configure, and operate moe-engine
using the code and scripts that exist in this repository.

## Install

1. Clone the project and enter the package root:
   ```bash
   git clone <this-repo> moe-engine
   cd moe-engine
   ```

2. Create a Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install runtime dependencies:
   ```bash
   pip install -U pip wheel
   pip install -r requirements.txt
   ```

4. Optional components:
   - GPU-only Triton kernels: `pip install triton==3.*`
   - S3/MinIO remote checkpoint mirror: `pip install boto3 botocore`

5. Verify the install:
   ```bash
   python -c "import torch, triton; print(torch.__version__, triton.__version__)"
   ```

## Configuration

`moe-engine/train.py` reads YAML config files via `pkg.utils.config.load_config`.
The repository ships two canonical configs:

- `configs/default.yaml`: hyperscale default settings.
- `configs/smoke.yaml`: minimal dimensions for laptop/CI smoke runs.

Important config sections:

- `model`: hidden_dim, num_layers, num_experts, top_k, ffn_dim, vocab_size,
  sequence_length, dtype.
- `training`: micro_batch_size, learning_rate, weight_decay, max_steps,
  log_interval, ckpt_interval, warmup_steps.
- `parallelism`: data_parallel, expert_parallel, tensor_parallel,
  pipeline_parallel.
- `checkpoint`: local_dir, remote_uri, async_workers, retention.
- `elastic`: min_nodes, max_nodes, rdzv_backend, rdzv_endpoint,
  health_check_interval_s, drop_grace_period_s.
- `telemetry`: log_dir, tensorboard_dir, json_path, mfu_target,
  hardware_peak_tflops.

### Parallelism constraints

The runtime enforces that the topology product corresponds to `WORLD_SIZE`.
Example from `configs/default.yaml`:

- `data_parallel: 8`
- `expert_parallel: 8`
- `tensor_parallel: 1`
- `pipeline_parallel: 1`

For a run with 64 total ranks, these axes must multiply to the launched world
size.

## Local development and regression

### Run the test suite

From `moe-engine`:

```bash
pytest -m "not chaos" -v
```

This runs the primary non-chaos regression tests on a laptop or CI worker.

### Smoke end-to-end run

A minimal local verification uses the smoke config and the `--smoke` flag:

```bash
python train.py --config configs/smoke.yaml --smoke
```

This exercises the full runtime path in a tiny model and training loop.

### CPU/Gloo regression mode

The repository is designed to run cleanly on a single rank without GPUs.
Use `GLOO_SOCKET_IFNAME=lo` to exercise local multi-rank Gloo-based chaos tests.

## Multi-GPU / cluster operation

### Single-node multi-GPU launch

Example for an 8-GPU node:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py --config configs/default.yaml
```

### Elastic launch helper

`moe-engine/scripts/launch.sh` wraps `torchrun` and is designed for
multi-node clusters.

Required environment variables:

- `NUM_NODES`
- `GPUS_PER_NODE`
- `RDZV_ENDPOINT`
- `RDZV_ID`
- `MAX_RESTARTS`
- `CONFIG`

Example:

```bash
NUM_NODES=32 \
GPUS_PER_NODE=8 \
RDZV_ENDPOINT=head-node:29500 \
RUN_ID=moe-run-001 \
bash scripts/launch.sh
```

The launcher uses `--rdzv_backend=c10d` and `--rdzv_conf=timeout=900`.

## Elastic checkpointing and recovery

`moe-engine/pkg/elastic/fault_monitor.py` implements the elastic harness.
Runtime behavior includes:

- `AsyncCheckpointer` streaming sharded checkpoints from local NVMe to remote
  storage.
- retention pruning controlled by `checkpoint.retention`.
- optional `S3Adapter` mirror support for `remote_uri` values beginning with
  `s3://`.
- local file fallback when `remote_uri` begins with `file://`.

The elastic harness can resume training from the latest available checkpoint.
This behavior is exercised by `moe-engine/tests/test_elastic.py` and
`moe-engine/tests/test_smoke_e2e.py`.

## Chaos and failure simulation

`moe-engine/scripts/simulate_node_failure.sh` is a helper for chaos testing.
It can SIGKILL local ranks or delete selected Kubernetes pods to verify that
surviving workers re-rendezvous and recover.

Usage examples:

- Kill random pods:
  ```bash
  ./scripts/simulate_node_failure.sh
  ```
- Kill specific ranks locally:
  ```bash
  ./scripts/simulate_node_failure.sh -r 4,5,6,7
  ```

## Monitoring and telemetry

Training emits structured telemetry into the directories defined by
`telemetry.tensorboard_dir` and `telemetry.json_path`.

- JSON step logs are written to `json_path`.
- TensorBoard summaries are written to `tensorboard_dir`.

## Operational notes

- Use environment variables rather than hardcoding secrets in configs.
- If using S3/MinIO, set `S3_ENDPOINT_URL` and credential environment variables.
- The `RDZV_ENDPOINT` may be an etcd service or a head node address.
- For production, prefer private networking and limited access to the object
  store and rendezvous endpoints.
