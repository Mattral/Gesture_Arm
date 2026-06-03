# Quickstart

Get started with moe-engine using the repository's installed package and
smoke-run workflow.

## 1. Install dependencies

```bash
cd moe-engine
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

If you want remote checkpointing support, install the S3/MinIO extras:

```bash
pip install boto3 botocore
```

## 2. Verify the Python environment

```bash
python -c "import torch, triton; print(torch.__version__, triton.__version__)"
```

## 3. Run the smoke demo

Use the small smoke config and the `--smoke` flag to exercise the full
entrypoint in a few steps:

```bash
python train.py --config configs/smoke.yaml --smoke
```

This run executes a tiny MoE transformer and validates the training,
checkpoint, and telemetry path.

## 4. Run the unit test suite

From `moe-engine`:

```bash
pytest -m "not chaos" -v
```

If you want to validate the Triton router numerics separately:

```bash
python tests/run_numerics_tests.py
```

## 5. Run a local multi-rank regression

For local Gloo-based regression and chaos test coverage on a single machine,
set the loopback interface:

```bash
GLOO_SOCKET_IFNAME=lo pytest -m chaos -v -k "baseline or scenario_b"
```

## 6. Inspect telemetry

The smoke config writes logs and metrics to `/tmp/moe-engine`.
You can open the TensorBoard log dir after the run:

```bash
tensorboard --logdir /tmp/moe-engine/logs/tb
```

## 7. Next step: cluster launch

When you are ready for multi-node testing, use `scripts/launch.sh` and a
`torchrun`-compatible rendezvous endpoint.

Example:

```bash
NUM_NODES=2 \
GPUS_PER_NODE=8 \
RDZV_ENDPOINT=head-node:29500 \
CONFIG=configs/default.yaml \
bash scripts/launch.sh
```

For a fast start, the smoke run path and unit tests are the recommended first
validation steps.
