"""End-to-end smoke tests for the training entrypoint.

These exercise the full `train.py` loop on CPU with toy-scale config
overrides (`--smoke`) so they finish in seconds and require no GPU.

We assert that:
  * The training loop completes without exceptions.
  * Structured telemetry is written as newline-delimited JSON with the
    documented envelope keys (loss, mfu, kernel{...}, infra{...}, ...).
  * Async checkpoints land in BOTH the local NVMe tier and the remote
    object-store tier (one test uses `file://`, another uses an
    in-process moto-mocked S3 bucket via `boto3`).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _smoke_yaml(work_dir: Path, remote_uri: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = work_dir / "smoke.yaml"
    cfg.write_text(textwrap.dedent(f"""
        model:
          hidden_dim: 32
          num_layers: 2
          num_experts: 4
          top_k: 2
          capacity_factor: 1.25
          ffn_dim: 64
          vocab_size: 128
          sequence_length: 8
          dtype: float32
        training:
          global_batch_size: 4
          micro_batch_size: 2
          learning_rate: 3.0e-4
          weight_decay: 0.1
          grad_clip: 1.0
          max_steps: 2
          log_interval: 1
          ckpt_interval: 1
          warmup_steps: 0
        parallelism:
          data_parallel: 1
          expert_parallel: 1
          tensor_parallel: 1
          pipeline_parallel: 1
        checkpoint:
          local_dir: {work_dir}/ckpts
          remote_uri: {remote_uri}
          async_workers: 1
          retention: 2
        elastic:
          min_nodes: 1
          max_nodes: 1
          rdzv_backend: c10d
          rdzv_endpoint: localhost:29400
          health_check_interval_s: 5
          drop_grace_period_s: 30
        telemetry:
          log_dir: {work_dir}/logs
          tensorboard_dir: {work_dir}/logs/tb
          json_path: {work_dir}/logs/step.jsonl
          mfu_target: 0.55
          hardware_peak_tflops: 989.0
    """).strip())
    return cfg


def _run_train(cfg_path: Path) -> None:
    """Invoke train.main() in-process so we share any mock contexts."""
    # Fresh import each call so torch/dist global state is rebuilt.
    for mod in list(sys.modules):
        if mod == "train" or mod.startswith("train."):
            del sys.modules[mod]
    sys.argv = ["train.py", "--config", str(cfg_path), "--max-steps", "2", "--smoke"]
    from train import main as train_main
    train_main()


def _assert_telemetry_envelope(jsonl_path: Path) -> None:
    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 2, f"expected >=2 telemetry lines, got {len(lines)}"
    for line in lines:
        rec = json.loads(line)
        for k in ("step", "loss", "mfu", "tokens_per_sec",
                  "kernel", "collective", "memory", "infra",
                  "wall_clock_ms", "rank", "ts"):
            assert k in rec, f"missing key {k!r} in telemetry record"
        # Kernel block must report Triton/CPU-fallback usage.
        assert "used_triton" in rec["kernel"]
        # Infra block must surface async ckpt commit time + cluster size.
        assert "async_ckpt_commit_ms" in rec["infra"]
        assert "active_nodes" in rec["infra"]


def test_smoke_local_file_tier(tmp_path: Path) -> None:
    work = tmp_path / "run"
    remote = tmp_path / "remote"
    cfg = _smoke_yaml(work, remote_uri=f"file://{remote}")
    _run_train(cfg)

    _assert_telemetry_envelope(work / "logs" / "step.jsonl")

    # Both tiers must hold step=1 and step=2 sharded checkpoints + meta.
    for tier in (work / "ckpts", remote):
        files = {p.name for p in tier.rglob("*") if p.is_file()}
        assert any(f.endswith(".pt") for f in files), f"no .pt in {tier}"
        assert any(f.endswith(".meta.json") for f in files), f"no meta in {tier}"


def test_smoke_s3_tier_with_moto(tmp_path: Path) -> None:
    moto = pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    bucket = "moe-engine-ckpts"
    work = tmp_path / "run"
    cfg = _smoke_yaml(work, remote_uri=f"s3://{bucket}/run-smoke/")

    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        _run_train(cfg)

        resp = s3.list_objects_v2(Bucket=bucket)
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        assert any(k.endswith(".pt") for k in keys), f"no .pt in mocked S3: {keys}"
        assert any(k.endswith(".meta.json") for k in keys), \
            f"no meta.json in mocked S3: {keys}"

    _assert_telemetry_envelope(work / "logs" / "step.jsonl")
