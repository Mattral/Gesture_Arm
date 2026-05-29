"""
train.py
========

Unified training entrypoint for the moe-engine. Designed to be launched
via TorchElastic:

    torchrun \
      --nnodes=$NUM_NODES \
      --nproc_per_node=$GPUS_PER_NODE \
      --rdzv_id=moe-run-001 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$RDZV_ENDPOINT \
      --max_restarts=10 \
      train.py --config configs/default.yaml

The script bootstraps a tiny MoE transformer (real architecture, just sized
down by the config) for end-to-end testing of the full stack. For
production runs swap the toy `MoEBlock` below for your real model code –
the surrounding distributed / elastic / telemetry harness is unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from pkg.distributed.parallel_mesh import (
    DistributedMoELayer,
    ParallelTopology,
    build_topology,
    apply_fsdp2,
)
from pkg.elastic.fault_monitor import (
    ElasticConfig,
    ElasticTrainerHarness,
)
from pkg.telemetry.logger import StructuredLogger, StepRecord
from pkg.utils.config import load_config
from pkg.utils.mfu import MFUAccountant, compute_moe_flops


# ----------------------------------------------------------------------
# Tiny test model: stack of (RMSNorm + MoEBlock).
# ----------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        norm = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


class _ToyMoEBlock(nn.Module):
    def __init__(self, model_cfg, topo: ParallelTopology, dtype: torch.dtype):
        super().__init__()
        H = model_cfg["hidden_dim"]
        self.norm = _RMSNorm(H)
        self.moe = DistributedMoELayer(
            hidden_dim=H,
            ffn_dim=model_cfg["ffn_dim"],
            num_experts=model_cfg["num_experts"],
            top_k=model_cfg["top_k"],
            topology=topo,
            capacity_factor=model_cfg["capacity_factor"],
            dtype=dtype,
        )

    def forward(self, x):
        return x + self.moe(self.norm(x))


class _ToyMoEModel(nn.Module):
    def __init__(self, cfg, topo: ParallelTopology):
        super().__init__()
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            cfg["model"]["dtype"]
        ]
        H = cfg["model"]["hidden_dim"]
        self.embed = nn.Embedding(cfg["model"]["vocab_size"], H, dtype=dtype)
        self.blocks = nn.ModuleList([
            _ToyMoEBlock(cfg["model"], topo, dtype) for _ in range(cfg["model"]["num_layers"])
        ])
        self.norm = _RMSNorm(H)
        self.lm_head = nn.Linear(H, cfg["model"]["vocab_size"], bias=False, dtype=dtype)

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=None,
                   help="override training.max_steps for smoke tests")
    p.add_argument("--smoke", action="store_true",
                   help="run a tiny end-to-end smoke pass and exit")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config).raw

    # ----------------------------------------------------------------
    # Process group bootstrap. When running under torchrun, the env vars
    # are set automatically.
    # ----------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size, rank=rank,
        )

    dp = cfg["parallelism"]["data_parallel"]
    ep = cfg["parallelism"]["expert_parallel"]
    # Sanity: if launched with a smaller world than the config expects,
    # collapse dp first then ep so the run still boots (production-friendly).
    if world_size < dp * ep:
        dp = max(1, world_size // ep)
        if dp * ep > world_size:
            ep = max(1, world_size)
            dp = max(1, world_size // ep)

    topology = build_topology(
        dp_size=dp, ep_size=ep,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ----------------------------------------------------------------
    # Toy/test-scale model. For real runs, bring your own model and
    # leave the rest of this scaffold untouched.
    # ----------------------------------------------------------------
    if args.smoke:
        cfg["model"]["hidden_dim"] = 64
        cfg["model"]["num_layers"] = 2
        cfg["model"]["ffn_dim"] = 128
        cfg["model"]["num_experts"] = 4
        cfg["model"]["sequence_length"] = 16
        cfg["model"]["vocab_size"] = 256
        cfg["training"]["micro_batch_size"] = 2

    model = _ToyMoEModel(cfg, topology)
    model = apply_fsdp2(model, topology, mixed_precision_dtype=torch.bfloat16
                       if cfg["model"]["dtype"] == "bfloat16" else None)
    model = model.to(topology.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.95),
    )

    # ----------------------------------------------------------------
    # Telemetry + MFU accountant.
    # ----------------------------------------------------------------
    logger = StructuredLogger(
        json_path=cfg["telemetry"]["json_path"],
        tensorboard_dir=cfg["telemetry"]["tensorboard_dir"],
        rank=topology.rank,
    )
    mfu_acct = MFUAccountant(
        peak_tflops=cfg["telemetry"]["hardware_peak_tflops"],
        mfu_target=cfg["telemetry"]["mfu_target"],
    )
    mfu_acct.configure(
        flops_per_token=compute_moe_flops(
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            ffn_dim=cfg["model"]["ffn_dim"],
            num_experts=cfg["model"]["num_experts"],
            top_k=cfg["model"]["top_k"],
            seq_length=cfg["model"]["sequence_length"],
            batch_tokens=1,
            vocab_size=cfg["model"]["vocab_size"],
        )
    )

    # ----------------------------------------------------------------
    # Elastic harness.
    # ----------------------------------------------------------------
    el_cfg = ElasticConfig(
        local_ckpt_dir=cfg["checkpoint"]["local_dir"],
        remote_uri=cfg["checkpoint"]["remote_uri"],
        s3_endpoint=os.environ.get("S3_ENDPOINT_URL"),
        retention=cfg["checkpoint"]["retention"],
        async_workers=cfg["checkpoint"]["async_workers"],
        health_interval_s=cfg["elastic"]["health_check_interval_s"],
        drop_grace_s=cfg["elastic"]["drop_grace_period_s"],
        min_nodes=cfg["elastic"]["min_nodes"],
    )
    harness = ElasticTrainerHarness(el_cfg, topology)
    harness.install_signal_handlers()

    # Resume from latest checkpoint, if any.
    latest = harness.async_ckpt.latest_step()
    start_step = 0
    if latest is not None:
        harness.async_ckpt.load(model, optimizer, latest, rank=topology.rank)
        start_step = latest + 1
        logging.info("resumed at step %d", start_step)

    # ----------------------------------------------------------------
    # Training loop.
    # ----------------------------------------------------------------
    max_steps = args.max_steps if args.max_steps is not None else cfg["training"]["max_steps"]
    if args.smoke:
        max_steps = min(max_steps, 5)

    B = cfg["training"]["micro_batch_size"]
    S = cfg["model"]["sequence_length"]
    V = cfg["model"]["vocab_size"]
    H = cfg["model"]["hidden_dim"]

    for step in range(start_step, max_steps):
        mfu_acct.start_step()

        ids = torch.randint(0, V, (B, S), device=topology.device)
        targets = torch.randint(0, V, (B, S), device=topology.device)

        # Forward
        logits = model(ids)                             # [B, S, V]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V).float(), targets.view(-1),
        )

        # Backward + optimizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        optimizer.step()

        mfu_res = mfu_acct.end_step(tokens=B * S)

        # Telemetry envelope.
        rec = StepRecord(
            step=step,
            loss=float(loss.detach().item()),
            mfu=mfu_res.mfu,
            tokens_per_sec=mfu_res.tokens_per_sec,
            wall_clock_ms=mfu_res.step_ms,
            kernel={},
            collective={},
            memory={},
            infra={
                "async_ckpt_commit_ms": harness.async_ckpt.last_commit_ms,
                "active_nodes": topology.world_size,
                "ep_world_size": topology.ep_size,
            },
        )
        # Pull last router profile from the first MoE block (best-effort).
        try:
            first_router = model.blocks[0].moe.router
            if first_router.last_profile is not None:
                rec.kernel = {
                    "sram_bytes_per_block": first_router.last_profile.sram_bytes_per_block,
                    "achieved_bw_gbps": first_router.last_profile.achieved_bandwidth_gbps,
                    "tokens_per_expert_mean": first_router.last_profile.tokens_per_expert_mean,
                    "tokens_per_expert_std": first_router.last_profile.tokens_per_expert_std,
                    "used_triton": first_router.last_profile.used_triton,
                }
        except (AttributeError, IndexError):
            pass

        if step % cfg["training"]["log_interval"] == 0:
            logger.emit(rec)

        if step > 0 and step % cfg["training"]["ckpt_interval"] == 0:
            harness.checkpoint(model, optimizer, step)

        if step > 0 and step % 50 == 0:
            dead = harness.health_check()
            if dead:
                logging.warning("rank drop detected: %s; entering recovery", dead)
                topology = harness.recover(model, optimizer,
                                           num_experts=cfg["model"]["num_experts"])

    # ----------------------------------------------------------------
    # Shutdown.
    # ----------------------------------------------------------------
    harness.checkpoint(model, optimizer, max_steps)
    harness.shutdown()
    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
