# `moe-engine` &nbsp;&middot;&nbsp; A Composed Mixture-of-Experts Engine

> **Production-grade sparse MoE training runtime.**
> Designed to keep large-scale pre-training jobs alive end-to-end:
> sparse Top-K routing in custom Triton, DP+EP distributed training with TP
> support in core layers and PP work in progress, on PyTorch 2.12+,
> asynchronous sharded checkpointing through a two-tier (NVMe → S3 / MinIO)
> durable store, and a TorchElastic state-machine that evicts dead ranks,
> reshards experts, and hot-resumes training without operator intervention.

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-3.x-9333ea.svg)](https://triton-lang.org/)

---

## Table of Contents
1. [Why this exists](#1-why-this-exists)
2. [System architecture](#2-system-architecture)
3. [Hardware & software requirements](#3-hardware--software-requirements)
4. [Installation](#4-installation)
5. [Local CPU / Gloo regression workflow](#5-local-cpu--gloo-regression-workflow)
6. [Cluster-scale multi-GPU training](#6-cluster-scale-multi-gpu-training)
7. [Configuration reference](#7-configuration-reference)
8. [Mathematical invariants & CI gates](#8-mathematical-invariants--ci-gates)
9. [Telemetry envelope](#9-telemetry-envelope)
10. [Fault-injection / chaos workflow](#10-fault-injection--chaos-workflow)
11. [Repository layout](#11-repository-layout)
12. [License](#12-license)

---

## 1. Why this exists

At frontier-lab scale three engineering disciplines that are normally separate
teams must instead be co-designed in a single repository:

| Layer | Concern | This repo's contribution |
|------|---------|---------------------------|
| **Hardware-aware kernels** | Memory coalescing, SRAM tiling, Tensor-Core feeding for sparse Top-K routing | `pkg/kernels/moe_router.py` — Triton forward + (planned) backward, dynamic-bound masking, 128-byte aligned loads |
| **Distributed runtime** | DP+EP training with TP support in core layers, FSDP2 DTensor sharding, EP `all_to_all_single` overlapped with compute | `pkg/distributed/parallel_mesh.py` — `init_device_mesh((dp, ep))` with TP axis reserved, dedicated comm stream overlap, dedicated CUDA streams |
| **Fault-tolerant infra** | Async pinned-memory checkpointing, S3/MinIO mirror, evict→reshard→reload state-machine | `pkg/elastic/fault_monitor.py` — TorchElastic harness, `SHARDED_STATE_DICT`, signal-driven flush |

`moe-engine` keeps these three layers in one binary so an MFU regression or a
checkpoint-stall bug can be isolated to a single line, not a six-team incident.

---

## 2. System architecture

```
                                 ┌─────────────────────────────────────────────┐
                                 │              train.py  (entrypoint)         │
                                 │  argparse → load_config → build_topology    │
                                 └──────────────┬──────────────────────────────┘
                                                │
            ┌───────────────────────────────────┼────────────────────────────────────┐
            │                                   │                                    │
            ▼                                   ▼                                    ▼
┌────────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────┐
│ pkg/distributed/           │   │ pkg/elastic/                 │   │ pkg/kernels/                 │
│   parallel_mesh.py         │   │   fault_monitor.py           │   │   moe_router.py              │
│                            │   │                              │   │                              │
│ • ParallelTopology         │   │ • ElasticTrainerHarness      │   │ • MoERouter (nn.Module)      │
│ • init_device_mesh((dp,ep))│   │ • AsyncCheckpointer          │   │   ├─ Triton fused forward /  │
│   with TP axis reserved    │   │ • _PinnedHostStager          │   │   │     autograd backward    │
│ • DistributedMoELayer      │   │ • ClusterStateMachine        │   │   │   - dynamic-bound mask   │
│ • apply_fsdp2(...)         │   │ • LocalNVMeAdapter           │   │   │   - 128B aligned loads   │
│ • all_to_all_dispatch      │   │ • S3Adapter (boto3)          │   │   └─ CPU fallback path       │
│   on dedicated comm stream │   │                              │   │                              │
└───────────┬────────────────┘   └─────────────┬────────────────┘   └──────────────┬───────────────┘
            │                                  │                                   │
            │   DeviceMesh sub-meshes          │   pinned-host snapshot queue      │  routing tokens
            │   ("pp","dp","ep","tp")          │                                   │  + gating weights
            │                                  │                                   │
            ▼                                  ▼                                   ▼
   ┌────────────────────┐         ┌───────────────────────────┐         ┌──────────────────────┐
   │ NCCL / Gloo        │         │ tier-1 NVMe (staging)     │         │ Triton runtime       │
   │ process groups     │         │ tier-2 S3 / MinIO mirror  │         │ (CUDA / ROCm)        │
   │ (one per axis)     │         │ background I/O thread×N   │         │                      │
   └────────────────────┘         └───────────┬───────────────┘         └──────────────────────┘
                                              │
                                              ▼
                                  ┌───────────────────────────┐
                                  │ TorchElastic agent        │
                                  │ (rdzv: c10d / etcd)       │
                                  │ rendezvous → restart loop │
                                  └───────────────────────────┘

                       Data-flow per training step
                       ───────────────────────────
   ids → embed → (TP shard) → block_0 ── … ── block_N → norm → lm_head → loss
                                  │
                                  ▼
                    DistributedMoELayer.forward
                    ┌─────────────────────────────────────────┐
                    │ 1. router (Triton fwd)                  │
                    │ 2. sort by target EP rank               │
                    │ 3. all_to_all_single                    │
                    │    on a dedicated comm stream ──► launch  ────────┐
                    │ 4. independent compute  ─── overlap ───►│  GPU compute
                    │ 5. work.wait() on dispatch              │  in flight
                    │ 6. local SwiGLU experts                 │
                    │ 7. all_to_all_combine on a dedicated comm stream ──┘
                    │ 8. weight ⊗ combine → reduce-K         │
                    └─────────────────────────────────────────┘
```

Per-rank a coordinate identifies its mesh slice. Sub-meshes are obtained by name:
`mesh["dp"]` (for FSDP2 sharding), `mesh["ep"]` (for `all_to_all_single`),
`mesh["tp"]` (for TP layer sharding when enabled), and `pp` support is reserved
for future pipeline stage mapping.

---

## 3. Hardware & software requirements

### Software

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Python | 3.10 | 3.11 | |
| PyTorch | 2.5 | 2.12+ | `init_device_mesh`, FSDP2 (`fully_shard`), DCP |
| Triton | 3.0 | 3.x latest | required for GPU forward kernel |
| CUDA | 12.1 | 12.4+ | for H100/H200/B200 BF16 paths |
| NCCL | 2.20 | 2.21+ | needed for `TORCH_NCCL_ASYNC_ERROR_HANDLING` |
| `boto3` | 1.34 | latest | only if streaming to S3/MinIO |
| `moto` | 5.x | latest | local S3 mock for the chaos suite |

### Hardware

| Profile | GPUs | Interconnect | Notes |
|---------|------|--------------|-------|
| **Smoke / CI** | none (CPU + Gloo) | localhost loopback | full unit + integration suite |
| **Single-node dev** | 1× H100 80GB | PCIe Gen5 | `world=1` degenerate path |
| **Pod (one node)** | 8× H100 SXM5 | NVLink 4 | TP across the NVLink island, EP within node |
| **Cluster** | 256–10 240 H100 | NVLink + InfiniBand 400G | TP intra-node (size 8), PP inter-node, DP via FSDP2, EP across all GPUs |

The default config (`configs/default.yaml`) targets H100 SXM5 with a peak of
989 TFLOP/s BF16. Override `telemetry.hardware_peak_tflops` for B200/MI300X.

---

## 4. Installation

```bash
git clone <this-repo> moe-engine && cd moe-engine

# Recommended: a fresh venv / conda env with python 3.11
python -m venv .venv && source .venv/bin/activate

pip install -U pip wheel
pip install -r requirements.txt

# Optional: GPU-only Triton kernels
pip install triton==3.*           # already pinned in requirements.txt for cu12

# Optional: S3/MinIO mirror
pip install boto3 botocore
```

Verify the install:

```bash
python -c "import torch, triton; print(torch.__version__, triton.__version__)"
```

---

## 5. Local CPU / Gloo regression workflow

Every code path in this repo no-ops cleanly on a 1-rank world. You can run
the **entire** non-chaos test suite on a laptop:

```bash
# Unit + integration tests, ~20 s on a modern laptop
pytest -m "not chaos" -v

# Single-rank end-to-end smoke (5 training steps, toy 64-d model)
python train.py --config configs/smoke.yaml --smoke
```

To exercise the elastic / chaos suite with **simulated** multi-rank Gloo
(spawned as subprocesses on localhost):

```bash
# Baseline + Scenario B (storage stall). Scenario A (sudden node failure)
# is currently quarantined — tracked in `roadmap.md` TD-05.
GLOO_SOCKET_IFNAME=lo pytest -m chaos -v -k "baseline or scenario_b"
```

---

## 6. Cluster-scale multi-GPU training

### 6.1 Single-node, 8× GPU

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py --config configs/default.yaml
```

### 6.2 Multi-node TorchElastic (rendezvous via c10d)

Set the same `RDZV_ENDPOINT` on every node (typically the head node's IP+port
or an etcd cluster). The launcher script `scripts/launch.sh` wraps this:

```bash
# On every node:
NUM_NODES=32 \
GPUS_PER_NODE=8 \
RDZV_ENDPOINT=head-node:29500 \
RUN_ID=moe-run-001 \
bash scripts/launch.sh
```

The launcher injects the NCCL fail-fast environment (see TD-03 in
`roadmap.md`) and points the elastic agent at `train.py`. Workers can be
added or removed mid-run; surviving ranks reshard experts and hot-resume from
the most recent async checkpoint.

### 6.3 Topology selection

`configs/default.yaml::parallelism` must satisfy
`tensor_parallel · pipeline_parallel · data_parallel · expert_parallel = WORLD_SIZE`.

Example for 256 GPUs (32 nodes × 8 H100):
- `tensor_parallel: 8` &nbsp;(intra-node, NVLink)
- `pipeline_parallel: 4` &nbsp;(inter-node, IB)
- `data_parallel: 4` &nbsp;(FSDP2 across remaining axis)
- `expert_parallel: 2`

The mesh constructor enforces this product equality; missized configs fail
fast at boot rather than mid-step.

---

## 7. Configuration reference

`configs/default.yaml` is the source of truth; `configs/smoke.yaml` shrinks
every dimension for laptop runs. Every block:

```yaml
model:           # transformer hyperparameters
parallelism:     # topology axes, must product to WORLD_SIZE
training:        # batch sizes, optimizer, schedule
checkpoint:      # local NVMe dir, remote URI, async workers, retention
elastic:         # rendezvous, heartbeat interval, drop grace, min_nodes
telemetry:       # log dirs, MFU target, hardware peak TFLOPs
```

---

## 8. Mathematical invariants & CI gates

| Invariant | Statement | Tested in |
|-----------|-----------|-----------|
| Mesh shape | `dp_size · ep_size · tp_size = world_size` for active axes | `moe-engine/tests/test_distributed.py`, `moe-engine/tests/test_distributed_invariants.py` |
| Token conservation | `Σ_r dispatched_r = B·S·K` | `moe-engine/tests/test_distributed.py`, `moe-engine/tests/test_distributed_invariants.py` |
| Numerical tolerance | `atol < 1e-5`, `rtol < 1e-5` (fp64 reference) | `moe-engine/tests/test_kernels.py` |
| Checksum identity | `hash(state_dict_post_load) == hash(state_dict_pre_save)` | `moe-engine/tests/test_elastic.py` |
| Monotonic progression | `step_{n+1} > step_n` across every restart generation | `moe-engine/tests/test_chaos.py` |
| Comm-compute overlap | EP dispatch/combine use a dedicated CUDA stream and event synchronization | `moe-engine/pkg/distributed/parallel_mesh.py::DistributedMoELayer.forward` |
| Async-ckpt zero-leak | After `harness.checkpoint()`, no device-resident references survive into the writer thread queue | `moe-engine/tests/test_elastic.py::test_async_ckpt_no_device_refs` |
| MFU target | `>= 0.55` of theoretical peak BF16 | `moe-engine/pkg/utils/mfu.py` |
| Dynamic MoE FLOP | `FLOPs_step = 2·T_active·P_dense + 2·T_routed·P_experts` | `moe-engine/pkg/utils/mfu.py` (TD-04) |

CI fails on violation of any of the above.

---

## 9. Telemetry envelope

Every training step emits one structured JSON record (also fanned out to
TensorBoard via `pkg/telemetry/logger.py`):

```json
{
  "step": 1024,
  "loss": 2.187,
  "mfu": 0.612,
  "tokens_per_sec": 184320,
  "wall_clock_ms": 412.3,
  "kernel": {
    "sram_bytes_per_block": 49152,
    "achieved_bw_gbps": 1843.0,
    "tokens_per_expert_mean": 256.4,
    "tokens_per_expert_std": 18.7,
    "used_triton": true
  },
  "collective": {
    "all_to_all_dispatch_ms": 0.87,
    "all_to_all_combine_ms": 0.91
  },
  "memory": {
    "peak_allocated_gb": 62.4,
    "reserved_gb": 70.0,
    "leak_delta_gb": 0.0
  },
  "infra": {
    "async_ckpt_commit_ms": 412.0,
    "active_nodes": 1250,
    "ep_world_size": 64
  }
}
```

---

## 10. Fault-injection / chaos workflow

The chaos suite spawns 4 Gloo workers as subprocesses on localhost and
exercises the full TorchElastic restart loop:

```bash
GLOO_SOCKET_IFNAME=lo pytest -m chaos -v
```

Scenarios:

| Scenario | What it injects | What it verifies |
|----------|-----------------|------------------|
| baseline | nothing | end-to-end correctness, monotonic step progression |
| A: sudden node failure | `SIGKILL` to one worker mid-step | TorchElastic agent re-rendezvous, surviving ranks reshard experts, training resumes from last checkpoint, **invariant: post-restart `step > step_at_kill`** |
| B: storage stall | injected 5-second `time.sleep` inside the storage adapter | async writer queue drains in background, training step never blocks, ckpt eventually commits |

> **Status, turn-001:** baseline + B pass; A is quarantined (TD-05 in
> `roadmap.md`) pending the Item #5 (uneven reshard) and Item #6
> (NCCL watchdog) deliveries.

---

## 11. Repository layout

```
README.md                          ← this file
docs/                              ← repository documentation
moe-engine/                        ← python package root
├── README.md                      ← package-specific README
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.yaml                ← cluster-scale config
│   └── smoke.yaml                  ← laptop CPU smoke
├── pkg/
│   ├── kernels/
│   │   └── moe_router.py           ← Triton Top-K routing kernel
│   ├── distributed/
│   │   └── parallel_mesh.py        ← DP+EP device mesh, TP layer sharding, PP shim
│   ├── elastic/
│   │   └── fault_monitor.py        ← AsyncCkpt + pinned staging + state-machine
│   ├── telemetry/
│   │   └── logger.py
│   └── utils/
│       ├── config.py
│       └── mfu.py
├── tests/
│   ├── test_kernels.py
│   ├── test_distributed.py
│   ├── test_elastic.py
│   ├── test_smoke_e2e.py
│   ├── test_chaos.py               ← TorchElastic chaos driver (4-rank Gloo)
│   └── _chaos_worker.py            ← subprocess entry-point
├── scripts/
│   ├── launch.sh                   ← TorchElastic multi-node launcher
│   └── simulate_node_failure.sh    ← drop N nodes mid-run
└── train.py                        ← unified training loop
```

---

## 12. License

Apache 2.0. See `LICENSE`.
