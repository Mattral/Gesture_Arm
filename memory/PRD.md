# moe-engine — Product Requirements Document

> Custom Mixture-of-Experts (MoE) Distributed Training Engine.
> Pure backend ML engine — **no** UI, **no** web server, **no** REST API.

## 1. Original problem statement

Write a fully comprehensive, production-ready, modular monorepo for a Custom
Mixture-of-Experts (MoE) Distributed Training Engine named **moe-engine**.
The engine must explicitly bridge:

1. **Hardware-Aware Kernel Optimization** — custom Triton kernel for Top-K
   routing with a CPU fallback for tests / GPU-less environments.
2. **3D Distributed Parallelism** — native PyTorch 2.5+ implementation built
   on `DeviceMesh`, `FSDP2`, and `DTensor`.
3. **Resilient Infrastructure** — elastic auto-recovery via TorchElastic and
   asynchronous checkpointing streaming local NVMe → S3 / MinIO.
4. **Pure Backend Engine** — structured JSON logging plus CLI / TensorBoard
   exporters only. No HTTP servers, no UIs.

## 2. Architecture overview

```
moe-engine/
├── configs/
│   ├── default.yaml         # hyperscale defaults (H100, 989 TFLOPs peak)
│   └── smoke.yaml           # toy CPU-only config used by launch.sh + tests
├── pkg/
│   ├── kernels/moe_router.py    # Triton Top-K router + CPU fallback
│   ├── distributed/parallel_mesh.py  # DeviceMesh, FSDP2 wrap, DistributedMoELayer
│   ├── elastic/fault_monitor.py # AsyncCheckpointer, ClusterStateMachine, Harness
│   ├── telemetry/logger.py      # NDJSON + TensorBoard sink (StepRecord envelope)
│   └── utils/{config.py,mfu.py} # YAML loader + MFU accountant
├── scripts/
│   ├── launch.sh                # torchrun elastic launcher
│   └── simulate_node_failure.sh # chaos test driver
├── tests/                       # 22 pytest tests (kernel, dist, elastic, smoke-e2e)
└── train.py                     # unified entrypoint (`--config`, `--smoke`, `--max-steps`)
```

### 2.1 Parallelism paradigm — **FSDP2 + Expert Parallel**

`build_topology(dp_size, ep_size, device_type)` constructs a 2D
`DeviceMesh` with axes `("dp", "ep")`. Weights are sharded along **dp**
via FSDP2 (`fully_shard`) for parameter / gradient / optimizer-state
partitioning. Expert tensors are sharded along **ep** as a `DTensor`,
and the routed tokens are dispatched via `all_to_all_single` along the
**ep** subgroup so each expert sees only its local share of tokens.
Tensor and pipeline parallelism axes are reserved in the config for
future extensions.

### 2.2 Triton Top-K router with CPU fallback

`pkg/kernels/moe_router.py` exposes `TritonTopKRouter`. On CUDA + a
working Triton runtime, a fused softmax → top-k → load-balance kernel is
launched and the SRAM / achieved-bandwidth profile is captured in
`RouterProfile`. On any other platform (incl. the test pod which lacks
GPUs), the same forward semantics run through a pure-PyTorch CPU path —
this is the path exercised by every CI run. The chosen path is
surfaced through `last_profile.used_triton` and reported in telemetry.

### 2.3 Async checkpointing — two-tier (NVMe → S3 / MinIO)

`AsyncCheckpointer` (in `pkg/elastic/fault_monitor.py`) snapshots a
`SHARDED_STATE_DICT` through `torch.distributed.checkpoint`
(`get_model_state_dict` / `get_optimizer_state_dict`, `cpu_offload=True`),
serialises to bytes, and enqueues to a worker thread pool. Workers
commit to:

* `LocalNVMeAdapter` — atomic write-and-rename to the local NVMe staging
  directory.
* `S3Adapter` — boto3 `put_object` against the configured `s3://bucket/prefix/`
  (works for AWS S3 and MinIO via `S3_ENDPOINT_URL`).

Retention is bounded (`retention=N`). The training loop is never blocked
on a save — `last_commit_ms` is published into the telemetry envelope so
operators can monitor staging latency.

### 2.4 Elastic recovery

`ClusterStateMachine` runs a monitored barrier each interval. When a
rank drops, the harness:

1. Destroys the process group.
2. Re-reads `WORLD_SIZE` / `RANK` (re-launched by the TorchElastic agent).
3. Recomputes the topology, preserving `ep_size` via the largest divisor
   that still divides the survivor count.
4. Reshards experts continuously (each survivor keeps its prior experts,
   orphans are absorbed round-robin) and reloads the latest checkpoint.
5. Marks phase = `resumed` and returns to training.

SIGTERM / SIGUSR1 trigger a synchronous flush of the async queue inside
the TorchElastic 30s grace window.

### 2.5 Telemetry envelope

`StructuredLogger.emit(StepRecord)` writes one JSON line per step to
`telemetry.json_path` and (rank-0 only) the matching `SummaryWriter`
scalars. Each record carries: `step`, `loss`, `mfu`, `tokens_per_sec`,
`wall_clock_ms`, `kernel{sram_bytes_per_block, achieved_bw_gbps,
tokens_per_expert_{mean,std}, used_triton}`, `collective{...}`,
`memory{...}`, `infra{async_ckpt_commit_ms, active_nodes, ep_world_size}`,
plus `rank` and `ts`.

## 3. CLI surface

```bash
# Hyperscale launch (Slurm / Kubeflow PyTorchJob)
NUM_NODES=128 GPUS_PER_NODE=8 RDZV_ENDPOINT=etcd:2379 ./scripts/launch.sh

# Local CPU smoke (used by CI and the launch.sh verification path)
python train.py --config configs/smoke.yaml --max-steps 2 --smoke
```

Flags:

| Flag           | Effect                                                       |
|----------------|--------------------------------------------------------------|
| `--config`     | YAML config file (required).                                 |
| `--max-steps`  | Override `training.max_steps`.                               |
| `--smoke`      | Downsize model + clamp `max_steps≤5` for a CPU smoke run.    |

## 4. Implementation status (snapshot — Feb 2026)

| Area                                | Status                            |
|-------------------------------------|-----------------------------------|
| Triton Top-K router + CPU fallback  | ✅ implemented & unit-tested      |
| DeviceMesh / FSDP2 / DistributedMoELayer | ✅ implemented & unit-tested |
| Async checkpointer (NVMe + S3)      | ✅ implemented & E2E-tested       |
| Elastic recovery harness            | ✅ implemented & unit-tested      |
| Structured telemetry (NDJSON + TB)  | ✅ implemented & E2E-tested       |
| End-to-end CPU smoke (launch.sh + train.py) | ✅ green (2 steps in ≈0.2s) |
| Moto-mocked S3 mirror path          | ✅ verified by `test_smoke_e2e`   |
| `pytest` suite                      | ✅ 22/22 green                    |

## 5. Test inventory

```
tests/test_kernels.py        Top-K router fwd + autograd
tests/test_distributed.py    Mesh wiring, FSDP2 wrap, DTensor sharding
tests/test_elastic.py        AsyncCheckpointer, cluster state machine
tests/test_smoke_e2e.py      train.py end-to-end (file:// + moto S3)
```

## 6. Backlog (P1 / P2)

* **P1** — Wire NCCL collective latency probes (`all_to_all` /
  `all_gather` / `reduce_scatter`) into the `collective{}` block.
* **P1** — Add a Prometheus exporter that tails `telemetry.json` so the
  same envelope feeds Grafana boards without extra plumbing.
* **P2** — Tensor / pipeline parallelism axes are reserved in the config
  but currently inert; flesh out 4D meshes when needed.
* **P2** — Provide a Helm chart + PyTorchJob CRD example under
  `deploy/` for the production launcher.

## 7. Non-goals (explicitly out of scope)

* Frontend / web UI / dashboard.
* HTTP / REST / gRPC servers.
* Cloud vendor lock-in (S3-compatible MinIO is supported via
  `S3_ENDPOINT_URL`).
* Real GPU training in CI — CI runs on CPU through the documented
  fallback paths.
