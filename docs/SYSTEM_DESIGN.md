# System Design

This document describes the current system design of moe-engine based on the
implemented code in `moe-engine/pkg/` and the actual runtime entrypoint in
`moe-engine/train.py`.

## System scope

moe-engine is designed to validate a distributed Mixture-of-Experts training
runtime with the following real behaviors:

- Data-parallel sharded model state using PyTorch DTensor/FSDP2.
- Expert parallel routing using `all_to_all_single` collectives.
- Async checkpoint writing with local NVMe staging and optional remote mirror.
- Elastic recovery and restart behavior under TorchElastic.
- Observable telemetry with step-level logging and TensorBoard output.

## Core components

### `moe-engine/train.py`

This is the runtime entrypoint. It:

- parses `--config`
- bootstraps distributed process groups for CPU or GPU
- builds a `ParallelTopology`
- constructs a toy MoE model and applies FSDP2 sharding
- configures `StructuredLogger` and `MFUAccountant`
- instantiates `ElasticTrainerHarness`
- resumes from the latest async checkpoint if available
- runs a training loop with checkpointing and telemetry emissions

### `pkg/distributed/parallel_mesh.py`

This module implements the topology and communication helpers:

- `ParallelTopology`: immutable record of rank layout and device mapping.
- `build_topology(...)`: creates a degenerate CPU topology for single-rank
  tests, or a distributed device mesh using PyTorch `init_device_mesh`.
- `experts_on_this_rank(...)`: assigns experts to EP ranks evenly with
  round-robin handling of remainder experts.
- `_CommStream`: a dedicated CUDA stream for EP collectives, enabling overlap
  between all-to-all communication and other work.
- `all_to_all_dispatch(...)` / `all_to_all_combine(...)`: wrappers around
  `dist.all_to_all_single` that record latency and honor the dedicated stream.

The implementation is currently centered on a 2D mesh `(dp, ep)` with reserved
support for `tp` (`tensor_parallel`) in the data structures and helper APIs.

### `pkg/kernels/moe_router.py`

This module provides the core MoE router kernel:

- `MoERouter`: generates top-K expert indices and combine weights.
- fused forward/backward behavior to compute sparse softmax and gradients in a
  single Triton pass where available.
- CPU fallback path for environments without Triton or GPU.

The correctness of this kernel is validated by numerics tests in
`moe-engine/tests/test_kernels_numerics.py`.

### `pkg/elastic/fault_monitor.py`

This module implements elastic training support:

- `AsyncCheckpointer`: background-thread checkpoint streamer that writes
  sharded state payloads from the training process.
- `S3Adapter` and `LocalNVMeAdapter`: remote mirror abstractions for checkpoint
  durability.
- `ElasticTrainerHarness`: high-level driver that combines checkpointing,
  resume semantics, and rendezvous management.
- default NCCL safety settings:
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
  - `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30`

The elastic stack is exercised by `moe-engine/tests/test_elastic.py` and
`moe-engine/tests/test_smoke_e2e.py`.

### `pkg/telemetry/logger.py`

Telemetry is structured and emitted for every training step.
The logger writes:

- JSON step logs to `telemetry.json_path`
- TensorBoard scalars to `telemetry.tensorboard_dir`

It is the source of truth for runtime observability and perf validation.

## Dataflow

A single training step flows as follows:

1. Input token IDs are embedded.
2. The toy model runs through RMS norm and a stack of MoE blocks.
3. Each MoE block:
   - routes tokens with `MoERouter`
   - sorts dispatched tokens by expert id
   - uses EP `all_to_all_single` to send tokens to expert ranks
   - computes local expert FFN outputs
   - uses another `all_to_all_single` to gather results back
   - reassembles and weights the outputs
4. Loss is computed, backward gradients are computed using Triton/CPU
   backprop, and the optimizer updates parameters.
5. The async checkpoint writer may enqueue a checkpoint commit based on
   `checkpoint.ckpt_interval`.

## Distributed design decisions

### Data parallelism

Data parallelism is handled along the `dp` axis using DTensor and FSDP2.
This avoids monolithic wrappers and retains a pure sharded state approach.

### Expert parallelism

Expert parallelism is implemented with a dedicated EP axis and overlapping
collectives. The design explicitly separates token dispatch and combine steps
and uses a dedicated CUDA stream to overlap communication with compute.

### Tensor and pipeline parallelism

The codebase reserves `tensor_parallel` and `pipeline_parallel` axes in the
topology, but the production default config currently sets them to `1`.
This makes the current system design a stable foundation while allowing
future extension.

## Resilience and recovery

The system is designed to resume from the latest valid checkpoint. The
elastic harness can stop and restart training while preserving model state.

Key resilience features:

- local NVMe staging for fast checkpoint writes
- remote checkpoint mirror support for durability
- checkpoint retention pruning to bound disk usage
- elastic rendezvous either via `c10d` or `etcd`

## Validation and testing

The design is grounded in the repository's test suite:

- Kernel numerics: `moe-engine/tests/test_kernels_numerics.py`
- Distributed correctness: `moe-engine/tests/test_distributed.py`
- Invariants: `moe-engine/tests/test_distributed_invariants.py`
- Elastic checkpoint lifecycle: `moe-engine/tests/test_elastic.py`
- Smoke end-to-end: `moe-engine/tests/test_smoke_e2e.py`
- Chaos resilience: `moe-engine/tests/test_chaos.py`

This file should be maintained as the implementation evolves so that system
behavior remains aligned with the actual code and tests.
