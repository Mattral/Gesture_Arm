# Architecture Overview

This document describes the high-level architecture of moe-engine and the
rationale behind key design decisions.

Goals
- Hardware-aware kernels (Triton) for Top-K routing (forward and backward)
- Multi-dimensional distributed parallelism (Data × Expert × Tensor × Pipeline)
- Fault-tolerant checkpointing with streaming NVMe and elastic recovery
- Observable telemetry with real CUDA measurements and MFU accounting

Components
- `train.py` — entrypoint; builds topology and configures harness
- `pkg/kernels/moe_router.py` — Top-K router (Triton forward + backward kernels)
- `pkg/distributed/parallel_mesh.py` — 4D DeviceMesh (DP/TP/PP/EP), DistributedMoELayer, TP/PP layers
- `pkg/elastic/fault_monitor.py` — AsyncCheckpointer with chunked NVMe, ClusterStateMachine, etcd rendezvous
- `pkg/telemetry/logger.py` — Structured per-step telemetry with real measurements
- `pkg/utils/mfu.py` — MFU accounting with sparse expert sparsity factor

Dataflow (per step)
1. Input tokens → embedding
2. Forward through transformer blocks with optional TP/PP sharding
3. Tokens through DistributedMoELayer:
   - `MoERouter` computes top-K indices and weights (Triton kernel)
   - Tokens sorted and dispatched via `all_to_all_single` across EP ranks (async_op=True)
   - Local experts compute on received tokens (with sequence parallel for TP>1)
   - Results combined via `all_to_all_single` and reassembled
4. Loss computed, backward through Triton backward kernel and autograd
5. Optimizer step; telemetry emitted with real CUDA measurements

Design Principles
- "Link, don't duplicate": documentation links to source files and tests
- Fail-fast correctness: runtime assertions for critical invariants (token conservation, mesh product)
- Real measurements: no fabricated telemetry numbers; all values measured via CUDA events or memory stats
- Production-grade: chunked I/O, etcd rendezvous, elastic recovery without manual intervention

Implementation Status (✅ Complete)
- Triton backward kernel: softmax Jacobian sparse gradients
- 4D parallelism fully mapped to compute: TP (column/row parallel), PP (1F1B schedule), DP (FSDP2), EP (all-to-all)
- NVMe streaming checkpointing: 256MB chunks with O_DIRECT fallback
- Etcd rendezvous: generation tracking for >100 node scale
- Sequence parallelism: prevents activation duplication on TP > 1
- Telemetry: real CUDA event timing, memory stats, kernel profiling

