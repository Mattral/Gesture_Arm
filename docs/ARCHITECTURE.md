# Architecture Overview

This document describes the high-level architecture of moe-engine and the
rationale behind key design decisions.

Goals
- Hardware-aware kernels (Triton) for Top-K routing
- Multi-dimensional distributed parallelism (Data × Expert; TP/PP scaffolded)
- Fault-tolerant checkpointing and elastic recovery
- Observable telemetry and measurable MFU

Components
- `train.py` — entrypoint; builds topology and configures harness
- `pkg/kernels/moe_router.py` — Top-K router (Triton kernel + FP64 reference)
- `pkg/distributed/parallel_mesh.py` — Distributed topology and `DistributedMoELayer`
- `pkg/elastic/fault_monitor.py` — AsyncCheckpointer and ClusterStateMachine
- `pkg/telemetry/logger.py` — Structured per-step telemetry
- `pkg/utils/mfu.py` — MFU accounting

Dataflow (per step)
1. Input tokens → embedding
2. Forward through transformer blocks with `DistributedMoELayer`
3. `MoERouter` computes top-K indices and weights
4. Tokens are sorted and dispatched via `all_to_all_single` across EP ranks
5. Local experts compute on received tokens
6. Results are combined via `all_to_all_single` and reassembled
7. Loss computed, backward, optimizer step

Design Principles
- "Link, don't duplicate": documentation simply links to source files and tests.
- Fail-fast correctness: runtime assertions for critical invariants (e.g. token conservation).
- Minimal reproducible building blocks that can be extended to TP/PP.

Limitations
- Current implementation provides Data × Expert parallelism only.
- TP and PP axes are scaffolded but not implemented.
- Some production-grade features (streaming NVMe checkpointing) are planned but not present.

For more details and the implementation roadmap see `ROADMAP.md` and `DESIGN.md`.
