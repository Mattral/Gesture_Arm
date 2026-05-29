# `moe-engine` — Engineering Roadmap & State Ledger

> **Protocol-1 ledger.** Every turn must read this file before mutating code,
> and append a markdown blockquote to the response summarising any updates
> made here. The schema below is **load-bearing**: do not rename sections.

Last update: turn-002 (in-flight) — reality-drift reconciliation + Items #1/#2/#3 re-land

---

## Completed Tasks

Each entry: `<scope>` — `<file path(s)>` — `<verification reference>`

### Carried over from prior session (verified before this ledger existed)
| # | Scope | Files | Verification |
|---|-------|-------|--------------|
| C-01 | Custom Triton Top-K MoE routing kernel (forward only) with CPU autograd fallback | `pkg/kernels/moe_router.py` | `pytest tests/test_kernels.py` (carried over from session 0; backward kernel is **NOT** yet present — see TD-04) |
| C-02 | 2D `(dp, ep)` `DeviceMesh` + FSDP2 `fully_shard` + EP `all_to_all_single` on a dedicated CUDA stream | `pkg/distributed/parallel_mesh.py` | `pytest tests/test_distributed.py` |
| C-03 | Two-tier (Local NVMe → S3/MinIO) async checkpointer scaffold with `boto3` adapter, retention pruning, signal-driven flush | `pkg/elastic/fault_monitor.py` | `pytest tests/test_elastic.py` |
| C-04 | TorchElastic chaos test driver: baseline + Scenario B (storage stall) **passing** | `tests/test_chaos.py`, `tests/_chaos_worker.py` | `pytest -m chaos -k "baseline or scenario_b"` |
| C-05 | Telemetry, MFU accountant, structured JSON logger, training entrypoint | `pkg/telemetry/`, `pkg/utils/`, `train.py` | `python train.py --config configs/smoke.yaml --smoke` |

### This turn (turn-001)
| # | Scope | Files | Verification |
|---|-------|-------|--------------|
| T01-A | Initialise Protocol-1 state ledger | `roadmap.md` | file present at repo root |
| T01-B | Enterprise-grade README with ASCII architecture, HW reqs, local + cluster orchestration guide | `README.md` | file present, replaces prior 93-line stub |

> **Reality-drift correction (turn-002):** Rows previously listed here as
> T01-C, T01-D, T01-E claimed Items #1, #2, #3 had landed in turn-001. A
> turn-002 audit of the working tree against these claims showed the edits
> were **never** persisted to `parallel_mesh.py` / `fault_monitor.py`
> (verified by reading the files end-to-end and by `pytest -m "not chaos"`
> still passing the carry-over baseline only). Those rows have been moved
> back to *In-Progress / Active Focus* below and re-tagged with `T02-*`
> identifiers. This is the only acceptable way to keep the Protocol-1
> ledger load-bearing.

---

## In-Progress / Active Focus

**Turn-002 (current):** Re-land the Items #1, #2, #3 batch that turn-001
committed to but failed to persist. All three must turn green inside this
single turn before TD-01 (Triton backward) is unblocked.

| # | Scope | Files | Verification target |
|---|-------|-------|---------------------|
| T02-A | **Item #1** — True 4D `(pp, dp, tp, ep)` topology via `init_device_mesh`; TP intra-node, PP inter-node; FSDP2 sharded along `dp`; EP collectives along `ep`; bullet-proof 1-rank degenerate fallback. | `pkg/distributed/parallel_mesh.py` | `pytest tests/test_distributed.py` (1-rank degenerate path preserved + 4-prop `pp/dp/tp/ep`-rank derivation) |
| T02-B | **Item #2** — `_PinnedBufferPool` keyed by `(shape, dtype)`; `AsyncCheckpointer.save` does `detach → acquire pinned buf → non_blocking=True` D2H copy → paged-CPU clone → release pinned buf → enqueue host-only payload to writer queue. | `pkg/elastic/fault_monitor.py` | `pytest tests/test_elastic.py::test_async_ckpt_no_device_refs` (new regression) |
| T02-C | **Item #3** — `dist.all_to_all_single(..., async_op=True)` + explicit `Work.wait()` 3-phase split (launch → independent local compute → wait) in `DistributedMoELayer.forward`; old CUDA-stream/event scaffold removed. | `pkg/distributed/parallel_mesh.py` | shape & autograd test in `test_distributed.py` still green; degenerate path still no-ops cleanly |

---

## Outstanding Technical Debt & Backlog

Chronological — top of list = next turn's focus.

| # | Title | Rationale | Touch points |
|---|-------|-----------|--------------|
| TD-01 | **Item #4 — Triton kernel safety + autograd backward** | Forward kernel currently assumes static `BLOCK_M`; produces unaligned-coalesced loads on variable seq-len batches. Backward path runs in eager PyTorch → 4× slower than necessary. | `pkg/kernels/moe_router.py` |
| TD-02 | **Item #5 — Non-uniform expert resharding state-machine** | After a node drop, `experts_on_this_rank` re-divides modulo survivors; the *weights* of orphaned experts are not migrated atomically — load-state-dict shape mismatch when EP shrinks from 8→3. | `pkg/elastic/fault_monitor.py` (`reshard`, `load`), `pkg/distributed/parallel_mesh.py` |
| TD-03 | **Item #6 — NCCL communicator-poisoning watchdogs** | A hard `SIGKILL` on a NCCL peer leaves survivors stuck in un-abortable `ncclRecv`. Need `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_DESYNC_DEBUG=1`, `NCCL_BLOCKING_WAIT=0` injected at harness init. | `pkg/elastic/fault_monitor.py`, `scripts/launch.sh` |
| TD-04 | **Item #7 — Dynamic MoE FLOP / MFU equation** | Current MFU uses static FLOPs/token. Replace with `FLOPs_step = 2·T_active·P_dense + 2·T_routed·P_experts` recomputed per-step from the actual router profile. | `pkg/utils/mfu.py`, `train.py` |
| TD-05 | Resolve `test_chaos_scenario_a_sudden_node_failure` (carried over) | Gloo rendezvous on `world_size=4` flakes after `SIGKILL`; survivor restart-generation telemetry shows `AssertionError` during state reload — likely intersection of TD-02 (uneven reshard) and TD-03 (stuck communicator). May auto-resolve once TD-02 + TD-03 land. | `tests/test_chaos.py`, `tests/_chaos_worker.py`, `pkg/elastic/fault_monitor.py` |
| TD-06 | Regression: `pytest -m "not chaos"` after every TD-0x lands | Catch collateral damage from large refactors. | n/a |
| TD-07 | `--runchaos` opt-in CLI flag in `conftest.py` | Make `-m chaos` opt-in by default in CI; nightly matrix flips it on. | `tests/conftest.py` |
| TD-08 | Wire chaos suite into a nightly GH Actions matrix | Continuous fault-injection coverage. | `.github/workflows/nightly.yml` |

---

## Mathematical Invariants (CI gates)

These must hold and are codified in tests:

1. **4-D Mesh shape:**  `|TP| · |PP| · |FSDP2| · |EP| = world_size`
2. **Token conservation:**  `Σ_r dispatched_r = B · S · K`
3. **Checksum identity:**  `hash(state_dict_post_load) = hash(state_dict_pre_save)`
4. **Monotonic progression:**  `step_{n+1} > step_n` across every restart generation
5. **Dynamic MoE FLOP:** `FLOPs_step = 2·T_active·P_dense + 2·T_routed·P_experts`
