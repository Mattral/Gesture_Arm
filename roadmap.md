# moe-engine Roadmap

**Last Updated:** June 3, 2026  
**Current Phase:** P0 (Correctness Foundation + 4D topology prework)

## Legend
- ✅ Complete + CI-verified  
- ⚠️ Partial  
- ❌ Not started  
- 🔒 Blocked

---

## v0.1 — Correctness Foundation

### P0 — CRITICAL CORRECTNESS

- [✅] **P0-1: Triton Backward Kernel** — `_router_bwd_kernel` implemented and tested
  - Backward through softmax + top-k + renormalization
  - Numerical validation: `test_backward_tolerance` passing on multiple random seeds
  - Acceptance: `atol=1e-5, rtol=1e-5` across H∈[64,128,256,512], E∈[8,16,32], K∈[1,2,4]

- [✅] **P0-2: Remove Dead Code** — No `if False`, placeholder, or `# TODO` branches remain
  - Grep sweep completed: `grep -n "if False|placeholder|# TODO" pkg/distributed/parallel_mesh.py` returns empty
  - Token dispatch path executes unconditionally

- [⚠️] **P0-3: Chaos Scenario A (Sudden Node Kill + Recovery)**
  - Environment: NCCL async error handling configured
    - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
    - `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30`
    - `TORCH_NCCL_TRACE_BUFFER_SIZE=1048576`
  - Expert rebalancing: `_rebalance_experts()` implements round-robin distribution
  - **Status**: Test execution reaches target but occasional timeouts on 4-rank restart due to Gloo connection handling
  - **Blockers**: Gloo connectFullMesh failures during PG reformation post-restart (infrastructure flakiness)
  - **Mitigation**: Increased synchronization delays and timeouts in worker init

---

## v0.2 — Complete 4D Parallelism (Planned)

- [⚠️] **P1-1: Tensor Parallelism** — ColumnParallelLinear + RowParallelLinear (topology placeholder path started)
- [⚠️] **P1-2: Pipeline Parallelism** — PipelineStage + 1F1B schedule (pp_size reserved and rank math generalized)
- [❌] **P1-3: Real MFU Calculation** — MoE-aware FLOPs accounting
- [⚠️] **P1-4: Telemetry Wired to Real Measurements** — CUDA event timing, peak memory stats (placeholder fields remain)

---

## v0.3 — Performance & Profiling (Planned)

- [❌] Async overlap ratio benchmark (target ≥ 0.60)
- [❌] Nsight/CUPTI integration
- [❌] Kernel latency profiling
- [❌] BENCHMARKS.md with actual run data

---

## v0.4 — Production Hardening (Planned)

- [❌] **P2-1: NVMe Streaming Checkpoint I/O** — Chunked writes (256MB chunks)
- [❌] **P2-2: Etcd Rendezvous** — Scale support > 100 nodes
- [❌] **P2-3: Sequence Parallelism** — For TP > 1
- [❌] Kubernetes / Kubeflow operator manifests

---

## Known Deficiencies & Honest Disclosure

### Current Limitations

1. **Chaos Test Timeout (P0-3)**
   - Root cause: Gloo backend connection timeouts during rank restart
   - Symptoms: 4-rank runs occasionally hit "connectFullMesh failed: timed out connecting"
   - Workaround: Increased init backoff/delay and extended PG timeout on restart
   - Likelihood: 80-90% pass rate on local runs; flakiness due to containerized environment socket binding

2. **Missing Parallelism Implementations**
   - TP, PP, SP axes exist in `DeviceMesh` but have no compute mapped to them
   - No tensor movement via `DTensor` or batch-wise sharding in the MoE layer

3. **Fabricated Telemetry Numbers**
   - Kernel SRAM bytes per block: estimated, not measured
   - Achieved bandwidth: placeholder, not timed via CUDA events
   - All numbers in README marked as `(illustrative)`

4. **Rendezvous & Scale**
   - Using Gloo + c10d rendezvous (suitable for < 100 nodes)
   - No etcd backend integration yet

---

## CI Status

### Passing Tests
- `tests/test_kernels.py::test_router_fwd` ✅
- `tests/test_kernels.py::test_backward_tolerance` ✅
- `tests/test_chaos.py::test_chaos_baseline_no_fault` ✅
- `tests/test_smoke_e2e.py` ✅

### Known Flaky
- `tests/test_chaos.py::test_chaos_scenario_a_sudden_node_failure` ⚠️ (Gloo connection issues)
- `tests/test_chaos.py::test_chaos_scenario_b_storage_stall` ✅ (passes)

---

## Prework: 4D topology placeholder work

- [⚠️] Add `pp_size` to `build_topology` and `ParallelTopology`; keep defaults at `1`
- [⚠️] Generalize rank calculation to support future `dp,tp,pp,ep` ordering
- [⚠️] Pass `tensor_parallel` and `pipeline_parallel` from config through `train.py`
- [⚠️] Reserve 4D `DeviceMesh` axis creation; do not change default DP/EP execution
- [⚠️] Add unit tests for `pp_size=1` regression and 4D rank semantics once stable

## Next Actions (High Priority)

1. **Stabilize Scenario A** — Investigate Gloo connection refused pattern on restart
   - Consider switching to nccl backend for scale tests
   - Add per-rank socket state logging (ss/lsof snapshots)

2. **Implement P1-1 (Tensor Parallelism)** — Required for 4D mesh validation
   - `ColumnParallelLinear` with all-gather on output
   - `RowParallelLinear` with reduce-scatter on output
   - Wire into `DistributedMoELayer` expert FFN

3. **Implement P1-2 (Pipeline Parallelism)** — Complete 1F1B staging and PP rank mapping
   - Define pipeline stage groups and `pp_rank` execution semantics
   - Add pipeline-aware parameter placement and activation routing

4. **Implement P1-3 (MFU Calculation)** — Required for perf claims
   - Account for sparse expert activation: `(K/E) * P_expert`
   - Validate against H100 peak FLOPs

---

## References

- **Directive**: `feedback.md` (principal engineering review)
- **Kernel Source**: `pkg/kernels/moe_router.py`
- **Chaos Test**: `tests/test_chaos.py`
- **Telemetry**: `pkg/telemetry/logger.py`
