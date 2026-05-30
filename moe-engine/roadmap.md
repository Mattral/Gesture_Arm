# moe-engine Roadmap

## Legend
✅ Complete + CI-verified  ⚠️ Partial  ❌ Not started  🔒 Blocked

---

## v0.1 — Correctness Foundation (current target)
- [❌] Triton backward kernel (`_router_bwd_kernel`) — P0-1
- [⚠️] Token conservation assertions — added, CI gate in place
- [❌] Chaos Scenario A: node kill + hot-resume — P0-3
- [⚠️] Chaos Scenario B: storage stall — passing
- [❌] Dead code removal (`if False` branches) — P0-2
 - [✅] Triton backward kernel (`_router_bwd_kernel`) — P0-1
 - [❌] Chaos Scenario A: node kill + hot-resume — P0-3
 - [⚠️] Chaos Scenario B: storage stall — passing
 - [✅] Dead code removal (`if False` branches) — P0-2

## v0.2 — Complete 4D Parallelism
- [❌] Tensor Parallelism: ColumnParallel + RowParallel linear — P1-1
- [❌] Sequence Parallelism for TP > 1 — P2-3
- [❌] Pipeline Parallelism: PipelineStage + 1F1B schedule — P1-2

## v0.3 — Verified Performance
- [❌] Real MFU calculation (MoE-correct formula) — P1-3
- [❌] Telemetry wired to real CUDA measurements — P1-4
- [❌] BENCHMARKS.md with actual run data (kernel latency, overlap ratio, MFU)
- [❌] Async overlap ratio benchmark (target ≥ 0.60)

## v0.4 — Production Hardening
- [❌] NVMe chunked streaming checkpoint I/O — P2-1
- [❌] Etcd rendezvous for > 100 nodes — P2-2
- [❌] Nsight/CUPTI profiling integration
- [❌] Kubernetes / Kubeflow operator manifests

## Known Deficiencies (honest disclosure)
- No Triton backward kernel yet: backward uses PyTorch autograd (slower, not fused)
- TP and PP axes exist in DeviceMesh but have no compute mapped to them
- MFU numbers in README are illustrative, not measured
- Chaos node-kill test is quarantined pending expert reshard fix
- No etcd integration; c10d rendezvous only (suitable for
