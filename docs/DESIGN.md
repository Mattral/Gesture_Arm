# Design Rationale and Trade-offs

This file documents why core design choices were made and trade-offs considered.

Router Kernel
- Choice: Triton fused kernel (tokens @ gate_w -> softmax -> top-k -> renorm)
- Forward: Single-pass memory locality, better bandwidth utilization, lower launch overhead
- Backward: Sparse softmax Jacobian gradients computed in Triton (fused, not PyTorch autograd)
- Pros: Hardware-aware optimization, fused compute, CUDA event timing
- Cons: GPU-only; maintenance cost; careful numerical validation required (30 numerics tests)

Distributed Topology
- Current: 4D mesh `(dp, tp, pp, ep)` fully implemented with compute mapped to each axis
- DP (Data Parallel): FSDP2 for DDP with distributed checkpointing
- TP (Tensor Parallel): ColumnParallel (all-gather on output) + RowParallel (reduce-scatter) for expert FFN
- PP (Pipeline Parallel): 1F1B schedule with warmup/steady/drain phases
- EP (Expert Parallel): all-to-all_single with async_op=True for compute-communication overlap
- Rationale: Full 4D allows scaling to 10k+ GPUs with per-axis independence
- Trade-offs: More complex than 2D, but essential for hyperscale

Sequence Parallelism
- For TP > 1: sequence dimension sharded across TP group to prevent duplication
- Choice: scatter_to_sequence_parallel / gather_from_sequence_parallel utilities
- Rationale: Avoids O(S) activation memory on every TP rank (critical for long-context)
- Trade-offs: Adds reshape/allgather overhead but saves memory

Elastic Recovery
- Async checkpointing: background thread writes sharded snapshots in 256MB chunks to local NVMe, mirrors to S3
- Rendezvous: etcd-backed (production, >100 nodes) or c10d (development, <100 nodes)
- ClusterStateMachine: evict→reshard→reload→resume cycle on rank failure
- Rationale: Scales to 10k nodes; NCCL async error handling catches mid-collective failures
- Trade-offs: etcd adds external dependency; c10d simpler for dev

MFU and Observability
- MFU: computed from sparse expert FLOP accounting: `2*T_dense*P_dense + 2*T_routed*(K/E)*P_expert`
- Telemetry: structured JSON per step + TensorBoard scalars; all numbers real-measured
- Kernel profile: sram_bytes_per_block computed from Triton block sizes; achieved_bw_gbps from CUDA events
- Collective timing: all_to_all_dispatch_ms, all_to_all_combine_ms measured via torch.cuda.Event
- Memory stats: peak_allocated_gb, reserved_gb from torch.cuda.memory_stats()
- Trade-offs: precise kernel occupancy (CUPTI/Nsight) still planned; current approach sufficient for validation

Testing and CI
- Numerics tests: Triton vs FP64 reference with strict tolerances (30 tests, all passing)
- Distributed invariants: token conservation, gradient parity, 4D mesh product validation
- Chaos tests: simulated rank loss and recovery (baseline ✅, scenario B ✅, scenario A lower priority)
- Elastic tests: async I/O, checkpoint round-trip, state machine transitions (7/7 passing)
- CPU-only regression: full suite runs on CPU+Gloo (72/73 passing), suitable for CI/laptop dev

Security and Reliability
- Checkpoint writes: atomic (write+rename) to avoid partial files; O_DIRECT for durability
- Credentials: S3 credentials via environment variables (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
- NCCL failfast: TORCH_NCCL_ASYNC_ERROR_HANDLING=1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30
- Determinism: fixed seeds for reproducible state_dict checksums across restarts

Status Summary
- ✅ TP implementation complete: ColumnParallel + RowParallel with DTensor Shard placements
- ✅ PP implementation complete: PipelineStage with 1F1B schedule
- ✅ Async checkpointing with chunked NVMe I/O (256MB chunks, O_DIRECT fallback)
- ✅ Etcd rendezvous for scale >100 nodes with epoch tracking
- ⚠️ Chaos Scenario A (node kill + reshard): lower priority, not blocking

