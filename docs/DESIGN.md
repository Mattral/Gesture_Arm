# Design Rationale and Trade-offs

This file documents why core design choices were made and trade-offs considered.

Router Kernel
- Choice: Triton fused kernel (tokens @ gate_w -> softmax -> top-k -> renorm)
- Pros: Single-pass memory locality, better bandwidth utilization, lower launch overhead
- Cons: GPU-only; maintenance cost; careful numerical validation required

Distributed Topology
- Current: 2D mesh `(dp, ep)` implemented to validate data and expert parallelism
- Rationale: EP is the most orthogonal to data parallelism for MoE workloads; TP/PP add complexity
- Trade-offs: Without TP/PP, per-node memory and compute balancing is limited

Elastic Recovery
- Async checkpointing: background thread writes sharded snapshots to local NVMe, mirrors to S3
- ClusterStateMachine uses a lightweight barrier-monitored heartbeat for now (adequate for <100 nodes)
- Trade-offs: barrier-based heartbeat is simple but not scalable to 10k nodes; planned etcd rendezvous will replace it

MFU and Observability
- MFU: computed from theoretical FLOP counts and measured step time
- Telemetry: structured JSON + TensorBoard scalars; kernel profile info filled from router
- Trade-offs: precise kernel occupancy metrics require CUPTI/Nsight integration (planned)

Testing and CI
- Numerics tests: Triton vs FP64 reference with strict tolerances
- Distributed invariants: token conservation / gradient parity tests
- Chaos tests: simulated rank loss and recovery (slow; run under `-m chaos`)


Security and Reliability
- Checkpoint writes are atomic (write+rename) to avoid partial files
- Credentials for S3 should be supplied via environment variables (do not hardcode)


Extensions
- TP: shard linear layers and coordinate matmul collectives
- PP: stage partitioning and microbatch scheduler
- NVMe streaming: chunked upload/download for very large shards
