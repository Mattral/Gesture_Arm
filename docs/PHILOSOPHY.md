# Philosophy

This repository is built around a single, evidence-driven philosophy: make distributed Mixture-of-Experts training reliable, measurable, and operable at scale.

## Core principles

1. **Measure, don’t guess**
   - Telemetry and performance counters are based on actual runtime measurements,
     not estimates. The code records CUDA event timings, memory stats, and MFU
     accounting from real execution paths.
   - The `moe-engine/tests/test_kernels_numerics.py` suite validates Triton
     kernels against FP64 reference implementations.

2. **Keep critical paths observable**
   - The runtime emits structured telemetry for each training step, allowing
     kernel, collective, and checkpoint behavior to be audited.
   - `pkg/telemetry/logger.py` is the source of truth for per-step logs and
     scalar summaries.

3. **Fail fast on invariants**
   - Distributed invariants such as token conservation and mesh product
     consistency are enforced in tests and runtime checks.
   - The repository contains dedicated invariant validation, e.g.
     `moe-engine/tests/test_distributed_invariants.py`.

4. **Prefer correctness over premature optimization**
   - Hardware-aware kernels are used where they provide measurable benefits.
   - The Triton router kernel is a targeted optimization, while the rest of the
     stack is intentionally explicit and debuggable.

5. **Design for operational resilience**
   - Async checkpointing, local NVMe staging, and remote mirror support are
     implemented to keep runs recoverable.
   - Elastic recovery is exercised by `moe-engine/tests/test_elastic.py` and
     `moe-engine/tests/test_chaos.py`.

6. **Link, don’t duplicate**
   - Documentation points to implementation and test files instead of repeating
     behavior descriptions.
   - This keeps docs grounded in code and makes maintenance easier.

## What this means in practice

- Changes should be backed by tests or clearly referenced runtime behavior.
- New features should expose observability, not hide it behind opaque layers.
- Production-oriented runtime trade-offs are preferred over academic
  micro-optimizations that are not validated by the repository's test set.
- Secrets are never stored in source; secret handling is explicitly delegated to
  environment variables.

## Evidence from the repo

- `README.md` and `docs/ARCHITECTURE.md` document the 4D mesh strategy (DP/TP/PP/EP).
- `moe-engine/train.py` demonstrates how a full distributed model is bootstrapped
  in a way that can be replaced with a real model while leaving the harness intact.
- `moe-engine/pkg/elastic/fault_monitor.py` is the implementation anchor for
  elastic checkpointing, NVMe staging, and recovery.
- `moe-engine/tests/test_smoke_e2e.py` verifies end-to-end behavior with both
  local file and mocked S3 storage.

## Practical guidance for contributors

- If you add a runtime path, add telemetry or test coverage for it.
- If you change a distributed algorithm, also update the corresponding invariant
  or numerics test.
- Keep the documentation and code aligned: if the behavior changes, update both.

