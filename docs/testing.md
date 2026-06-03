# Testing

This repository includes unit, numerics, distributed, elastic, and chaos tests.
The test suite is designed to validate correctness across the full runtime.

## Run the main test suite

From the `moe-engine` package root:

```bash
pytest -q
```

For a faster local regression set that excludes injected chaos scenarios:

```bash
pytest -m "not chaos" -v
```

## Triton numerics validation

The dedicated numerics driver validates the router against FP64 reference
behavior without requiring the full `pytest` harness.

```bash
python tests/run_numerics_tests.py
```

This covers forward/backward correctness, token conservation, weight
normalization, and deterministic behavior.

## Test coverage areas

### Kernel correctness

- `moe-engine/tests/test_kernels_numerics.py`
- `moe-engine/tests/test_kernels.py`

These tests validate the Triton/CPU router kernel and its numerical
properties.

### Distributed semantics

- `moe-engine/tests/test_distributed.py`
- `moe-engine/tests/test_distributed_invariants.py`

They verify topology construction, MoE layer forward/backward shapes,
expert-to-rank mapping, token conservation, and gradient sanity.

### Elastic checkpointing

- `moe-engine/tests/test_elastic.py`

These tests cover async checkpoint save/load, local NVMe round-trip,
retention pruning, and zero-divergence resharding behavior.

### End-to-end smoke validation

- `moe-engine/tests/test_smoke_e2e.py`

This suite validates full-stack behavior, including checkpoint tiers,
telemetry output, and S3/MinIO mirror paths.

### Chaos resilience

- `moe-engine/tests/test_chaos.py`

Chaos tests are gated with `@pytest.mark.chaos` and validate crash recovery,
storage stall handling, and TorchElastic re-rendezvous.

## Running chaos tests locally

The repo supports local multi-rank chaos regression via Gloo.
Use the loopback interface when running chaos tests on a single machine:

```bash
GLOO_SOCKET_IFNAME=lo pytest -m chaos -v -k "baseline or scenario_b"
```

## Smoke and minimal test paths

A lightweight validation path is provided by `configs/smoke.yaml` and the
`--smoke` flag in `train.py`.

```bash
python train.py --config configs/smoke.yaml --smoke
```

This is the recommended first step for new contributors and CI smoke checks.

## Test development guidance

- Add or update tests for any behavioral change.
- Prefer focused tests that fail before the fix and pass afterward.
- Document the verification command in PR descriptions.
- Keep the change set minimal and linked to the relevant test category.

## Notes

- The CPU-only and single-rank paths are intentionally supported so the full
  non-chaos suite can run on developer laptops.
- Use `pytest -m "not chaos"` for fast validation when chaos injection is not
  required.
- The `tests/run_numerics_tests.py` script is useful when working on the
  router implementation and Triton kernel correctness.
