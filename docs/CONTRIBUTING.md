# Contributing to moe-engine

Thank you for contributing to moe-engine. This repository is a reference
implementation for production-grade distributed mixture-of-experts training,
so contributions should improve correctness, observability, reproducibility,
and operational resilience.

## Project context

The codebase is centered on:
- `moe-engine/train.py`: distributed training entrypoint and elastic harness.
- `moe-engine/pkg/kernels/moe_router.py`: Triton-based Top-K routing forward/backward.
- `moe-engine/pkg/distributed/parallel_mesh.py`: 4D mesh support for DP/TP/PP/EP.
- `moe-engine/pkg/elastic/fault_monitor.py`: async checkpointing, NVMe staging, and elastic recovery.
- `moe-engine/tests/`: numerics, distributed invariants, elastic, chaos, and smoke regression tests.

## Getting started

1. Read the core docs:
   - `README.md`
   - `docs/ARCHITECTURE.md`
   - `docs/DESIGN.md`
   - `docs/SECURITY.md`

2. Install dependencies from `moe-engine/requirements.txt`.
3. Run the base regression suite from the package root:
   ```bash
   cd moe-engine
   python -m pytest -q
   ```
4. Validate Triton kernel correctness with the dedicated numerics driver:
   ```bash
   python tests/run_numerics_tests.py
   ```

## Contribution workflow

1. Open an issue if the bug, feature, or gap is not already tracked.
2. Fork the repository and create a focused feature branch.
3. Implement the change with minimal scope.
4. Add or update tests that reproduce the issue and verify the fix.
5. Submit a pull request with:
   - a concise description of the change,
   - reasoning for the design choices,
   - the verification commands you used.

## Testing expectations

Changes should include tests whenever they affect behavior or correctness.
Important test areas in this repo include:
- Kernel numerics: `moe-engine/tests/test_kernels_numerics.py`
- Distributed correctness: `moe-engine/tests/test_distributed.py`
- Distributed invariants: `moe-engine/tests/test_distributed_invariants.py`
- Elastic checkpointing: `moe-engine/tests/test_elastic.py`
- Chaos recovery: `moe-engine/tests/test_chaos.py`
- Smoke end-to-end: `moe-engine/tests/test_smoke_e2e.py`
- Tensor parallelism semantics: `moe-engine/tests/test_tensor_parallel.py`
- MFU accounting: `moe-engine/tests/test_mfu.py`

If a change touches runtime behavior, document the expected regression strategy
and include relevant `pytest` markers, such as `@pytest.mark.chaos` for chaos tests.

## Code quality

- Prefer clarity and explicit correctness over cleverness.
- Avoid magic numbers; explain constants with comments or configuration.
- Maintain public API stability in `moe-engine/pkg/` when possible.
- Keep changes well-scoped and easy to review.

## Security and secrets

- Do not commit secrets, keys, or credentials.
- Use environment variables for credentials, especially S3-related values.
- If you find a security issue, open a private issue rather than posting secrets.

## Licensing

This project is licensed under Apache 2.0. See `LICENSE` for full terms.

