# Contributing to moe-engine

Thank you for contributing! This project aims to be a validated reference
implementation for distributed MoE training. Contributions that improve
correctness, observability, and reproducibility are most welcome.

Where to start
- Read `README.md`, `DESIGN.md`, and `ARCHITECTURE.md` to understand scope.
- Run unit and numerics tests locally: `pytest -q` (or `python tests/run_numerics_tests.py`).

How to contribute
1. Open a GitHub issue describing the change or bug.
2. Fork the repo and create a feature branch per change.
3. Write tests that fail before your fix and pass after.
4. Keep changes minimal and focused; explain reasoning in PR description.

Testing and CI
- New PRs must include tests for correctness-sensitive changes.
- Numerics and assertion checks run in CI gating workflows.

Coding guidelines
- Prefer clarity over brevity.
- Avoid magic numbers; explain constants with comments or config.
- Preserve public APIs in `pkg/` where practical.

Security
- Do not commit secrets; use environment variables for credentials!

License
- This repo is MIT licensed. See `LICENSE` for details.

