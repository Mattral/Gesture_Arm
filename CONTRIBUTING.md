# Contributing

Thank you for your interest in contributing to Gesture Arm. This document covers how to set up a development environment, the code standards enforced in CI, and the process for submitting changes.

---

## Table of contents

1. [Development setup](#1-development-setup)
2. [Code standards](#2-code-standards)
3. [Testing](#3-testing)
4. [Submitting a pull request](#4-submitting-a-pull-request)
5. [Adding a new feature](#5-adding-a-new-feature)
6. [Reporting a bug](#6-reporting-a-bug)

---

## 1. Development setup

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/gesture_arm.git
cd gesture_arm

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# Install in editable mode with all dev dependencies
pip install -e ".[ml,dev]"

# Verify
pytest tests/ -v                # all tests should pass
ruff check gesture_arm/         # no lint errors
```

---

## 2. Code standards

All of these are enforced automatically in CI. A PR that fails any check will not be merged.

### Formatting — black

```bash
black gesture_arm/ scripts/ tests/
```

Line length is 100 characters. Do not configure your editor to use a different line length.

### Linting — ruff

```bash
ruff check gesture_arm/ scripts/ tests/
```

Rules enforced: pyflakes (F), pycodestyle (E, W), isort (I), pyupgrade (UP), bugbear (B). The full config is in `pyproject.toml`.

Fix automatically:
```bash
ruff check --fix gesture_arm/
```

### Type checking — mypy

```bash
mypy gesture_arm/ --ignore-missing-imports
```

All public function signatures must have type annotations. `-> None` is required on functions that return nothing. `Optional[X]` is required instead of `X | None` for Python 3.9 compatibility.

### Running all checks at once

```bash
black --check gesture_arm/ scripts/ tests/ && \
ruff check gesture_arm/ scripts/ tests/ && \
mypy gesture_arm/ --ignore-missing-imports && \
pytest tests/ -v
```

---

## 3. Testing

### Running the test suite

```bash
pytest tests/ -v                          # all tests
pytest tests/ -v -k "Config"             # one class
pytest tests/ -v --cov=gesture_arm       # with coverage
```

### Writing new tests

All tests live in `tests/test_core.py`. Tests must:

- Run without hardware, network access, or TensorFlow (mark TF tests with `pytest.skip` if TF unavailable)
- Follow the `TestClassName.test_method_name` naming pattern
- Cover the normal path, boundary conditions, and at least one error path for any new function

**Do not:**
- Open a real camera (`cv2.VideoCapture`) in a test
- Connect to a real serial port
- Make network requests

**Use fixtures** for shared setup:
```python
@pytest.fixture
def mapper(self):
    from gesture_arm.models.stabilizer import BaselineMapper
    return BaselineMapper({"x":(60,180),"y":(40,140),"z":(100,150)})
```

### Coverage

Aim for >80% line coverage on any new module. Check with:
```bash
pytest tests/ --cov=gesture_arm --cov-report=term-missing
```

---

## 4. Submitting a pull request

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/transformer-stabilizer
   ```

2. **Make your changes.** Keep commits small and focused. One logical change per commit.

3. **Write or update tests** for your change.

4. **Run the full check suite** (see Section 2) and confirm everything passes locally before pushing.

5. **Push and open a PR** against `main`. Fill in the PR template:
   - What does this change do?
   - Why is it needed?
   - How was it tested?
   - Does it change any public API? (update `docs/API_REFERENCE.md` if so)

6. **CI runs automatically.** Fix any failures — do not ask reviewers to ignore CI.

7. **One approval** from a maintainer is required to merge.

### PR size

Keep PRs small. A PR that changes 5 files is reviewed in 20 minutes. A PR that changes 30 files takes days or gets rubber-stamped. If your feature is large, split it into multiple PRs: interface first, then implementation, then tests, then documentation.

---

## 5. Adding a new feature

### Adding a new stabilization model

1. Add a new class in `gesture_arm/models/` that implements this interface:

   ```python
   class MyStabilizer:
       def update(self, feature_vector: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
           ...
       def reset(self) -> None:
           ...
   ```

2. Add a config section in `config/default.yaml` if the model has tunable parameters.

3. Add a `--model` CLI flag to `gesture_arm/run.py` to select between stabilizers.

4. Add benchmark comparison to `notebooks/benchmark_analysis.ipynb`.

5. Add unit tests in `tests/test_core.py`.

### Adding a new hardware backend

1. Create `gesture_arm/hardware/<backend>.py` with `ArmController` and `BaseController` classes that match the existing interface.

2. Add a `--hardware` CLI flag or config option to `run.py`.

3. No hardware-specific code should appear outside the `hardware/` module.

### Adding a new voice command

1. Add the command string and motor args to `speech.commands` in `config/default.yaml`.

2. No code changes required — the ASR listener reads the command vocabulary from config at startup.

---

## 6. Reporting a bug

Open a GitHub issue with:

1. **Description** — what happened vs what you expected
2. **Reproduction steps** — exact commands run
3. **Environment** — OS, Python version, `pip freeze` output
4. **Logs** — full terminal output including any tracebacks
5. **Hardware** — Arduino model, camera model (if hardware-related)

For crashes, run with `PYTHONFAULTHANDLER=1` to get a full traceback:
```bash
PYTHONFAULTHANDLER=1 python -m gesture_arm.run
```
