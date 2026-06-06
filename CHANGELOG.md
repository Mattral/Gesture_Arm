# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] — 2025

### Added — Geometric Inverse Kinematics

- **`gesture_arm/kinematics/ik_solver.py`** — `GeometricIKSolver`: closed-form
  analytical IK for the 2-DoF positional arm (base rotation θ₁ + shoulder
  elevation θ₂). Given a desired TCP position (px, py, pz) in centimetres,
  computes joint angles via:

  ```
  r  = √(px² + py²)        θ₁ = atan2(py, px)
  L  = √(r²  + pz²)        θ₂ = atan2(pz, r)
  ```

  Returns a typed `IKResult` dataclass with solution status
  (`OK`, `UNREACHABLE`, `IN_DEADZONE`, `JOINT_LIMIT`), servo angles,
  raw geometric angles, and a human-readable message.

- **`IKResult` and `IKSolution`** — immutable result type and status enum
  for structured error handling without exceptions in the hot path.

- **`GeometricIKSolver.forward()`** — forward kinematics for FK/workspace
  verification and visualization.

- **`GeometricIKSolver.fk_check()`** — convenience round-trip consistency check.

- **`GeometricIKSolver.workspace_bounds()`** — returns Cartesian workspace
  envelope (x_range, z_range, max/min reach) for HUD scaling.

- **`GeometricIKSolver.hand_position_to_target()`** — maps a normalized hand
  position (from `HandState.features`) to a desired TCP position in the arm's
  workspace, bridging the gesture pipeline to the IK solver.

- **`gesture_arm/kinematics/__init__.py`** — package init exporting
  `GeometricIKSolver`, `IKResult`, `IKSolution`.

### Changed — Control pipeline

- **`gesture_arm/run.py`** — arm-control path now implements a three-stage
  priority cascade:
  1. `GeometricIKSolver` (IK mode, if `--ik` or `kinematics.enabled: true`)
  2. `LSTMStabilizer` (if trained model loaded and buffer full)
  3. `BaselineMapper` (always available fallback)

  IK gracefully falls through to LSTM/baseline on
  `UNREACHABLE`, `IN_DEADZONE`, or `JOINT_LIMIT` results.

- **`gesture_arm/run.py`** — added `--ik` CLI flag (overrides
  `kinematics.enabled` in config without editing the YAML file).

- **`gesture_arm/run.py`** — HUD: method badge now shows three colours
  (blue = lstm, amber = ik, grey = baseline); "IK MODE" banner displayed
  at top-centre of frame when IK is active.

- **`gesture_arm/run.py`** — module docstring updated to document all run
  modes including `--ik --no-hardware` combo.

### Changed — Configuration

- **`gesture_arm/config/default.yaml`** — added `kinematics:` section:
  ```yaml
  kinematics:
    enabled: false
    link1_cm: 10.0
    link2_cm: 8.0
    servo_x_neutral_deg: 120.0
    servo_y_zero_deg: 40.0
  ```

- **`gesture_arm/config/settings.py`** — added `IKConfig` dataclass and
  wired it into `AppConfig` and `load_config()`.

### Changed — Tests

- **`tests/test_core.py`** — added `TestGeometricIKSolver` with 23 tests
  covering: reachability (OK, UNREACHABLE, IN_DEADZONE, JOINT_LIMIT), angle
  bounds, direction consistency (left/right/forward/elevation), gripper
  passthrough, FK consistency (forward pointing, elevation, left rotation),
  IK θ₁ direction consistency, workspace bounds, hand-position-to-target
  finite output, invalid constructor arguments, and config loading.
  All tests pass without hardware, network, or TensorFlow.

### Changed — Documentation

- **`docs/ARCHITECTURE.md`** — added Section 11: Geometric IK Module,
  covering arm morphology, coordinate frame, IK equations, integration
  cascade diagram, activation instructions, and link-length tuning.

- **`docs/RESEARCH.md`** — fully updated: paper title updated to include IK;
  paper-to-code mapping table extended with all IK sections and equations;
  key equations section extended with IK equations (15–24); reproducing
  results updated for three-mode comparison; limitations updated; roadmap
  updated with two-link elbow IK and Kalman filter baseline.

- **`docs/SYSTEM_DESIGN.md`** — updated Section 2 (package structure) and
  Section 10 (what was deliberately left out) to reflect IK addition and
  note the elbow-IK extension as a roadmap item.

- **`docs/API_REFERENCE.md`** — added full `gesture_arm.kinematics.ik_solver`
  section covering `GeometricIKSolver`, `IKResult`, `IKSolution`, and all
  public methods with parameters, return types, and usage examples.

- **`docs/SETUP.md`** — added IK mode section: how to measure link lengths,
  update config, and activate via CLI flag.

- **`README.md`** — updated: title includes IK; architecture diagram updated;
  results table shows all three modes; quickstart shows `--ik` flag; new
  IK mode section in controls table.

- **`CHANGELOG.md`** — this entry.

- **`gesture_arm/__init__.py`** — version bumped `1.0.0` → `1.1.0`; `kinematics`
  added to `__all__`; docstring updated.

- **`pyproject.toml`** — version bumped to `1.1.0`.

### Paper

- Title updated: *"...with LSTM-Based Temporal Stabilization and Geometric
  Inverse Kinematics for Real-Time Robotics"*
- Abstract updated to describe both contributions.
- Section III-A: Eq (1) (feature vector) restored — had been dropped.
- Section IV restructured: A (Feature Extraction), B (Baseline), C (LSTM),
  D (Training), E (Metrics), F (IK equations 15–22), G (Gesture mapping
  Eqs 23–24), H (Cascade).
- Section VI: Table II extended to three methods; IK reachability distribution
  subsection added.
- Section VII: IK vs LSTM complementary analysis; workspace calibration
  sensitivity; elbow-IK limitation; Kalman filter comparison discussion.
- Fixes: year corrected to 2025; duplicated sentence in IV-D removed;
  "pneumatic gel muscles" sentence removed from abstract; reference [13]
  flagged for verification.

---

## [1.0.0] — 2025

Initial public release.

### Added

- `gesture_arm` Python package with six submodules: `vision`, `models`,
  `hardware`, `speech`, `evaluation`, `config`
- `HandTracker` — cvzone/MediaPipe wrapper emitting typed `HandState`
  dataclasses with normalized 42-dimensional feature vectors
- `LSTMStabilizer` — sliding-window LSTM temporal stabilization (core
  contribution); reduces control variance S by ~30% vs baseline
- `BaselineMapper` — direct linear frame-by-frame mapping used as comparison
  baseline and warm-up fallback
- `ArmController` — typed interface to three SG90/MG996R servos via
  pyFirmata; angle clamping as hardware safety backstop
- `BaseController` — L298N motor driver interface
- `TTSEngine` — non-blocking pyttsx3 TTS in a daemon thread
- `ASRListener` — continuous Google Speech Recognition in a daemon thread
- `MetricsLogger` — real-time CSV logging of servo commands, latency L,
  and stability variance S
- `AppConfig` — typed dataclass hierarchy loaded from `config/default.yaml`
- `scripts/collect.py` — training data collection
- `scripts/train.py` — LSTM training with early stopping
- `notebooks/benchmark_analysis.ipynb` — evaluation plots
- `firmware/server.ino` — StandardFirmata sketch for Arduino Uno
- `docker/Dockerfile.sim` — simulation image
- `.github/workflows/ci.yml` — CI pipeline
- Full documentation suite: ARCHITECTURE, SETUP, SYSTEM_DESIGN,
  API_REFERENCE, HARDWARE, TROUBLESHOOTING, RESEARCH, CONTRIBUTING

---

## [Unreleased]

### Planned

- Two-link elbow IK extension (law of cosines, elbow-up/elbow-down)
- Kalman filter stabilizer baseline for LSTM comparison
- Transformer-based temporal stabilizer (TFT)
- Few-shot user adaptation (10s re-calibration)
- ROS2 publisher node (`/servo_angles`, `/cmd_vel`)
- Offline ASR via Vosk
