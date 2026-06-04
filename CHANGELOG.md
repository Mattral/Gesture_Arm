# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2025

Initial public release.

### Added

- `gesture_arm` Python package with six submodules: `vision`, `models`, `hardware`, `speech`, `evaluation`, `config`
- `HandTracker` — cvzone/MediaPipe wrapper emitting typed `HandState` dataclasses with normalized 42-dimensional feature vectors
- `LSTMStabilizer` — sliding-window LSTM temporal stabilization (core contribution); reduces control variance S by ~30% vs baseline
- `BaselineMapper` — direct linear frame-by-frame mapping used as comparison baseline and warm-up fallback
- `ArmController` — typed interface to three SG90/MG996R servos via pyFirmata; angle clamping as hardware safety backstop
- `BaseController` — L298N motor driver interface with `forward()`, `reverse()`, `turn_left()`, `turn_right()`, `stop()` methods
- `TTSEngine` — non-blocking pyttsx3 TTS in a daemon thread; suppresses consecutive duplicate utterances
- `ASRListener` — continuous Google Speech Recognition in a daemon thread with substring command matching
- `MetricsLogger` — real-time CSV logging of servo commands, latency L, and stability variance S
- `AppConfig` — typed dataclass hierarchy loaded from `config/default.yaml`; all parameters configurable without code changes
- `scripts/collect.py` — 90-second guided training data collection with live progress bar
- `scripts/train.py` — LSTM training with early stopping and model checkpointing
- `notebooks/benchmark_analysis.ipynb` — four publication-ready evaluation plots reproducing paper Section VI results
- `firmware/server.ino` — StandardFirmata sketch for Arduino Uno
- `docker/Dockerfile.sim` — simulation image; runs `--no-hardware` demo without Arduino
- `.github/workflows/ci.yml` — CI pipeline: ruff lint, black format check, mypy type check, pytest across Python 3.9/3.10/3.11, Docker build
- `pytest` unit tests covering config loading, feature extraction, baseline mapper, LSTM buffer logic, metrics logger; all hardware-free

### Architecture decisions

- Single YAML config file; environment variable override for serial port (`GESTURE_ARM_PORT`)
- Immutable `HandState` / `TrackerOutput` dataclasses for thread-safe inter-module communication
- Optional TensorFlow dependency; graceful baseline fallback when not installed
- `--no-hardware` flag for CI and demo without physical robot
- `board_session()` context manager guaranteeing clean Arduino disconnect on exit

---

## [Unreleased]

### Planned

- Transformer-based temporal stabilizer with attention over the landmark sequence
- Offline ASR via Vosk (no internet required)
- ROS2 publisher node (`/servo_angles`, `/cmd_vel` topics)
- Few-shot user adaptation for 10-second re-calibration
- Web metrics dashboard (FastAPI + WebSocket)
