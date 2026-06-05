

# Gesture Arm

**Real-time multimodal robot control via hand gestures and voice, with LSTM temporal stabilization and geometric inverse kinematics.**

[![CI](https://github.com/Mattral/gesture_arm/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/gesture_arm/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- Replace with your actual demo GIF -->
<p align="center">
  <img src="demo.gif" alt="Gesture Arm Demo" width="300"/>
</p>

---

## What it does

A 3DoF robotic arm + mobile base controlled by:

| Input | Controls | Hardware |
|---|---|---|
| **Left hand** (in right camera zone) | Arm — pan, tilt, grip | 3× servo motors |
| **Right hand** (in left camera zone) | Mobile base — forward, reverse, turn | L298N + 2× DC motors |
| **Voice** | Any base direction + stop | Microphone |


Frame-by-frame landmark coordinates are fed through a sliding-window LSTM that smooths jitter before writing to servo pins, reducing control variance S by ~30% vs direct mapping. An optional geometric IK mode (--ik) lets the operator point to a Cartesian end-effector position; angles are solved analytically (O(1), two atan2 calls) with workspace reachability checking, achieving ~45% variance reduction.

---

## Architecture

```
Webcam (1280×720)
    │
    ▼
HandTracker          ← cvzone / MediaPipe, 21 landmarks
    │  features (42,) = normalized [x/W, y/H] × 21
    ▼
GeometricIKSolver    ← (optional, --ik flag) hand pos → TCP → joint angles
    │  IKSolution.OK → servo angles directly
    │  UNREACHABLE / JOINT_LIMIT / IN_DEADZONE → fallthrough ↓
    ▼
LSTMStabilizer       ← seq-15 sliding window → smoothed û_t
    │  or BaselineMapper (fallback / comparison)
    ▼
ArmController        ← pyfirmata → Arduino → servos (pins 3, 5, 6)
BaseController       ← pyfirmata → Arduino → L298N  (pins 7–13)
    │
    ▼
MetricsLogger        ← latency L, stability S, method tag → CSV

ASRListener  ──────────────────────────────► BaseController
TTSEngine    ◄─────────────────────────────── command feedback
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Mattral/gesture_arm.git
cd gesture_arm
pip install -e ".[ml,dev]"      # ml = TensorFlow; dev = pytest, notebooks
```

### 2. Flash firmware

Open `firmware/server.ino` in the Arduino IDE and upload it to your Arduino Uno.
Set the serial port in `gesture_arm/config/default.yaml`:

```yaml
hardware:
  port: "COM6"    # Windows: COMx   Linux/macOS: /dev/ttyUSBx
```

### 3. Collect training data

```bash
python scripts/collect.py --duration 90
```

Move your **left hand** slowly across the full range — left/right, up/down, open/close grip.
Data is saved to `data/training_data.csv`.

### 4. Train the LSTM

```bash
python scripts/train.py
```

Model is saved to `models/lstm_gesture_model.h5`.
Training takes ~2 minutes on CPU for 90 seconds of data.

### 5. Run

```bash
python -m gesture_arm.run
# Demo mode (no Arduino):
python -m gesture_arm.run --no-hardware
```

Press **Q** to quit. Metrics summary is printed on exit.

---

## Hardware

| Component | Part | Pin(s) |
|---|---|---|
| Microcontroller | Arduino Uno (StandardFirmata) | — |
| Arm servo X | SG90 / MG996R | D3 |
| Arm servo Y | SG90 / MG996R | D5 |
| Arm servo Z (grip) | SG90 / MG996R | D6 |
| Motor driver | L298N | D7, D8, D9 (left) · D10, D12, D13 (right) |
| Camera | Any USB webcam | USB |
| Microphone | Any USB/built-in mic | USB |

Total BOM cost: ~$25 USD.

---

## Evaluation results

Run the benchmark notebook after collecting data:

```bash
jupyter notebook notebooks/benchmark_analysis.ipynb
```

| Method | S (stability ↓) | L mean (ms) ↓ | L p95 (ms) |
|---|---|---|---|
| Baseline (frame-by-frame) | ~18.4 | ~55 | ~90 |
| **LSTM stabilized** | **~12.8** | **~42** | **~68** |
| **Geometric IK** | **~10.1** | **~43** | **~70** |

Metrics definitions (paper Section VI):
- **S** = `(1/T) Σ (u_t − ū)²` — rolling variance of servo commands
- **L** = `t_actuation − t_capture` — end-to-end frame-to-servo latency

---

## Project structure

```
gesture_arm/
├── gesture_arm/
│   ├── vision/         # HandTracker — landmark extraction + normalization
│   ├── models/         # LSTMStabilizer, BaselineMapper, train()
│   ├── hardware/       # ArmController, BaseController (pyfirmata)
│   ├── speech/         # ASRListener (thread), TTSEngine (thread)
│   ├── evaluation/     # MetricsLogger — S, L → CSV
│   ├── config/         # default.yaml + typed settings dataclasses
│   └── run.py          # Main control loop entry point
├── scripts/
│   ├── collect.py      # Data collection mode
│   └── train.py        # LSTM training
├── tests/
│   └── test_core.py    # pytest unit tests (hardware-free)
├── notebooks/
│   └── benchmark_analysis.ipynb
├── firmware/
│   └── server.ino      # StandardFirmata for Arduino Uno
├── docker/
│   └── Dockerfile.sim  # Simulation image (no Arduino needed)
└── .github/workflows/
    └── ci.yml          # Lint + test + Docker build on every push
```

---

## Configuration

All parameters live in `gesture_arm/config/default.yaml`. No hardcoded constants in source.

Override the serial port without editing the file:
```bash
GESTURE_ARM_PORT=/dev/ttyUSB0 python -m gesture_arm.run
```

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=gesture_arm   # with coverage
```

Tests run without hardware or TensorFlow — safe for CI.

---

## Roadmap

- [ ] Transformer-based stabilizer (attention over landmark sequence)
- [ ] Few-shot user adaptation (10-second re-calibration)
- [ ] ROS2 publisher node for integration with full robot stacks
- [ ] Web dashboard for live metrics visualization

---

## Citation

If you use this work, please cite:

```bibtex
@article{Mattral2025gesture,
  title   = {A Low-Cost Multimodal Gesture Control System with
             LSTM-Based Temporal Stabilization for Real-Time Robotics},
  author  = {Min Htet Myet},
  year    = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
