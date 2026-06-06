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
Webcam (1280×720, ~30 fps)
    │
    ▼
HandTracker  ←  cvzone / MediaPipe
    │  HandState: features (42,) normalized  |  landmarks (21,3)  |  pinch_distance
    │
    ▼
┌─────────────────── IK layer (optional, --ik) ────────────────────┐
│  hand_position_to_target()  →  GeometricIKSolver.solve(px,py,pz) │
│  reachable? ──yes──► angles[ ]    not reachable? ──► fallthrough  │
└──────────────────────────────────────────────────────────────────┘
    │  (if IK disabled or target unreachable)
    ▼
LSTMStabilizer   ←  k=15 sliding window  →  LSTM(64)→Dense(32)→Dense(3,σ)
    │  or BaselineMapper  (warm-up / fallback)
    ▼
ArmController.write(angles)  →  pyFirmata  →  Arduino Uno
    │  SG90 servos: X→D3  Y→D5  Z→D6  |  L298N motors: D7–D13
    ▼
MetricsLogger  →  data/metrics_log.csv  (S, L, method per frame)

ASRListener  ──────────────────────────────► BaseController
TTSEngine    ◄─────────────────────────────── command confirmation
```


 
---



## Results

| Method | S (stability ↓) | L median (ms) ↓ | L p95 (ms) | Real-time |
|---|---|---|---|---|
| Baseline (linear, per-frame) | 18.4 | 52 | 90 | ✓ |
| LSTM (k = 15) | 12.8 | 40 | 68 | ✓ |
| **Geometric IK** | **10.1** | **41** | **70** | **✓** |
| Δ IK vs baseline | −45.1% | −21.2% | −22.2% | — |

Metrics: **S** = `(1/T) Σ(u_t − ū)²` (lower = smoother).  **L** = `t_actuation − t_capture` (ms).

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


## Controls

### Right hand → Mobile base (left camera zone)

| Gesture | Action |
|---|---|
| Hand in upper zone | Forward |
| Hand in lower zone | Reverse |
| Hand in left zone | Turn left |
| Hand in right zone | Turn right |
| Closed fist | **Stop** (highest priority override) |

### Left hand → Arm (right camera zone)

**LSTM / Baseline mode** (default):

| Motion | Servo | Range |
|---|---|---|
| Move left/right | X — pan | 60°–180° |
| Move up/down | Y — tilt | 40°–140° |
| Pinch open/close | Z — grip | 100°–150° |

**IK mode** (`--ik` flag):

| Motion | Effect |
|---|---|
| Move hand anywhere in zone | Arm tip moves to corresponding Cartesian position |
| Pinch open/close | Grip servo (pass-through, unchanged) |
| Hand outside workspace | Graceful fallback to LSTM/baseline |

### Voice commands

| Word | Action |
|---|---|
| "forward" / "go" | Move forward |
| "reverse" / "back" | Move backward |
| "left" | Turn left |
| "right" | Turn right |
| "stop" | Stop motors |

---

## IK mode: tuning link lengths

Measure your physical arm and update `gesture_arm/config/default.yaml`:

```yaml
kinematics:
  enabled: false           # or use --ik flag at runtime
  link1_cm: 10.0           # shoulder pivot → end-effector mount
  link2_cm: 8.0            # end-effector mount → TCP (fingertip)
  servo_x_neutral_deg: 120.0
  servo_y_zero_deg: 40.0
```

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

Total BOM cost: ~$45 USD.

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
│   ├── vision/         # HandTracker — landmark extraction + normalization (Eq. 1)
│   ├── models/         # LSTMStabilizer (Eqs. 3–12), BaselineMapper (Eq. 2), train()
│   ├── kinematics/     # GeometricIKSolver (Eqs. 15–24)  ← new in v1.1
│   ├── hardware/       # ArmController, BaseController (pyFirmata)
│   ├── speech/         # ASRListener (thread), TTSEngine (thread)
│   ├── evaluation/     # MetricsLogger — S (Eq. 13), L (Eq. 14) → CSV
│   ├── config/         # default.yaml + typed settings dataclasses
│   └── run.py          # Main control loop — IK → LSTM → baseline cascade
├── scripts/
│   ├── collect.py      # Training data collection
│   └── train.py        # LSTM training
├── tests/
│   └── test_core.py    # pytest unit tests (hardware-free, 23 IK tests)
├── notebooks/
│   └── benchmark_analysis.ipynb   # Reproduces Table II from metrics_log.csv
├── firmware/
│   └── server.ino      # StandardFirmata for Arduino Uno
├── docker/
│   └── Dockerfile.sim  # Simulation image (no Arduino needed)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── HARDWARE.md
│   ├── RESEARCH.md     # Paper-to-code mapping, all equations
│   ├── SETUP.md
│   ├── SYSTEM_DESIGN.md
│   └── TROUBLESHOOTING.md
└── .github/workflows/
    └── ci.yml          # Lint + test + Docker build on every push
```

---

## Benchmark notebook

```bash
jupyter notebook notebooks/benchmark_analysis.ipynb
```

Reads `data/metrics_log.csv` and produces:
- Rolling stability S comparison (all three modes)
- Latency L histogram
- Servo trajectory plots
- Table II (auto-generated, matches the paper)

---

## Configuration

All parameters in `gesture_arm/config/default.yaml`. Override the serial port without editing the file:

```bash
GESTURE_ARM_PORT=/dev/ttyUSB0 python -m gesture_arm.run --ik
```


---

## Running tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=gesture_arm   # with coverage
```

All 23 IK tests and all existing tests run without hardware, network, or TensorFlow.


---

## Documentation

| Doc | Contents |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Hardware assembly, installation, first run, IK tuning |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Data flow, LSTM design, IK model, threading |
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Engineering decisions, trade-offs |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | All public classes and methods |
| [docs/HARDWARE.md](docs/HARDWARE.md) | BOM, wiring diagrams, power supply |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Paper-to-code mapping, all equations |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Error messages and fixes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, PR process |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

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
@article{Mattral2026gesture,
  title   = {A Low-Cost Multimodal Gesture Control System with
             LSTM-Based Temporal Stabilization for Real-Time Robotics},
  author  = {Min Htet Myet},
  year    = {2025}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
