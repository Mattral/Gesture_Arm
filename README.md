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

Left hand controls the arm (pan, tilt, grip). Right hand controls the mobile base (forward, reverse, turn). Voice commands work in parallel for any direction plus stop. All three inputs run simultaneously.

---


## Why this is different from other gesture control projects

Most gesture-control robotics projects map hand landmarks directly to servo angles frame-by-frame. This creates jitter — the servo hunts continuously because MediaPipe produces slightly different landmark coordinates on every frame even with a still hand.

This project adds a **sliding-window LSTM** between the landmark extractor and the servo controller. The LSTM is trained on your own gesture data (90 seconds to collect, 2 minutes to train on CPU), learns the smooth intent behind your movement, and outputs stabilized servo commands.

Measured result on the included benchmark:

| Method | Stability variance S (↓) | Mean latency ms (↓) | p95 latency ms |
|---|---|---|---|
| Direct mapping (baseline) | ~18.4 | ~55 | ~90 |
| **LSTM stabilized** | **~12.8** | **~42** | **~68** |

~30% reduction in variance. ~25% reduction in latency (the LSTM runs fast enough that the sequence buffer adds less time than the jitter-correction loop it replaces).

**Caveat:** These numbers are from a single hardware setup. Your numbers will differ based on your webcam, CPU, and Arduino serial speed. The benchmark notebook generates these figures from your own collected data.

---

## Hardware

Total BOM: **~$45 USD**

| Component | Part | Pin(s) |
|---|---|---|
| Microcontroller | Arduino Uno (StandardFirmata) | — |
| Arm servo X | SG90 or MG996R | D3 |
| Arm servo Y | SG90 or MG996R | D5 |
| Arm servo Z (grip) | SG90 or MG996R | D6 |
| Motor driver | L298N | D7, D8, D9 (left) · D10, D12, D13 (right) |
| Camera | Any USB webcam (720p minimum) | USB |
| Microphone | Any USB or built-in mic | USB |

You can run and test everything **without the hardware** using `--no-hardware` mode — the software stack runs fully in simulation.


---

---

## Quickstart (5 steps, ~10 minutes)

### 1. Install

```bash
git clone https://github.com/Mattral/Gesture_Arm
cd Gesture_Arm
pip install -e ".[ml,dev]"   # ml = TensorFlow for LSTM; dev = pytest + notebooks
```

### 2. Try it immediately — no hardware needed

```bash
python -m gesture_arm.run --no-hardware
```

Opens the webcam, runs hand tracking, and prints what servo commands would be sent. No Arduino required.

### 3. Flash firmware (hardware only)

Open `firmware/server.ino` in Arduino IDE. Upload to your Arduino Uno. Set the serial port in `gesture_arm/config/default.yaml`:

```yaml
hardware:
  port: "COM6"      # Windows: COMx  |  Linux/macOS: /dev/ttyUSBx
```

### 4. Collect training data and train the LSTM

```bash
# 90 seconds of gesture data collection
python scripts/collect.py --duration 90

# Train LSTM (~2 minutes on CPU)
python scripts/train.py
# Saves to: models/lstm_gesture_model.h5
```

Move your left hand slowly through the full range during collection: left/right, up/down, open/close grip.

### 5. Run

```bash
python -m gesture_arm.run
# Press Q to quit. Metrics summary prints on exit.
```

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

The key design decision: the LSTM operates on a **sliding window of 15 frames** rather than the current frame alone. This gives it temporal context to distinguish intentional movement from hand tremor or tracking noise. The window slides forward one frame at a time, so there's no batch latency — inference runs on every frame.

 
---

## LSTM stabilizer

The stabilizer is a lightweight single-layer LSTM trained per-user on their own gesture style. This matters: MediaPipe landmark coordinates vary between individuals based on hand size, camera angle, and lighting. A model trained on your data outperforms a pre-trained model on someone else's data.

```python
# gesture_arm/models/lstm_stabilizer.py — the core
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, N_FEATURES)),
    Dense(N_OUTPUTS, activation='linear')
])
```

Training input: sequences of 15 frames of 42-dim landmark vectors.
Training target: smoothed servo angles (moving-average of the next 5 frames, a simple teacher signal).
Training time: ~2 minutes on CPU for 90 seconds of data.

---


## Stability metric

S is defined per paper Section VI:

```
S = (1/T) Σ_t (u_t − ū)²
```

where u_t is the servo command at frame t and ū is the rolling mean over a 1-second window. Lower is better — high S means the servo is hunting. The baseline direct-mapping score (~18.4) is dominated by MediaPipe's per-frame landmark jitter. The LSTM score (~12.8) reflects only intentional movement variance.

---

## Voice commands

ASR runs in a background thread (Python `speech_recognition`, Google Speech API by default). Any of these phrases trigger the base controller:

| Phrase | Action |
|---|---|
| "forward" / "go" | Base forward |
| "back" / "reverse" | Base reverse |
| "left" | Base turn left |
| "right" | Base turn right |
| "stop" | All motors stop |

Voice and hand control run simultaneously — you can steer with your right hand and say "stop" to halt immediately.


---

## Simulation / Docker

No Arduino? No problem.

```bash
# Docker simulation (no hardware at all)
docker build -f docker/Dockerfile.sim -t gesture_arm_sim .
docker run --device /dev/video0 gesture_arm_sim   # passes through webcam

# Or just use --no-hardware locally
python -m gesture_arm.run --no-hardware
```

---

## Running tests

```bash
pytest tests/ -v                          # all tests, no hardware required
pytest tests/ -v --cov=gesture_arm        # with coverage report
```

Tests are hardware-free and TensorFlow-free — safe for CI, fast to run locally.

---

## Benchmark your own setup

```bash
jupyter notebook notebooks/benchmark_analysis.ipynb
```

Reads `data/metrics_log.csv` and produces:
- Rolling stability S comparison (all three modes)
- Latency L histogram
- Servo trajectory plots
- Table II (auto-generated, matches the paper)

---

## Evaluation results

| Method | S (stability ↓) | L mean (ms) ↓ | L p95 (ms) |
|---|---|---|---|
| Baseline (frame-by-frame) | ~18.4 | ~55 | ~90 |
| **LSTM stabilized** | **~12.8** | **~42** | **~68** |
| **Geometric IK** | **~10.1** | **~43** | **~70** |

Metrics definitions (paper Section VI):
- **S** = `(1/T) Σ (u_t − ū)²` — rolling variance of servo commands
- **L** = `t_actuation − t_capture` — end-to-end frame-to-servo latency

---

## Configuration

Everything in `gesture_arm/config/default.yaml`. No hardcoded constants in source files.

Override without editing:
```bash
GESTURE_ARM_PORT=/dev/ttyUSB0 python -m gesture_arm.run
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
  year    = {2026}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
