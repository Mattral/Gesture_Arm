# System Architecture

This document describes the full technical architecture of the Gesture Arm system — how data flows from camera pixels to physical servo motion, how each module is designed, and the reasoning behind each design decision.

---

## Table of contents

1. [High-level overview](#1-high-level-overview)
2. [Runtime data flow](#2-runtime-data-flow)
3. [Module breakdown](#3-module-breakdown)
4. [LSTM temporal stabilization](#4-lstm-temporal-stabilization)
5. [Dual-hand control model](#5-dual-hand-control-model)
6. [Threading model](#6-threading-model)
7. [Hardware communication layer](#7-hardware-communication-layer)
8. [Configuration system](#8-configuration-system)
9. [Evaluation pipeline](#9-evaluation-pipeline)
10. [Design decisions and trade-offs](#10-design-decisions-and-trade-offs)

---

## 1. High-level overview

The system is a real-time closed-loop controller. On every camera frame (~30 fps), it:

1. Detects hand landmarks using MediaPipe (via cvzone)
2. Extracts and normalizes a 42-dimensional feature vector
3. Passes the vector through an LSTM stabilizer that smooths jitter across a 15-frame sliding window
4. Writes the smoothed servo angles to the Arduino over serial (pyFirmata / StandardFirmata)
5. Logs the command and its latency to CSV for evaluation

A parallel speech thread continuously listens for voice commands and dispatches them to the motor controller independently of the vision pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST COMPUTER                            │
│                                                                 │
│  Webcam ──► HandTracker ──► LSTMStabilizer ──► ArmController   │
│                                     │                           │
│                              BaselineMapper                     │
│                             (warm-up / fallback)               │
│                                                                 │
│  Microphone ──► ASRListener ──────────────────► BaseController  │
│                     │                                           │
│                  TTSEngine ◄── command confirmation             │
│                                                                 │
│  MetricsLogger ◄──────────────── every servo write             │
└───────────────────────────────────┬─────────────────────────────┘
                                    │ USB Serial (57600 baud)
                              ┌─────▼──────┐
                              │  Arduino   │
                              │    Uno     │
                              │            │
                              │ Servo X ───► pin 3
                              │ Servo Y ───► pin 5
                              │ Servo Z ───► pin 6
                              │ L298N  ───► pins 7–13
                              └────────────┘
```

---

## 2. Runtime data flow

### Per-frame pipeline (vision thread, ~30 fps)

```
cap.read()
    │  BGR frame (1280×720)
    ▼
HandTracker.process(frame)
    │  Runs MediaPipe Hands
    │  Returns TrackerOutput(left: HandState, right: HandState)
    │
    ├── right hand present?
    │       │  palm_center (x, y)
    │       ▼
    │   BaseController dispatch
    │       ├── y < cy*0.75  → forward()
    │       ├── y > cy*1.25  → reverse()
    │       ├── x < cx*0.75  → turn_left()
    │       ├── x > cx*1.25  → turn_right()
    │       └── fist         → stop()
    │
    └── left hand present?
            │  features: (42,) normalized landmarks
            ▼
        LSTMStabilizer.update(features)
            │
            ├── buffer < 15 frames → BaselineMapper.map(landmarks) → angles
            │
            └── buffer = 15 frames → model.predict(seq) → angles
                    │
                    ▼
                ArmController.write(angles)   ← clamped to servo bounds
                    │
                    ▼
                MetricsLogger.log(angles, t_capture)
```

### Speech pipeline (daemon thread, continuous)

```
Microphone
    │  raw audio stream
    ▼
sr.Recognizer.listen()
    │  blocks until phrase detected
    ▼
Google Speech API (online) or local engine (offline fallback)
    │  transcribed text
    ▼
ASRListener._match(text)
    │  substring match against command vocabulary
    ▼
handle_speech_command(cmd)
    │
    ├── BaseController.forward() / reverse() / turn_left() / turn_right()
    └── TTSEngine.say("Command: {cmd}")  ← non-blocking queue
```

---

## 3. Module breakdown

### `gesture_arm/vision/tracker.py`

**Responsibility:** Convert a raw camera frame into structured `HandState` objects.

**Key design:** The `HandTracker` class owns the MediaPipe detector and exposes a single `process(frame) -> TrackerOutput` method. All downstream code works with `HandState` dataclasses — immutable, typed snapshots — never with raw landmark lists. This makes the vision layer independently testable without a camera.

**Feature vector construction:**

```
lm = hand['lmList']                    # 21 landmarks, each [x_px, y_px, z_raw]
features = flatten([x/W, y/H] × 21)   # → (42,) float32, values in [0, 1]
```

Normalization by frame dimensions makes the model resolution-independent — the same trained model works whether the camera is 720p or 1080p.

---

### `gesture_arm/models/stabilizer.py`

**Responsibility:** LSTM-based temporal smoothing and the baseline linear mapper.

See [Section 4](#4-lstm-temporal-stabilization) for the full model description.

---

### `gesture_arm/hardware/arduino.py`

**Responsibility:** Abstract all pin-level hardware access behind typed controllers.

**Key design:** Two classes, one job each:
- `ArmController` — writes to three servo pins, clamps angles to configured bounds, exposes `write(angles)` and `home()`
- `BaseController` — drives the L298N motor driver, exposes `forward()`, `reverse()`, `turn_left()`, `turn_right()`, `stop()`

No pin numbers appear anywhere outside this module. The `board_session()` context manager guarantees `board.exit()` is called even if the process is killed.

---

### `gesture_arm/speech/multimodal.py`

**Responsibility:** Non-blocking speech I/O.

**Key design:**
- `ASRListener` runs `recognizer.listen()` in a daemon thread, never blocking the vision loop
- `TTSEngine` drains a queue in a daemon thread; consecutive duplicate messages are suppressed
- Both classes accept callbacks / queues as constructor arguments — easy to swap the speech backend without touching the control logic

---

### `gesture_arm/evaluation/metrics.py`

**Responsibility:** Collect, store, and report the paper's evaluation metrics.

**Key design:** Every `log()` call appends one row to CSV immediately (no buffering), so data is never lost on crash. The `stability()` and `avg_latency()` methods compute rolling statistics over the last N frames for real-time HUD display.

---

### `gesture_arm/config/settings.py`

**Responsibility:** Load YAML configuration into typed Python dataclasses.

**Key design:** All parameters in one file (`config/default.yaml`). The loader validates structure at startup and raises clear errors for missing fields. The serial port can be overridden via the `GESTURE_ARM_PORT` environment variable without editing any file — important for Docker and CI.

---

## 4. LSTM temporal stabilization

This is the core technical contribution of the system.

### Problem: hand tracking jitter

Raw MediaPipe landmark coordinates fluctuate by ±5–15 pixels per frame even when the hand is held still. When mapped linearly to servo angles, this produces rapid small oscillations that:
- Wear servo gears and increase power consumption
- Make the arm look uncontrolled
- Reduce positioning accuracy

### Solution: sliding-window LSTM

Rather than mapping each frame independently, we feed a **sequence of 15 consecutive frames** through an LSTM and let it learn the smoothed trajectory.

```
Frame t-14  →  ┐
Frame t-13  →  │  sequence X_t ∈ ℝ^(15×42)
   ⋮         →  │
Frame t     →  ┘
                │
                ▼
           LSTM(64 units)   h_t ∈ ℝ^64
                │
                ▼
           Dense(32, relu)
                │
                ▼
           Dense(3, sigmoid)   û_t ∈ [0,1]^3
                │
                ▼
           denormalize → [servoX°, servoY°, servoZ°]
```

**Paper equations:**

```
X_t  = [x_{t-k}, x_{t-k+1}, …, x_t]     (sliding window of feature vectors)
h_t  = LSTM(X_t)                          (hidden state encodes temporal context)
û_t  = sigmoid(W · h_t + b)              (normalized servo command)
```

### Training

The LSTM is trained in a supervised fashion using ground-truth labels generated by the baseline mapper:

1. Collect 90+ seconds of hand movement with `scripts/collect.py`
2. The collector records `(features, baseline_angles)` pairs to CSV
3. `scripts/train.py` builds 15-frame sliding-window sequences and fits the model

The rationale for using baseline labels: we want the LSTM to reproduce the *intended* motion while suppressing the frame-to-frame noise, not to predict a fundamentally different motion.

### Warm-up

The LSTM buffer requires 15 frames before it can predict. During warm-up, the system transparently falls back to the baseline mapper. The `method` field in the metrics CSV records which path produced each command.

---

## 5. Dual-hand control model

The camera frame is divided into two vertical zones:

```
┌──────────────────┬──────────────────┐
│                  │                  │
│   RIGHT HAND     │   LEFT HAND      │
│   Mobile base    │   Arm control    │
│   (gesture zone) │   (gesture zone) │
│                  │                  │
│   0 … W/2        │   W/2 … W        │
└──────────────────┴──────────────────┘
```

**Right hand** (must be in the left camera zone, i.e. x < W/2):

The palm centre (landmark 9) position drives the mobile base:

```
            y < cy - cy/4   →  forward
            y > cy + cy/4   →  reverse
x < cx - cx/4               →  left
x > cx + cx/4               →  right
fist (all fingers down)      →  stop  (safety override — highest priority)
```

**Left hand** (must be in the right camera zone, i.e. x > W/2):

The LSTM (or baseline mapper) converts the full 42-dim feature vector to three servo angles controlling pan (X), tilt (Y), and grip (Z).

**Z-axis grip mapping:**

```
pinch_distance = ‖landmark[4][:2] − landmark[8][:2]‖₂
servoZ = interp(pinch_distance, [20px, 220px], [100°, 150°])
```

Using thumb-tip to index-tip distance as grip proxy is consistent, robust to hand rotation, and naturally intuitive for users.

---

## 6. Threading model

The system runs three concurrent threads:

```
Main thread (vision loop)
├── cv2.VideoCapture.read()    — blocks until frame ready (~33ms at 30fps)
├── HandTracker.process()      — MediaPipe inference (~10–15ms)
├── LSTMStabilizer.update()    — LSTM inference (~5ms, CPU)
├── ArmController.write()      — serial write via pyFirmata
└── cv2.imshow() / waitKey()   — rendering

ASR daemon thread
├── sr.Recognizer.listen()     — blocks on microphone
├── Google Speech API call     — network I/O (~200–500ms)
└── handle_speech_command()    — dispatches to BaseController

TTS daemon thread
└── queue.get() → engine.say() — audio output (~1–3s per utterance)
```

**Thread safety:** The `last_motor_time` reference is shared between the main thread (gesture-based motor commands) and the ASR thread (speech-based motor commands). It is wrapped in a list `[time.time()]` so assignment is atomic in CPython (GIL protects single-element list item assignment). A `threading.Lock` would be the production-grade approach; this is noted in the code.

**Why daemon threads?** Both ASR and TTS are set as daemon threads (`daemon=True`). This means they are automatically killed when the main thread exits, preventing the process from hanging on `engine.runAndWait()` or a blocking `listen()` call at shutdown.

---

## 7. Hardware communication layer

### Protocol: StandardFirmata over serial

The Arduino runs `firmware/server.ino` — a standard Firmata sketch that exposes every pin over a simple serial protocol. The host Python process communicates via pyFirmata, which translates `pin.write(value)` calls into Firmata MIDI-like byte messages.

```
Python: servo_pinX.write(120)
    │
    ▼
pyFirmata: encode as ANALOG_MESSAGE (0xE0) + pin + value (7-bit pairs)
    │
    ▼
Serial (USB, 57600 baud, ~0.2ms per message)
    │
    ▼
Arduino: Firmata.processInput() → analogWriteCallback()
    │
    ▼
Servo.write(120)  →  1.5ms PWM pulse on pin 3
```

### Latency budget

| Stage | Typical time |
|---|---|
| Frame capture (USB webcam) | ~5 ms |
| MediaPipe hand detection | ~10–15 ms |
| LSTM inference (CPU) | ~5 ms |
| Serial write (pyFirmata) | ~1–2 ms |
| Arduino processing | ~1 ms |
| Servo mechanical response | ~100–200 ms |
| **Total frame-to-motion** | **~120–230 ms** |

The dominant latency is servo mechanical response, not software. The software pipeline (L) measured in the paper is the time from frame capture to `write()` call, which is ~40–55 ms.

### Pin assignment

| Pin | Mode | Connected to |
|---|---|---|
| D3 | Servo | Arm servo X (horizontal) |
| D5 | Servo | Arm servo Y (vertical) |
| D6 | Servo | Arm servo Z (grip) |
| D7 | Output | L298N left IN1 |
| D8 | Output | L298N left IN2 |
| D9 | PWM | L298N left ENA |
| D10 | PWM | L298N right ENB |
| D12 | Output | L298N right IN4 |
| D13 | Output | L298N right IN3 |

---

## 8. Configuration system

All runtime parameters live in `gesture_arm/config/default.yaml`. The loader (`settings.py`) parses this file into typed Python dataclasses at startup.

**Why YAML + dataclasses, not argparse or `.env`?**

- YAML is human-readable and supports nested structure (servo X has `pin`, `min_deg`, `max_deg`, `default` — hard to express in flat key-value)
- Dataclasses provide IDE autocomplete and catch typos at import time
- `argparse` is appropriate for run-time flags (`--no-hardware`, `--config`) but not for the 30+ parameters this system has
- The `GESTURE_ARM_PORT` environment variable override allows the serial port to be set in Docker / CI without modifying the file

**Override precedence:**

```
Environment variable  >  custom YAML (--config)  >  default.yaml
```

---

## 9. Evaluation pipeline

The `MetricsLogger` implements the two metrics reported in the paper:

### Stability variance S

```
S = (1/T) Σ_{t=1}^{T} (u_t − ū)²
```

where `u_t` is the 3-vector of servo angles at frame t, and `ū` is the rolling mean over the last T frames.

Lower S = smoother, more stable arm motion. The LSTM achieves lower S than the baseline because it averages over 15 frames rather than reacting to every noisy frame.

### End-to-end latency L

```
L = t_actuation − t_capture   (milliseconds)
```

where `t_capture = time.time()` before `cap.read()` returns, and `t_actuation = time.time()` immediately after `arm.write()`.

### Generating the paper's results table

```bash
# 1. Run baseline mode and collect metrics
python -m gesture_arm.run --no-lstm --out-metrics data/baseline_metrics.csv

# 2. Run LSTM mode and collect metrics  
python -m gesture_arm.run --out-metrics data/lstm_metrics.csv

# 3. Open the benchmark notebook
jupyter notebook notebooks/benchmark_analysis.ipynb
```

---

## 10. Design decisions and trade-offs

### Why pyFirmata + StandardFirmata instead of a custom protocol?

**Pro:** Zero Arduino code to maintain. StandardFirmata is battle-tested and handles servo, PWM, and digital I/O out of the box. pyFirmata gives a clean Python API.

**Con:** ~0.2ms per serial message (vs ~0.05ms for a custom binary protocol). For a ~30fps vision loop this is completely acceptable. A custom protocol would only matter if we needed >100 Hz update rates.

### Why cvzone instead of raw MediaPipe?

cvzone provides a simpler API (`findHands`, `fingersUp`) that reduces boilerplate. The underlying MediaPipe model is identical. Switching to raw MediaPipe would be straightforward — only `tracker.py` would need to change.

### Why train the LSTM on baseline labels instead of recording "ideal" motion?

Recording truly ideal motion would require an expert operator and a separate ground-truth system (e.g. optical motion capture). Using baseline labels is self-supervised: we collect data with the same system, then train the LSTM to reproduce the baseline's intended trajectory while filtering the noise. This is practical and reproducible with consumer hardware.

### Why a sigmoid output activation?

Servo angles are bounded. A sigmoid output maps to [0, 1], which is then linearly denormalized to the configured degree range. This prevents the model from ever predicting an out-of-bounds angle, even without explicit clamping — though `ArmController.write()` clamps anyway as a hardware safety backstop.

### Why not end-to-end learning (raw pixels → angles)?

The system separates hand tracking (MediaPipe) from motion smoothing (LSTM) deliberately. MediaPipe is a mature, optimized model that generalizes across users and lighting conditions. Training an end-to-end model from pixels would require orders of magnitude more data and a GPU. The modular design lets each component be replaced independently.
