# System Design

This document covers the engineering decisions behind the Gesture Arm codebase — why it is structured the way it is, the patterns used, and what trade-offs were made. It is intended for contributors and for anyone reviewing the code for research or hiring purposes.

---

## Table of contents

1. [Design principles](#1-design-principles)
2. [Package structure rationale](#2-package-structure-rationale)
3. [Data flow and the immutable state pattern](#3-data-flow-and-the-immutable-state-pattern)
4. [Error handling strategy](#4-error-handling-strategy)
5. [Dependency management](#5-dependency-management)
6. [Real-time constraints](#6-real-time-constraints)
7. [Testability without hardware](#7-testability-without-hardware)
8. [Configuration-driven design](#8-configuration-driven-design)
9. [The LSTM integration decision](#9-the-lstm-integration-decision)
10. [What was deliberately left out](#10-what-was-deliberately-left-out)
11. [Known limitations](#11-known-limitations)
12. [Planned improvements](#12-planned-improvements)

---

## 1. Design principles

Five principles guided every decision in this codebase.

**Single responsibility.** Each class does one thing. `HandTracker` detects hands. `LSTMStabilizer` smooths commands. `ArmController` writes to servos. None of them know about each other's internals.

**Fail gracefully.** The system runs in a degraded but functional state without TensorFlow (baseline mode), without a microphone (speech disabled), and without an Arduino (`--no-hardware` flag). A missing optional dependency is a warning, not a crash.

**No magic constants.** Every tunable value — servo bounds, sequence length, camera resolution, speech rate — lives in `config/default.yaml`. Source files contain zero hardcoded numbers.

**Typed interfaces.** Every inter-module boundary passes typed dataclasses, not raw dicts or lists. `HandState`, `TrackerOutput`, `AppConfig` — downstream code gets autocomplete, static analysis, and clear contracts.

**Observable by default.** The metrics logger is always on. Every servo write is timestamped and recorded. The system generates its own evaluation data during normal operation.

---

## 2. Package structure rationale

```
gesture_arm/
├── vision/        ← perception
├── models/        ← intelligence
├── hardware/      ← actuation
├── speech/        ← multimodal I/O
├── evaluation/    ← observability
├── config/        ← configuration
└── run.py         ← composition root
```

This mirrors the **hexagonal architecture** (ports and adapters) pattern. The core logic (`models/`) has no dependencies on any I/O system. It receives normalized numpy arrays and returns angle arrays. It has no idea whether those arrays came from a webcam or a unit test.

The `run.py` module is the only place where all the pieces are wired together. This means you can swap out any subsystem (e.g. replace pyFirmata with ROS2 publishers) by only changing the hardware module and the composition in `run.py`.

### Why not a single `main.py`?

The original codebase was a single 300-line script. This works fine for prototyping but has three problems at scale:

1. **Untestable.** You cannot unit test `np.interp(lmList[9][0], ...)` without a real camera running.
2. **Opaque.** A reviewer cannot understand the system by reading the file — they have to read every line to build a mental model.
3. **Brittle.** Changing the servo angle range requires finding every `np.interp` call and updating it. With typed config, you change one YAML value.

---

## 3. Data flow and the immutable state pattern

The `HandState` dataclass is frozen (`@dataclass(frozen=True)`). Once created by `HandTracker.process()`, it cannot be modified. This is intentional.

**Why immutable?**

In a threaded system, mutable shared state is a source of subtle bugs. If `HandState` were a mutable object and the speech thread happened to read it while the vision thread was writing landmark updates, the result would be a partially-updated state. Immutability eliminates this class of bug entirely.

**Why dataclasses instead of dicts?**

```python
# Bad — happens in the original codebase:
x = hand['lmList'][9][0]   # What is lmList[9]? What units is [0] in?

# Good — what the refactored code does:
x = hand_state.palm_center[0]   # Unambiguous. IDE shows the type. Tests can check it.
```

Dataclasses also serve as living documentation. Reading `HandState`'s field definitions tells you exactly what information the tracker produces — you do not need to trace through MediaPipe's output format.

---

## 4. Error handling strategy

The system distinguishes between three categories of failure:

### Category 1 — Configuration errors (fail at startup, loudly)

Missing YAML keys, wrong types, invalid servo ranges. These raise exceptions before any hardware is touched. Better to crash at startup with a clear message than to silently misbehave mid-operation.

```python
# settings.py
if not path.exists():
    raise FileNotFoundError(f"Config file not found: {path}")
```

### Category 2 — Optional dependency missing (warn, degrade gracefully)

TensorFlow missing → log warning, run in baseline mode. PyAudio missing → log error, disable speech. The system is still useful without these components.

```python
# stabilizer.py
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not found. Running in baseline mode.")
```

### Category 3 — Runtime I/O errors (log and continue)

Speech recognition network errors, occasional frame drops, serial hiccups. These are logged at WARNING level and the loop continues. A single failed frame is not a reason to stop the system.

```python
# multimodal.py
except sr.RequestError as exc:
    logger.warning("ASR service error: %s", exc)
    # loop continues
```

### Hardware safety

`ArmController.write()` always clamps angles to configured bounds before writing:

```python
clamped = float(np.clip(angles[i], lo, hi))
self._servos[axis].write(clamped)
```

This is a hard safety backstop. Even if the LSTM outputs a nonsensical value, the servo never moves outside its physical range.

---

## 5. Dependency management

Dependencies are split into three groups in `pyproject.toml`:

**Core** (always installed): `opencv-python`, `cvzone`, `numpy`, `pyfirmata`, `pyserial`, `SpeechRecognition`, `pyttsx3`, `PyYAML`. These are required for any functionality.

**ML** (`pip install -e ".[ml]"`): `tensorflow`. Optional — the system runs in baseline mode without it. Kept separate because TensorFlow is ~500MB and slow to install, which would break CI speed and Docker image size.

**Dev** (`pip install -e ".[ml,dev]"`): `pytest`, `mypy`, `ruff`, `black`, `jupyter`, `matplotlib`, `pandas`, `seaborn`. Only needed for development and benchmarking.

### Why `pyproject.toml` and not `requirements.txt`?

`requirements.txt` with pinned versions (e.g. `numpy==1.24.3`) creates an installation that works exactly once — until any other package in the environment has a different opinion about numpy's version. `pyproject.toml` with minimum-version bounds (`numpy>=1.24`) expresses the real constraint: "we need at least this version," which lets pip resolve a consistent environment for any user's machine.

A `requirements.txt` is still useful for reproducing exact environments (e.g. CI cache, Docker). One can be generated with:
```bash
pip freeze > requirements-lock.txt
```

---

## 6. Real-time constraints

The system must complete one iteration of the vision loop in under ~33ms to keep up with a 30fps camera. The time budget per frame:

| Operation | Budget | Typical actual |
|---|---|---|
| `cap.read()` | 5ms | 5ms |
| `HandTracker.process()` | 15ms | 10–15ms |
| `LSTMStabilizer.update()` | 8ms | 4–6ms |
| `ArmController.write()` | 3ms | 1–2ms |
| `MetricsLogger.log()` | 1ms | <1ms |
| `cv2.imshow()` | 3ms | 2–3ms |
| **Total** | **35ms** | **22–32ms** |

The LSTM inference (~5ms on CPU for a 15×42 sequence) is well within budget. If it were not, options would include: reducing sequence length, reducing LSTM units, or moving inference to a separate thread with a command queue.

**The most important constraint: never block on I/O in the main thread.**

This is why speech recognition runs in a daemon thread. `sr.Recognizer.listen()` blocks until a phrase is detected — sometimes for several seconds. If it ran in the main loop, the arm would freeze while the system waited for speech. The threading design ensures vision, arm control, and speech are all concurrent.

---

## 7. Testability without hardware

All unit tests in `tests/test_core.py` run with:

```bash
pytest tests/ -v
```

No Arduino. No camera. No microphone. No GPU. This is possible because of three design choices:

**1. The vision layer is mockable.** `HandTracker` can be constructed without a camera — the `process()` method takes a raw numpy array. Tests create fake frames or fake `HandState` objects directly.

**2. Hardware is behind an interface.** `ArmController` and `BaseController` are instantiated with a `board` object. In tests, that board can be a mock. The classes themselves contain only the logic of mapping angles to pin writes — testable without a real pin.

**3. TF model is mockable.** `LSTMStabilizer` is initialized with a `model` object. Tests can pass a mock that returns a fixed array, testing the buffer logic and denormalization without any TF dependency.

The CI pipeline runs these tests on every push across Python 3.9, 3.10, and 3.11. If a change breaks the feature extractor's normalization or the metrics logger's CSV format, CI catches it before it reaches main.

---

## 8. Configuration-driven design

### The problem with hardcoded constants

The original three Python files had constants scattered throughout:

```python
# In Complete.py
minDegX, maxDegX = 60, 180
# In SpeechL298XYZ.py
minDegX, maxDegX = 60, 180   # duplicated
# In finalV5VoiceSplit.py
minDegX, maxDegX = 60, 180   # duplicated again
```

This creates three failure modes: forgetting to update one copy, updating them inconsistently, and not knowing which copy is authoritative.

### The solution: a single YAML source of truth

```yaml
# config/default.yaml
hardware:
  servos:
    x: { pin: 3, min_deg: 60, max_deg: 180, default: 120 }
```

Every module that needs the servo X range does:
```python
cfg = load_config()
bounds = (cfg.hardware.servos["x"].min_deg, cfg.hardware.servos["x"].max_deg)
```

One change in one file. Immediately reflected everywhere. Validated at startup by the dataclass loader.

---

## 9. The LSTM integration decision

### Why not a Kalman filter?

A Kalman filter is a natural choice for smoothing noisy sensor data and would add zero dependencies. We chose LSTM for one reason: it is the paper's contribution and needs to be implemented to make the paper's results reproducible.

That said, for a production system a Kalman filter would be worth benchmarking. It has lower latency (microseconds vs milliseconds), no training requirement, and a well-understood mathematical foundation. This comparison is documented as a roadmap item.

### Why not an exponential moving average (EMA)?

EMA (`α * u_{t-1} + (1-α) * u_t`) is even simpler than Kalman and adds no dependencies. It is also already equivalent to a 1-layer RNN with a fixed weight. The LSTM learns a more complex smoothing function — one that can model the human wrist's dynamics, which involve not just noise but anticipatory motion. Whether this actually helps over EMA for this specific task is an empirical question the benchmark notebook can answer.

### Why supervised learning on baseline labels?

The LSTM is trained to reproduce the baseline mapper's output for a given sequence of frames. This is self-supervised: no external ground truth is needed, and the same operator who will use the system can collect the training data in 90 seconds.

The trade-off: the LSTM cannot produce outputs that the baseline mapper couldn't. It learns to smooth, not to improve. A reinforcement learning formulation (reward = arm accuracy at a target) could potentially learn a better controller — but requires a physical target-reaching setup to generate rewards, which is beyond the hardware budget of the project.

---

## 10. What was deliberately left out

### ROS2 integration

ROS2 is the standard for industrial robotics. A `gesture_arm` ROS2 node that publishes `std_msgs/Float32MultiArray` on a `/servo_angles` topic would make this system composable with any ROS2 robot stack. This was scoped out of the current version to keep the dependency footprint manageable, but the architecture fully supports it — `ArmController` would simply subscribe to the topic instead of being called directly. The roadmap item is documented.

### Offline speech recognition

The current ASR uses Google's online API, which requires internet access and introduces 200–500ms latency. Offline alternatives (Vosk, Whisper) exist and would be straightforward to plug into `ASRListener` — only `_run()` would change. Scoped out to avoid adding a large model download to the default setup.

### Multi-person / multi-hand disambiguation

If two people put their hands in front of the camera, the system picks the first detected right and left hand. No disambiguation logic exists. For a single-operator setup this is fine. For a shared workspace, a user ID / registration system would be needed.

### Servo feedback / position sensing

The system is open-loop: it writes an angle and assumes the servo reached it. If the arm is mechanically blocked (e.g. hitting a joint limit), the servo will stall silently. Closing the loop would require either servo feedback (more expensive servos with encoders) or inverse kinematics with joint limit modelling.

---

## 11. Known limitations

**Google ASR requires internet.** Recognized voice commands have 200–500ms network latency. In an offline environment, voice control does not work. Use gesture control exclusively in offline settings.

**LSTM warm-up delay.** The first 15 frames (~0.5s) after a hand enters frame use the baseline mapper, not the LSTM. This is visible as a brief period of slightly less smooth motion at the start of each interaction.

**Single-user, static lighting assumption.** The LSTM is trained on one user's hand gestures in one lighting environment. If a different user with significantly different hand proportions uses the system, accuracy will degrade until re-training. Similarly, a dramatic change in lighting affects MediaPipe's landmark confidence.

**No inverse kinematics.** Servo X, Y, Z are controlled independently. The arm tip does not follow a Cartesian trajectory — moving servo X while servo Y is at an extreme angle will trace an arc, not a straight line. For manipulation tasks requiring Cartesian accuracy, IK would be needed.

**pyFirmata serial throughput.** At 57600 baud, each Firmata message takes ~0.2ms. Writing three servo values per frame adds ~0.6ms, which is negligible at 30fps. At higher update rates (e.g. 100Hz force feedback), the serial bandwidth would become a bottleneck.

---

## 12. Planned improvements

| Item | Priority | Complexity | Notes |
|---|---|---|---|
| Transformer-based stabilizer | High | Medium | Replace LSTM with temporal attention; compare S and L |
| Kalman / EMA baseline comparison | High | Low | Add to benchmark notebook |
| Offline ASR (Vosk) | Medium | Low | Drop-in replacement for `ASRListener._run()` |
| ROS2 publisher node | Medium | Medium | `hardware/ros2_bridge.py` publishing to `/servo_angles` |
| Few-shot user adaptation | Low | High | Meta-learning layer for 10s re-calibration |
| Web dashboard (FastAPI + WebSocket) | Low | Medium | Live metrics stream in browser |
| Docker Compose with mock hardware | Low | Low | Full integration test without physical robot |
