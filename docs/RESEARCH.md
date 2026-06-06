# Research Notes

This document bridges the paper and the codebase — showing exactly where
each claim, equation, and experimental result is implemented.

> **Paper:** *A Low-Cost Multimodal Gesture Control System with LSTM-Based
> Temporal Stabilization and Geometric Inverse Kinematics for Real-Time Robotics*
> — Min Htet Myet, 2025

---

## Table of contents

1. [Paper-to-code mapping](#1-paper-to-code-mapping)
2. [Key equations implemented](#2-key-equations-implemented)
3. [Reproducing the results table](#3-reproducing-the-results-table)
4. [Experimental conditions](#4-experimental-conditions)
5. [Limitations of the current implementation](#5-limitations-of-the-current-implementation)
6. [Roadmap](#6-roadmap)

---

## 1. Paper-to-code mapping

| Paper section | Claim / contribution | Implemented in |
|---|---|---|
| Abstract | LSTM temporal stabilization reduces jitter | `gesture_arm/models/stabilizer.py` — `LSTMStabilizer` |
| Abstract | Geometric IK converts hand position to joint angles | `gesture_arm/kinematics/ik_solver.py` — `GeometricIKSolver` |
| Section IV-A | Feature vector: 21 landmarks × (x/W, y/H), Eq. (1) | `gesture_arm/vision/tracker.py` — `HandTracker._normalize()` |
| Section III | Dual-hand split: right=base, left=arm | `gesture_arm/run.py` — main loop, zones `x < W/2` and `x > W/2` |
| Section IV-B | Baseline: u_t = α·x_t + β, Eq. (2) | `gesture_arm/models/stabilizer.py` — `BaselineMapper.map()` |
| Section IV-C | LSTM: X_t → LSTM → û_t, Eqs. (3)–(11) | `gesture_arm/models/stabilizer.py` — `build_model()`, `LSTMStabilizer.update()` |
| Section IV-D | Self-supervised training, Eq. (12) | `gesture_arm/models/stabilizer.py` — `train()`, `scripts/train.py` |
| Section IV-E | Stability metric S, Eq. (13) | `gesture_arm/evaluation/metrics.py` — `MetricsLogger.stability()` |
| Section IV-E | Latency metric L, Eq. (14) | `gesture_arm/evaluation/metrics.py` — `MetricsLogger.log()` |
| Section IV-F | IK equations (15)–(22) | `gesture_arm/kinematics/ik_solver.py` — `GeometricIKSolver.solve()` |
| Section IV-G | Gesture-to-task-space mapping, Eqs. (23)–(24) | `gesture_arm/kinematics/ik_solver.py` — `GeometricIKSolver.hand_position_to_target()` |
| Section IV-H | IK → LSTM → baseline cascade | `gesture_arm/run.py` — left-hand arm control block |
| Section III | TTS confirms executed actions | `gesture_arm/speech/multimodal.py` — `TTSEngine` |
| Section III | Speech recognition in separate thread | `gesture_arm/speech/multimodal.py` — `ASRListener` |
| Section V-A | Hardware: Arduino Uno, SG90 servos, L298N | `gesture_arm/hardware/arduino.py`, `firmware/server.ino` |
| Section VI | Results Table II (all three modes) | `notebooks/benchmark_analysis.ipynb` — Section 3 |

---

## 2. Key equations implemented

### Eq. (1) — Normalized feature vector (Section IV-A)

```
x_t = [x₀/W, y₀/H,  x₁/W, y₁/H,  …,  x₂₀/W, y₂₀/H]  ∈ ℝ⁴²
```

```python
# gesture_arm/vision/tracker.py — HandTracker._normalize()
xy = landmarks[:, :2].copy()   # (21, 2) raw pixel coords
xy[:, 0] /= self._fw            # normalize x by frame width W
xy[:, 1] /= self._fh            # normalize y by frame height H
return xy.flatten()              # (42,) normalized feature vector
```

### Eq. (2) — Baseline linear mapping (Section IV-B)

```
u_t = u_min + (p_t − p_low) / (p_high − p_low) × (u_max − u_min)
```

```python
# gesture_arm/models/stabilizer.py — BaselineMapper.map()
servoX = np.interp(x_pos, [frame_w/2, frame_w], [min_deg_x, max_deg_x])
servoY = np.interp(y_pos, [0, frame_h],          [min_deg_y, max_deg_y])
pinch  = np.linalg.norm(landmarks[4, :2] - landmarks[8, :2])
servoZ = np.interp(pinch, [20, 220],             [min_deg_z, max_deg_z])
```

### Eqs. (3)–(11) — LSTM temporal stabilization (Section IV-C)

```
X_t  = [x_{t−k+1}, …, x_t]  ∈ ℝ^(k×42)   (k = 15)
h_t  = LSTM(X_t)             ∈ ℝ⁶⁴
û_t  = σ(W₂·ReLU(W₁·h_t + b₁) + b₂)  ∈ [0,1]³
```

```python
# gesture_arm/models/stabilizer.py — build_model()
model = Sequential([
    LSTM(64, input_shape=(sequence_length, feature_dim)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(output_dim, activation="sigmoid"),
])

# LSTMStabilizer.update()
seq  = np.array(self._buffer)[np.newaxis, ...]   # (1, k, 42)
norm = self._model.predict(seq, verbose=0)[0]    # (3,) in [0, 1]
angles = self._denormalize(norm)                  # → degrees
```

### Eq. (12) — MSE training loss (Section IV-D)

```
ℒ_MSE = (1/N) Σ ‖ û_t − normalize(u_t) ‖²
```

```python
# gesture_arm/models/stabilizer.py — train()
model.compile(optimizer=Adam(lr=0.001), loss="mse", metrics=["mae"])
```

### Eq. (13) — Control stability variance S (Section IV-E)

```
S = (1/T) Σ_{t=1}^{T} ‖ u_t − ū ‖²
```

```python
# gesture_arm/evaluation/metrics.py — MetricsLogger.stability()
recent = np.array(self._servo_history[-window:])   # (T, 3)
mean   = np.mean(recent, axis=0)                   # ū
S      = float(np.mean(np.sum((recent - mean)**2, axis=1)))
```

### Eq. (14) — End-to-end latency L (Section IV-E)

```
L = t_actuation − t_capture   (ms)
```

```python
# gesture_arm/evaluation/metrics.py — MetricsLogger.log()
t_actuation = time.time()                        # after arm.write()
latency = (t_actuation - t_capture) * 1000.0    # → ms
```

### Eqs. (15)–(22) — Geometric IK (Section IV-F)

```
r  = √(px² + py²)              (horizontal reach)
θ₁ = atan2(py, px)             (base rotation)
L  = √(r² + pz²)               (straight-line shoulder→TCP distance)
θ₂ = atan2(pz, r)              (elevation angle)

Reachability:
  L > l₁ + l₂   →  UNREACHABLE
  L < |l₁ − l₂| →  IN_DEADZONE

Servo mapping:
  θ₁_servo = 120° + degrees(θ₁)
  θ₂_servo = 40°  + degrees(θ₂) × (140° − 40°) / 90°
```

```python
# gesture_arm/kinematics/ik_solver.py — GeometricIKSolver.solve()
r_desired = math.sqrt(px**2 + py**2)
theta1    = math.atan2(py, px)
L         = math.sqrt(r_desired**2 + pz**2)

if L > self.l_max:  return IKResult(solution=IKSolution.UNREACHABLE, ...)
if L < self.l_min:  return IKResult(solution=IKSolution.IN_DEADZONE,  ...)

theta2  = math.atan2(pz, r_desired)
servo_x = self._theta1_to_servo_x(theta1)
servo_y = self._theta2_to_servo_y(theta2)
```

### Eqs. (23)–(24) — Gesture-to-task-space mapping (Section IV-G)

```
px = x_lo + norm_x × (x_hi − x_lo)
pz = z_hi + norm_y × (z_lo − z_hi)    (inverted: high pixel-y = low pz)
```

```python
# gesture_arm/kinematics/ik_solver.py — GeometricIKSolver.hand_position_to_target()
wb = self.workspace_bounds()
px = wb["x_range_cm"][0] + norm_x * (wb["x_range_cm"][1] - wb["x_range_cm"][0])
pz = wb["z_range_cm"][1] + norm_y * (wb["z_range_cm"][0] - wb["z_range_cm"][1])
```

---

## 3. Reproducing the results table

### Step 1 — Collect training data

```bash
python scripts/collect.py --duration 90 --out data/training_data.csv
```

Move your left hand through the full range: left/right sweeps, up/down sweeps,
open/close pinch cycles. Three 90-second sessions (concatenated) gives ~8,100 frames.

### Step 2 — Train the LSTM

```bash
python scripts/train.py
```

### Step 3 — Run all three evaluation sessions

```bash
# Session 1: Baseline only
# In run.py, temporarily set stabilizer = None and ik_solver = None, or:
python -m gesture_arm.run   # interact 3 min, ensure LSTM not loaded yet
# (run before training so no model file exists)

# Session 2: LSTM mode
python -m gesture_arm.run   # interact 3 min

# Session 3: IK mode
python -m gesture_arm.run --ik   # interact 3 min

# All three sessions write to data/metrics_log.csv with different method labels:
# "baseline", "lstm", "baseline (warming)", "ik", "ik_fallback"
```

### Step 4 — Open the benchmark notebook

```bash
jupyter notebook notebooks/benchmark_analysis.ipynb
```

Run all cells. The notebook generates:
- `docs/stability_comparison.png` — rolling S over time, all three modes
- `docs/latency_distribution.png` — histogram of L values per mode
- `docs/trajectory_comparison.png` — servo angle trajectories
- Table II (auto-generated, matches the paper exactly)

### Expected results

| Method | S (↓ better) | L median ms (↓ better) | L p95 ms |
|---|---|---|---|
| Baseline | ~18.4 | ~52 | ~90 |
| LSTM | ~12.8 | ~40 | ~68 |
| IK | ~10.1 | ~41 | ~70 |

Exact values vary by operator, hardware, and lighting. The key result is:
S(IK) < S(LSTM) < S(baseline) — each level of the cascade reduces variance.

---

## 4. Experimental conditions

**Physical setup:**
- Webcam at 60–80 cm from operator, at eye height
- Room lighting: overhead fluorescent or LED, ≥ 300 lux
- Plain, non-skin-toned background
- Arm mounted on a flat stable surface, free to rotate full range
- Link lengths measured and set in config: `link1_cm`, `link2_cm`

**Data collection:**
- 90 seconds minimum per session, 3 sessions recommended
- Deliberately move through the full range of all three axes
- Include both slow deliberate motions and faster transitions
- Collect in the same lighting as evaluation

**Evaluation session:**
- 3 minutes of continuous interaction per mode
- Mix of arm positioning and base navigation
- Same operator who collected training data (LSTM sessions)
- For IK sessions: move arm tip to five visually marked workspace positions,
  two repetitions each

---

## 5. Limitations of the current implementation

**LSTM is user-specific and lighting-specific.** The model is trained on one
operator's gesture patterns in one environment. New users should collect their
own training data (90 seconds). The IK solver has no learned parameters and
requires no re-training.

**Online speech API.** Google Speech Recognition requires internet access.
Latency (200–500 ms) is not measured in the paper's L metric — speech commands
target the motor controller, not the arm.

**No task accuracy metric.** The paper reports S and L, not whether the arm
successfully reached a target position. Measuring task accuracy requires a
physical marker setup. See roadmap item below.

**IK simplified arm model.** The single-rigid-link model (θ₂ controls
total arm elevation) is exact for the current SG90 arm, which has no
independent elbow joint. Adding an elbow joint requires extending to the
two-link law-of-cosines IK with elbow-up/down configuration selection.

**LSTM vs baseline overlap in metrics CSV.** Both "lstm" and "baseline (warming)"
rows appear in the same session CSV. For the cleanest comparison, run two
separate sessions (one with LSTM disabled, one enabled) and filter by method
label in the notebook.

**Reference [13] unverified.** The citation for pyFirmata use in a research
context needs verification against IEEE Xplore before journal submission. The
pyFirmata library itself can be cited directly: github.com/tino/pyFirmata.

---

## 6. Roadmap

### Two-link elbow IK extension

Extend `GeometricIKSolver` to support an independent elbow joint using the
law of cosines for the two-link planar case:

```
cos(θ_elbow) = (L² − l₁² − l₂²) / (2·l₁·l₂)
```

This adds an elbow-up/elbow-down configuration selection step.
Implementation path: extend `solve()` with an optional `elbow_joint` flag.

### Kalman filter stabilizer

Add `KalmanStabilizer` to `gesture_arm/models/` using a constant-velocity
Kalman filter on the three servo dimensions. This is the principled classical
comparison baseline: parameter-free, ~microsecond latency, no training data.
If S(LSTM) ≈ S(Kalman), the LSTM adds no value. If S(LSTM) < S(Kalman),
the learned frequency response is genuinely useful.

### Transformer-based stabilizer

Replace the LSTM with a lightweight Temporal Fusion Transformer (TFT) or
multi-head self-attention over the 15-frame sequence. Match the
`update(features) -> (angles, method)` interface; add `--model transformer`
CLI flag; add comparison column to the benchmark notebook.

### Few-shot user adaptation

Fine-tune only the final dense layer (`W₂`, `b₂`) on 10 seconds of new-user
data. This requires saving the base model and providing a `adapt()` method.
Expected result: new-user S approaches trained-user S after a brief
calibration session.

### Task accuracy metric

Place five colored markers at known positions in the arm workspace. Measure
the fraction of IK-commanded moves that bring the TCP within 5 mm of the
target. Compare IK vs LSTM vs baseline task accuracy. This requires a
physical target setup and a measurement procedure (e.g. camera overhead view).

### ROS2 publisher node

Add `gesture_arm/hardware/ros2_bridge.py` publishing:
- `/servo_angles` (`std_msgs/Float32MultiArray`)
- `/cmd_vel` (`geometry_msgs/Twist`)

This enables the gesture_arm controller to drive any ROS2-compatible robot.

### Offline ASR (Vosk)

Replace Google ASR with Vosk in `ASRListener._run()`. This removes the
internet dependency and reduces speech latency to ~50 ms for short commands.
