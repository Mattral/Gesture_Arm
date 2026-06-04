# Research Notes

This document bridges the paper and the codebase — showing exactly where each claim, equation, and experimental result is implemented.

> **Paper:** *A Low-Cost Multimodal Gesture Control System with LSTM-Based Temporal Stabilization for Real-Time Robotics*

---

## Table of contents

1. [Paper-to-code mapping](#1-paper-to-code-mapping)
2. [Key equations implemented](#2-key-equations-implemented)
3. [Reproducing the results table](#3-reproducing-the-results-table)
4. [Experimental conditions](#4-experimental-conditions)
5. [Limitations of the current implementation](#5-limitations-of-the-current-implementation)
6. [Extending the research](#6-extending-the-research)

---

## 1. Paper-to-code mapping

| Paper section | Claim | Implemented in |
|---|---|---|
| Abstract | LSTM temporal stabilization reduces jitter | `gesture_arm/models/stabilizer.py` — `LSTMStabilizer` |
| Section III-A | Feature vector: 21 landmarks × (x/W, y/H) | `gesture_arm/vision/tracker.py` — `HandTracker._normalize()` |
| Section III-B | Dual-hand split: right=base, left=arm | `gesture_arm/run.py` — main loop, zones `x < W/2` and `x > W/2` |
| Section III-C | Sliding-window LSTM architecture | `gesture_arm/models/stabilizer.py` — `build_model()`, `LSTMStabilizer` |
| Section III-D | TTS confirms executed actions | `gesture_arm/speech/multimodal.py` — `TTSEngine` |
| Section III-D | Speech recognition in separate thread | `gesture_arm/speech/multimodal.py` — `ASRListener` |
| Section IV-A | Baseline: u_t = α·x_t + β | `gesture_arm/models/stabilizer.py` — `BaselineMapper.map()` |
| Section IV-B | LSTM: h_t = LSTM(X_t), û_t = W·h_t + b | `gesture_arm/models/stabilizer.py` — `LSTMStabilizer.update()` |
| Section V-A | Hardware: Arduino Uno, SG90 servos, L298N | `gesture_arm/hardware/arduino.py`, `firmware/server.ino` |
| Section VI | Stability metric S | `gesture_arm/evaluation/metrics.py` — `MetricsLogger.stability()` |
| Section VI | Latency metric L | `gesture_arm/evaluation/metrics.py` — `MetricsLogger.avg_latency()` |
| Section VI | Results table | `notebooks/benchmark_analysis.ipynb` — Section 3, summary table |

---

## 2. Key equations implemented

### Feature vector (Section III-A)

The paper defines the per-frame input as:

```
x_t = [x₁/W, y₁/H,  x₂/W, y₂/H,  ...,  x₂₁/W, y₂₁/H]   ∈ ℝ⁴²
```

Implemented in `HandTracker._normalize()`:

```python
xy = landmarks[:, :2].copy()   # (21, 2) raw pixel coords
xy[:, 0] /= self._fw           # normalize x by frame width
xy[:, 1] /= self._fh           # normalize y by frame height
return xy.flatten()             # (42,) feature vector
```

### Baseline mapping (Section IV-A)

```
u_t = α · x_t + β
```

Implemented as `np.interp()` calls in `BaselineMapper.map()`:

```python
servoX = np.interp(x_pos, [frame_w/2, frame_w], [min_deg_x, max_deg_x])
servoY = np.interp(y_pos, [0, frame_h],          [min_deg_y, max_deg_y])
servoZ = np.interp(pinch, [20, 220],             [min_deg_z, max_deg_z])
```

### LSTM temporal stabilization (Section IV-B)

```
X_t  = [x_{t-k},  x_{t-k+1},  …,  x_t]   ∈ ℝ^(k×42)
h_t  = LSTM(X_t)                           ∈ ℝ^64
û_t  = sigmoid(W · h_t + b)               ∈ [0, 1]³
```

Implemented in `LSTMStabilizer.update()` and `build_model()`:

```python
# build_model()
model = Sequential([
    LSTM(64, input_shape=(sequence_length, feature_dim)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(output_dim, activation="sigmoid"),  # û_t ∈ [0,1]³
])

# LSTMStabilizer.update()
seq = np.array(self._buffer)[np.newaxis, ...]    # (1, k, 42)
norm = self._model.predict(seq, verbose=0)[0]    # (3,) in [0,1]
angles = denormalize(norm)                        # → degrees
```

### Stability variance S (Section VI)

```
S = (1/T) Σ_{t=1}^{T} (u_t − ū)²
```

Implemented in `MetricsLogger.stability()`:

```python
recent = np.array(self._servo_history[-window:])   # (T, 3)
mean   = np.mean(recent, axis=0)                   # ū
S      = np.mean(np.sum((recent - mean)**2, axis=1))
```

### End-to-end latency L (Section VI)

```
L = t_actuation − t_capture   (ms)
```

Implemented in `MetricsLogger.log()`:

```python
t_actuation = time.time()                       # called immediately after arm.write()
latency = (t_actuation - t_capture) * 1000.0   # ms
```

---

## 3. Reproducing the results table

### Step 1 — Collect data and run both modes

```bash
# Collect training data
python scripts/collect.py --duration 90 --out data/training_data.csv

# Train the LSTM
python scripts/train.py

# Run LSTM mode — collect metrics
python -m gesture_arm.run
# (interact for ~3 minutes, press Q)
# → data/metrics_log.csv contains both "lstm" and "baseline (warming)" rows

# For a clean baseline-only run, temporarily comment out the LSTMStabilizer
# lines in run.py and re-run. Save as data/baseline_metrics.csv.
```

### Step 2 — Open the benchmark notebook

```bash
jupyter notebook notebooks/benchmark_analysis.ipynb
```

Run all cells. The notebook generates:
- `docs/stability_comparison.png` — rolling S over time, LSTM vs baseline
- `docs/latency_distribution.png` — histogram of L values
- `docs/trajectory_comparison.png` — servo angle trajectories
- A markdown summary table matching paper Table I

### Expected results

| Method | S (↓ better) | L mean ms (↓ better) | L p95 ms |
|---|---|---|---|
| Baseline | ~18 | ~55 | ~90 |
| LSTM | ~13 | ~42 | ~68 |

Exact values will differ based on your hand movement patterns and hardware. The key result is that LSTM S < baseline S — the LSTM achieves lower variance, i.e. smoother motion.

---

## 4. Experimental conditions

To reproduce results closest to the paper:

**Physical setup:**
- Webcam at 60–80 cm from operator, at eye height
- Room lighting: overhead fluorescent or LED, ≥300 lux
- Plain background (avoid skin-toned walls)
- Arm mounted on a flat surface, free to rotate full range

**Data collection:**
- 90 seconds minimum
- Deliberately move through the full range of all three axes
- Include both slow deliberate motions and faster gestures
- Collect in the same lighting conditions as evaluation

**Evaluation session:**
- 3 minutes of continuous interaction
- Equal mix of base navigation and arm control
- Use the same operator who collected training data

---

## 5. Limitations of the current implementation

**LSTM trained on one user.** The model does not generalize well across users with different hand proportions or gesture styles. Each user should collect their own training data.

**Online speech API.** Google Speech Recognition requires internet. Latency (200–500ms) is not measured in the paper's L metric since speech commands go to the motor controller, not the arm.

**No ground truth for accuracy.** The paper reports stability S and latency L, but not task accuracy (e.g. % of times the arm correctly reaches a target). Computing task accuracy requires a physical target setup that is beyond the current hardware.

**LSTM vs baseline on the same session.** The metrics CSV contains both "lstm" and "baseline (warming)" rows from the same session. For a clean comparison, run two separate sessions — one with LSTM disabled and one with it enabled — and analyze them separately in the notebook.

---

## 6. Extending the research

### Transformer-based stabilizer

Replace the LSTM with a Temporal Fusion Transformer or a simple multi-head self-attention over the 15-frame sequence. Expected benefit: better handling of long-range dependencies (e.g. the arm's current position influencing the expected trajectory).

Implementation path:
1. Add `TransformerStabilizer` to `gesture_arm/models/stabilizer.py`
2. Match the same `update(features) -> (angles, method)` interface
3. Add `--model transformer` CLI flag to `run.py`
4. Add a comparison column to the benchmark notebook

### Few-shot user adaptation

Train a meta-learning wrapper (e.g. MAML) on data from multiple users. At deployment, a new user collects 10 seconds of data and the model fine-tunes in seconds. This would make the system genuinely user-agnostic.

### Kalman filter baseline

Add a `KalmanStabilizer` that uses a constant-velocity Kalman filter on the three servo dimensions. This is a strong baseline — it is parameter-free, has ~microsecond latency, and requires no training data. If LSTM S ≈ Kalman S, the LSTM adds no value over a classical filter. If LSTM S < Kalman S, that is a genuine result worth reporting.

### Task accuracy metric

Design a target-reaching experiment: place colored markers at five positions in the arm's workspace. Measure what fraction of commanded positions (via gesture) the arm reaches within a 5mm radius. Compare LSTM vs baseline task accuracy. This would strengthen the paper significantly.
