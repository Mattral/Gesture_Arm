# API Reference

Complete reference for all public classes, functions, and data structures in the `gesture_arm` package.

---

## Table of contents

- [`gesture_arm.vision.tracker`](#gesture_armvisiontracker)
- [`gesture_arm.models.stabilizer`](#gesture_armmodelsstabilizer)
- [`gesture_arm.hardware.arduino`](#gesture_armhardwarearduino)
- [`gesture_arm.speech.multimodal`](#gesture_armspeechmultimodal)
- [`gesture_arm.evaluation.metrics`](#gesture_armevaluationmetrics)
- [`gesture_arm.config.settings`](#gesture_armconfigsettings)

---

## `gesture_arm.vision.tracker`

Hand detection and normalized feature extraction.

---

### `HandState`

```python
@dataclass(frozen=True)
class HandState:
    hand_type:      str            # "Left" or "Right"
    landmarks:      np.ndarray     # shape (21, 3), raw pixel [x, y, z]
    features:       np.ndarray     # shape (42,),  normalized [x/W, y/H] × 21
    palm_center:    Tuple[float, float]  # pixel position of landmark 9
    pinch_distance: float          # px distance: thumb tip → index tip
    fingers_up:     List[bool]     # [thumb, index, middle, ring, pinky]
    is_fist:        bool           # True when all five fingers are closed
```

Immutable snapshot of one detected hand for a single frame. All coordinates in `landmarks` are raw pixel values; all values in `features` are normalized to `[0, 1]`.

**Example:**
```python
if hand_state.is_fist:
    base.stop()

x, y = hand_state.palm_center
if y < frame_height / 4:
    base.forward()
```

---

### `TrackerOutput`

```python
@dataclass(frozen=True)
class TrackerOutput:
    frame: np.ndarray          # BGR image with optional landmark overlay
    left:  Optional[HandState] # Left hand state, or None if not detected
    right: Optional[HandState] # Right hand state, or None if not detected
```

Output from one call to `HandTracker.process()`.

---

### `HandTracker`

```python
class HandTracker:
    def __init__(
        self,
        detection_confidence: float = 0.8,
        max_hands: int = 2,
        frame_width: int = 1280,
        frame_height: int = 720,
        draw_landmarks: bool = False,
    ) -> None
```

Wraps cvzone/MediaPipe and converts raw landmark dicts into typed `HandState` objects.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detection_confidence` | float | 0.8 | MediaPipe minimum detection confidence [0, 1] |
| `max_hands` | int | 2 | Maximum number of hands to detect simultaneously |
| `frame_width` | int | 1280 | Frame width in pixels (used for normalization) |
| `frame_height` | int | 720 | Frame height in pixels (used for normalization) |
| `draw_landmarks` | bool | False | Draw MediaPipe skeleton overlay on the frame |

**Methods:**

#### `process(frame) -> TrackerOutput`

```python
def process(self, frame: np.ndarray) -> TrackerOutput
```

Detect hands in one BGR frame.

| Parameter | Type | Description |
|---|---|---|
| `frame` | np.ndarray | BGR image from `cv2.VideoCapture.read()` |

Returns a `TrackerOutput`. `left` and `right` fields are `None` if the respective hand is not detected.

**Example:**
```python
tracker = HandTracker(detection_confidence=0.8, frame_width=1280, frame_height=720)
ret, frame = cap.read()
output = tracker.process(frame)

if output.left is not None:
    print(output.left.palm_center)   # (x_px, y_px)
    print(output.left.features)      # (42,) normalized array
```

---

## `gesture_arm.models.stabilizer`

LSTM temporal stabilization and baseline mapper.

---

### `LSTMStabilizer`

```python
class LSTMStabilizer:
    def __init__(
        self,
        model: tf.keras.Model,
        servo_bounds: Dict[str, Tuple[float, float]],
        sequence_length: int = 15,
    ) -> None
```

Maintains a sliding window of feature vectors and produces LSTM-smoothed servo angles. Falls back transparently while the buffer fills.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | `tf.keras.Model` | Trained LSTM model from `build_model()` or `load_or_build()` |
| `servo_bounds` | dict | `{"x": (min°, max°), "y": (min°, max°), "z": (min°, max°)}` |
| `sequence_length` | int | Sliding window size (k in paper). Must match model's input shape. |

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `is_warmed_up` | bool | True once the buffer has reached `sequence_length` frames |

**Methods:**

#### `update(feature_vector) -> Tuple[Optional[np.ndarray], str]`

```python
def update(self, feature_vector: np.ndarray) -> Tuple[Optional[np.ndarray], str]
```

Push one feature vector and return a smoothed servo command.

| Parameter | Type | Description |
|---|---|---|
| `feature_vector` | np.ndarray | Shape `(42,)`, normalized landmarks from `HandState.features` |

Returns `(angles, method)` where:
- `angles` is `np.ndarray` shape `(3,)` of `[servoX°, servoY°, servoZ°]`, or `None` if the buffer is still filling
- `method` is `"lstm"` or `"baseline (warming up)"`

#### `reset() -> None`

Clear the sequence buffer. Call this when the hand leaves the frame to avoid stale history contaminating the next interaction.

**Example:**
```python
stabilizer = LSTMStabilizer(model, servo_bounds={"x":(60,180),"y":(40,140),"z":(100,150)})

if output.left:
    angles, method = stabilizer.update(output.left.features)
    if angles is not None:
        arm.write(angles)
else:
    stabilizer.reset()
```

---

### `BaselineMapper`

```python
class BaselineMapper:
    def __init__(
        self,
        servo_bounds: Dict[str, Tuple[float, float]],
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> None
```

Direct frame-by-frame linear mapping. Used as the comparison baseline and as warm-up fallback.

**Methods:**

#### `map(landmarks) -> np.ndarray`

```python
def map(self, landmarks: np.ndarray) -> np.ndarray
```

Map raw landmark array to servo angles.

| Parameter | Type | Description |
|---|---|---|
| `landmarks` | np.ndarray | Shape `(21, 3)` raw pixel coordinates from `HandState.landmarks` |

Returns `np.ndarray` shape `(3,)` of `[servoX°, servoY°, servoZ°]`.

Mapping logic:
- `servoX` ← horizontal position of landmark 9 (palm centre)
- `servoY` ← vertical position of landmark 9
- `servoZ` ← Euclidean distance between landmark 4 (thumb tip) and landmark 8 (index tip)

---

### `build_model(...) -> tf.keras.Model`

```python
def build_model(
    sequence_length: int = 15,
    feature_dim: int = 42,
    output_dim: int = 3,
    lstm_units: int = 64,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> tf.keras.Model
```

Build and compile the LSTM stabilization model. Architecture: `LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(3, sigmoid)`.

Raises `RuntimeError` if TensorFlow is not installed.

---

### `load_or_build(model_path, **build_kwargs) -> tf.keras.Model`

```python
def load_or_build(model_path: str | Path, **build_kwargs) -> tf.keras.Model
```

Load a saved `.h5` model if it exists; otherwise build a new untrained model and log a warning.

---

### `train(...) -> None`

```python
def train(
    data_path: str | Path,
    model_path: str | Path,
    sequence_length: int = 15,
    feature_dim: int = 42,
    output_dim: int = 3,
    epochs: int = 80,
    batch_size: int = 16,
    validation_split: float = 0.15,
    **build_kwargs,
) -> None
```

Train the LSTM on collected CSV data and save the model. Uses `EarlyStopping(patience=10)` and `ModelCheckpoint(save_best_only=True)`.

Raises `FileNotFoundError` if `data_path` does not exist. Raises `RuntimeError` if TensorFlow is not installed.

---

## `gesture_arm.hardware.arduino`

Arduino communication layer.

---

### `connect(port, baudrate) -> pyfirmata.Arduino`

```python
def connect(port: str, baudrate: int = 57600)
```

Connect to the Arduino and return the board object. Also starts the pyFirmata iterator thread.

Raises `serial.SerialException` if the port cannot be opened.

---

### `board_session(port, baudrate)`

```python
@contextmanager
def board_session(port: str, baudrate: int = 57600)
```

Context manager that connects and guarantees `board.exit()` on cleanup.

```python
with board_session("COM6") as board:
    arm = ArmController(board, cfg.hardware.servos)
    # ... use arm
# board.exit() called automatically
```

---

### `ArmController`

```python
class ArmController:
    def __init__(self, board, servo_configs: Dict[str, ServoConfig]) -> None
```

Controls the three-servo robot arm.

**Methods:**

#### `write(angles) -> None`

```python
def write(self, angles: np.ndarray) -> None
```

Write servo angles to hardware. Values are clamped to configured bounds before writing.

| Parameter | Type | Description |
|---|---|---|
| `angles` | np.ndarray | Shape `(3,)`: `[servoX°, servoY°, servoZ°]` |

#### `home() -> None`

Return all servos to their midpoint default positions.

---

### `BaseController`

```python
class BaseController:
    def __init__(self, board, motor_cfg: MotorConfig) -> None
```

Controls the L298N motor driver.

**Methods:**

| Method | Description |
|---|---|
| `forward()` | Both motors forward at `max_speed` |
| `reverse()` | Both motors reverse at `max_speed` |
| `turn_left()` | Left motor at `turn_speed`, right at `max_speed` |
| `turn_right()` | Left motor at `max_speed`, right at `turn_speed` |
| `stop()` | Set both PWM pins to 0 |

---

## `gesture_arm.speech.multimodal`

Threaded speech I/O.

---

### `TTSEngine`

```python
class TTSEngine:
    def __init__(self, rate: int = 160, volume: float = 0.8) -> None
```

Non-blocking text-to-speech. Runs pyttsx3 in a daemon thread.

**Methods:**

#### `start() -> None`
Start the daemon thread. Must be called before `say()`.

#### `say(text) -> None`
Queue an utterance. Non-blocking. Suppresses consecutive duplicates.

#### `stop() -> None`
Send sentinel to the queue; the thread exits after draining.

---

### `ASRListener`

```python
class ASRListener:
    def __init__(
        self,
        commands: Set[str],
        on_command: Callable[[str], None],
        tts: Optional[TTSEngine] = None,
    ) -> None
```

Continuous speech recognition in a daemon thread.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `commands` | set[str] | Vocabulary of recognized commands (lowercase) |
| `on_command` | callable | Callback invoked with the matched command string |
| `tts` | TTSEngine | Optional — if provided, confirms commands by speaking |

**Methods:**

#### `start() -> None`
Start the daemon listener thread.

**Example:**
```python
def handle(cmd):
    if cmd == "forward": base.forward()
    elif cmd == "stop":  base.stop()

tts = TTSEngine()
tts.start()
asr = ASRListener(commands={"forward","reverse","left","right","stop"}, on_command=handle, tts=tts)
asr.start()
```

---

## `gesture_arm.evaluation.metrics`

Metrics logging and computation.

---

### `MetricsLogger`

```python
class MetricsLogger:
    def __init__(
        self,
        log_path: str | Path = "data/metrics_log.csv",
        stability_window: int = 100,
        latency_window: int = 100,
    ) -> None
```

Streams servo commands and latency to CSV. Computes rolling stability `S` and latency `L`.

**Methods:**

#### `log(angles, t_capture, method) -> None`

```python
def log(self, angles: np.ndarray, t_capture: float, method: str = "lstm") -> None
```

Record one servo command. Appends one row to CSV immediately.

| Parameter | Type | Description |
|---|---|---|
| `angles` | np.ndarray | Shape `(3,)`: `[servoX°, servoY°, servoZ°]` |
| `t_capture` | float | `time.time()` value when the frame was captured |
| `method` | str | `"lstm"` or `"baseline"` |

#### `stability(window) -> Optional[float]`

Compute rolling stability variance `S = (1/T) Σ (u_t − ū)²` over the last `window` frames. Returns `None` if fewer than 2 frames logged.

#### `avg_latency(window) -> Optional[float]`

Compute rolling mean latency `L` (ms) over the last `window` frames. Returns `None` if no frames logged.

#### `summary() -> dict`

Return a dict with keys: `n_frames`, `stability_S`, `avg_latency_ms`, `min_latency_ms`, `max_latency_ms`, `log_path`.

#### `print_summary() -> None`

Print a formatted summary to stdout. Called automatically on clean exit from `gesture_arm.run`.

---

## `gesture_arm.config.settings`

Configuration loading.

---

### `load_config(path) -> AppConfig`

```python
def load_config(path: str | Path = <default.yaml>) -> AppConfig
```

Load and parse a YAML config file into a typed `AppConfig` dataclass.

Raises `FileNotFoundError` if the file does not exist. The serial port is overridable via the `GESTURE_ARM_PORT` environment variable.

---

### `AppConfig`

Top-level config dataclass with fields: `hardware: HardwareConfig`, `vision: VisionConfig`, `model: ModelConfig`, `speech: SpeechConfig`, `evaluation: EvaluationConfig`.

All nested dataclasses follow the structure of `config/default.yaml`. Access example:

```python
cfg = load_config()
print(cfg.hardware.servos["x"].pin)          # 3
print(cfg.hardware.motors.max_speed)         # 1.0
print(cfg.model.sequence_length)             # 15
print(cfg.speech.commands["forward"])        # [1, 0, 1.0, 0, 1, 1.0]
```
