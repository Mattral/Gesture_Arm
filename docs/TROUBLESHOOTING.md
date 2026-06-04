# Troubleshooting

Diagnostic guide for every common failure mode, with exact error messages and fixes.

---

## Quick diagnostics

Run this first to get a system status snapshot:

```bash
python -c "
import sys
print(f'Python {sys.version}')

import cv2; print(f'OpenCV {cv2.__version__}')
import numpy; print(f'NumPy {numpy.__version__}')
import yaml; print('PyYAML OK')

try:
    import pyfirmata; print('pyFirmata OK')
except ImportError: print('pyFirmata MISSING')

try:
    import speech_recognition as sr; print(f'SpeechRecognition OK')
except ImportError: print('SpeechRecognition MISSING')

try:
    import pyttsx3; print('pyttsx3 OK')
except ImportError: print('pyttsx3 MISSING')

try:
    import tensorflow as tf; print(f'TensorFlow {tf.__version__}')
except ImportError: print('TensorFlow not installed (optional)')

try:
    import cvzone; print('cvzone OK')
except ImportError: print('cvzone MISSING')
"
```

---

## Contents

- [Arduino / serial issues](#arduino--serial-issues)
- [Camera issues](#camera-issues)
- [Hand detection issues](#hand-detection-issues)
- [Servo issues](#servo-issues)
- [Motor issues](#motor-issues)
- [Speech recognition issues](#speech-recognition-issues)
- [LSTM / TensorFlow issues](#lstm--tensorflow-issues)
- [Python / dependency issues](#python--dependency-issues)
- [Performance issues](#performance-issues)
- [CI / Docker issues](#ci--docker-issues)

---

## Arduino / serial issues

### `serial.SerialException: could not open port 'COM6'`

The configured port does not exist or is in use by another process (e.g. the Arduino IDE's serial monitor).

**Fix:**
1. Find the correct port:
   - Windows: Device Manager → Ports (COM & LPT) → look for "Arduino Uno"
   - Linux: `ls /dev/ttyUSB* /dev/ttyACM*`
   - macOS: `ls /dev/cu.usbmodem*`
2. Update `port` in `gesture_arm/config/default.yaml`, or:
   ```bash
   GESTURE_ARM_PORT=COM4 python -m gesture_arm.run
   ```
3. Close the Arduino IDE serial monitor if it is open.

---

### `serial.SerialException: [Errno 13] Permission denied`

User does not have read/write permission on the serial port.

**Fix (Linux/macOS):**
```bash
# Temporary (until next reboot):
sudo chmod a+rw /dev/ttyUSB0

# Permanent (requires logout):
sudo usermod -aG dialout $USER
```

---

### Arduino connects but servos do not move

The firmware is not StandardFirmata, or pyFirmata's iterator thread did not start.

**Fix:**
1. Re-upload `firmware/server.ino` from the Arduino IDE
2. Confirm the board is "Arduino Uno" in Tools → Board
3. The code calls `pyfirmata.util.Iterator(board).start()` — check `hardware/arduino.py` `connect()` function includes this

---

### `pyfirmata.pyfirmata.PinAlreadyDefinedError`

A pin is being defined twice — usually because `connect()` is called more than once.

**Fix:** Ensure `connect()` is called exactly once. Use the `board_session()` context manager to prevent accidental double-initialization.

---

## Camera issues

### Camera window does not open / black screen

```
Camera read failed — is camera_index correct?
```

**Fix:**
```python
# Find available cameras
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: OK")
    cap.release()
```

Update `camera_index` in `config/default.yaml`.

---

### Camera opens but shows wrong device

You have multiple cameras (e.g. built-in + USB webcam) and the wrong one is selected.

**Fix:** Change `camera_index` from `0` to `1` (or higher) in the config.

---

### Low frame rate (<15 fps)

The camera is set to a resolution the USB bandwidth cannot sustain.

**Fix:** Reduce resolution in the config:
```yaml
vision:
  width: 640
  height: 480
```

Or force the frame rate:
```python
cap.set(cv2.CAP_PROP_FPS, 30)
```

---

### `cv2.error: (-215:Assertion failed) !_src.empty()`

`cap.read()` returned an empty frame.

**Fix:** Check that no other application has the camera open (Zoom, Teams, OBS, etc.). Only one process can access a webcam at a time on most operating systems.

---

## Hand detection issues

### Hand not detected at all

- Room is too dark or strongly backlit from behind you
- Hand is too close (<20cm) or too far (>120cm) from camera
- Background color is close to skin tone

**Fix:**
- Add a light source facing you
- Move your hand to 50–80cm from the camera
- Use a plain, non-skin-toned background
- Lower `detection_confidence` in config to `0.6`

---

### Only one hand detected when both are in frame

MediaPipe may drop one hand if both overlap significantly or if the frame is crowded.

**Fix:**
- Keep hands clearly separated in the frame
- Confirm `max_hands: 2` in `config/default.yaml`

---

### Wrong hand type assigned (left/right swapped)

Your camera may horizontally mirror the image (common with front-facing webcams).

**Fix:** Add a flip after `cap.read()` in `run.py`:
```python
ret, frame = cap.read()
frame = cv2.flip(frame, 1)   # horizontal mirror
```

---

## Servo issues

### Servos jitter constantly

- Servos are drawing power from the Arduino 5V pin (insufficient current)
- Signal wire picking up interference

**Fix:**
1. Power servos from a dedicated 5V supply, not the Arduino's 5V pin
2. Shorten or twist the signal wire
3. Add a 100µF capacitor between the servo power rail and GND

---

### Servo moves to one extreme and stays there

The `min_deg` / `max_deg` bounds in the config are incorrect, or the servo is receiving a value outside the 0–180° range.

**Fix:**
1. Print the angle being commanded: check the `servoX/Y/Z` display in the HUD
2. Narrow the bounds in `config/default.yaml`
3. `ArmController.write()` clamps angles — if the HUD shows a valid angle but the servo is at an extreme, check the physical attachment of the servo horn

---

### Servo makes grinding noise

The commanded angle is beyond the servo's physical range of motion.

**Fix:** Narrow `min_deg` / `max_deg` in the config until the grinding stops. The grinding means the servo output shaft is hitting its internal mechanical stop while the motor continues to push.

---

### Arm drifts slowly even when hand is still

The LSTM buffer is filling with slightly different feature vectors due to hand tracking micro-jitter, causing slow output drift.

**Fix:**
1. Increase `sequence_length` in the config (e.g. from 15 to 20) for more smoothing
2. Check that the model is trained — if it is untrained, the output is random

---

## Motor issues

### Motors do not turn

Common causes:

1. **ENA/ENB jumper caps still on** — caps lock speed at full, but only if PWM is not connected. With PWM connected and caps on, the caps are overridden. Actually more likely: the jumpers are off but PWM is not connected.
2. **Common ground missing** — L298N logic GND not connected to Arduino GND
3. **Motor power supply off or wrong voltage**

**Fix:**
1. Remove ENA and ENB jumper caps
2. Connect L298N GND to Arduino GND
3. Verify motor power supply is 7–9V and its GND is connected to common GND

---

### Motors turn but very weakly

Motor supply voltage is too low. The L298N has ~2V dropout, so a 5V supply delivers only ~3V to the motors.

**Fix:** Use a 7–9V power supply for the motor rail.

---

### Left and right motors reversed

Motor wiring is reversed (common if you swap OUT1/OUT2 or OUT3/OUT4).

**Fix:** Swap the two leads of the affected motor at the L298N terminals. Do not change code.

---

### Robot turns in wrong direction

The code's "left" command turns the robot right, or vice versa.

**Fix:** In `config/default.yaml`, swap the `left` and `right` command motor arguments:
```yaml
commands:
  left:  [1, 0, 1.0, 0, 1, 0.5]   # ← swap these two lines
  right: [0, 1, 0.5, 0, 1, 1.0]
```

---

## Speech recognition issues

### `Could not request results from Google Speech Recognition service`

No internet connection, or the Google API is temporarily unavailable.

**Fix:**
- Check internet connectivity
- The system continues in gesture-only mode; speech commands will not work until connectivity is restored
- For offline use, implement `vosk` as an alternative backend in `ASRListener._run()`

---

### Speech commands recognized but nothing happens

The recognized word does not exactly match a command key.

**Fix:** Check the terminal output for `[Speech] Heard: '...'`. If the word is close but not exact (e.g. "forwards" instead of "forward"), add the variant to `config/default.yaml`:
```yaml
commands:
  forward:  [1, 0, 1.0, 0, 1, 1.0]
  forwards: [1, 0, 1.0, 0, 1, 1.0]   # add synonym
```

---

### Microphone not found

```
OSError: [Errno -9996] Invalid input device (no default)
```

No microphone is available or the default audio device is wrong.

**Fix:**
```python
import speech_recognition as sr
for i, mic in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {mic}")
```

Then in `gesture_arm/speech/multimodal.py`, change `sr.Microphone()` to `sr.Microphone(device_index=N)` where N is the index of your microphone.

---

## LSTM / TensorFlow issues

### `RuntimeError: TensorFlow is required`

```
Running in baseline (frame-by-frame mapping) mode.
```

TensorFlow is not installed. The system still works in baseline mode.

**Fix:**
```bash
pip install tensorflow
# macOS M1/M2:
pip install tensorflow-macos tensorflow-metal
```

---

### `No model found at models/lstm_gesture_model.h5`

Training has not been run yet.

**Fix:**
```bash
python scripts/collect.py   # collect training data first
python scripts/train.py     # then train
```

---

### Training loss does not decrease

- Insufficient training data (fewer than ~1000 samples)
- Learning rate too high

**Fix:**
1. Collect more data: `python scripts/collect.py --duration 180`
2. Lower the learning rate in `config/default.yaml`: `learning_rate: 0.0005`

---

### `ValueError: Input 0 of layer lstm is incompatible`

The loaded model's input shape does not match the current `sequence_length` or `feature_dim` config.

**Fix:** Either retrain the model with the current config, or revert the config to match the model's original settings. The model's expected input shape can be inspected with:
```python
from tensorflow.keras.models import load_model
model = load_model("models/lstm_gesture_model.h5")
print(model.input_shape)   # (None, seq_len, feat_dim)
```

---

## Python / dependency issues

### `ModuleNotFoundError: No module named 'cvzone'`

```bash
pip install cvzone
```

If cvzone installs but fails to import with a MediaPipe error:
```bash
pip install mediapipe --upgrade
```

---

### `pip install pyaudio` fails on Windows

```bash
pip install pipwin
pipwin install pyaudio
```

---

### `pip install pyaudio` fails on Ubuntu

```bash
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio
```

---

### `ImportError: libGL.so.1: cannot open shared object file`

OpenCV requires libGL on headless Linux.

```bash
sudo apt-get install libgl1-mesa-glx
# or for headless use:
pip install opencv-python-headless
```

---

## Performance issues

### Frame rate below 15 fps

Typical causes and fixes:

| Cause | Fix |
|---|---|
| Camera USB bandwidth | Reduce resolution in config: `width: 640, height: 480` |
| LSTM inference slow | Reduce `sequence_length` or `lstm_units` in config |
| MediaPipe slow | Reduce `detection_confidence` to 0.6 |
| Too many windows open | Close unused applications |

To measure frame rate:
```python
import time
t0 = time.time()
frame_count = 0
# ... in the loop:
frame_count += 1
if frame_count % 30 == 0:
    fps = 30 / (time.time() - t0)
    t0 = time.time()
    print(f"FPS: {fps:.1f}")
```

---

### LSTM inference takes >20ms

This can cause the vision loop to drop below 30fps.

**Fix:** Reduce model complexity in `config/default.yaml`:
```yaml
model:
  sequence_length: 10   # was 15
  lstm_units: 32        # was 64
```

Then retrain.

---

## CI / Docker issues

### CI fails on `ruff check`

```bash
ruff check --fix gesture_arm/ scripts/ tests/
```

Review the remaining unfixable warnings and address them manually.

---

### CI fails on `black --check`

```bash
black gesture_arm/ scripts/ tests/
git add -u
git commit -m "style: apply black formatting"
```

---

### Docker build fails at `pip install pyaudio`

The `Dockerfile.sim` uses `portaudio19-dev` from apt. If the apt cache is stale:
```dockerfile
RUN apt-get update && apt-get install -y portaudio19-dev
```

Ensure `apt-get update` runs before the install in the Dockerfile.

---

### `python -m gesture_arm.run --no-hardware` exits immediately in Docker

The `--no-hardware` mode still tries to open a camera. In a headless Docker container, there is no camera device.

**Fix:** For CI, test with `--help` instead:
```bash
docker run --rm gesture_arm-sim python -m gesture_arm.run --no-hardware --help
```

For a full headless demo, a video file mock would be needed instead of `cv2.VideoCapture(0)`.
