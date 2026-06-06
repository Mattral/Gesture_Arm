# Setup Guide

Complete step-by-step instructions for assembling the hardware, installing the software, and running the system for the first time.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Hardware assembly](#2-hardware-assembly)
3. [Arduino firmware](#3-arduino-firmware)
4. [Python environment](#4-python-environment)
5. [Configuration](#5-configuration)
6. [Verify the installation](#6-verify-the-installation)
7. [First run](#7-first-run)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### Hardware required

| Item | Qty | Notes |
|---|---|---|
| Arduino Uno R3 | 1 | Or compatible (Nano works on pins 3–13) |
| SG90 or MG996R servo motors | 3 | SG90 for light loads; MG996R for heavier arm |
| L298N dual H-bridge module | 1 | The common red breakout board |
| DC gear motors (3–6V) | 2 | For the mobile base wheels |
| USB webcam | 1 | Any 720p+ camera; built-in laptop cam works |
| USB cable (Type-A to Type-B) | 1 | Arduino to host PC |
| Jumper wires | ~30 | Male-to-male and male-to-female |
| 5V power supply or USB power bank | 1 | For servo power rail |
| 7–12V power supply | 1 | For L298N motor power (separate from logic) |
| Breadboard | 1 | Optional but recommended |

### Software required on host PC

| Software | Version | Purpose |
|---|---|---|
| Python | 3.9 or higher | Runtime |
| Arduino IDE | 2.x | Firmware upload |
| Git | Any | Clone repo |

### Operating system support

| OS | Status | Notes |
|---|---|---|
| Windows 10/11 | ✅ Fully supported | COM port format: `COM6` |
| Ubuntu 20.04 / 22.04 | ✅ Fully supported | Port format: `/dev/ttyUSB0` |
| macOS 12+ | ✅ Fully supported | Port format: `/dev/cu.usbmodem*` |

---

## 2. Hardware assembly

### 2a. Robot arm servo wiring

Wire each servo's three leads to the Arduino and power rail:

```
Servo X (horizontal pan)
├── Signal (orange/yellow) → Arduino D3
├── VCC    (red)           → 5V power rail
└── GND    (brown/black)   → Common GND

Servo Y (vertical tilt)
├── Signal → Arduino D5
├── VCC    → 5V power rail
└── GND    → Common GND

Servo Z (grip)
├── Signal → Arduino D6
├── VCC    → 5V power rail
└── GND    → Common GND
```

> ⚠️ **Do not power servos from the Arduino 5V pin.** Three servos draw 600–900mA under load, which will reset or damage the Arduino. Use a dedicated 5V supply or USB power bank. Connect GND of the power supply to Arduino GND.

### 2b. L298N motor driver wiring

```
L298N module          Arduino / motors
─────────────────────────────────────
IN1               →   D7
IN2               →   D8
ENA (PWM)         →   D9      ← remove the jumper cap on ENA
IN3               →   D13
IN4               →   D12
ENB (PWM)         →   D10     ← remove the jumper cap on ENB

OUT1 + OUT2       →   Left motor (red/black leads)
OUT3 + OUT4       →   Right motor (red/black leads)

12V               →   7–12V motor power supply +
GND               →   Motor power supply − AND Arduino GND
5V (logic out)    →   (optional) can power Arduino if no USB
```

> ⚠️ **Keep motor power supply GND connected to Arduino GND.** Without a common ground, the L298N signal lines will not work.

### 2c. Full wiring diagram (text)

```
HOST PC ──USB──► Arduino Uno
                     │
          D3 ────► Servo X signal
          D5 ────► Servo Y signal
          D6 ────► Servo Z signal
          D7 ────► L298N IN1
          D8 ────► L298N IN2
          D9 ────► L298N ENA (PWM)
         D10 ────► L298N ENB (PWM)
         D12 ────► L298N IN4
         D13 ────► L298N IN3
         GND ────► Common GND rail
                     │
5V supply ──────────► Servo VCC rail
GND ────────────────► Common GND rail

7–12V supply ───────► L298N 12V
GND ────────────────► Common GND rail

L298N OUT1/OUT2 ────► Left DC motor
L298N OUT3/OUT4 ────► Right DC motor
```

### 2d. Camera placement

Mount the webcam directly above or below the screen, facing you. The system splits the frame vertically — your **right hand** should be able to reach the **left half** of the frame, and your **left hand** the **right half**. A camera-to-operator distance of 60–90 cm works well.

---

## 3. Arduino firmware

### 3a. Install the Arduino IDE

Download from [arduino.cc/en/software](https://www.arduino.cc/en/software) and install.

### 3b. Install required library

The firmware uses **StandardFirmata**, which ships with the Arduino IDE:

```
Arduino IDE → File → Examples → Firmata → StandardFirmata
```

Alternatively, use the custom firmware in this repo (identical to StandardFirmata but with confirmed servo support):

```
firmware/server.ino
```

### 3c. Upload the firmware

1. Connect Arduino Uno via USB
2. Open `firmware/server.ino` in the Arduino IDE
3. Select board: **Tools → Board → Arduino AVR Boards → Arduino Uno**
4. Select port: **Tools → Port → COM6** (Windows) or **/dev/ttyUSB0** (Linux/macOS)
5. Click **Upload** (→ button)
6. Wait for "Done uploading"

### 3d. Confirm the port

**Windows:**
```
Device Manager → Ports (COM & LPT) → Arduino Uno (COMx)
```

**Linux:**
```bash
ls /dev/ttyUSB* /dev/ttyACM*
# Expected: /dev/ttyUSB0 or /dev/ttyACM0

# Grant permission if needed:
sudo chmod a+rw /dev/ttyUSB0
# Or add yourself to the dialout group (permanent):
sudo usermod -aG dialout $USER
```

**macOS:**
```bash
ls /dev/cu.usbmodem* /dev/cu.usb*
```

---

## 4. Python environment

### 4a. Clone the repository

```bash
git clone https://github.com/minhtetmyet/gesture-arm.git
cd gesture-arm
```

### 4b. Create a virtual environment

```bash
# Create
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 4c. Install the package

**Minimum install** (gesture control only, no LSTM training):
```bash
pip install -e .
```

**Full install** (includes TensorFlow for LSTM training):
```bash
pip install -e ".[ml]"
```

**Development install** (adds pytest, notebooks, linting tools):
```bash
pip install -e ".[ml,dev]"
```

### 4d. Install PyAudio (required for speech recognition)

PyAudio requires PortAudio. Install the system dependency first:

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Ubuntu / Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

### 4e. Verify all dependencies

```bash
python -c "
import cv2, numpy, pyfirmata, speech_recognition, pyttsx3, yaml
print('Core dependencies OK')
try:
    import tensorflow as tf
    print(f'TensorFlow {tf.__version__} OK')
except ImportError:
    print('TensorFlow not installed (optional for LSTM training)')
"
```

Expected output:
```
Core dependencies OK
TensorFlow 2.x.x OK
```

---

## 5. Configuration

All settings are in `gesture_arm/config/default.yaml`. Open it and update the serial port:

```yaml
hardware:
  port: "COM6"      # ← change this to your Arduino port
  baudrate: 57600
```

**Common port values:**

| OS | Typical port |
|---|---|
| Windows | `"COM3"`, `"COM4"`, `"COM6"` |
| Linux | `"/dev/ttyUSB0"`, `"/dev/ttyACM0"` |
| macOS | `"/dev/cu.usbmodem14101"` |

You can also override the port at runtime without editing the file:

```bash
# Windows PowerShell:
$env:GESTURE_ARM_PORT = "COM4"
python -m gesture_arm.run

# Linux / macOS:
GESTURE_ARM_PORT=/dev/ttyUSB0 python -m gesture_arm.run
```

### Camera index

If the system opens the wrong camera, change `camera_index` in the config:

```yaml
vision:
  camera_index: 0   # 0 = default/first camera, 1 = second camera, etc.
```

To find your camera index:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: available")
    cap.release()
```

---

## 6. Verify the installation

Run the hardware-free smoke test before connecting the Arduino:

```bash
python -m gesture_arm.run --no-hardware
```

This opens the camera feed with hand tracking overlays but does not attempt to connect to the Arduino. You should see:
- Live camera window titled "Gesture Arm Control"
- Blue rectangle on the left half (right-hand zone)
- Green rectangle on the right half (left-hand zone)
- Hand landmarks drawn when hands are visible

Press **Q** to quit.

### Run the unit tests

```bash
pytest tests/ -v
```

All tests should pass without hardware or TensorFlow connected.

---

## 7. First run

### Step 1 — Connect the Arduino

Plug in the Arduino USB cable. Confirm the port in `config/default.yaml`.

### Step 2 — Collect training data

```bash
python scripts/collect.py --duration 90
```

Move your **left hand** slowly and deliberately through the full range of motion in the **right half** of the camera frame:
- Left to right (full X range)
- Top to bottom (full Y range)
- Pinch open and closed repeatedly (full Z range)

The script displays a progress bar and sample counter. 90 seconds yields approximately 2,700 samples at 30 fps — sufficient for training. Press **Q** to save and exit early.

Data is saved to `data/training_data.csv`.

### Step 3 — Train the LSTM model

```bash
python scripts/train.py
```

Training takes approximately 2–4 minutes on CPU. You will see the Keras training progress with loss and validation MAE per epoch. Early stopping will halt training if validation loss stops improving.

The trained model is saved to `models/lstm_gesture_model.h5`.

### Step 4 — Run the full system

```bash
python -m gesture_arm.run
```

**Right hand controls (left camera zone):**

| Gesture | Action |
|---|---|
| Hand in upper zone | Forward |
| Hand in lower zone | Reverse |
| Hand in left zone | Turn left |
| Hand in right zone | Turn right |
| Closed fist | Stop (emergency override) |

**Left hand controls (right camera zone):**

| Motion | Servo | Range |
|---|---|---|
| Move left/right | Servo X (pan) | 60° – 180° |
| Move up/down | Servo Y (tilt) | 40° – 140° |
| Pinch open/close | Servo Z (grip) | 100° – 150° |

**Voice commands** (speak clearly):

| Word | Action |
|---|---|
| "forward" / "go" | Move forward |
| "reverse" / "back" | Move backward |
| "left" | Turn left |
| "right" | Turn right |
| "stop" | Stop motors |

Press **Q** to exit. A metrics summary is printed to the terminal.

---

## 8. Troubleshooting

### Arduino not detected

```
serial.SerialException: could not open port 'COM6'
```

- Confirm the port in Device Manager / `ls /dev/ttyUSB*`
- Update `port` in `config/default.yaml`
- Try unplugging and replugging the USB cable
- On Linux, run `sudo chmod a+rw /dev/ttyUSB0`

---

### Camera not opening

```
Camera read failed — is camera_index correct?
```

- Run the camera detection script in Section 5 to find the correct index
- Update `camera_index` in `config/default.yaml`
- On Linux, ensure your user is in the `video` group: `sudo usermod -aG video $USER`

---

### Hand not detected

- Ensure lighting is adequate — MediaPipe struggles in dark or strongly backlit conditions
- Keep your hand within 30–80 cm of the camera
- Avoid backgrounds with skin-like colors
- Try increasing `detection_confidence` to 0.9 or decreasing it to 0.7 in the config

---

### Servos jittering without input

- Confirm servos are powered from a dedicated 5V supply, not the Arduino 5V pin
- Check that the signal wires are connected to the correct pins (D3, D5, D6)
- Verify GND of the servo power supply is connected to Arduino GND

---

### Speech recognition not working

```
ASR service error: recognition connection failed
```

- The default speech engine (Google) requires an internet connection
- Check firewall / proxy settings
- For offline use, install `vosk`: `pip install vosk` — a future update will add offline ASR support

---

### `pip install pyaudio` fails on Windows

Use `pipwin` instead of pip directly:
```bash
pip install pipwin
pipwin install pyaudio
```

---

### TensorFlow install fails

TensorFlow 2.x requires Python 3.9–3.11 and a 64-bit OS. Common fix:

```bash
pip install tensorflow --upgrade
```

If you are on an M1/M2 Mac:
```bash
pip install tensorflow-macos tensorflow-metal
```

---

### Model file not found

```
No model found at models/lstm_gesture_model.h5 — building untrained model.
```

You need to collect data and train before running the full LSTM mode. The system will still work in baseline mode. Run:

```bash
python scripts/collect.py
python scripts/train.py
```

---

## 9. IK mode setup

IK mode maps your hand position to a Cartesian end-effector target instead
of directly to servo angles. This requires two additional steps.

### 9a. Measure your link lengths

With the arm powered off, measure physically with a ruler:

```
l1 = distance from the servo-Y pivot (shoulder) to the end-effector mount point
l2 = distance from the mount point to the TCP (fingertip of the gripper)
```

Typical values for a standard 3D-printed SG90 arm: l1 ≈ 10 cm, l2 ≈ 8 cm.

### 9b. Update the config

```yaml
# gesture_arm/config/default.yaml
kinematics:
  enabled: false          # keep false; use --ik flag at runtime instead
  link1_cm: 10.0          # ← your measured l1
  link2_cm: 8.0           # ← your measured l2
  servo_x_neutral_deg: 120.0   # servo X angle when arm faces forward
  servo_y_zero_deg: 40.0       # servo Y angle when arm is horizontal
```

### 9c. Run in IK mode

```bash
python -m gesture_arm.run --ik
```

Or enable permanently in the config (`kinematics.enabled: true`) and run normally.

### 9d. Verify workspace

Check that the arm reaches the expected positions. If the arm overshoots or
undershoots, your link lengths may be slightly off. Run the FK check in a
Python console:

```python
from gesture_arm.kinematics import GeometricIKSolver
solver = GeometricIKSolver(link1_cm=10.0, link2_cm=8.0)

# Check: what TCP position does servo X=120, Y=90 produce?
px, py, pz = solver.forward(120, 90)
print(f"TCP at: ({px:.1f}, {py:.1f}, {pz:.1f}) cm")

# Check: can the arm reach (10, 0, 5)?
r = solver.solve(10, 0, 5)
print(r.message)
print(f"Servo X={r.angles[0]:.1f}°  Y={r.angles[1]:.1f}°")
```

Adjust `link1_cm` and `link2_cm` until `forward()` output matches what
you observe physically.

### 9e. IK fallback behaviour

If IK is enabled but the target is outside the workspace, the system
falls through to LSTM/baseline automatically. The HUD method badge
shows the current path: `[ik]`, `[lstm]`, or `[baseline]`.
The metrics CSV method column records which path produced each frame's
servo command, enabling comparison in the benchmark notebook.
