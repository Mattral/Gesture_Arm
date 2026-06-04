# Hardware Reference

Complete reference for the physical components, wiring, power requirements, and mechanical assembly of the Gesture Arm system.

---

## Table of contents

1. [Bill of materials](#1-bill-of-materials)
2. [Pin assignment table](#2-pin-assignment-table)
3. [Wiring diagrams](#3-wiring-diagrams)
4. [Power supply design](#4-power-supply-design)
5. [Servo selection and calibration](#5-servo-selection-and-calibration)
6. [Motor driver configuration](#6-motor-driver-configuration)
7. [Mechanical assembly notes](#7-mechanical-assembly-notes)
8. [Safety considerations](#8-safety-considerations)

---

## 1. Bill of materials

| # | Component | Specification | Qty | Approx. cost (USD) |
|---|---|---|---|---|
| 1 | Arduino Uno R3 | ATmega328P, USB-B | 1 | $5–10 |
| 2 | Servo motor (arm) | SG90 (light) or MG996R (heavy) | 3 | $3–8 each |
| 3 | L298N motor driver module | Dual H-bridge, 2A per channel | 1 | $3–5 |
| 4 | DC gear motor | 3–6V, 100–200 RPM | 2 | $3–6 each |
| 5 | USB webcam | 720p or higher, USB 2.0 | 1 | $10–20 |
| 6 | USB Type-A to Type-B cable | For Arduino connection | 1 | $2 |
| 7 | 5V DC power supply | ≥2A for servos | 1 | $5–8 |
| 8 | 7–9V DC power supply | For motors via L298N | 1 | $5–10 |
| 9 | Jumper wires | Male-to-male and male-to-female | 30 | $3 |
| 10 | Breadboard | 400-point or larger | 1 | $3 |
| 11 | Robot arm frame | 3D printed or acrylic laser cut | 1 | $5–15 |
| 12 | Mobile base chassis | 2WD Arduino car kit | 1 | $8–15 |
| **Total** | | | | **~$55–120** |

### Component notes

**Servos:** SG90 is sufficient for a lightweight 3D-printed arm. Use MG996R if the arm carries any payload or if the links are longer than 10 cm. Both use the same 50Hz PWM signal and work identically with StandardFirmata.

**L298N module:** Buy the common red breakout board that includes the logic-level regulators and flyback diodes. Bare L298N ICs require external protection components.

**DC motors:** The speed and torque specification depends on your mobile base chassis. Most 2WD Arduino car kits come with 3V–6V motors that work with the L298N's output.

---

## 2. Pin assignment table

| Arduino Pin | Mode | Connected to | Notes |
|---|---|---|---|
| D3 | Servo PWM | Servo X signal wire | Horizontal pan |
| D5 | Servo PWM | Servo Y signal wire | Vertical tilt |
| D6 | Servo PWM | Servo Z signal wire | Grip open/close |
| D7 | Digital out | L298N IN1 | Left motor direction |
| D8 | Digital out | L298N IN2 | Left motor direction |
| D9 | PWM out | L298N ENA | Left motor speed |
| D10 | PWM out | L298N ENB | Right motor speed |
| D12 | Digital out | L298N IN4 | Right motor direction |
| D13 | Digital out | L298N IN3 | Right motor direction |
| GND | Ground | Common ground rail | — |
| 5V | Power | Logic supply only | Do NOT power servos here |

> **Pins D3, D5, D6, D9, D10** require hardware PWM support. On the Arduino Uno these are hardware PWM pins. Do not substitute with non-PWM digital pins.

---

## 3. Wiring diagrams

### 3a. Servo arm wiring

```
Arduino Uno                Servo motor (×3)
───────────                ────────────────
D3  ──────────────────────► Signal  (orange or white wire)
5V rail (external) ───────► VCC     (red wire)
GND (common) ─────────────► GND     (brown or black wire)

D5  ──────────────────────► Signal  (Servo Y)
5V rail ──────────────────► VCC
GND ──────────────────────► GND

D6  ──────────────────────► Signal  (Servo Z)
5V rail ──────────────────► VCC
GND ──────────────────────► GND
```

### 3b. L298N motor driver wiring

```
Arduino Uno          L298N Module
───────────          ────────────
D7  ────────────────► IN1
D8  ────────────────► IN2
D9  ────────────────► ENA        (remove ENA jumper cap first)
D10 ────────────────► ENB        (remove ENB jumper cap first)
D12 ────────────────► IN4
D13 ────────────────► IN3
GND ────────────────► GND

7–9V supply + ──────► 12V (motor power)
7–9V supply − ──────► GND

L298N Module         DC Motors
────────────         ─────────
OUT1  ──────────────► Left motor  (terminal 1)
OUT2  ──────────────► Left motor  (terminal 2)
OUT3  ──────────────► Right motor (terminal 1)
OUT4  ──────────────► Right motor (terminal 2)
```

### 3c. Complete system overview

```
HOST PC
   │
   │ USB (data + Arduino power)
   │
Arduino Uno
   │
   ├── D3 ─────────────────────────────────────► Servo X signal
   ├── D5 ─────────────────────────────────────► Servo Y signal
   ├── D6 ─────────────────────────────────────► Servo Z signal
   │
   ├── D7, D8, D9  ────────────────────────────► L298N IN1/IN2/ENA
   ├── D10, D12, D13 ──────────────────────────► L298N ENB/IN4/IN3
   │
   └── GND ────────────────────────────────────┐
                                               │
5V power supply ────────────────────────────── ┤ Common GND rail
   │                                           │
   └── 5V ────► Servo VCC rail (all 3 servos)  │
                                               │
7–9V power supply ────► L298N 12V              │
   └── GND ──────────────────────────────────── ┘
                │
           L298N OUT1/2 ──────────────► Left motor
           L298N OUT3/4 ──────────────► Right motor
```

---

## 4. Power supply design

### Why three power domains?

The system uses three separate power sources for reliability and hardware protection.

**Domain 1 — Arduino logic (USB, 5V, ~500mA):** The Arduino is powered by the USB connection to the host PC. This also powers the Arduino's digital I/O and provides the Firmata serial communication. Never run motors or servos from this domain.

**Domain 2 — Servo power (5V, ≥2A dedicated supply):** Three SG90 servos draw 150–250mA each at idle and up to 700mA under stall load. Three MG996R servos can draw 2.5A under stall. A 5V/2A USB power bank or wall adapter works well for SG90s. For MG996Rs, use a 5V/3A supply.

**Domain 3 — Motor power (7–9V, ≥1A):** The L298N expects 7–12V on its motor power pin. Lower voltage reduces motor torque and top speed. The L298N also has a ~2V voltage drop across each H-bridge, so a 9V supply delivers ~7V to the motors.

### Common ground rule

All three power domains must share a common ground. Connect:
- Arduino GND
- 5V servo supply GND
- 7–9V motor supply GND

to the same ground rail. Without a common ground, the L298N direction signals (D7, D8, etc.) will not be interpreted correctly because there is no voltage reference.

### Current capacity check

For SG90 servos with a 2WD mobile base:

| Consumer | Current (peak) |
|---|---|
| Arduino Uno | ~50mA |
| 3× SG90 servos (stall) | ~600mA |
| 2× DC motors (stall) | ~600mA |
| L298N quiescent | ~50mA |
| **Total** | **~1.3A** |

A 5V/2A supply for servos and a 9V/1A supply for motors are both sufficient.

---

## 5. Servo selection and calibration

### SG90 vs MG996R comparison

| Property | SG90 | MG996R |
|---|---|---|
| Torque | 1.8 kg·cm | 9.4 kg·cm |
| Speed | 0.1 s/60° | 0.17 s/60° |
| Weight | 9g | 55g |
| Gear material | Plastic | Metal |
| Cost | ~$2 | ~$5 |
| Best for | Lightweight arm links < 10cm | Heavier arms, longer links |

### Angle range configuration

Default servo ranges in `config/default.yaml`:

```yaml
servos:
  x: { min_deg: 60,  max_deg: 180 }   # 120° range, horizontal pan
  y: { min_deg: 40,  max_deg: 140 }   # 100° range, vertical tilt
  z: { min_deg: 100, max_deg: 150 }   # 50° range, grip
```

Adjust these ranges to match your physical arm's actual safe range of motion. If a servo makes a grinding noise or stalls at a commanded position, the configured range exceeds the physical limit — narrow the range in the config.

### Calibration procedure

1. Set all servos to their midpoint: `x=120, y=90, z=125`
2. Attach servo horns so the arm is at a neutral upright position at these angles
3. Manually move the arm to its maximum extent in each direction
4. Note the physical limit angles and set `min_deg` / `max_deg` accordingly
5. Test the full range by running `python -m gesture_arm.run --no-hardware` and moving your hand through the full frame

---

## 6. Motor driver configuration

### Removing the ENA/ENB jumper caps

The L298N breakout board ships with jumper caps on ENA and ENB. These caps connect the enable pins to 5V, which locks the motors at full speed. **Remove both jumper caps** before connecting D9 and D10. With the caps removed, the Arduino controls speed via PWM on those pins.

### Motor direction

If a motor turns the wrong direction, swap its two output wires (OUT1/OUT2 for left, OUT3/OUT4 for right). This is purely a mechanical fix — do not change the firmware or Python code to compensate for reversed wiring.

### Motor stall protection

The L298N does not have built-in overcurrent protection. If a motor stalls (e.g. wheels blocked), the H-bridge will overheat. For extended use, add a small heatsink to the L298N chip. The module's thermal shutdown will trigger at ~150°C, which appears as the motor suddenly stopping.

---

## 7. Mechanical assembly notes

### 3DoF arm geometry

The three servos control:
- **Servo X (pin 3):** Rotation around the vertical axis — sweeps the arm left and right
- **Servo Y (pin 5):** Rotation around the horizontal axis — raises and lowers the arm
- **Servo Z (pin 6):** Opens and closes the gripper — controlled by thumb-index pinch distance

For a well-balanced arm, mount Servo X at the base, Servo Y at the shoulder, and Servo Z at the gripper. Keep link lengths under 12 cm for SG90 servos to stay within torque limits.

### Mobile base

A standard 2WD Arduino car chassis with two rear drive wheels and one front caster wheel works well. The differential steering (one motor per side) maps cleanly to the left/right turn commands in `BaseController`.

### Camera mount

Mount the camera so that both hands can comfortably reach their respective control zones. A clip-on mount on a laptop screen at eye height is the simplest setup. The camera should face the operator directly — tilting the camera down or sideways degrades MediaPipe's hand detection accuracy.

---

## 8. Safety considerations

**Servo stall:** Never command a servo to a position beyond its physical stop. Use the `min_deg` / `max_deg` configuration to stay within safe bounds. `ArmController.write()` clamps angles to these bounds before writing.

**Motor runaway:** The code has a 2-second auto-stop: if no gesture is detected for 2 seconds, the motors stop. This prevents the robot from driving away if the operator's hand leaves the camera frame. The `stop_timeout_sec` value is configurable.

**Servo power:** Do not power servos from the Arduino 5V pin. Three servos under load can draw more current than the Arduino's onboard regulator can supply, causing resets or permanent damage.

**Arduino reset during operation:** If the Arduino resets mid-operation (caused by power brownout or a firmware crash), pyFirmata will raise a serial exception. The code catches this in the `finally` block and calls `board.exit()`. The servos will hold their last position until power is removed.

**Eye safety:** The webcam is standard consumer hardware with no additional laser or IR emitters beyond the camera sensor. No specific eye safety precautions beyond normal computer use are required.
