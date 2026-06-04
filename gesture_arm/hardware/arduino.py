"""
gesture_arm.hardware.arduino
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clean, typed interface to the Arduino over pyFirmata.

Abstracts away all pin-level detail so higher-level code
simply calls arm.write(angles) or base.forward().

Hardware (paper Section V-A):
  Servos  : X → pin 3, Y → pin 5, Z(grip) → pin 6
  L298N   : left  dir1=7, dir2=8, pwm=9
             right dir1=13, dir2=12, pwm=10
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Arm controller — 3DoF servo arm
# ══════════════════════════════════════════════════════════════════════════════

class ArmController:
    """
    Controls the three-servo robotic arm.

    Angles are clamped to their configured bounds before writing,
    so out-of-range values never reach the hardware.

    Usage::

        arm = ArmController(board, cfg.hardware)
        arm.write(np.array([120.0, 90.0, 125.0]))
    """

    def __init__(self, board, servo_configs: dict) -> None:
        import pyfirmata

        self._servos = {}
        self._bounds = {}
        for axis, cfg in servo_configs.items():
            pin = board.get_pin(f"d:{cfg.pin}:s")
            self._servos[axis] = pin
            self._bounds[axis] = (cfg.min_deg, cfg.max_deg)
            pin.write(float(cfg.default))
            logger.debug("Servo %s → pin %d, range [%d, %d]°", axis, cfg.pin, cfg.min_deg, cfg.max_deg)

        logger.info("ArmController ready — axes: %s", list(self._servos))

    def write(self, angles: np.ndarray) -> None:
        """
        Write servo angles to the Arduino.

        Args:
            angles: (3,) array [x_deg, y_deg, z_deg].
                    Values are clamped to their configured bounds.
        """
        for i, axis in enumerate(["x", "y", "z"]):
            lo, hi = self._bounds[axis]
            clamped = float(np.clip(angles[i], lo, hi))
            self._servos[axis].write(clamped)

    def home(self) -> None:
        """Return all servos to their default (home) positions."""
        for axis, servo in self._servos.items():
            lo, hi = self._bounds[axis]
            mid = (lo + hi) / 2
            servo.write(mid)
        logger.info("Arm homed.")


# ══════════════════════════════════════════════════════════════════════════════
# Mobile base controller — L298N DC motors
# ══════════════════════════════════════════════════════════════════════════════

class BaseController:
    """
    Controls the two-wheeled mobile base via an L298N motor driver.

    Direction is encoded as two complementary digital signals (dir1, dir2).
    Speed is a PWM duty cycle [0.0, 1.0].

    Usage::

        base = BaseController(board, cfg.hardware.motors)
        base.forward()
        time.sleep(1.0)
        base.stop()
    """

    def __init__(self, board, motor_cfg) -> None:
        mc = motor_cfg
        self._ld1  = board.get_pin(f"d:{mc.left.dir1}:o")
        self._ld2  = board.get_pin(f"d:{mc.left.dir2}:o")
        self._lpwm = board.get_pin(f"d:{mc.left.pwm}:p")

        self._rd1  = board.get_pin(f"d:{mc.right.dir1}:o")
        self._rd2  = board.get_pin(f"d:{mc.right.dir2}:o")
        self._rpwm = board.get_pin(f"d:{mc.right.pwm}:p")

        self._max   = mc.max_speed
        self._turn  = mc.turn_speed
        self.stop()
        logger.info("BaseController ready — L298N on pins L[%d,%d,%d] R[%d,%d,%d]",
                    mc.left.dir1, mc.left.dir2, mc.left.pwm,
                    mc.right.dir1, mc.right.dir2, mc.right.pwm)

    # ── Directional commands ───────────────────────────────────────────────────

    def forward(self) -> None:
        self._drive(1, 0, self._max, 0, 1, self._max)
        logger.debug("Base: forward")

    def reverse(self) -> None:
        self._drive(0, 1, self._max, 1, 0, self._max)
        logger.debug("Base: reverse")

    def turn_left(self) -> None:
        self._drive(0, 1, self._turn, 0, 1, self._max)
        logger.debug("Base: left")

    def turn_right(self) -> None:
        self._drive(1, 0, self._max, 0, 1, self._turn)
        logger.debug("Base: right")

    def stop(self) -> None:
        self._lpwm.write(0.0)
        self._rpwm.write(0.0)
        logger.debug("Base: stop")

    # ── Low-level drive ────────────────────────────────────────────────────────

    def _drive(
        self,
        ld1: int, ld2: int, lspd: float,
        rd1: int, rd2: int, rspd: float,
    ) -> None:
        self._ld1.write(ld1);  self._ld2.write(ld2);  self._lpwm.write(lspd)
        self._rd1.write(rd1);  self._rd2.write(rd2);  self._rpwm.write(rspd)


# ══════════════════════════════════════════════════════════════════════════════
# Board factory
# ══════════════════════════════════════════════════════════════════════════════

def connect(port: str, baudrate: int = 57600):
    """
    Connect to the Arduino and return the pyfirmata Board object.

    Args:
        port:     Serial port string, e.g. "COM6" or "/dev/ttyUSB0".
        baudrate: Firmata baud rate (default 57600 matches StandardFirmata).

    Returns:
        pyfirmata.Arduino board object.

    Raises:
        serial.SerialException: if the port cannot be opened.
    """
    try:
        import pyfirmata
    except ImportError as exc:
        raise ImportError(
            "pyfirmata is required. Install with: pip install pyfirmata"
        ) from exc

    logger.info("Connecting to Arduino on %s @ %d baud …", port, baudrate)
    board = pyfirmata.Arduino(port)

    # Start the iterator thread so analog/digital reads don't block
    it = pyfirmata.util.Iterator(board)
    it.start()

    logger.info("Arduino connected on %s", port)
    return board


@contextmanager
def board_session(port: str, baudrate: int = 57600):
    """
    Context manager that connects to the Arduino and ensures clean exit.

    Usage::

        with board_session("COM6") as board:
            arm = ArmController(board, cfg.hardware.servos)
            ...
    """
    board = connect(port, baudrate)
    try:
        yield board
    finally:
        board.exit()
        logger.info("Arduino disconnected.")
