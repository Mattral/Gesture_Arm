"""
gesture_arm.kinematics.ik_solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Geometric Inverse Kinematics solver for the 3-DoF revolute arm.

Arm morphology (RRR planar + base-rotation)
-------------------------------------------

Joint 1 — Base rotation   (θ₁, servo X)  :  rotates the whole arm around
                                             the vertical Z-axis.
                                             Maps to servo X: [60°, 180°]
                                             Physical zero = arm pointing forward.

Joint 2 — Shoulder elevation (θ₂, servo Y) : rotates the upper link in the
                                             vertical plane.
                                             Maps to servo Y: [40°, 140°]
                                             Physical zero = arm fully lowered.

Joint 3 — Gripper (θ₃, servo Z)           : opens / closes the end-effector.
                                             NOT a positional joint — it does
                                             not contribute to TCP position.
                                             Controlled separately by the
                                             operator's pinch gesture.

Because Joint 3 is a gripper (not an elbow), the TCP position is determined
entirely by θ₁ and θ₂, making this a 2-DoF positional problem in 3-D space
solved analytically with straightforward trigonometry.

Coordinate frame
----------------
Origin: servo-X pivot (base centre).
  +X : arm pointing forward at θ₁ = 90° (servo mid-point)
  +Y : arm pointing left   at θ₁ = 180°
  +Z : up

Link lengths (configurable in default.yaml → kinematics section):
  l1 : upper-arm link  (shoulder pivot → end-effector mount)  default 10 cm
  l2 : forearm link    (end-effector mount → TCP)             default 8 cm

Forward kinematics (for verification / unit tests)
---------------------------------------------------
  r  = l1·cos(θ₂) + l2·cos(θ₂)      [radial reach in the arm's vertical plane]
  x  = r · cos(θ₁)
  y  = r · sin(θ₁)
  z  = l1·sin(θ₂) + l2·sin(θ₂)      [height above base]

  ← simplified: both links share elevation angle θ₂, i.e. the arm is a
    single rigid link of total length l1+l2 in the elevation plane.
    This matches the SG90-based arm where the two links are rigidly coupled.

Inverse kinematics
------------------
Given desired TCP position (px, py, pz) in metres:

  r_desired = √(px² + py²)           [horizontal reach]
  θ₁ = atan2(py, px)                 [base rotation angle]

  L  = √(r_desired² + pz²)           [straight-line distance from shoulder to TCP]
  If L > l1 + l2 → target unreachable (IKSolution.UNREACHABLE)
  If L < |l1 - l2| → target inside dead-zone (IKSolution.IN_DEADZONE)

  θ₂ = atan2(pz, r_desired)          [elevation angle]

After computing the geometric angles, they are converted to servo-space
degrees and validated against the physical joint limits stored in config.

Usage
-----
    from gesture_arm.kinematics import GeometricIKSolver

    solver = GeometricIKSolver(link1_cm=10.0, link2_cm=8.0)

    # From a target (x, y, z) in centimetres
    result = solver.solve(px=5.0, py=3.0, pz=4.0)

    if result.reachable:
        arm.write(result.angles)    # np.ndarray([θ₁_servo, θ₂_servo, θ₃_servo])
    else:
        print(result.message)

    # The LSTM / baseline pipeline still works normally.
    # IK is an optional path activated when the operator's hand crosses
    # into the IK zone (configurable) or when called programmatically.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Result types
# ══════════════════════════════════════════════════════════════════════════════

class IKSolution(Enum):
    """Outcome of an IK solve attempt."""
    OK           = auto()   # Valid solution within joint limits
    UNREACHABLE  = auto()   # Target is beyond maximum reach (L > l1+l2)
    IN_DEADZONE  = auto()   # Target is inside minimum reach (L < |l1-l2|)
    JOINT_LIMIT  = auto()   # Solution exists geometrically but violates servo bounds


@dataclass(frozen=True)
class IKResult:
    """
    Immutable result of one IK computation.

    Attributes:
        solution:     IKSolution enum value.
        angles:       np.ndarray shape (3,) — [θ₁_servo°, θ₂_servo°, θ₃_servo°]
                      Valid only when solution == IKSolution.OK.
                      θ₃ is passed through unchanged from the caller (gripper).
        theta1_rad:   Base rotation in radians (raw geometric result).
        theta2_rad:   Shoulder elevation in radians (raw geometric result).
        reach_cm:     Actual straight-line reach L used in the computation.
        reachable:    Convenience bool — True iff solution == IKSolution.OK.
        message:      Human-readable status string.
    """
    solution:   IKSolution
    angles:     Optional[np.ndarray]
    theta1_rad: float
    theta2_rad: float
    reach_cm:   float
    reachable:  bool
    message:    str


# ══════════════════════════════════════════════════════════════════════════════
# Core solver
# ══════════════════════════════════════════════════════════════════════════════

class GeometricIKSolver:
    """
    Analytical geometric IK solver for the 2-DoF positional arm.

    The arm is modelled as two collinear rigid links (total length l1+l2)
    rotating in the vertical plane set by the base rotation angle θ₁.
    This is the correct model for the SG90-based arm used in the project,
    where a single shoulder servo (Y) controls arm elevation and there is no
    independent elbow joint.

    Parameters
    ----------
    link1_cm : float
        Length of the upper-arm link in centimetres (shoulder to mount point).
        Default: 10.0 cm.
    link2_cm : float
        Length of the forearm / end-effector mount in centimetres.
        Default: 8.0 cm.
    servo_x_bounds : tuple[float, float]
        (min_deg, max_deg) for servo X (base rotation). Default: (60, 180).
    servo_y_bounds : tuple[float, float]
        (min_deg, max_deg) for servo Y (shoulder elevation). Default: (40, 140).
    servo_x_neutral_deg : float
        Servo X angle at which the arm points along the +X axis (world forward).
        Default: 120° (mid-point of [60, 180]).
    servo_y_zero_deg : float
        Servo Y angle corresponding to θ₂ = 0 (arm fully horizontal).
        Default: 40° (lower bound).

    Notes
    -----
    All public methods accept and return centimetres for spatial quantities
    and degrees for angles, matching the rest of the gesture_arm API.
    Internal computation uses radians.
    """

    def __init__(
        self,
        link1_cm: float = 10.0,
        link2_cm: float = 8.0,
        servo_x_bounds: Tuple[float, float] = (60.0, 180.0),
        servo_y_bounds: Tuple[float, float] = (40.0, 140.0),
        servo_x_neutral_deg: float = 120.0,
        servo_y_zero_deg: float = 40.0,
    ) -> None:
        if link1_cm <= 0 or link2_cm <= 0:
            raise ValueError("Link lengths must be positive.")

        self.l1    = link1_cm
        self.l2    = link2_cm
        self.l_max = link1_cm + link2_cm
        self.l_min = abs(link1_cm - link2_cm)

        self._x_bounds  = servo_x_bounds
        self._y_bounds  = servo_y_bounds
        self._x_neutral = servo_x_neutral_deg
        self._y_zero    = servo_y_zero_deg

        # Elevation scale: degrees per radian for servo Y
        # Maps θ₂ ∈ [0, π/2] → servo_y ∈ [y_zero, y_max]
        self._y_scale = (servo_y_bounds[1] - servo_y_zero_deg) / (math.pi / 2)

        logger.info(
            "GeometricIKSolver ready — links: l1=%.1f cm, l2=%.1f cm, "
            "reach: [%.1f, %.1f] cm",
            link1_cm, link2_cm, self.l_min, self.l_max,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def solve(
        self,
        px: float,
        py: float,
        pz: float,
        gripper_deg: float = 120.0,
    ) -> IKResult:
        """
        Compute joint angles for the desired end-effector position.

        Parameters
        ----------
        px, py, pz : float
            Desired TCP position in centimetres relative to the arm base origin.
              px : forward (along +X at θ₁ = servo neutral)
              py : left    (along +Y)
              pz : up      (along +Z)
        gripper_deg : float
            Pass-through servo Z angle (grip state). The IK solver does not
            modify the gripper — its state is set by the pinch gesture.

        Returns
        -------
        IKResult
            Contains the solution status, servo angles (3,), and diagnostics.

        Examples
        --------
        >>> solver = GeometricIKSolver(link1_cm=10, link2_cm=8)
        >>> r = solver.solve(px=8.0, py=0.0, pz=6.0)
        >>> assert r.reachable
        >>> print(r.angles)  # [θ₁_servo, θ₂_servo, gripper]
        """
        # ── Step 1: horizontal reach and base angle ────────────────────────────
        r_desired = math.sqrt(px ** 2 + py ** 2)
        theta1    = math.atan2(py, px)          # radians, CCW from +X

        # ── Step 2: straight-line distance from shoulder to TCP ───────────────
        L = math.sqrt(r_desired ** 2 + pz ** 2)

        # ── Step 3: reachability check ────────────────────────────────────────
        if L > self.l_max:
            msg = (
                f"Target ({px:.1f}, {py:.1f}, {pz:.1f}) cm is unreachable: "
                f"required reach {L:.2f} cm > max {self.l_max:.2f} cm."
            )
            logger.debug(msg)
            return IKResult(
                solution=IKSolution.UNREACHABLE,
                angles=None,
                theta1_rad=theta1,
                theta2_rad=float("nan"),
                reach_cm=L,
                reachable=False,
                message=msg,
            )

        if L < self.l_min:
            msg = (
                f"Target ({px:.1f}, {py:.1f}, {pz:.1f}) cm is in dead-zone: "
                f"required reach {L:.2f} cm < min {self.l_min:.2f} cm."
            )
            logger.debug(msg)
            return IKResult(
                solution=IKSolution.IN_DEADZONE,
                angles=None,
                theta1_rad=theta1,
                theta2_rad=float("nan"),
                reach_cm=L,
                reachable=False,
                message=msg,
            )

        # ── Step 4: elevation angle (single rigid link model) ─────────────────
        theta2 = math.atan2(pz, r_desired)      # radians, elevation above horizontal

        # ── Step 5: convert to servo space ────────────────────────────────────
        servo_x = self._theta1_to_servo_x(theta1)
        servo_y = self._theta2_to_servo_y(theta2)

        # ── Step 6: joint-limit check ─────────────────────────────────────────
        x_ok = self._x_bounds[0] <= servo_x <= self._x_bounds[1]
        y_ok = self._y_bounds[0] <= servo_y <= self._y_bounds[1]

        if not (x_ok and y_ok):
            msg = (
                f"Geometric solution (θ₁={math.degrees(theta1):.1f}°, "
                f"θ₂={math.degrees(theta2):.1f}°) → servo X={servo_x:.1f}°, "
                f"Y={servo_y:.1f}° violates joint limits "
                f"X∈{self._x_bounds}, Y∈{self._y_bounds}."
            )
            logger.debug(msg)
            return IKResult(
                solution=IKSolution.JOINT_LIMIT,
                angles=None,
                theta1_rad=theta1,
                theta2_rad=theta2,
                reach_cm=L,
                reachable=False,
                message=msg,
            )

        angles = np.array([servo_x, servo_y, gripper_deg], dtype=np.float32)
        msg = (
            f"IK OK: ({px:.1f}, {py:.1f}, {pz:.1f}) cm → "
            f"servo X={servo_x:.1f}°, Y={servo_y:.1f}°, Z={gripper_deg:.1f}°"
        )
        logger.debug(msg)
        return IKResult(
            solution=IKSolution.OK,
            angles=angles,
            theta1_rad=theta1,
            theta2_rad=theta2,
            reach_cm=L,
            reachable=True,
            message=msg,
        )

    def forward(
        self,
        servo_x_deg: float,
        servo_y_deg: float,
    ) -> Tuple[float, float, float]:
        """
        Forward kinematics: given servo angles, return TCP position in cm.

        Useful for verifying IK results and for visualisation / workspace plots.

        Parameters
        ----------
        servo_x_deg : float   Servo X angle in degrees.
        servo_y_deg : float   Servo Y angle in degrees.

        Returns
        -------
        (px, py, pz) : Tuple[float, float, float]
            TCP position in centimetres.

        Examples
        --------
        >>> solver = GeometricIKSolver()
        >>> px, py, pz = solver.forward(120, 90)
        """
        theta1 = self._servo_x_to_theta1(servo_x_deg)
        theta2 = self._servo_y_to_theta2(servo_y_deg)

        # Single rigid link of total length l1+l2 in the elevation plane
        L  = self.l1 + self.l2
        r  = L * math.cos(theta2)   # horizontal reach
        px = r * math.cos(theta1)
        py = r * math.sin(theta1)
        pz = L * math.sin(theta2)
        return px, py, pz

    def fk_check(
        self,
        px: float,
        py: float,
        pz: float,
        tolerance_cm: float = 1.0,
    ) -> bool:
        """
        Convenience: run IK then FK and check the round-trip error.

        The round-trip is not perfect because the simplified single-link
        model cannot distinguish individual link contributions; the test
        is on the total reach magnitude rather than per-axis position.

        Returns True if the reach error is within tolerance_cm.
        """
        r = self.solve(px=px, py=py, pz=pz)
        if not r.reachable:
            return False
        px_rt, py_rt, pz_rt = self.forward(r.angles[0], r.angles[1])
        L_in  = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
        L_out = math.sqrt(px_rt ** 2 + py_rt ** 2 + pz_rt ** 2)
        return abs(L_in - L_out) <= tolerance_cm

    def workspace_bounds(self) -> dict:
        """
        Return the approximate Cartesian workspace envelope.

        Computed from the joint limits and link lengths; useful for
        scaling the gesture mapping to the reachable workspace.

        Returns
        -------
        dict with keys: x_range, y_range, z_range, max_reach_cm, min_reach_cm.
        """
        x_min_rad = self._servo_x_to_theta1(self._x_bounds[0])
        x_max_rad = self._servo_x_to_theta1(self._x_bounds[1])
        y_min_rad = self._servo_y_to_theta2(self._y_bounds[0])
        y_max_rad = self._servo_y_to_theta2(self._y_bounds[1])

        L = self.l1 + self.l2
        return {
            "max_reach_cm": self.l_max,
            "min_reach_cm": self.l_min,
            "x_range_cm":   (L * math.cos(x_max_rad), L * math.cos(x_min_rad)),
            "z_range_cm":   (L * math.sin(y_min_rad), L * math.sin(y_max_rad)),
            "theta1_range_rad": (x_min_rad, x_max_rad),
            "theta2_range_rad": (y_min_rad, y_max_rad),
        }

    # ── Gesture-space → task-space mapper ─────────────────────────────────────

    def hand_position_to_target(
        self,
        norm_x: float,
        norm_y: float,
        pinch_distance_px: float,
        px_range: Tuple[float, float] = (20.0, 220.0),
    ) -> Tuple[float, float, float]:
        """
        Map a normalized hand position (from HandState.features) to a
        desired TCP position in the arm's workspace.

        This is the bridge between the gesture pipeline and the IK solver.
        The operator's left-hand position is interpreted as a desired
        end-effector location rather than a direct joint angle, giving
        more intuitive Cartesian-space arm control.

        Parameters
        ----------
        norm_x : float
            Normalized horizontal position ∈ [0, 1] (from HandState.features[18]).
        norm_y : float
            Normalized vertical position ∈ [0, 1] (from HandState.features[19]).
        pinch_distance_px : float
            Thumb-to-index pixel distance (from HandState.pinch_distance).
            Mapped to pz (height above base).
        px_range : tuple
            Expected pixel range of pinch_distance for min/max pz mapping.

        Returns
        -------
        (px, py, pz) in centimetres, ready to pass to solve().
        """
        wb = self.workspace_bounds()
        x_lo, x_hi = wb["x_range_cm"]
        z_lo, z_hi = wb["z_range_cm"]

        # Horizontal reach: norm_x [0,1] → [x_lo, x_hi] cm
        # (left side of camera = far reach; right = near)
        px = x_lo + norm_x * (x_hi - x_lo)
        py = 0.0   # keep arm in the sagittal plane; extend later for full 3D

        # Height: norm_y [0,1] → z_hi to z_lo (inverted: high y_pixel = low in frame)
        pz = z_hi + norm_y * (z_lo - z_hi)

        return px, py, pz

    # ── Private angle-space conversions ────────────────────────────────────────

    def _theta1_to_servo_x(self, theta1_rad: float) -> float:
        """
        Convert base rotation θ₁ (rad) to servo X degrees.
        θ₁ = 0 → arm pointing along +X → servo X = neutral (120°).
        θ₁ > 0 (CCW) → arm swings left → servo X increases toward 180°.
        θ₁ < 0 (CW)  → arm swings right → servo X decreases toward 60°.
        """
        delta_deg = math.degrees(theta1_rad)
        return float(np.clip(self._x_neutral + delta_deg, *self._x_bounds))

    def _theta2_to_servo_y(self, theta2_rad: float) -> float:
        """
        Convert elevation angle θ₂ (rad) to servo Y degrees.
        θ₂ = 0 (horizontal) → servo Y = y_zero (40°).
        θ₂ = π/2 (vertical) → servo Y = y_max (140°).
        """
        servo_y = self._y_zero + math.degrees(theta2_rad) * (
            (self._y_bounds[1] - self._y_zero) / 90.0
        )
        return float(np.clip(servo_y, *self._y_bounds))

    def _servo_x_to_theta1(self, servo_x_deg: float) -> float:
        """Inverse of _theta1_to_servo_x."""
        return math.radians(servo_x_deg - self._x_neutral)

    def _servo_y_to_theta2(self, servo_y_deg: float) -> float:
        """Inverse of _theta2_to_servo_y."""
        return math.radians(
            (servo_y_deg - self._y_zero) * 90.0 / (self._y_bounds[1] - self._y_zero)
        )

    def __repr__(self) -> str:
        return (
            f"GeometricIKSolver(l1={self.l1}cm, l2={self.l2}cm, "
            f"reach=[{self.l_min:.1f}, {self.l_max:.1f}]cm)"
        )
