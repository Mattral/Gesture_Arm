"""
gesture_arm.run
~~~~~~~~~~~~~~~~
Main real-time control loop.

Wires all subsystems together via the three-stage arm-control cascade:

  1. GeometricIKSolver  → Cartesian IK (optional, --ik flag)
  2. LSTMStabilizer     → temporal smoothing of servo commands
  3. BaselineMapper     → direct per-frame mapping (fallback / warm-up)

Other subsystems:
  - HandTracker      → per-frame landmark extraction + feature normalization
  - ArmController    → 3DoF servo arm (pins D3, D5, D6)
  - BaseController   → L298N mobile base (pins D7–D13)
  - ASRListener      → continuous speech recognition daemon thread
  - TTSEngine        → text-to-speech feedback daemon thread
  - MetricsLogger    → latency L + stability S → data/metrics_log.csv

Run modes:
    python -m gesture_arm.run                        # LSTM mode (default)
    python -m gesture_arm.run --ik                   # IK → LSTM → baseline cascade
    python -m gesture_arm.run --no-hardware          # demo / CI, no Arduino
    python -m gesture_arm.run --config my.yaml       # custom config file
    python -m gesture_arm.run --ik --no-hardware     # IK demo without hardware
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config.settings import AppConfig, load_config
from .evaluation.metrics import MetricsLogger
from .kinematics.ik_solver import GeometricIKSolver, IKSolution
from .models.stabilizer import BaselineMapper, LSTMStabilizer, TF_AVAILABLE, load_or_build
from .speech.multimodal import ASRListener, TTSEngine

# HandTracker imported lazily inside run() to avoid importing cvzone at module
# load time, which would break CI environments without the hardware stack.

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HUD drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_hud(
    frame: np.ndarray,
    angles: Optional[np.ndarray],
    method: str,
    S: Optional[float],
    L: Optional[float],
    half_w: int,
    gesture_label: str,
) -> None:
    h, w = frame.shape[:2]

    # Zone borders
    cv2.rectangle(frame, (0, 0),      (half_w, h), (255, 80, 80),  2)
    cv2.rectangle(frame, (half_w, 0), (w, h),      (80, 255, 80),  2)

    # Zone labels
    cv2.putText(frame, "RIGHT — mobile base",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "LEFT — arm control",
                (half_w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1, cv2.LINE_AA)

    # Gesture label
    if gesture_label:
        cv2.putText(frame, gesture_label,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 120), 2, cv2.LINE_AA)

    # Servo angle bars (bottom of right panel)
    if angles is not None:
        cfg_bounds = {"x": (60, 180), "y": (40, 140), "z": (100, 150)}
        labels = ["X", "Y", "Z"]
        for i, (axis, lbl) in enumerate(zip(["x", "y", "z"], labels)):
            lo, hi = cfg_bounds[axis]
            bar_x0 = half_w + 20
            bar_y  = h - 30 - i * 28
            bar_w  = w - half_w - 40
            fill   = int(np.interp(angles[i], [lo, hi], [0, bar_w]))
            cv2.rectangle(frame, (bar_x0, bar_y - 10), (bar_x0 + bar_w, bar_y + 8), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x0, bar_y - 10), (bar_x0 + fill,  bar_y + 8), (80, 220, 100), -1)
            cv2.putText(frame, f"{lbl}: {int(angles[i])}°",
                        (bar_x0 + bar_w + 6, bar_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Method badge
        color = (80, 200, 255) if method == "lstm" else \
                (255, 180,  60) if method == "ik"   else (160, 160, 160)
        cv2.putText(frame, f"[{method}]",
                    (half_w + 20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Metrics overlay (bottom-left)
    if S is not None:
        cv2.putText(frame, f"S={S:.2f}  L={L:.1f}ms" if L else f"S={S:.2f}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # IK mode indicator (top centre)
    if method == "ik":
        cv2.putText(frame, "IK MODE",
                    (w // 2 - 42, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 60), 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Main control loop
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: AppConfig, no_hardware: bool = False) -> None:
    """
    Start the real-time gesture control system.

    Args:
        cfg:          Loaded AppConfig (from config/default.yaml).
        no_hardware:  If True, skip Arduino connection (demo / CI mode).
    """
    # Lazy import — cvzone only needed at runtime
    from .vision.tracker import HandTracker

    # ── Hardware ───────────────────────────────────────────────────────────────
    arm = None
    base = None
    board = None

    if not no_hardware:
        from .hardware.arduino import ArmController, BaseController, board_session, connect
        board = connect(cfg.hardware.port, cfg.hardware.baudrate)
        arm   = ArmController(board, cfg.hardware.servos)
        base  = BaseController(board, cfg.hardware.motors)

    # ── Vision ─────────────────────────────────────────────────────────────────
    tracker = HandTracker(
        detection_confidence=cfg.vision.detection_confidence,
        max_hands=cfg.vision.max_hands,
        frame_width=cfg.vision.width,
        frame_height=cfg.vision.height,
    )

    cap = cv2.VideoCapture(cfg.vision.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.vision.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.vision.height)

    # ── LSTM model ─────────────────────────────────────────────────────────────
    servo_bounds = {
        k: (v.min_deg, v.max_deg)
        for k, v in cfg.hardware.servos.items()
    }

    stabilizer: Optional[LSTMStabilizer] = None
    if TF_AVAILABLE:
        model      = load_or_build(cfg.model.model_path)
        stabilizer = LSTMStabilizer(model, servo_bounds, cfg.model.sequence_length)

    baseline = BaselineMapper(servo_bounds, cfg.vision.width, cfg.vision.height)

    # ── IK solver (optional Cartesian mode) ───────────────────────────────────
    ik_solver: Optional[GeometricIKSolver] = None
    if cfg.kinematics.enabled:
        ik_cfg = cfg.kinematics
        ik_solver = GeometricIKSolver(
            link1_cm=ik_cfg.link1_cm,
            link2_cm=ik_cfg.link2_cm,
            servo_x_bounds=(
                cfg.hardware.servos["x"].min_deg,
                cfg.hardware.servos["x"].max_deg,
            ),
            servo_y_bounds=(
                cfg.hardware.servos["y"].min_deg,
                cfg.hardware.servos["y"].max_deg,
            ),
            servo_x_neutral_deg=ik_cfg.servo_x_neutral_deg,
            servo_y_zero_deg=ik_cfg.servo_y_zero_deg,
        )
        wb = ik_solver.workspace_bounds()
        logger.info(
            "IK mode enabled — workspace reach [%.1f, %.1f] cm",
            wb["min_reach_cm"], wb["max_reach_cm"],
        )

    # ── TTS + ASR ──────────────────────────────────────────────────────────────
    tts = TTSEngine(rate=cfg.speech.tts_rate, volume=cfg.speech.tts_volume)
    tts.start()

    last_motor_time = [time.time()]

    def handle_speech_command(cmd: str) -> None:
        args = cfg.speech.commands.get(cmd)
        if args is None:
            if base:
                base.stop()
            tts.say("Stopping")
        elif base:
            ld1, ld2, lspd, rd1, rd2, rspd = args
            base._drive(int(ld1), int(ld2), lspd, int(rd1), int(rd2), rspd)
            last_motor_time[0] = time.time()

    asr = ASRListener(
        commands=set(cfg.speech.commands.keys()),
        on_command=handle_speech_command,
        tts=tts,
    )
    asr.start()

    # ── Metrics logger ─────────────────────────────────────────────────────────
    mlog = MetricsLogger(
        cfg.evaluation.log_path,
        stability_window=cfg.evaluation.stability_window,
        latency_window=cfg.evaluation.latency_window,
    )

    tts.say("System ready")
    logger.info("Main loop started. Press Q to quit.")

    half_w = cfg.vision.width // 2
    half_h = cfg.vision.height // 2

    current_angles = np.array([120.0, 90.0, 125.0], dtype=np.float32)
    gesture_label  = ""

    try:
        while True:
            t_capture = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera read failed — is camera_index correct?")
                break

            output = tracker.process(frame)

            # ── RIGHT HAND → mobile base ─────────────────────────────────────
            if output.right is not None:
                rh = output.right
                x, y = rh.palm_center

                if rh.is_fist:
                    if base:
                        base.stop()
                    gesture_label = "STOP (fist)"
                    last_motor_time[0] = time.time()

                elif x < half_w:
                    cx, cy = half_w / 2, half_h / 2
                    if y < cy * 0.75:
                        if base: base.forward()
                        gesture_label = "Forward"
                    elif y > cy * 1.25:
                        if base: base.reverse()
                        gesture_label = "Reverse"
                    elif x < cx * 0.75:
                        if base: base.turn_left()
                        gesture_label = "Left"
                    elif x > cx * 1.25:
                        if base: base.turn_right()
                        gesture_label = "Right"
                    else:
                        if base: base.stop()
                        gesture_label = "Idle"
                    last_motor_time[0] = time.time()
                else:
                    if base: base.stop()

            # Auto-stop motors after timeout
            if time.time() - last_motor_time[0] >= cfg.hardware.motors.stop_timeout_sec:
                if base:
                    base.stop()

            # ── LEFT HAND → 3DoF arm (IK → LSTM → baseline cascade) ─────────
            method = "no hand"
            if output.left is not None:
                lh = output.left

                if ik_solver is not None:
                    # ── IK path: hand position → desired TCP → joint angles ──
                    # Extract normalized palm position from the feature vector.
                    # Features layout: [x0/W, y0/H, x1/W, y1/H, ..., x20/W, y20/H]
                    # Landmark 9 (palm centre) is at indices 18, 19.
                    norm_x = float(lh.features[18])   # x9 / W
                    norm_y = float(lh.features[19])   # y9 / H

                    px, py, pz = ik_solver.hand_position_to_target(
                        norm_x=norm_x,
                        norm_y=norm_y,
                        pinch_distance_px=lh.pinch_distance,
                    )
                    ik_result = ik_solver.solve(
                        px=px, py=py, pz=pz,
                        gripper_deg=float(current_angles[2]),  # preserve current grip
                    )

                    if ik_result.reachable:
                        angles = ik_result.angles
                        method = "ik"
                    else:
                        # Target outside workspace — fall through to LSTM / baseline
                        logger.debug("IK fallback: %s", ik_result.message)
                        angles = None
                        method = "ik_fallback"

                    if angles is None:
                        # IK failed or disabled — use LSTM / baseline
                        if stabilizer is not None:
                            angles, method = stabilizer.update(lh.features)
                            if angles is None:
                                angles = baseline.map(lh.landmarks)
                                method = "baseline (warming)"
                        else:
                            angles = baseline.map(lh.landmarks)
                            method = "baseline"

                elif stabilizer is not None:
                    angles, method = stabilizer.update(lh.features)
                    if angles is None:
                        angles = baseline.map(lh.landmarks)
                        method = "baseline (warming)"
                else:
                    angles = baseline.map(lh.landmarks)
                    method = "baseline"

                current_angles = angles
                if arm:
                    arm.write(angles)

                mlog.log(angles, t_capture=t_capture, method=method)
            else:
                if stabilizer:
                    stabilizer.reset()

            # ── HUD ───────────────────────────────────────────────────────────
            _draw_hud(
                output.frame,
                current_angles,
                method,
                mlog.stability(),
                mlog.avg_latency(),
                half_w,
                gesture_label,
            )

            cv2.imshow("Gesture Arm Control", output.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if base:
            base.stop()
        if board:
            board.exit()
        tts.say("System shutdown")
        mlog.print_summary()


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture Arm real-time control")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--no-hardware", action="store_true",
                        help="Run without Arduino (demo / CI mode)")
    parser.add_argument("--ik", action="store_true",
                        help="Enable IK Cartesian mode (overrides config kinematics.enabled)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config) if args.config else load_config()

    # CLI flag overrides config file
    if args.ik:
        from dataclasses import replace
        cfg = replace(cfg, kinematics=replace(cfg.kinematics, enabled=True))

    run(cfg, no_hardware=args.no_hardware)


if __name__ == "__main__":
    main()
