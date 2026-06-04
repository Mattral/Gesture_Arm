"""
scripts/collect.py
~~~~~~~~~~~~~~~~~~
Collect hand-landmark training data for the LSTM model.

Usage:
    python scripts/collect.py
    python scripts/collect.py --duration 120 --out data/training_data.csv

Controls during collection:
    Q  — quit and save
    S  — save immediately and continue
    R  — reset / discard current samples

Output CSV columns:
    feat_00 … feat_41  (normalized x,y for all 21 landmarks)
    label_x, label_y, label_z  (normalized servo angles [0,1])
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from gesture_arm.config.settings import load_config
from gesture_arm.models.stabilizer import BaselineMapper
from gesture_arm.vision.tracker import HandTracker


def collect(duration: int, out_path: str, config_path: str | None) -> None:
    cfg      = load_config(config_path) if config_path else load_config()
    tracker  = HandTracker(
        detection_confidence=cfg.vision.detection_confidence,
        max_hands=1,
        frame_width=cfg.vision.width,
        frame_height=cfg.vision.height,
    )
    servo_bounds = {k: (v.min_deg, v.max_deg) for k, v in cfg.hardware.servos.items()}
    mapper   = BaselineMapper(servo_bounds, cfg.vision.width, cfg.vision.height)

    cap = cv2.VideoCapture(cfg.vision.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.vision.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.vision.height)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build CSV header
    feat_cols  = [f"feat_{i:02d}" for i in range(cfg.model.feature_dim)]
    label_cols = ["label_x", "label_y", "label_z"]

    samples: list = []
    t_start = time.time()

    print(f"\n{'='*55}")
    print("  DATA COLLECTION MODE")
    print(f"  Duration: {duration}s   Output: {out_path}")
    print("  Move your LEFT hand slowly across the full range.")
    print("  Q=quit+save  S=save now  R=reset samples")
    print(f"{'='*55}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output  = tracker.process(frame)
        elapsed = time.time() - t_start

        if output.left is not None:
            lh      = output.left
            angles  = mapper.map(lh.landmarks)

            # Normalize angles to [0,1] for training labels
            norm_labels = np.array([
                (angles[i] - servo_bounds[ax][0]) / (servo_bounds[ax][1] - servo_bounds[ax][0])
                for i, ax in enumerate(["x", "y", "z"])
            ])
            samples.append(np.concatenate([lh.features, norm_labels]))

        # HUD
        cv2.putText(frame,
                    f"Samples: {len(samples):5d}   Time: {elapsed:.0f}/{duration}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    "Move LEFT hand across full range",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Progress bar
        pct  = min(elapsed / duration, 1.0)
        bw   = cfg.vision.width - 40
        cv2.rectangle(frame, (20, 100), (20 + bw, 120), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 100), (20 + int(bw * pct), 120), (0, 200, 100), -1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or elapsed >= duration:
            break
        if key == ord("r"):
            samples.clear()
            print(f"\r[collect] Samples reset.")
        if key == ord("s"):
            _save(samples, out, feat_cols, label_cols)
            print(f"\r[collect] Saved {len(samples)} samples (continuing).")

    cap.release()
    cv2.destroyAllWindows()
    _save(samples, out, feat_cols, label_cols)


def _save(samples, path, feat_cols, label_cols):
    if not samples:
        print("[collect] No samples to save.")
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(feat_cols + label_cols)
        for row in samples:
            w.writerow([round(float(v), 6) for v in row])
    print(f"[collect] ✓ Saved {len(samples)} samples → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=90,
                        help="Recording duration in seconds (default: 90)")
    parser.add_argument("--out", default="data/training_data.csv",
                        help="Output CSV path")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    collect(args.duration, args.out, args.config)
