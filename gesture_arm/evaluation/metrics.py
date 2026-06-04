"""
gesture_arm.evaluation.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation metrics logger.

Implements the two primary metrics from paper Section VI:

  Stability variance:
      S = (1/T) Σ_{t=1}^{T} (u_t − ū)²
      where u_t is the servo command at frame t and ū is the rolling mean.
      Lower S = smoother, more stable motion.

  End-to-end latency:
      L = t_actuation − t_capture   (milliseconds)
      where t_capture is when the frame was read and
      t_actuation is when write() was called on the servo.

All data is streamed to CSV for offline analysis in the benchmark notebook.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Collects, stores, and reports gesture control metrics in real time.

    Usage::

        logger = MetricsLogger("data/metrics_log.csv")
        t0 = time.time()
        arm.write(angles)
        logger.log(angles, t_capture=t0, method="lstm")

        print("Stability S =", logger.stability())
        print("Latency  L =", logger.avg_latency(), "ms")
    """

    # CSV header
    _COLUMNS = ["timestamp", "servo_x", "servo_y", "servo_z", "latency_ms", "method"]

    def __init__(
        self,
        log_path: str | Path = "data/metrics_log.csv",
        stability_window: int = 100,
        latency_window: int = 100,
    ) -> None:
        self._path             = Path(log_path)
        self._stability_window = stability_window
        self._latency_window   = latency_window

        self._servo_history:   List[np.ndarray] = []
        self._latency_history: List[float]      = []

        self._init_csv()
        logger.info("MetricsLogger writing to %s", self._path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def log(
        self,
        angles: np.ndarray,
        t_capture: float,
        method: str = "lstm",
    ) -> None:
        """
        Record one servo command.

        Args:
            angles:    (3,) array [servoX_deg, servoY_deg, servoZ_deg].
            t_capture: time.time() value when the video frame was captured.
            method:    "lstm" or "baseline" — which path produced this command.
        """
        t_actuation = time.time()
        latency     = (t_actuation - t_capture) * 1000.0   # → ms

        self._servo_history.append(angles.copy())
        self._latency_history.append(latency)

        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow([
                round(t_capture, 6),
                round(float(angles[0]), 3),
                round(float(angles[1]), 3),
                round(float(angles[2]), 3),
                round(latency, 3),
                method,
            ])

    def stability(self, window: Optional[int] = None) -> Optional[float]:
        """
        Compute rolling stability variance S over the last `window` frames.

        S = (1/T) Σ (u_t − ū)²

        Returns None if fewer than 2 frames have been logged.
        """
        w      = window or self._stability_window
        recent = np.array(self._servo_history[-w:])
        if len(recent) < 2:
            return None
        mean = np.mean(recent, axis=0)
        S    = float(np.mean(np.sum((recent - mean) ** 2, axis=1)))
        return S

    def avg_latency(self, window: Optional[int] = None) -> Optional[float]:
        """
        Compute rolling average end-to-end latency L (ms).

        Returns None if no frames logged yet.
        """
        w      = window or self._latency_window
        recent = self._latency_history[-w:]
        if not recent:
            return None
        return float(np.mean(recent))

    def summary(self) -> dict:
        """Return a summary dict of all metrics."""
        return {
            "n_frames":   len(self._servo_history),
            "stability_S": self.stability(),
            "avg_latency_ms": self.avg_latency(),
            "min_latency_ms": float(np.min(self._latency_history)) if self._latency_history else None,
            "max_latency_ms": float(np.max(self._latency_history)) if self._latency_history else None,
            "log_path":   str(self._path),
        }

    def print_summary(self) -> None:
        """Print a formatted metrics summary to stdout."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("  EVALUATION METRICS  (paper Section VI)")
        print("=" * 50)
        print(f"  Frames logged   : {s['n_frames']}")
        if s['stability_S'] is not None:
            print(f"  Stability S     : {s['stability_S']:.4f}  (lower = smoother)")
        if s['avg_latency_ms'] is not None:
            print(f"  Avg latency L   : {s['avg_latency_ms']:.2f} ms")
            print(f"  Min / Max L     : {s['min_latency_ms']:.2f} / {s['max_latency_ms']:.2f} ms")
        print(f"  Log saved to    : {s['log_path']}")
        print("=" * 50 + "\n")

    # ── Private ────────────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", newline="") as f:
            csv.writer(f).writerow(self._COLUMNS)
