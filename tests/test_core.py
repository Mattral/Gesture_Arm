"""
tests/test_core.py
~~~~~~~~~~~~~~~~~~~
Unit tests for core modules — runs without hardware or TensorFlow.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_load_default_config(self):
        from gesture_arm.config.settings import load_config
        cfg = load_config()
        assert cfg.hardware.port is not None
        assert cfg.vision.width == 1280
        assert cfg.vision.height == 720
        assert cfg.model.sequence_length == 15
        assert cfg.model.feature_dim == 42

    def test_servo_bounds_present(self):
        from gesture_arm.config.settings import load_config
        cfg = load_config()
        for axis in ["x", "y", "z"]:
            assert axis in cfg.hardware.servos
            s = cfg.hardware.servos[axis]
            assert s.min_deg < s.max_deg


# ══════════════════════════════════════════════════════════════════════════════
# Vision — feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_hand(hand_type: str = "Left") -> dict:
    """Create a minimal fake cvzone hand dict for testing."""
    lm = [[int(i * 10), int(i * 5), 0] for i in range(21)]
    return {"type": hand_type, "lmList": lm}


class TestFeatureExtraction:
    def test_normalize_shape(self):
        """Normalized feature vector must be (42,)."""
        from gesture_arm.vision.tracker import HandTracker
        tracker = HandTracker.__new__(HandTracker)
        tracker._fw = 1280
        tracker._fh = 720

        lm = np.array([[i * 10, i * 5, 0] for i in range(21)], dtype=np.float32)
        feat = tracker._normalize(lm)
        assert feat.shape == (42,)

    def test_normalize_range(self):
        """All feature values must be in [0, 1] for valid pixel coordinates."""
        from gesture_arm.vision.tracker import HandTracker
        tracker = HandTracker.__new__(HandTracker)
        tracker._fw = 1280
        tracker._fh = 720

        lm = np.array([[100, 200, 0] for _ in range(21)], dtype=np.float32)
        feat = tracker._normalize(lm)
        assert np.all(feat >= 0.0)
        assert np.all(feat <= 1.0)

    def test_normalize_boundary_values(self):
        """Landmark at (W, H) should normalize to (1.0, 1.0)."""
        from gesture_arm.vision.tracker import HandTracker
        tracker = HandTracker.__new__(HandTracker)
        tracker._fw = 1280
        tracker._fh = 720

        lm = np.zeros((21, 3), dtype=np.float32)
        lm[0] = [1280, 720, 0]
        feat = tracker._normalize(lm)
        assert pytest.approx(feat[0]) == 1.0   # x normalized
        assert pytest.approx(feat[1]) == 1.0   # y normalized


# ══════════════════════════════════════════════════════════════════════════════
# Models — BaselineMapper
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineMapper:
    @pytest.fixture
    def mapper(self):
        from gesture_arm.models.stabilizer import BaselineMapper
        bounds = {"x": (60, 180), "y": (40, 140), "z": (100, 150)}
        return BaselineMapper(bounds, frame_width=1280, frame_height=720)

    def test_output_shape(self, mapper):
        lm = np.array([[i * 10, i * 5, 0] for i in range(21)], dtype=np.float32)
        angles = mapper.map(lm)
        assert angles.shape == (3,)

    def test_output_within_bounds(self, mapper):
        lm = np.array([[500, 300, 0] for _ in range(21)], dtype=np.float32)
        angles = mapper.map(lm)
        assert 60 <= angles[0] <= 180, f"servoX={angles[0]} out of [60,180]"
        assert 40 <= angles[1] <= 140, f"servoY={angles[1]} out of [40,140]"
        assert 100 <= angles[2] <= 150, f"servoZ={angles[2]} out of [100,150]"

    def test_pinch_distance_affects_z(self, mapper):
        """Wider pinch → different Z angle."""
        lm_tight = np.array([[100, 100, 0] for _ in range(21)], dtype=np.float32)
        lm_wide  = np.array([[100, 100, 0] for _ in range(21)], dtype=np.float32)
        lm_tight[4]  = [100, 100, 0]   # thumb tip
        lm_tight[8]  = [110, 100, 0]   # index tip (10px apart)
        lm_wide[4]   = [100, 100, 0]
        lm_wide[8]   = [220, 100, 0]   # index tip (120px apart)

        z_tight = mapper.map(lm_tight)[2]
        z_wide  = mapper.map(lm_wide)[2]
        assert z_tight != z_wide


# ══════════════════════════════════════════════════════════════════════════════
# Models — LSTMStabilizer buffer logic (no TF needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestLSTMStabilizerBuffer:
    def test_returns_none_while_warming(self):
        """Should return None until buffer reaches sequence_length."""
        try:
            from gesture_arm.models.stabilizer import LSTMStabilizer
            import tensorflow as tf
            from gesture_arm.models.stabilizer import build_model
        except ImportError:
            pytest.skip("TensorFlow not installed")

        model = build_model()
        bounds = {"x": (60, 180), "y": (40, 140), "z": (100, 150)}
        stab = LSTMStabilizer(model, bounds, sequence_length=15)

        for i in range(14):
            feat = np.random.rand(42).astype(np.float32)
            angles, method = stab.update(feat)
            assert angles is None, f"Expected None at frame {i}"
            assert method == "baseline (warming up)"

    def test_returns_array_when_full(self):
        """After sequence_length frames, should return (3,) angle array."""
        try:
            from gesture_arm.models.stabilizer import LSTMStabilizer, build_model
        except ImportError:
            pytest.skip("TensorFlow not installed")

        model = build_model()
        bounds = {"x": (60, 180), "y": (40, 140), "z": (100, 150)}
        stab = LSTMStabilizer(model, bounds, sequence_length=15)

        for _ in range(15):
            angles, _ = stab.update(np.random.rand(42).astype(np.float32))

        assert angles is not None
        assert angles.shape == (3,)

    def test_reset_clears_buffer(self):
        """After reset(), buffer should be empty and is_warmed_up=False."""
        try:
            from gesture_arm.models.stabilizer import LSTMStabilizer, build_model
        except ImportError:
            pytest.skip("TensorFlow not installed")

        model = build_model()
        bounds = {"x": (60, 180), "y": (40, 140), "z": (100, 150)}
        stab = LSTMStabilizer(model, bounds, sequence_length=5)

        for _ in range(5):
            stab.update(np.random.rand(42).astype(np.float32))

        assert stab.is_warmed_up
        stab.reset()
        assert not stab.is_warmed_up
        assert len(stab._buffer) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation — MetricsLogger
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsLogger:
    @pytest.fixture(tmp_path=Path("/tmp"))
    def logger(self, tmp_path):
        from gesture_arm.evaluation.metrics import MetricsLogger
        return MetricsLogger(log_path=tmp_path / "test_metrics.csv")

    def test_stability_none_when_empty(self, logger):
        assert logger.stability() is None

    def test_stability_zero_for_constant_signal(self, logger):
        """Constant servo position → S = 0 (perfect stability)."""
        import time
        angles = np.array([120.0, 90.0, 125.0])
        t0 = time.time()
        for _ in range(20):
            logger.log(angles, t_capture=t0)
        s = logger.stability()
        assert s is not None
        assert pytest.approx(s, abs=1e-4) == 0.0

    def test_stability_higher_for_noisy_signal(self, logger):
        """Noisy signal → S > 0."""
        import time
        t0 = time.time()
        for _ in range(50):
            angles = np.array([
                np.random.uniform(60, 180),
                np.random.uniform(40, 140),
                np.random.uniform(100, 150),
            ])
            logger.log(angles, t_capture=t0)
        assert logger.stability() > 0.0

    def test_latency_is_positive(self, logger):
        import time
        t0 = time.time()
        logger.log(np.array([120.0, 90.0, 125.0]), t_capture=t0)
        assert logger.avg_latency() >= 0.0

    def test_csv_written(self, tmp_path):
        from gesture_arm.evaluation.metrics import MetricsLogger
        import time
        path = tmp_path / "out.csv"
        ml = MetricsLogger(log_path=path)
        ml.log(np.array([120.0, 90.0, 125.0]), t_capture=time.time())
        lines = path.read_text().splitlines()
        assert len(lines) == 2   # header + 1 row
        assert "servo_x" in lines[0]
