"""
tests/test_core.py
~~~~~~~~~~~~~~~~~~~
Unit tests for core modules — runs without hardware or TensorFlow.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import math
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
# Kinematics — GeometricIKSolver
# ══════════════════════════════════════════════════════════════════════════════

class TestGeometricIKSolver:
    """
    All tests use the default arm: l1=10 cm, l2=8 cm, max_reach=18 cm.
    Servo X ∈ [60°, 180°], neutral=120° (arm forward).
    Servo Y ∈ [40°, 140°], zero=40° (arm horizontal).
    """

    @pytest.fixture
    def solver(self):
        from gesture_arm.kinematics.ik_solver import GeometricIKSolver
        return GeometricIKSolver(link1_cm=10.0, link2_cm=8.0)

    # ── Constructor ───────────────────────────────────────────────────────────

    def test_repr(self, solver):
        r = repr(solver)
        assert "10.0" in r and "8.0" in r

    def test_invalid_links_raises(self):
        from gesture_arm.kinematics.ik_solver import GeometricIKSolver
        with pytest.raises(ValueError):
            GeometricIKSolver(link1_cm=-1.0, link2_cm=8.0)
        with pytest.raises(ValueError):
            GeometricIKSolver(link1_cm=10.0, link2_cm=0.0)

    # ── Reachability ──────────────────────────────────────────────────────────

    def test_reachable_point_returns_ok(self, solver):
        from gesture_arm.kinematics.ik_solver import IKSolution
        r = solver.solve(px=12.0, py=0.0, pz=0.0)
        assert r.solution == IKSolution.OK
        assert r.reachable is True
        assert r.angles is not None
        assert r.angles.shape == (3,)

    def test_target_beyond_max_reach_is_unreachable(self, solver):
        from gesture_arm.kinematics.ik_solver import IKSolution
        # max_reach = 18 cm; target at 25 cm
        r = solver.solve(px=25.0, py=0.0, pz=0.0)
        assert r.solution == IKSolution.UNREACHABLE
        assert r.reachable is False
        assert r.angles is None

    def test_target_just_at_max_reach_is_ok(self, solver):
        from gesture_arm.kinematics.ik_solver import IKSolution
        # Exactly on the sphere of max reach (with small epsilon tolerance)
        r = solver.solve(px=18.0 - 0.01, py=0.0, pz=0.0)
        assert r.solution == IKSolution.OK

    def test_origin_is_deadzone(self, solver):
        from gesture_arm.kinematics.ik_solver import IKSolution
        # l1=10, l2=8 → min_reach = |10-8| = 2 cm; origin is dead zone
        r = solver.solve(px=0.0, py=0.0, pz=0.0)
        assert r.solution == IKSolution.IN_DEADZONE
        assert r.reachable is False

    # ── Output bounds ─────────────────────────────────────────────────────────

    def test_servo_angles_within_bounds_for_valid_target(self, solver):
        r = solver.solve(px=10.0, py=0.0, pz=5.0)
        assert r.reachable
        assert 60.0  <= r.angles[0] <= 180.0, f"Servo X={r.angles[0]:.1f} out of [60,180]"
        assert 40.0  <= r.angles[1] <= 140.0, f"Servo Y={r.angles[1]:.1f} out of [40,140]"

    def test_gripper_passes_through_unchanged(self, solver):
        gripper = 135.0
        r = solver.solve(px=12.0, py=0.0, pz=2.0, gripper_deg=gripper)
        assert r.reachable
        assert pytest.approx(r.angles[2], abs=0.01) == gripper

    # ── Angle mapping ─────────────────────────────────────────────────────────

    def test_forward_pointing_target_gives_neutral_servo_x(self, solver):
        """Target directly in front (py=0) → servo X should be at neutral (120°)."""
        r = solver.solve(px=14.0, py=0.0, pz=0.0)
        assert r.reachable
        assert pytest.approx(r.angles[0], abs=1.0) == 120.0

    def test_elevated_target_increases_servo_y(self, solver):
        """Raising the target height should increase servo Y angle."""
        r_low  = solver.solve(px=14.0, py=0.0, pz=0.0)
        r_high = solver.solve(px=10.0, py=0.0, pz=6.0)
        assert r_low.reachable and r_high.reachable
        assert r_high.angles[1] > r_low.angles[1], \
            "Higher target should produce higher servo Y angle"

    def test_left_target_increases_servo_x(self, solver):
        """Target to the left (py > 0) should increase servo X above neutral."""
        r = solver.solve(px=10.0, py=6.0, pz=0.0)
        assert r.reachable
        assert r.angles[0] > 120.0, "Left target should increase servo X"

    def test_right_target_decreases_servo_x(self, solver):
        """Target to the right (py < 0) should decrease servo X below neutral."""
        r = solver.solve(px=10.0, py=-6.0, pz=0.0)
        assert r.reachable
        assert r.angles[0] < 120.0, "Right target should decrease servo X"

    # ── Forward kinematics consistency ────────────────────────────────────────

    def test_fk_forward_pointing_gives_positive_x(self, solver):
        """At neutral servo X (120°), FK should give positive px and py≈0."""
        px, py, pz = solver.forward(servo_x_deg=120.0, servo_y_deg=40.0)
        assert px > 0.0, f"Forward-pointing arm should have px > 0, got {px:.2f}"
        assert abs(py) < 0.5, f"py should be ~0 at neutral X, got {py:.2f}"

    def test_fk_elevated_gives_positive_pz(self, solver):
        """Higher servo Y should produce higher pz."""
        _, _, pz_low  = solver.forward(servo_x_deg=120.0, servo_y_deg=40.0)
        _, _, pz_high = solver.forward(servo_x_deg=120.0, servo_y_deg=100.0)
        assert pz_high > pz_low, "Higher servo Y should give higher pz"

    def test_fk_left_rotation_gives_positive_py(self, solver):
        """Left rotation (servo X > neutral) should give positive py."""
        _, py, _ = solver.forward(servo_x_deg=160.0, servo_y_deg=40.0)
        assert py > 0.0, f"Left rotation should give positive py, got {py:.2f}"

    def test_ik_direction_consistency(self, solver):
        """IK-derived θ₁ should point toward the target's (px, py) direction."""
        r = solver.solve(px=10.0, py=5.0, pz=0.0)
        assert r.reachable
        # The base angle from IK should match atan2(py, px)
        expected_theta1 = math.atan2(5.0, 10.0)
        assert pytest.approx(r.theta1_rad, abs=0.05) == expected_theta1

    # ── Workspace bounds ──────────────────────────────────────────────────────

    def test_workspace_bounds_structure(self, solver):
        wb = solver.workspace_bounds()
        assert "max_reach_cm" in wb
        assert "min_reach_cm" in wb
        assert pytest.approx(wb["max_reach_cm"]) == 18.0
        assert pytest.approx(wb["min_reach_cm"]) == 2.0

    # ── hand_position_to_target ────────────────────────────────────────────────

    def test_hand_to_target_output_range(self, solver):
        """Any normalized hand position in [0,1]² should map to a finite target."""
        for nx, ny in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.2, 0.8)]:
            px, py, pz = solver.hand_position_to_target(nx, ny, 100.0)
            assert math.isfinite(px)
            assert math.isfinite(py)
            assert math.isfinite(pz)

    # ── Config integration ────────────────────────────────────────────────────

    def test_ik_config_loads(self):
        """IKConfig should load from default.yaml without error."""
        from gesture_arm.config.settings import load_config
        cfg = load_config()
        assert hasattr(cfg, "kinematics")
        assert cfg.kinematics.link1_cm == 10.0
        assert cfg.kinematics.link2_cm == 8.0
        assert cfg.kinematics.enabled is False  # off by default

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
