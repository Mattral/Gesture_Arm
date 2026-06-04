"""
gesture_arm.models.stabilizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LSTM temporal stabilization model (core contribution).

Contains:
  - LSTMStabilizer  : sliding-window LSTM → smoothed servo angles
  - BaselineMapper  : direct linear mapping (comparison baseline)
  - build_model()   : Keras model factory
  - train()         : training pipeline from CSV data

Paper reference: Section III-C / IV-B
  h_t   = LSTM(X_t)          X_t = [x_{t-k}, …, x_t]
  û_t   = W · h_t + b        stabilized servo command
  S     = Var(u_t)            stability metric (lower = better)
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional TensorFlow — gracefully falls back to baseline mode
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
    logger.info("TensorFlow %s detected — LSTM mode available.", tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    logger.warning(
        "TensorFlow not found. Install with: pip install tensorflow\n"
        "Running in baseline (frame-by-frame mapping) mode."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    sequence_length: int = 15,
    feature_dim: int = 42,
    output_dim: int = 3,
    lstm_units: int = 64,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> "tf.keras.Model":
    """
    Build the LSTM stabilization model.

    Architecture (Section III-C):
        Input(seq_len, feat_dim)
        → LSTM(64)
        → Dropout(0.2)
        → Dense(32, relu)
        → Dense(output_dim, sigmoid)    ← normalized [0, 1] servo angles

    Args:
        sequence_length: Number of frames in the sliding window (k in paper).
        feature_dim:     Length of each feature vector (42 for 21 landmarks × xy).
        output_dim:      Number of servo outputs (3 for X, Y, Z).
        lstm_units:      Hidden units in the LSTM layer.
        dense_units:     Units in the intermediate Dense layer.
        learning_rate:   Adam optimizer learning rate.

    Returns:
        Compiled Keras Sequential model.

    Raises:
        RuntimeError: if TensorFlow is not installed.
    """
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is required to build the LSTM model.\n"
            "Install with: pip install tensorflow"
        )

    model = Sequential(
        [
            LSTM(lstm_units, input_shape=(sequence_length, feature_dim)),
            Dropout(0.2),
            Dense(dense_units, activation="relu"),
            Dense(output_dim, activation="sigmoid"),
        ],
        name="lstm_gesture_stabilizer",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    logger.info(
        "Built LSTM model: seq=%d, feat=%d, out=%d, units=%d",
        sequence_length, feature_dim, output_dim, lstm_units,
    )
    return model


def load_or_build(
    model_path: str | Path,
    **build_kwargs,
) -> "tf.keras.Model":
    """Load a saved model, or build a new untrained one if not found."""
    path = Path(model_path)
    if path.exists():
        model = load_model(str(path))
        logger.info("Loaded trained model from %s", path)
        return model

    logger.warning(
        "No model found at %s — building untrained model.\n"
        "Run:  python -m gesture_arm.scripts.train",
        path,
    )
    return build_model(**build_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train(
    data_path: str | Path,
    model_path: str | Path,
    sequence_length: int = 15,
    feature_dim: int = 42,
    output_dim: int = 3,
    epochs: int = 80,
    batch_size: int = 16,
    validation_split: float = 0.15,
    **build_kwargs,
) -> None:
    """
    Train the LSTM model on collected landmark data.

    The CSV produced by the collect script has columns:
        [feat_0 … feat_41, label_0, label_1, label_2]
    where features are normalized landmarks and labels are normalized
    servo angles in [0, 1].

    Sliding window sequences are built automatically from the flat CSV.

    Args:
        data_path:       Path to training_data.csv.
        model_path:      Where to save the trained .h5 file.
        sequence_length: Window size k.
        epochs:          Training epochs.
        batch_size:      Mini-batch size.
        validation_split: Fraction held out for validation.

    Raises:
        FileNotFoundError: if data_path does not exist.
        RuntimeError:      if TensorFlow is not installed.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required. pip install tensorflow")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Run:  python scripts/collect.py  to collect data first."
        )

    data   = np.loadtxt(data_path, delimiter=",", skiprows=1).astype(np.float32)
    X_all  = data[:, :feature_dim]
    y_all  = data[:, feature_dim : feature_dim + output_dim]

    # Build sliding-window sequences
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X_all)):
        X_seq.append(X_all[i - sequence_length : i])
        y_seq.append(y_all[i])

    X_seq = np.array(X_seq)   # (N, seq_len, feat_dim)
    y_seq = np.array(y_seq)   # (N, output_dim)

    logger.info(
        "Training on %d sequences, shape %s → labels %s",
        len(X_seq), X_seq.shape, y_seq.shape,
    )

    model = build_model(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        output_dim=output_dim,
        **build_kwargs,
    )

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_path), save_best_only=True, verbose=1),
    ]

    model.fit(
        X_seq, y_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Model saved to %s", model_path)


# ══════════════════════════════════════════════════════════════════════════════
# Runtime stabilizer
# ══════════════════════════════════════════════════════════════════════════════

class LSTMStabilizer:
    """
    Real-time LSTM temporal stabilizer.

    Maintains a sliding window deque of feature vectors. Once the buffer
    reaches sequence_length frames, every new frame produces a stabilized
    servo command via the LSTM. While the buffer is filling, it delegates
    to BaselineMapper.

    Paper Section III-C / IV-B:
        X_t   = [x_{t-k}, …, x_t]
        h_t   = LSTM(X_t)
        û_t   = W · h_t + b    (sigmoid output, then denormalized)

    Usage::

        stabilizer = LSTMStabilizer(model, servo_bounds, cfg.model.sequence_length)
        angles = stabilizer.update(hand_state.features)
    """

    def __init__(
        self,
        model: "tf.keras.Model",
        servo_bounds: dict,
        sequence_length: int = 15,
    ) -> None:
        self._model          = model
        self._bounds         = servo_bounds     # {"x": (min, max), "y": …, "z": …}
        self._seq_len        = sequence_length
        self._buffer: deque  = deque(maxlen=sequence_length)
        self.is_warmed_up    = False

    def update(self, feature_vector: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Push one feature vector and return a (angles, method) tuple.

        Args:
            feature_vector: (42,) normalized landmark array from HandTracker.

        Returns:
            angles: (3,) array [servoX_deg, servoY_deg, servoZ_deg]
            method: "lstm" | "baseline (warming up)"
        """
        self._buffer.append(feature_vector.astype(np.float32))

        if len(self._buffer) < self._seq_len:
            return None, "baseline (warming up)"

        self.is_warmed_up = True
        seq    = np.array(self._buffer, dtype=np.float32)[np.newaxis, ...]   # (1, k, 42)
        norm   = self._model.predict(seq, verbose=0)[0]                       # (3,)
        angles = self._denormalize(norm)
        return angles, "lstm"

    def reset(self) -> None:
        """Clear the sequence buffer (e.g. when the hand leaves frame)."""
        self._buffer.clear()
        self.is_warmed_up = False

    def _denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Convert sigmoid [0, 1] output back to degree ranges."""
        keys   = ["x", "y", "z"]
        angles = np.array([
            normalized[i] * (self._bounds[k][1] - self._bounds[k][0]) + self._bounds[k][0]
            for i, k in enumerate(keys)
        ])
        return angles.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Baseline mapper (comparison / fallback)
# ══════════════════════════════════════════════════════════════════════════════

class BaselineMapper:
    """
    Direct linear frame-by-frame mapping.

    Paper Section IV-A baseline:  u_t = α · x_t + β

    Maps left hand landmarks directly to servo angles with no temporal
    smoothing. Used as:
      1. The comparison baseline in evaluation (higher variance S).
      2. Warm-up fallback while the LSTM buffer fills.
      3. Sole method when TensorFlow is unavailable.
    """

    def __init__(
        self,
        servo_bounds: dict,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> None:
        self._bounds = servo_bounds
        self._fw     = frame_width
        self._fh     = frame_height

    def map(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Map raw landmarks to servo angles.

        Args:
            landmarks: (21, 3) array of raw pixel coordinates.

        Returns:
            (3,) array [servoX_deg, servoY_deg, servoZ_deg]
        """
        # X — horizontal palm position (landmark 9)
        x_pos  = landmarks[9, 0]
        servoX = np.interp(x_pos, [self._fw / 2, self._fw], self._bounds["x"])

        # Y — vertical palm position (landmark 9)
        y_pos  = landmarks[9, 1]
        servoY = np.interp(y_pos, [0, self._fh], self._bounds["y"])

        # Z — pinch distance: thumb tip (4) to index tip (8)
        # Canonical grip mapping, consistent across all code versions.
        pinch  = np.linalg.norm(landmarks[4, :2] - landmarks[8, :2])
        servoZ = np.interp(pinch, [20, 220], self._bounds["z"])

        return np.array([servoX, servoY, servoZ], dtype=np.float32)
