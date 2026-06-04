"""
scripts/train.py
~~~~~~~~~~~~~~~~~
Train the LSTM temporal stabilization model.

Usage:
    python scripts/train.py
    python scripts/train.py --data data/training_data.csv --epochs 100

Requires: pip install tensorflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gesture_arm.config.settings import load_config
from gesture_arm.models.stabilizer import TF_AVAILABLE, train


def main() -> None:
    if not TF_AVAILABLE:
        print("[train] ERROR: TensorFlow not installed.")
        print("        pip install tensorflow")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train LSTM gesture stabilizer")
    parser.add_argument("--data",    default="data/training_data.csv")
    parser.add_argument("--out",     default="models/lstm_gesture_model.h5")
    parser.add_argument("--epochs",  type=int, default=80)
    parser.add_argument("--config",  default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    print(f"\n{'='*55}")
    print("  LSTM TRAINING")
    print(f"  Data   : {args.data}")
    print(f"  Output : {args.out}")
    print(f"  Epochs : {args.epochs}")
    print(f"{'='*55}\n")

    train(
        data_path=args.data,
        model_path=args.out,
        sequence_length=cfg.model.sequence_length,
        feature_dim=cfg.model.feature_dim,
        output_dim=cfg.model.output_dim,
        lstm_units=cfg.model.lstm_units,
        dense_units=cfg.model.dense_units,
        learning_rate=cfg.model.learning_rate,
        epochs=args.epochs,
        batch_size=cfg.model.batch_size,
        validation_split=cfg.model.validation_split,
    )


if __name__ == "__main__":
    main()
