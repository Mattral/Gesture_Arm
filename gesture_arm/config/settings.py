"""
gesture_arm.config.settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads and validates configuration from a YAML file into typed dataclasses.
All other modules import from here — never hardcode constants elsewhere.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ── Default config path ────────────────────────────────────────────────────────
_DEFAULT_CFG = Path(__file__).parent / "default.yaml"


# ── Typed config dataclasses ───────────────────────────────────────────────────

@dataclass
class ServoConfig:
    pin: int
    min_deg: float
    max_deg: float
    default: float


@dataclass
class MotorSideConfig:
    dir1: int
    dir2: int
    pwm: int


@dataclass
class MotorConfig:
    left: MotorSideConfig
    right: MotorSideConfig
    max_speed: float
    turn_speed: float
    stop_timeout_sec: float


@dataclass
class HardwareConfig:
    port: str
    baudrate: int
    servos: Dict[str, ServoConfig]
    motors: MotorConfig


@dataclass
class VisionConfig:
    camera_index: int
    width: int
    height: int
    detection_confidence: float
    max_hands: int


@dataclass
class ModelConfig:
    sequence_length: int
    feature_dim: int
    output_dim: int
    lstm_units: int
    dense_units: int
    learning_rate: float
    epochs: int
    batch_size: int
    validation_split: float
    model_path: str
    training_data_path: str


@dataclass
class SpeechConfig:
    tts_rate: int
    tts_volume: float
    commands: Dict[str, Optional[List[float]]]


@dataclass
class EvaluationConfig:
    log_path: str
    stability_window: int
    latency_window: int


@dataclass
class AppConfig:
    hardware: HardwareConfig
    vision: VisionConfig
    model: ModelConfig
    speech: SpeechConfig
    evaluation: EvaluationConfig


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = _DEFAULT_CFG) -> AppConfig:
    """
    Load and parse a YAML config file into a typed AppConfig object.

    Args:
        path: Path to the YAML file. Defaults to gesture_arm/config/default.yaml.

    Returns:
        AppConfig dataclass instance.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if a required field is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    logger.debug("Loaded config from %s", path)

    hw_raw = raw["hardware"]
    servos = {
        k: ServoConfig(**v) for k, v in hw_raw["servos"].items()
    }
    motors = MotorConfig(
        left=MotorSideConfig(**hw_raw["motors"]["left"]),
        right=MotorSideConfig(**hw_raw["motors"]["right"]),
        max_speed=hw_raw["motors"]["max_speed"],
        turn_speed=hw_raw["motors"]["turn_speed"],
        stop_timeout_sec=hw_raw["motors"]["stop_timeout_sec"],
    )
    hardware = HardwareConfig(
        port=os.environ.get("GESTURE_ARM_PORT", hw_raw["port"]),
        baudrate=hw_raw["baudrate"],
        servos=servos,
        motors=motors,
    )

    vision = VisionConfig(**raw["vision"])
    model  = ModelConfig(**raw["model"])
    speech = SpeechConfig(**raw["speech"])
    evaluation = EvaluationConfig(**raw["evaluation"])

    return AppConfig(
        hardware=hardware,
        vision=vision,
        model=model,
        speech=speech,
        evaluation=evaluation,
    )
