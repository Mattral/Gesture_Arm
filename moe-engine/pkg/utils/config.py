"""YAML config loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    model: Dict[str, Any]
    training: Dict[str, Any]
    parallelism: Dict[str, Any]
    checkpoint: Dict[str, Any]
    elastic: Dict[str, Any]
    telemetry: Dict[str, Any]
    raw: Dict[str, Any]


def load_config(path: str | Path) -> Config:
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=raw["model"],
        training=raw["training"],
        parallelism=raw["parallelism"],
        checkpoint=raw["checkpoint"],
        elastic=raw["elastic"],
        telemetry=raw["telemetry"],
        raw=raw,
    )
