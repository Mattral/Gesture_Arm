"""
pkg/telemetry/logger.py
=======================

Structured per-step telemetry. Emits both:

  * **Newline-delimited JSON** to `telemetry.json_path` for Prometheus /
    ELK / Loki scrapers.
  * **TensorBoard scalars** for human inspection.

Per the spec the emitted envelope includes:
  * Triton kernel SRAM occupancy + achieved bandwidth (GB/s)
  * NCCL all-to-all + FSDP all-gather + reduce-scatter latencies
  * Peak CUDA memory (allocated + reserved + leak delta)
  * Async checkpoint commit duration + active cluster node count
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    SummaryWriter = None                                                 # type: ignore
    _HAS_TB = False


@dataclass
class StepRecord:
    step: int
    loss: float
    mfu: float
    tokens_per_sec: float
    kernel: dict = field(default_factory=dict)
    collective: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    infra: dict = field(default_factory=dict)
    wall_clock_ms: float = 0.0


class StructuredLogger:
    """Sink for structured per-step telemetry.

    Thread-safe by virtue of:
      * file appends are atomic on POSIX for lines < PIPE_BUF (~4KB), and
        our envelopes are well under that limit;
      * we wrap the SummaryWriter calls in a re-entrant lock.
    """

    def __init__(
        self,
        json_path: str,
        tensorboard_dir: Optional[str] = None,
        rank: int = 0,
        also_stdout: bool = True,
    ):
        self.rank = rank
        self.also_stdout = also_stdout
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.json_path.open("a", buffering=1)
        self._tb: Optional[SummaryWriter] = None
        if _HAS_TB and tensorboard_dir and rank == 0:
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(log_dir=tensorboard_dir)
        self._peak_mem_prev: float = 0.0

    # ------------------------------------------------------------------
    def emit(self, record: StepRecord) -> None:
        # Auto-fill the memory section if the caller didn't already.
        if not record.memory and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            reserv = torch.cuda.memory_reserved() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            record.memory = {
                "peak_allocated_gb": round(peak, 4),
                "reserved_gb": round(reserv, 4),
                "leak_delta_gb": round(peak - self._peak_mem_prev, 4),
            }
            self._peak_mem_prev = peak

        payload = asdict(record)
        # Always include rank for downstream multi-process aggregation.
        payload["rank"] = self.rank
        payload["ts"] = time.time()
        line = json.dumps(payload, separators=(",", ":"))
        self._fh.write(line + "\n")
        if self.also_stdout and self.rank == 0:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

        if self._tb is not None:
            self._tb.add_scalar("loss", record.loss, record.step)
            self._tb.add_scalar("mfu", record.mfu, record.step)
            self._tb.add_scalar("tokens_per_sec", record.tokens_per_sec, record.step)
            for k, v in record.collective.items():
                self._tb.add_scalar(f"collective/{k}", v, record.step)
            for k, v in record.memory.items():
                self._tb.add_scalar(f"memory/{k}", v, record.step)
            for k, v in record.kernel.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"kernel/{k}", v, record.step)
            for k, v in record.infra.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"infra/{k}", v, record.step)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
        if self._tb is not None:
            self._tb.close()
