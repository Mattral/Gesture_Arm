"""MFU (Model FLOPs Utilization) accountant.

Computes per-step MFU as:

    MFU = achieved_tflops / hardware_peak_tflops

Where `achieved_tflops` is derived from a theoretical FLOP count for the
forward+backward pass of the model architecture, divided by the measured
step time.

For a Transformer MoE layer the per-token FLOP count is:

    flops_attn        = 4 * H^2 * S    (qkv + out projection, summed)
    flops_router      = H * E          (gating gemm)
    flops_expert_ffn  = 2 * H * F * 2  (gate + up + down, each ~H*F)
                      = 6 * H * F      (per active token)
    flops_per_token   = flops_attn / S + flops_router + K * flops_expert_ffn

The factor 3 covers forward + backward + recomputation (standard convention
used by Chinchilla, PaLM, Llama papers).
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class MFUResult:
    achieved_tflops: float
    peak_tflops: float
    mfu: float
    step_ms: float
    tokens_per_sec: float


def compute_moe_flops(
    hidden_dim: int,
    num_layers: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    seq_length: int,
    batch_tokens: int,
    vocab_size: int = 0,
) -> int:
    """Total FLOPs for a forward+backward step on `batch_tokens` tokens."""
    H = hidden_dim
    F = ffn_dim
    E = num_experts
    K = top_k
    S = seq_length

    # Attention: 4*H^2 per token + 4*H*S per token for QK^T and AV.
    flops_attn_per_token = 4 * H * H + 4 * H * S
    # Router: tokens * gate matrix.
    flops_router_per_token = H * E
    # Expert FFN: per active token = 3 GEMMs of size H*F (SwiGLU has gate+up+down).
    # Each GEMM = 2*H*F mac ops. Total = 6*H*F per active token.
    flops_expert_per_token = K * 6 * H * F
    # Final LM head projection (if vocab_size > 0).
    flops_lm_head_per_token = 2 * H * vocab_size if vocab_size else 0

    flops_per_token_fwd = num_layers * (
        flops_attn_per_token + flops_router_per_token + flops_expert_per_token
    ) + flops_lm_head_per_token
    # Forward + backward + activation recompute = 3x forward.
    flops_per_token_total = 3 * flops_per_token_fwd
    return flops_per_token_total * batch_tokens


class MFUAccountant:
    """Streaming MFU tracker. Call `start_step()` / `end_step(tokens)` per iter."""

    def __init__(self, peak_tflops: float, mfu_target: float = 0.55):
        self.peak_tflops = peak_tflops
        self.mfu_target = mfu_target
        self._t0: float = 0.0
        self._flops_per_token: int = 0
        self.history: list[MFUResult] = []
        self._running_mfu: float = 0.0
        self._steps: int = 0

    def configure(self, flops_per_token: int) -> None:
        self._flops_per_token = flops_per_token

    def start_step(self) -> None:
        self._t0 = time.perf_counter()

    def end_step(self, tokens: int) -> MFUResult:
        dt = max(time.perf_counter() - self._t0, 1e-9)
        achieved = (self._flops_per_token * tokens) / dt / 1e12
        mfu = achieved / max(self.peak_tflops, 1e-9)
        res = MFUResult(
            achieved_tflops=achieved,
            peak_tflops=self.peak_tflops,
            mfu=mfu,
            step_ms=dt * 1000.0,
            tokens_per_sec=tokens / dt,
        )
        self.history.append(res)
        self._steps += 1
        self._running_mfu = (self._running_mfu * (self._steps - 1) + mfu) / self._steps
        return res

    @property
    def running_mfu(self) -> float:
        return self._running_mfu

    def is_above_target(self) -> bool:
        return self._running_mfu >= self.mfu_target
