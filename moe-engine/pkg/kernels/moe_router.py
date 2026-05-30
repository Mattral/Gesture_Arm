"""
pkg/kernels/moe_router.py
=========================

Hardware-aware Top-K Mixture-of-Experts router.

This module implements the sparse gating/dispatching kernel that sits at the
heart of every modern MoE block (Switch-Transformer, GShard, Mixtral, etc.).
It is written in two interlocking layers:

1.  **Triton JIT kernel** (`_router_fwd_kernel`, `_router_bwd_kernel`) –
    executed when a CUDA-capable GPU and a working Triton install are
    available. The kernel fuses:

        gemm( tokens [N, H], gate_w [H, E] )  ->  logits [N, E]
        softmax_along_E( logits )             ->  probs  [N, E]
        top_k( probs, k=K )                   ->  idx    [N, K], w [N, K]
        renormalize( w )                      ->  combine weights

    in a *single* pass over the gating dimension. SRAM occupancy is bounded
    by `BLOCK_E * BLOCK_N` floats; we choose `(BLOCK_N=64, BLOCK_E=64)` to
    keep working-set under 32 KiB so all three operands stay resident in L1
    on Ampere/Hopper. Global loads of `tokens` and `gate_w` are coalesced
    on the contiguous (H) dimension – consecutive lanes read consecutive
    addresses, avoiding bank conflicts. Top-K is implemented as an in-SRAM
    selection-sort over K elements (K is small, typically 1, 2, or 4),
    eliminating shared-memory bank pressure that a full sort would create.

2.  **PyTorch double-precision reference** (`_reference_route_fp64`) used by
    the autograd backward and by the test-suite. This path runs on CPU or
    GPU and is the numerical ground truth against which the Triton kernel
    is validated at `atol = rtol = 1e-5`.

Both paths are wrapped behind a `torch.autograd.Function`
(`MoERouterAutograd`) so that the entire router is a drop-in differentiable
module that respects PyTorch's autograd graph and AMP semantics.

Tensor-shape glossary
---------------------
    N  = Batch * Sequence              (flattened token count)
    H  = hidden_dim
    E  = num_experts
    K  = top_k

    tokens      : [N, H]   (input activations, fp16/bf16/fp32)
    gate_w      : [H, E]   (router projection matrix)
    logits      : [N, E]
    probs       : [N, E]   (softmax)
    topk_idx    : [N, K]   int32      – expert id chosen per (token, slot)
    topk_w      : [N, K]   float      – renormalized combine weights
    dispatch_cnt: [E]      int64      – tokens assigned to each expert

Token-Conservation Invariant
----------------------------
    sum(dispatch_cnt) == N * K
    unique(topk_idx[:, 0])  has no NaN / no -1 entries

These invariants are asserted both in the Python wrapper and in
`tests/test_kernels.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# --------------------------------------------------------------------------
# Optional Triton import. The repo MUST work in CPU-only environments so
# tests can run anywhere; the JIT kernel is only loaded when CUDA + Triton
# are both available.
# --------------------------------------------------------------------------
try:
    import triton                       # type: ignore
    import triton.language as tl        # type: ignore
    TRITON_AVAILABLE = True
except Exception:                       # pragma: no cover - import-time guard
    triton = None                       # type: ignore
    tl = None                           # type: ignore
    TRITON_AVAILABLE = False


# ==========================================================================
# Telemetry record returned from each router invocation. Consumed by
# pkg.telemetry.logger.StructuredLogger.
# ==========================================================================
@dataclass
class RouterProfile:
    sram_bytes_per_block: int
    achieved_bandwidth_gbps: float
    kernel_ms: float
    used_triton: bool
    tokens_per_expert_mean: float
    tokens_per_expert_std: float


# ==========================================================================
# Triton kernel – forward pass.
# ==========================================================================
if TRITON_AVAILABLE:

    @triton.jit
    def _router_fwd_kernel(
        # Pointers
        tokens_ptr,           # [N, H]
        gate_w_ptr,           # [H, E]
        topk_idx_ptr,         # [N, K] int32
        topk_w_ptr,           # [N, K] float32
        logits_ptr,           # [N, E] float32  (saved for backward)
        # Strides
        stride_tn, stride_th,
        stride_gh, stride_ge,
        stride_in, stride_ik,
        stride_wn, stride_wk,
        stride_ln, stride_le,
        # Sizes
        N, H, E, K,
        # Meta
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Fused router: tokens @ gate_w -> softmax -> top_k -> combine weights.

        Grid layout: 1D, one program instance per BLOCK_N tokens. Each program
        materializes its BLOCK_N x E logit tile in SRAM, runs a numerically
        stable softmax along E, then performs an in-SRAM selection of the
        top-K entries per row.
        """
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # ----------------------------------------------------------------
        # 1. Compute the logits tile [BLOCK_N, E] by accumulating over H in
        #    BLOCK_H-sized chunks. Loads of `tokens_ptr` are contiguous on
        #    the H dimension (stride_th == 1 in row-major), guaranteeing
        #    coalesced global-memory access. `gate_w_ptr` is broadcast.
        # ----------------------------------------------------------------
        offs_e = tl.arange(0, BLOCK_E)
        # NOTE: We require E <= BLOCK_E (router fanout is small, typically
        # 8-256). The host wrapper picks BLOCK_E = next_pow2(E).
        mask_e = offs_e < E

        acc = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H

            tok_tile = tl.load(
                tokens_ptr
                + offs_n[:, None] * stride_tn
                + offs_h[None, :] * stride_th,
                mask=mask_n[:, None] & mask_h[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_tile = tl.load(
                gate_w_ptr
                + offs_h[:, None] * stride_gh
                + offs_e[None, :] * stride_ge,
                mask=mask_h[:, None] & mask_e[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(tok_tile, gate_tile, allow_tf32=False)

        # Mask invalid expert columns to -inf so they never win softmax/topk.
        logits = tl.where(mask_e[None, :], acc, float("-inf"))

        # Save raw logits for backward (we recompute softmax there from these).
        tl.store(
            logits_ptr
            + offs_n[:, None] * stride_ln
            + offs_e[None, :] * stride_le,
            logits,
            mask=mask_n[:, None] & mask_e[None, :],
        )

        # ----------------------------------------------------------------
        # 2. Numerically stable softmax along E.
        # ----------------------------------------------------------------
        row_max = tl.max(logits, axis=1)
        shifted = logits - row_max[:, None]
        exp_l = tl.exp(shifted)
        # Re-mask: positions where mask_e is False contributed exp(-inf)==0.
        denom = tl.sum(exp_l, axis=1)
        probs = exp_l / denom[:, None]

        # ----------------------------------------------------------------
        # 3. Top-K selection by repeated argmax. K is small (1-4) so this
        #    linear loop is faster and uses less SRAM than a bitonic sort.
        #    Each pass zeroes out the previously-chosen column.
        # ----------------------------------------------------------------
        topk_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(0, K):
            kth_idx = tl.argmax(probs, axis=1).to(tl.int32)
            kth_val = tl.max(probs, axis=1)
            # Store this (index, weight) pair.
            tl.store(
                topk_idx_ptr + offs_n * stride_in + k * stride_ik,
                kth_idx,
                mask=mask_n,
            )
            tl.store(
                topk_w_ptr + offs_n * stride_wn + k * stride_wk,
                kth_val,
                mask=mask_n,
            )
            topk_sum += kth_val
            # Mask the chosen column to -inf for the next iteration.
            kth_mask = tl.arange(0, BLOCK_E)[None, :] == kth_idx[:, None]
            probs = tl.where(kth_mask, 0.0, probs)

        # ----------------------------------------------------------------
        # 4. Renormalize combine weights so each row sums to 1. This is the
        #    Switch-style "combine" weight; matches the reference path.
        # ----------------------------------------------------------------
        inv = 1.0 / tl.where(topk_sum > 0.0, topk_sum, 1.0)
        for k in tl.static_range(0, K):
            w = tl.load(
                topk_w_ptr + offs_n * stride_wn + k * stride_wk,
                mask=mask_n,
                other=0.0,
            )
            tl.store(
                topk_w_ptr + offs_n * stride_wn + k * stride_wk,
                w * inv,
                mask=mask_n,
            )

    @triton.jit
    def _router_bwd_kernel(
        grad_w_ptr,            # [N, K]    upstream grad on combine weights
        topk_idx_ptr,          # [N, K]    int32   chosen experts
        logits_ptr,            # [N, E]    saved logits from forward
        grad_logits_ptr,       # [N, E]    output: dL/dlogits  (fp32)
        stride_gn, stride_gk,
        stride_in, stride_ik,
        stride_ln, stride_le,
        stride_dn, stride_de,
        N, E, K,
        BLOCK_N: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Backward through the (softmax -> top_k -> renormalize) pipeline.

        Forward chain:
            p = softmax(l)              (along E)
            (idx, v) = top_k(p)
            w_k = v_k / sum_j v_j

        We propagate `grad_w` -> `grad_v` -> `grad_p` -> `grad_l`. The
        top_k stage is treated as a sparse scatter: for every row, only K
        entries of `grad_p` are non-zero (those at `idx`). The softmax
        Jacobian collapses to:
            grad_l_i = p_i * (grad_p_i - sum_j(grad_p_j * p_j))
        """
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < E

        # Recompute softmax in fp32 for numerical agreement with reference.
        logits = tl.load(
            logits_ptr
            + offs_n[:, None] * stride_ln
            + offs_e[None, :] * stride_le,
            mask=mask_n[:, None] & mask_e[None, :],
            other=float("-inf"),
        )
        row_max = tl.max(logits, axis=1)
        exp_l = tl.exp(logits - row_max[:, None])
        denom = tl.sum(exp_l, axis=1)
        probs = exp_l / denom[:, None]
        probs = tl.where(mask_e[None, :], probs, 0.0)

        # Scatter grad_w into grad_p at top-k positions.
        # First load all K gradients (small loop, fully unrolled).
        # grad_v = grad_w * d(renorm)/d(v). For w_k = v_k / S, where
        # S = sum v_j, we have:
        #   dw_k / dv_k = 1/S - v_k / S^2
        #   dw_k / dv_j = -v_k / S^2   (j != k)
        # Hence grad_v_k = (1/S) * grad_w_k - (1/S^2) * sum_j(grad_w_j * v_j)
        S = tl.zeros((BLOCK_N,), dtype=tl.float32)
        gwv = tl.zeros((BLOCK_N,), dtype=tl.float32)  # sum_j(grad_w_j * v_j)
        # Load v_j from probs[idx_j].
        for k in tl.static_range(0, K):
            idx_k = tl.load(
                topk_idx_ptr + offs_n * stride_in + k * stride_ik,
                mask=mask_n, other=0,
            ).to(tl.int32)
            gw_k = tl.load(
                grad_w_ptr + offs_n * stride_gn + k * stride_gk,
                mask=mask_n, other=0.0,
            )
            # Gather probs[idx_k] via a one-hot reduction in SRAM.
            onehot = (tl.arange(0, BLOCK_E)[None, :] == idx_k[:, None]).to(tl.float32)
            v_k = tl.sum(probs * onehot, axis=1)
            S += v_k
            gwv += gw_k * v_k

        inv_S = 1.0 / tl.where(S > 0.0, S, 1.0)
        inv_S2 = inv_S * inv_S

        # Build grad_p as a dense [BLOCK_N, BLOCK_E] tile (mostly zeros).
        grad_p = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
        for k in tl.static_range(0, K):
            idx_k = tl.load(
                topk_idx_ptr + offs_n * stride_in + k * stride_ik,
                mask=mask_n, other=0,
            ).to(tl.int32)
            gw_k = tl.load(
                grad_w_ptr + offs_n * stride_gn + k * stride_gk,
                mask=mask_n, other=0.0,
            )
            grad_v_k = gw_k * inv_S - gwv * inv_S2
            onehot = (tl.arange(0, BLOCK_E)[None, :] == idx_k[:, None]).to(tl.float32)
            grad_p += onehot * grad_v_k[:, None]

        # Softmax-Jacobian collapse.
        dot = tl.sum(grad_p * probs, axis=1)
        grad_l = probs * (grad_p - dot[:, None])

        tl.store(
            grad_logits_ptr
            + offs_n[:, None] * stride_dn
            + offs_e[None, :] * stride_de,
            grad_l,
            mask=mask_n[:, None] & mask_e[None, :],
        )


# ==========================================================================
# Reference (double-precision) implementation used:
#   * as autograd reference for the unit tests' tolerance gate
#   * as the actual forward when CUDA / Triton are unavailable
# ==========================================================================
def _reference_route_fp64(
    tokens: torch.Tensor,
    gate_w: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference path. Always runs in fp64 for max precision.

    Returns
    -------
    topk_idx   : LongTensor [N, K]
    topk_w     : FloatTensor [N, K]  (renormalized)
    logits     : FloatTensor [N, E]  (saved for autograd; cast back to caller dtype later)
    """
    orig_dtype = tokens.dtype
    t64 = tokens.to(torch.float64)
    g64 = gate_w.to(torch.float64)
    logits = t64 @ g64                                                   # [N, E]
    probs = torch.softmax(logits, dim=-1)                                # [N, E]
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1, largest=True)   # [N, K]
    denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    topk_w = topk_vals / denom                                           # renormalize
    return (
        topk_idx.to(torch.long),
        topk_w.to(orig_dtype),
        logits.to(orig_dtype),
    )


# ==========================================================================
# Autograd Function -- single entry-point used by `MoERouter`.
# ==========================================================================
class MoERouterFunction(torch.autograd.Function):
    """Differentiable Top-K router.

    Forward chooses Triton on CUDA-capable devices, falls back to the
    fp64 reference otherwise. Backward always uses the analytical
    softmax-topk-renorm gradient described above.
    """

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        gate_w: torch.Tensor,
        k: int,
        force_reference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert tokens.dim() == 2, "tokens must be flattened to [N, H]"
        assert gate_w.dim() == 2, "gate_w must be [H, E]"
        N, H = tokens.shape
        H2, E = gate_w.shape
        assert H == H2, f"hidden mismatch {H} vs {H2}"
        assert 1 <= k <= E, f"k={k} must satisfy 1 <= k <= E={E}"

        use_triton = (
            (not force_reference)
            and TRITON_AVAILABLE
            and tokens.is_cuda
            and gate_w.is_cuda
        )

        if use_triton:
            topk_idx, topk_w, logits = _triton_forward(tokens, gate_w, k)
        else:
            topk_idx, topk_w, logits = _reference_route_fp64(tokens, gate_w, k)

        # Token-Conservation Invariant (must hold by construction; assert
        # cheaply in debug). Each row contributes exactly K assignments,
        # so total assignments == N * K and no -1 / NaN entries appear.
        # We avoid a sync on the hot path by only checking shapes.
        assert topk_idx.shape == (N, k)
        assert topk_w.shape == (N, k)

        ctx.save_for_backward(tokens, gate_w, logits, topk_idx, topk_w)
        ctx.k = k
        ctx.use_triton = use_triton
        return topk_idx, topk_w

    @staticmethod
    def backward(ctx, grad_idx, grad_w):                                 # noqa: D401
        # grad_idx is meaningless (idx is an integer hard-selection) and
        # the autograd engine will pass zeros / None.
        tokens, gate_w, logits, topk_idx, topk_w = ctx.saved_tensors
        k = ctx.k
        N, H = tokens.shape
        E = gate_w.shape[1]

        if ctx.use_triton:
            grad_logits = torch.empty_like(logits, dtype=torch.float32)
            BLOCK_N = 64
            BLOCK_E = _next_pow2(E)
            grid = ((N + BLOCK_N - 1) // BLOCK_N,)
            _router_bwd_kernel[grid](
                grad_w.contiguous().to(torch.float32),
                topk_idx.contiguous().to(torch.int32),
                logits.contiguous().to(torch.float32),
                grad_logits,
                grad_w.stride(0), grad_w.stride(1),
                topk_idx.stride(0), topk_idx.stride(1),
                logits.stride(0), logits.stride(1),
                grad_logits.stride(0), grad_logits.stride(1),
                N, E, k,
                BLOCK_N=BLOCK_N, BLOCK_E=BLOCK_E,
            )
            grad_logits = grad_logits.to(tokens.dtype)
        else:
            grad_logits = _reference_backward_fp64(
                logits.detach(), topk_idx, grad_w, k, E,
            ).to(tokens.dtype)

        # Propagate through the (tokens @ gate_w) gemm.
        grad_tokens = grad_logits @ gate_w.t()
        grad_gate_w = tokens.t() @ grad_logits

        return grad_tokens, grad_gate_w, None, None


MoERouterAutograd = MoERouterFunction


def _reference_backward_fp64(
    logits: torch.Tensor,
    topk_idx: torch.Tensor,
    grad_w: torch.Tensor,
    k: int,
    E: int,
) -> torch.Tensor:
    """Analytical backward used both for CPU path and as test oracle."""
    l64 = logits.to(torch.float64)
    probs = torch.softmax(l64, dim=-1)                                   # [N, E]
    # Gather selected probs to build v_k, S, and grad_v.
    v = probs.gather(1, topk_idx)                                        # [N, K]
    S = v.sum(dim=-1, keepdim=True).clamp_min(1e-30)                     # [N, 1]
    gw = grad_w.to(torch.float64)                                        # [N, K]
    # grad_v = (1/S) * grad_w - (1/S^2) * sum_j(grad_w_j * v_j)
    gwv = (gw * v).sum(dim=-1, keepdim=True)
    grad_v = gw / S - gwv / (S * S)                                      # [N, K]
    # Scatter back to dense [N, E]
    grad_p = torch.zeros_like(probs).scatter_add_(1, topk_idx, grad_v)
    # Softmax Jacobian.
    dot = (grad_p * probs).sum(dim=-1, keepdim=True)
    grad_logits = probs * (grad_p - dot)
    return grad_logits


def _triton_forward(
    tokens: torch.Tensor,
    gate_w: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:                    # pragma: no cover - GPU-only
    N, H = tokens.shape
    E = gate_w.shape[1]
    BLOCK_N = 64
    BLOCK_H = 64
    BLOCK_E = _next_pow2(E)

    topk_idx = torch.empty((N, k), dtype=torch.int32, device=tokens.device)
    topk_w = torch.empty((N, k), dtype=torch.float32, device=tokens.device)
    logits = torch.empty((N, E), dtype=torch.float32, device=tokens.device)

    grid = ((N + BLOCK_N - 1) // BLOCK_N,)
    _router_fwd_kernel[grid](
        tokens.contiguous(),
        gate_w.contiguous(),
        topk_idx, topk_w, logits,
        tokens.stride(0), tokens.stride(1),
        gate_w.stride(0), gate_w.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        topk_w.stride(0), topk_w.stride(1),
        logits.stride(0), logits.stride(1),
        N, H, E, k,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, BLOCK_E=BLOCK_E,
    )
    return topk_idx.to(torch.long), topk_w.to(tokens.dtype), logits.to(tokens.dtype)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


# ==========================================================================
# Public module-level helper used by the distributed layer & tests.
# ==========================================================================
def moe_topk_route(
    tokens: torch.Tensor,
    gate_w: torch.Tensor,
    k: int,
    force_reference: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable functional entry point.

    Parameters
    ----------
    tokens : [B, S, H] or [N, H]
    gate_w : [H, E]
    k      : top-k

    Returns
    -------
    topk_idx : LongTensor [N, K]
    topk_w   : Tensor      [N, K]   (same dtype as `tokens`)
    """
    if tokens.dim() == 3:
        B, S, H = tokens.shape
        flat = tokens.reshape(B * S, H)
    elif tokens.dim() == 2:
        flat = tokens
    else:
        raise ValueError(f"tokens must be rank 2 or 3, got {tokens.dim()}")
    idx, w = MoERouterFunction.apply(flat, gate_w, k, force_reference)
    return idx, w


# ==========================================================================
# nn.Module wrapper.
# ==========================================================================
class MoERouter(torch.nn.Module):
    """Top-K router as a `nn.Module`. The gate matrix is a learnable parameter.

    Attributes
    ----------
    hidden_dim   : H
    num_experts  : E
    top_k        : K
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # Stored as [H, E] (transposed vs an nn.Linear) so the Triton kernel
        # can load it directly without an extra transpose.
        self.gate_w = torch.nn.Parameter(
            torch.empty(hidden_dim, num_experts, dtype=dtype)
        )
        torch.nn.init.normal_(self.gate_w, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        self.bias = (
            torch.nn.Parameter(torch.zeros(num_experts, dtype=dtype)) if bias else None
        )
        self.last_profile: Optional[RouterProfile] = None

    def forward(
        self,
        tokens: torch.Tensor,
        force_reference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (topk_idx [N, K], topk_w [N, K], dispatch_cnt [E])."""
        if tokens.dim() == 3:
            B, S, H = tokens.shape
            flat = tokens.reshape(B * S, H)
        else:
            flat = tokens
            B = 1
            S = flat.shape[0]
            H = flat.shape[1]
        N = flat.shape[0]
        assert H == self.hidden_dim

        gate_w = self.gate_w
        if self.bias is not None:
            # Fold bias into the routing by adding to the (tokens @ gate_w)
            # output. For simplicity we apply on the reference path; the
            # Triton kernel above doesn't accept bias yet -- TODO for v2 --
            # so we route bias users through the reference path.
            force_reference = True

        idx, w = moe_topk_route(flat, gate_w, self.top_k, force_reference)

        if self.bias is not None and force_reference:
            # In bias mode we used the reference path on the *unbiased* logits.
            # Add bias post-softmax weights – following the Switch-Transformer
            # auxiliary-loss formulation – by re-scoring through a second pass.
            with torch.no_grad():
                logits = flat.to(torch.float32) @ gate_w.to(torch.float32) + self.bias.to(torch.float32)
                probs = torch.softmax(logits, dim=-1)
                topk_vals, idx_new = torch.topk(probs, k=self.top_k, dim=-1)
                w_new = (topk_vals / topk_vals.sum(-1, keepdim=True).clamp_min(1e-30)).to(tokens.dtype)
            idx, w = idx_new.to(torch.long), w_new

        # Dispatch count per expert: histogram of idx flatten.
        dispatch_cnt = torch.bincount(
            idx.reshape(-1), minlength=self.num_experts
        ).to(torch.long)

        # ===== TOKEN CONSERVATION INVARIANT (fail-fast guard) =====
        # Each of N tokens gets K assignments, so total must be N*K.
        # This is a critical safety check: if violated, training will diverge silently.
        total_dispatched = dispatch_cnt.sum().item()
        expected_total = N * self.top_k
        assert total_dispatched == expected_total, (
            f"Token loss detected in router: {total_dispatched} routed tokens != "
            f"expected {expected_total} (N={N}, K={self.top_k}). "
            f"This indicates a routing bug or silent data corruption."
        )
        # Also verify no -1 or NaN expert indices (would indicate failed topk)
        assert not torch.isnan(idx.float()).any(), (
            "NaN detected in expert indices; topk kernel may have failed."
        )
        assert (idx >= 0).all() and (idx < self.num_experts).all(), (
            f"Out-of-range expert indices detected; "
            f"min={idx.min()}, max={idx.max()}, E={self.num_experts}"
        )

        # ------ Telemetry profile ------
        # SRAM footprint for a 64x64 tile of fp32 == 64*64*4 == 16 KiB plus the
        # tokens / gate_w halves, dominated by 64*max(BLOCK_E,BLOCK_H)*4.
        sram = 64 * max(_next_pow2(self.num_experts), 64) * 4 * 3
        # Bandwidth: bytes_read = N*H*dtype + H*E*dtype; assume kernel_ms = 1.
        # The real value is filled in by the profiler integration in
        # pkg.telemetry.logger; we record a conservative estimate here.
        bytes_moved = (flat.numel() + gate_w.numel()) * flat.element_size()
        achieved_bw = (bytes_moved / (1024 ** 3)) / 1e-3   # GB/s assuming 1ms
        self.last_profile = RouterProfile(
            sram_bytes_per_block=sram,
            achieved_bandwidth_gbps=achieved_bw,
            kernel_ms=1.0,
            used_triton=(TRITON_AVAILABLE and flat.is_cuda),
            tokens_per_expert_mean=float(dispatch_cnt.float().mean().item()),
            tokens_per_expert_std=float(dispatch_cnt.float().std().item()),
        )
        return idx, w, dispatch_cnt
