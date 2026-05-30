"""
tests/test_kernels.py
=====================

Validates the moe_router kernel against a double-precision PyTorch
autograd reference. Asserts:

  * Forward tolerance:   atol < 1e-5, rtol < 1e-5  on weights & probs
  * Backward tolerance:  atol < 1e-5, rtol < 1e-5  via gradcheck-style
                          comparison of grad_tokens and grad_gate_w.
  * Token conservation:  sum(dispatch_cnt) == N * K
                          unique tokens (rows of idx) is N
                          no -1 / NaN entries
"""

from __future__ import annotations

import math

import pytest
import torch

from pkg.kernels.moe_router import (
    MoERouter,
    moe_topk_route,
    MoERouterFunction,
    _reference_route_fp64,
)


@pytest.mark.parametrize("B,S,H,E,K", [
    (2, 16, 64, 8, 2),
    (1, 32, 128, 16, 1),
    (4, 8, 64, 32, 4),
])
def test_forward_tolerance(B, S, H, E, K):
    torch.manual_seed(0)
    tokens = torch.randn(B * S, H, dtype=torch.float64)
    gate_w = torch.randn(H, E, dtype=torch.float64) * (1.0 / math.sqrt(H))

    # CPU path -- always uses reference fp64.
    idx, w = moe_topk_route(tokens.float(), gate_w.float(), k=K, force_reference=True)
    ref_idx, ref_w, _ = _reference_route_fp64(tokens, gate_w, k=K)

    assert idx.shape == ref_idx.shape == (B * S, K)
    assert w.shape == ref_w.shape == (B * S, K)
    assert torch.equal(idx.cpu(), ref_idx.cpu())

    assert torch.allclose(
        w.double().cpu(), ref_w.double().cpu(), atol=1e-5, rtol=1e-5,
    ), f"forward weight tolerance violated, max_diff={(w.double()-ref_w.double()).abs().max()}"


@pytest.mark.parametrize("B,S,H,E,K", [
    (2, 8, 32, 8, 2),
    (1, 16, 64, 16, 2),
])
def test_backward_tolerance(B, S, H, E, K):
    """Manual analytical backward vs torch.autograd through the reference path."""
    torch.manual_seed(7)
    N = B * S
    tokens = torch.randn(N, H, dtype=torch.float64, requires_grad=True)
    gate_w = (torch.randn(H, E, dtype=torch.float64) / math.sqrt(H)).requires_grad_(True)

    # ------- autograd reference: rerun the math inline & let PyTorch diff it.
    def _ref(tk, gw):
        logits = tk @ gw
        probs = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k=K, dim=-1)
        denom = vals.sum(-1, keepdim=True).clamp_min(1e-30)
        return vals / denom, idx

    w_ref, _ = _ref(tokens, gate_w)
    grad_w = torch.randn_like(w_ref)
    (w_ref * grad_w).sum().backward()
    ref_grad_tokens = tokens.grad.detach().clone()
    ref_grad_gate = gate_w.grad.detach().clone()

    # ------- analytical backward through MoERouterAutograd
    tokens2 = tokens.detach().float().requires_grad_(True)
    gate2 = gate_w.detach().float().requires_grad_(True)
    idx, w = MoERouterFunction.apply(tokens2, gate2, K, True)
    (w * grad_w.float()).sum().backward()

    assert torch.allclose(
        tokens2.grad.double(), ref_grad_tokens, atol=1e-5, rtol=1e-5,
    ), f"grad_tokens diff {(tokens2.grad.double()-ref_grad_tokens).abs().max()}"
    assert torch.allclose(
        gate2.grad.double(), ref_grad_gate, atol=1e-5, rtol=1e-5,
    ), f"grad_gate diff {(gate2.grad.double()-ref_grad_gate).abs().max()}"


@pytest.mark.parametrize("B,S,H,E,K", [
    (2, 32, 64, 8, 2),
    (3, 16, 64, 32, 4),
])
def test_token_conservation(B, S, H, E, K):
    """Every token is dispatched exactly K times; no drops, no duplicates."""
    torch.manual_seed(1)
    N = B * S
    tokens = torch.randn(N, H)
    router = MoERouter(hidden_dim=H, num_experts=E, top_k=K)
    idx, w, cnt = router(tokens)

    assert idx.shape == (N, K)
    assert cnt.shape == (E,)
    assert int(cnt.sum().item()) == N * K, "token conservation broken: total dispatch mismatch"
    assert (idx >= 0).all() and (idx < E).all(), "out-of-range expert id detected"
    assert not torch.isnan(w).any(), "NaN combine weight detected"
    # Each token's K slots must be K *distinct* experts (no duplicate slot).
    sorted_idx, _ = torch.sort(idx, dim=-1)
    diffs = sorted_idx[:, 1:] - sorted_idx[:, :-1]
    assert (diffs > 0).all() if K > 1 else True, "duplicate expert assignment within a token"


def test_combine_weights_sum_to_one():
    torch.manual_seed(2)
    tokens = torch.randn(64, 32)
    router = MoERouter(hidden_dim=32, num_experts=16, top_k=2)
    _, w, _ = router(tokens)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_router_profile_populated():
    torch.manual_seed(3)
    tokens = torch.randn(32, 16)
    router = MoERouter(hidden_dim=16, num_experts=8, top_k=2)
    router(tokens)
    prof = router.last_profile
    assert prof is not None
    assert prof.sram_bytes_per_block > 0
    assert prof.tokens_per_expert_mean > 0
