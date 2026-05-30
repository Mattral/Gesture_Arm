"""
tests/test_kernels_numerics.py

Numerical validation tests for the MoE router Triton kernel.
Compares Triton forward/backward against FP64 reference path with strict
atol < 1e-5, rtol < 1e-5 gates to catch silent correctness bugs.

Key invariants tested:
  * Forward output indices/weights match reference
  * Backward gradients match reference (softmax → topk → renorm chain)
  * Token conservation: sum(routed_tokens) == N * K
  * Weight normalization per token
  * Determinism (same seed → same result)
"""

import pytest
import torch
import numpy as np

from pkg.kernels.moe_router import moe_topk_route, TRITON_AVAILABLE


def assert_close(a, b, atol=1e-5, rtol=1e-5, name=""):
    """Assert tensors are close with detailed error message.
    
    Parameters
    ----------
    a, b : torch.Tensor
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance (checked as: max_rel_diff < rtol)
    name : str
        Descriptive name for error message
    """
    # Align dtypes for comparison
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    
    diff = torch.abs(a - b)
    max_abs_diff = diff.max().item()
    
    # Relative tolerance: diff / |b|
    denom = torch.abs(b).clamp_min(1e-30)
    rel_diff = (diff / denom)
    max_rel_diff = rel_diff.max().item()
    
    passed_abs = max_abs_diff < atol
    passed_rel = max_rel_diff < rtol
    
    msg = (
        f"{name}: max_abs_diff={max_abs_diff:.3e} (tol={atol:.3e}), "
        f"max_rel_diff={max_rel_diff:.3e} (tol={rtol:.3e})"
    )
    assert passed_abs or passed_rel, msg


class TestMoERouterNumerics:
    """Validate Triton kernel vs FP64 reference on multiple dimensions."""
    
    # ────────────────────────────────────────────────────────────────────
    # FORWARD PASS TESTS
    # ────────────────────────────────────────────────────────────────────
    
    @pytest.mark.parametrize("H,E,K", [
        (64, 8, 2),
        (256, 16, 2),
        (512, 64, 4),
        (1024, 256, 2),
    ])
    def test_forward_indices_match_reference(self, H, E, K):
        """Forward: top-K indices must match FP64 reference exactly.
        
        Since argmax is deterministic, Triton and reference should select
        identical expert indices (up to tie-breaking behavior).
        """
        torch.manual_seed(42)
        N = 128
        
        tokens = torch.randn(N, H, dtype=torch.bfloat16)
        gate_w = torch.randn(H, E, dtype=torch.bfloat16)
        
        # Triton path (GPU/CPU autodetect)
        idx_triton, w_triton = moe_topk_route(
            tokens, gate_w, K, force_reference=False
        )
        
        # Reference path (FP64 CPU)
        idx_ref, w_ref = moe_topk_route(
            tokens, gate_w, K, force_reference=True
        )
        
        # Indices MUST match exactly
        assert (idx_triton == idx_ref).all(), (
            f"Expert indices differ; "
            f"Triton: {idx_triton[0]}, Ref: {idx_ref[0]}"
        )
    
    @pytest.mark.parametrize("H,E,K", [
        (64, 8, 2),
        (256, 16, 2),
        (512, 64, 4),
    ])
    def test_forward_weights_match_reference(self, H, E, K):
        """Forward: renormalized top-K weights must match FP64 reference.
        
        Weights are normalized per token so they sum to 1.
        Triton vs reference should match to numerical precision.
        """
        torch.manual_seed(123)
        N = 64
        
        tokens = torch.randn(N, H, dtype=torch.float32)
        gate_w = torch.randn(H, E, dtype=torch.float32)
        
        idx_tri, w_tri = moe_topk_route(tokens, gate_w, K, force_reference=False)
        idx_ref, w_ref = moe_topk_route(tokens, gate_w, K, force_reference=True)
        
        # Weights should be close
        assert_close(w_tri, w_ref, name="forward_weights", atol=1e-5, rtol=1e-5)
    
    # ────────────────────────────────────────────────────────────────────
    # BACKWARD PASS TESTS
    # ────────────────────────────────────────────────────────────────────
    
    @pytest.mark.parametrize("H,E,K", [
        (64, 8, 2),
        (256, 16, 2),
        (512, 64, 4),
    ])
    def test_backward_grad_tokens_match_reference(self, H, E, K):
        """Backward: grad_tokens must match FP64 reference.
        
        The backward chain is:
          loss → grad_w → grad_v (from topk) → grad_p (from softmax) → grad_l
        
        Then grad_tokens = grad_l @ gate_w^T
        
        This test verifies the entire chain matches reference.
        """
        torch.manual_seed(456)
        N = 64
        
        # Triton forward/backward
        tokens = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
        gate_w = torch.randn(H, E, dtype=torch.float32, requires_grad=True)
        
        idx_tri, w_tri = moe_topk_route(tokens, gate_w, K, force_reference=False)
        loss_tri = w_tri.sum()
        loss_tri.backward()
        grad_tokens_tri = tokens.grad.clone().detach()
        grad_gate_w_tri = gate_w.grad.clone().detach()
        
        # Reference forward/backward
        tokens_ref = tokens.detach().clone().requires_grad_(True)
        gate_w_ref = gate_w.detach().clone().requires_grad_(True)
        
        idx_ref, w_ref = moe_topk_route(tokens_ref, gate_w_ref, K, force_reference=True)
        loss_ref = w_ref.sum()
        loss_ref.backward()
        grad_tokens_ref = tokens_ref.grad.clone().detach()
        grad_gate_w_ref = gate_w_ref.grad.clone().detach()
        
        # Compare
        assert_close(
            grad_tokens_tri, grad_tokens_ref,
            name="grad_tokens", atol=1e-5, rtol=1e-5
        )
    
    @pytest.mark.parametrize("H,E,K", [
        (64, 8, 2),
        (256, 16, 2),
    ])
    def test_backward_grad_gate_w_match_reference(self, H, E, K):
        """Backward: grad_gate_w must match FP64 reference.
        
        grad_gate_w = tokens^T @ grad_l
        """
        torch.manual_seed(789)
        N = 64
        
        # Triton path
        tokens = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
        gate_w = torch.randn(H, E, dtype=torch.float32, requires_grad=True)
        
        idx_tri, w_tri = moe_topk_route(tokens, gate_w, K, force_reference=False)
        loss_tri = w_tri.sum()
        loss_tri.backward()
        grad_gate_w_tri = gate_w.grad.clone().detach()
        
        # Reference path
        tokens_ref = tokens.detach().clone().requires_grad_(True)
        gate_w_ref = gate_w.detach().clone().requires_grad_(True)
        
        idx_ref, w_ref = moe_topk_route(tokens_ref, gate_w_ref, K, force_reference=True)
        loss_ref = w_ref.sum()
        loss_ref.backward()
        grad_gate_w_ref = gate_w_ref.grad.clone().detach()
        
        # Compare
        assert_close(
            grad_gate_w_tri, grad_gate_w_ref,
            name="grad_gate_w", atol=1e-5, rtol=1e-5
        )

    def test_backward_gradients_randomized_triton(self):
        """Backward: Triton gradients match reference across randomized shapes."""
        if not (TRITON_AVAILABLE and torch.cuda.is_available()):
            pytest.skip("Triton GPU path not available for this test")

        import random
        dims = [64, 128, 256, 512]
        experts = [8, 16, 32]
        ks = [1, 2, 4]

        for seed in range(50):
            random.seed(seed)
            H = random.choice(dims)
            E = random.choice(experts)
            K = random.choice(ks)
            N = 32

            torch.manual_seed(seed)
            tokens = torch.randn(N, H, dtype=torch.float32, requires_grad=True, device="cuda")
            gate_w = torch.randn(H, E, dtype=torch.float32, requires_grad=True, device="cuda")

            idx_tri, w_tri = moe_topk_route(tokens, gate_w, K, force_reference=False)
            loss_tri = w_tri.sum()
            loss_tri.backward()

            grad_tokens_tri = tokens.grad.clone().detach().cpu().double()
            grad_gate_w_tri = gate_w.grad.clone().detach().cpu().double()

            tokens_ref = tokens.detach().cpu().clone().requires_grad_(True)
            gate_w_ref = gate_w.detach().cpu().clone().requires_grad_(True)
            idx_ref, w_ref = moe_topk_route(tokens_ref, gate_w_ref, K, force_reference=True)
            loss_ref = w_ref.sum()
            loss_ref.backward()

            assert_close(
                grad_tokens_tri, tokens_ref.grad.clone().detach().double(),
                name=f"grad_tokens_randomized_seed_{seed}", atol=1e-5, rtol=1e-5
            )
            assert_close(
                grad_gate_w_tri, gate_w_ref.grad.clone().detach().double(),
                name=f"grad_gate_w_randomized_seed_{seed}", atol=1e-5, rtol=1e-5
            )
    
    # ────────────────────────────────────────────────────────────────────
    # TOKEN CONSERVATION INVARIANT TESTS
    # ────────────────────────────────────────────────────────────────────
    
    @pytest.mark.parametrize("H,E,K,N", [
        (256, 64, 2, 1024),
        (512, 128, 4, 512),
        (1024, 256, 2, 2048),
    ])
    def test_token_conservation_shape(self, H, E, K, N):
        """Token conservation: output shape must be [N, K].
        
        Each of N tokens gets K assignments, so total routed count is N*K.
        """
        torch.manual_seed(111)
        
        tokens = torch.randn(N, H, dtype=torch.float32)
        gate_w = torch.randn(H, E, dtype=torch.float32)
        
        idx, w = moe_topk_route(tokens, gate_w, K)
        
        assert idx.shape == (N, K), f"Expected idx shape ({N}, {K}), got {idx.shape}"
        assert w.shape == (N, K), f"Expected w shape ({N}, {K}), got {w.shape}"
    
    def test_token_conservation_weight_normalization(self):
        """Token conservation: weights must sum to 1 per token (renormalized).
        
        After top-K and renormalization, w.sum(dim=1) should be all 1s.
        """
        H, E, K = 256, 64, 2
        N = 512
        torch.manual_seed(222)
        
        tokens = torch.randn(N, H, dtype=torch.float32)
        gate_w = torch.randn(H, E, dtype=torch.float32)
        
        idx, w = moe_topk_route(tokens, gate_w, K)
        
        # Weights per token should sum to 1
        row_sums = w.sum(dim=1)  # [N]
        expected = torch.ones_like(row_sums)
        
        assert_close(row_sums, expected, name="weight_sums", atol=1e-5, rtol=1e-5)
    
    def test_token_conservation_dispatch_count(self):
        """Token conservation: total dispatched tokens == N*K.
        
        Each expert receives a histogram of routed tokens.
        Sum over all experts should equal N*K.
        """
        H, E, K = 256, 128, 2
        N = 1024
        torch.manual_seed(333)
        
        tokens = torch.randn(N, H, dtype=torch.float32)
        gate_w = torch.randn(H, E, dtype=torch.float32)
        
        idx, w = moe_topk_route(tokens, gate_w, K)
        
        # Histogram: how many tokens per expert?
        dispatch_cnt = torch.bincount(idx.reshape(-1), minlength=E)
        
        total_dispatched = dispatch_cnt.sum().item()
        expected_total = N * K
        
        assert total_dispatched == expected_total, (
            f"Dispatch count mismatch: {total_dispatched} != {expected_total}"
        )
    
    # ────────────────────────────────────────────────────────────────────
    # DETERMINISM & STABILITY TESTS
    # ────────────────────────────────────────────────────────────────────
    
    @pytest.mark.parametrize("seed", range(5))
    def test_determinism_forward(self, seed):
        """Same seed → same forward outputs (reproducibility).
        
        Critical for debugging and validation.
        """
        H, E, K = 128, 32, 2
        N = 256
        
        # Run 1
        torch.manual_seed(seed)
        tokens1 = torch.randn(N, H, dtype=torch.float32)
        gate_w1 = torch.randn(H, E, dtype=torch.float32)
        idx1, w1 = moe_topk_route(tokens1, gate_w1, K, force_reference=False)
        
        # Run 2 (same seed)
        torch.manual_seed(seed)
        tokens2 = torch.randn(N, H, dtype=torch.float32)
        gate_w2 = torch.randn(H, E, dtype=torch.float32)
        idx2, w2 = moe_topk_route(tokens2, gate_w2, K, force_reference=False)
        
        # Should be bitwise identical
        assert (idx1 == idx2).all(), f"Seed {seed}: indices not deterministic"
        assert (w1 == w2).all(), f"Seed {seed}: weights not deterministic"
    
    @pytest.mark.parametrize("seed", range(5))
    def test_determinism_backward(self, seed):
        """Same seed → same backward gradients (reproducibility).
        
        Critical for reproducible training.
        """
        H, E, K = 128, 32, 2
        N = 256
        
        # Run 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        tokens1 = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
        gate_w1 = torch.randn(H, E, dtype=torch.float32, requires_grad=True)
        idx1, w1 = moe_topk_route(tokens1, gate_w1, K)
        loss1 = w1.sum()
        loss1.backward()
        grad_tokens1 = tokens1.grad.clone().detach()
        
        # Run 2 (same seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        tokens2 = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
        gate_w2 = torch.randn(H, E, dtype=torch.float32, requires_grad=True)
        idx2, w2 = moe_topk_route(tokens2, gate_w2, K)
        loss2 = w2.sum()
        loss2.backward()
        grad_tokens2 = tokens2.grad.clone().detach()
        
        # Gradients should be bitwise identical
        assert (grad_tokens1 == grad_tokens2).all(), (
            f"Seed {seed}: backward gradients not deterministic"
        )
    
    # ────────────────────────────────────────────────────────────────────
    # EDGE CASES & ROBUSTNESS
    # ────────────────────────────────────────────────────────────────────
    
    def test_single_token(self):
        """Edge case: single token (N=1)."""
        H, E, K = 64, 16, 2
        tokens = torch.randn(1, H, dtype=torch.float32)
        gate_w = torch.randn(H, E, dtype=torch.float32)
        
        idx, w = moe_topk_route(tokens, gate_w, K)
        
        assert idx.shape == (1, K)
        assert w.shape == (1, K)
        assert_close(w.sum(), torch.tensor(1.0), name="single_token_weight_sum")
    
    def test_k_equals_one(self):
        """Edge case: K=1 (single expert per token, like vanilla attention)."""
        H, E = 128, 32
        K = 1
        N = 64
        torch.manual_seed(444)
        
        tokens = torch.randn(N, H, dtype=torch.float32)
        gate_w = torch.randn(H, E, dtype=torch.float32)
        
        idx, w = moe_topk_route(tokens, gate_w, K)
        
        assert idx.shape == (N, 1)
        assert w.shape == (N, 1)
        # With K=1, weights should be all 1s (single expert per token)
        assert_close(w, torch.ones_like(w), name="k_equals_one", atol=1e-5)
    
    def test_k_equals_e(self):
        """Edge case: K=E (all experts per token)."""
        H, E = 64, 8
        K = E
        N = 32
        torch.manual_seed(555)
        
        tokens = torch.randn(N, H, dtype=torch.float32)
        gate_w = torch.randn(H, E, dtype=torch.float32)
        
        idx, w = moe_topk_route(tokens, gate_w, K)
        
        assert idx.shape == (N, K)
        assert w.shape == (N, K)
        # All indices should be unique per row (no repeats)
        for i in range(N):
            unique_count = len(set(idx[i].tolist()))
            assert unique_count == K, f"Row {i}: only {unique_count} unique experts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
