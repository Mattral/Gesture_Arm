#!/usr/bin/env python3
"""
Standalone numerics test runner for moe_router.py

This script validates Triton kernel correctness WITHOUT requiring pytest,
making it suitable for quick CI validation or local testing.

Exit codes:
  0 = all tests passed
  1 = at least one test failed
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the repo root to sys.path so imports work correctly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pkg.kernels.moe_router import moe_topk_route


def assert_close(a, b, atol=1e-5, rtol=1e-5, name=""):
    """Assert tensors are close with detailed error message."""
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    
    diff = torch.abs(a - b)
    max_abs_diff = diff.max().item()
    
    denom = torch.abs(b).clamp_min(1e-30)
    rel_diff = (diff / denom).max().item()
    
    passed_abs = max_abs_diff < atol
    passed_rel = rel_diff < rtol
    
    if not (passed_abs or passed_rel):
        msg = (
            f"FAIL [{name}]: max_abs_diff={max_abs_diff:.3e} (tol={atol:.3e}), "
            f"max_rel_diff={rel_diff:.3e} (tol={rtol:.3e})"
        )
        return False, msg
    
    return True, f"PASS [{name}]"


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_names = []
    
    def run_test(self, test_func, *args, **kwargs):
        """Run a single test function."""
        test_name = test_func.__name__
        try:
            test_func(*args, **kwargs)
            self.passed += 1
            self.test_names.append((test_name, "PASS"))
            print(f"  ✓ {test_name}")
        except AssertionError as e:
            self.failed += 1
            self.test_names.append((test_name, "FAIL"))
            print(f"  ✗ {test_name}: {e}")
        except Exception as e:
            self.failed += 1
            self.test_names.append((test_name, "ERROR"))
            print(f"  ✗ {test_name} (ERROR): {e}")
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"  ✗ {self.failed} failures")
        print(f"{'='*60}\n")
        return self.failed == 0


# ────────────────────────────────────────────────────────────────────────────
# TEST FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def test_forward_matches_reference():
    """Forward: Triton output matches FP64 reference."""
    torch.manual_seed(42)
    H, E, K, N = 256, 64, 2, 128
    
    tokens = torch.randn(N, H, dtype=torch.float32)
    gate_w = torch.randn(H, E, dtype=torch.float32)
    
    # Triton path
    idx_tri, w_tri = moe_topk_route(tokens, gate_w, K, force_reference=False)
    
    # Reference path
    idx_ref, w_ref = moe_topk_route(tokens, gate_w, K, force_reference=True)
    
    # Compare
    assert (idx_tri == idx_ref).all(), "Indices mismatch"
    ok, msg = assert_close(w_tri, w_ref, name="weights")
    assert ok, msg


def test_backward_gradients_match_reference():
    """Backward: gradients match FP64 reference."""
    torch.manual_seed(123)
    H, E, K, N = 256, 64, 2, 64
    
    tokens = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
    gate_w = torch.randn(H, E, dtype=torch.float32, requires_grad=True)
    
    # Triton forward/backward
    idx_tri, w_tri = moe_topk_route(tokens, gate_w, K, force_reference=False)
    loss_tri = w_tri.sum()
    loss_tri.backward()
    grad_tokens_tri = tokens.grad.clone().detach()
    
    # Reference forward/backward
    tokens_ref = tokens.detach().clone().requires_grad_(True)
    gate_w_ref = gate_w.detach().clone().requires_grad_(True)
    
    idx_ref, w_ref = moe_topk_route(tokens_ref, gate_w_ref, K, force_reference=True)
    loss_ref = w_ref.sum()
    loss_ref.backward()
    grad_tokens_ref = tokens_ref.grad.clone().detach()
    
    # Compare
    ok, msg = assert_close(grad_tokens_tri, grad_tokens_ref, name="grad_tokens")
    assert ok, msg


def test_token_conservation_shape():
    """Token conservation: output shape must be [N, K]."""
    torch.manual_seed(111)
    H, E, K, N = 512, 128, 4, 1024
    
    tokens = torch.randn(N, H, dtype=torch.float32)
    gate_w = torch.randn(H, E, dtype=torch.float32)
    
    idx, w = moe_topk_route(tokens, gate_w, K)
    
    assert idx.shape == (N, K), f"Expected idx ({N}, {K}), got {idx.shape}"
    assert w.shape == (N, K), f"Expected w ({N}, {K}), got {w.shape}"


def test_weight_normalization():
    """Weights must sum to 1 per token (renormalized)."""
    torch.manual_seed(222)
    H, E, K = 256, 64, 2
    N = 512
    
    tokens = torch.randn(N, H, dtype=torch.float32)
    gate_w = torch.randn(H, E, dtype=torch.float32)
    
    idx, w = moe_topk_route(tokens, gate_w, K)
    
    row_sums = w.sum(dim=1)
    expected = torch.ones_like(row_sums)
    
    ok, msg = assert_close(row_sums, expected, name="weight_sums")
    assert ok, msg


def test_dispatch_count_conservation():
    """Dispatch count: total routed == N*K."""
    torch.manual_seed(333)
    H, E, K = 256, 128, 2
    N = 1024
    
    tokens = torch.randn(N, H, dtype=torch.float32)
    gate_w = torch.randn(H, E, dtype=torch.float32)
    
    idx, w = moe_topk_route(tokens, gate_w, K)
    
    dispatch_cnt = torch.bincount(idx.reshape(-1), minlength=E)
    total_dispatched = dispatch_cnt.sum().item()
    expected_total = N * K
    
    assert total_dispatched == expected_total, (
        f"Dispatch mismatch: {total_dispatched} != {expected_total}"
    )


def test_determinism_forward():
    """Same seed → same forward outputs."""
    H, E, K, N = 128, 32, 2, 256
    
    torch.manual_seed(42)
    tokens1 = torch.randn(N, H, dtype=torch.float32)
    gate_w1 = torch.randn(H, E, dtype=torch.float32)
    idx1, w1 = moe_topk_route(tokens1, gate_w1, K)
    
    torch.manual_seed(42)
    tokens2 = torch.randn(N, H, dtype=torch.float32)
    gate_w2 = torch.randn(H, E, dtype=torch.float32)
    idx2, w2 = moe_topk_route(tokens2, gate_w2, K)
    
    assert (idx1 == idx2).all(), "Indices not deterministic"
    assert (w1 == w2).all(), "Weights not deterministic"


def test_edge_case_k_equals_one():
    """Edge case: K=1."""
    H, E, K = 128, 32, 1
    N = 64
    torch.manual_seed(444)
    
    tokens = torch.randn(N, H, dtype=torch.float32)
    gate_w = torch.randn(H, E, dtype=torch.float32)
    
    idx, w = moe_topk_route(tokens, gate_w, K)
    
    assert idx.shape == (N, 1)
    # With K=1, weights should be all 1s (single expert per token)
    ok, msg = assert_close(w, torch.ones_like(w), name="k_equals_one")
    assert ok, msg


def test_edge_case_single_token():
    """Edge case: N=1 (single token)."""
    H, E, K = 64, 16, 2
    N = 1
    
    tokens = torch.randn(N, H, dtype=torch.float32)
    gate_w = torch.randn(H, E, dtype=torch.float32)
    
    idx, w = moe_topk_route(tokens, gate_w, K)
    
    assert idx.shape == (1, K)
    ok, msg = assert_close(w.sum(), torch.tensor(1.0), name="single_token_sum")
    assert ok, msg


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("MoE Router Numerics Test Suite (Standalone)")
    print("="*60 + "\n")
    
    runner = TestRunner()
    
    # Forward tests
    print("Forward Pass Tests:")
    runner.run_test(test_forward_matches_reference)
    runner.run_test(test_token_conservation_shape)
    runner.run_test(test_weight_normalization)
    runner.run_test(test_dispatch_count_conservation)
    
    # Backward tests
    print("\nBackward Pass Tests:")
    runner.run_test(test_backward_gradients_match_reference)
    
    # Determinism tests
    print("\nDeterminism Tests:")
    runner.run_test(test_determinism_forward)
    
    # Edge case tests
    print("\nEdge Case Tests:")
    runner.run_test(test_edge_case_k_equals_one)
    runner.run_test(test_edge_case_single_token)
    
    # Summary and exit
    success = runner.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
