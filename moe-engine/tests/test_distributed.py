"""
tests/test_distributed.py
=========================

Single-process exercises of the distributed primitives. These tests
deliberately run in the 1-rank degenerate topology so they can be executed
on any machine; the relevant code paths inside `parallel_mesh.py` no-op
cleanly when world_size==1, so we are testing the *plumbing* (shape
preservation, gradient flow, expert assignment) rather than the actual NCCL
collectives.
"""

from __future__ import annotations

import torch
import pytest

from pkg.distributed.parallel_mesh import (
    DistributedMoELayer,
    ParallelTopology,
    build_topology,
)


def test_topology_singleton_cpu():
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    assert topo.world_size == 1
    assert topo.rank == 0
    assert topo.ep_size == 1
    assert topo.experts_on_this_rank(64) == list(range(64))


def test_topology_expert_split_remainder():
    """Even-split with remainder must keep every expert assigned exactly once."""
    fake = ParallelTopology(
        world_size=4, rank=0, dp_size=1, ep_size=4, mesh=None,
        device=torch.device("cpu"),
    )
    seen = set()
    for r in range(4):
        fake_r = ParallelTopology(
            world_size=4, rank=r, dp_size=1, ep_size=4, mesh=None,
            device=torch.device("cpu"),
        )
        local = fake_r.experts_on_this_rank(10)
        seen.update(local)
    assert seen == set(range(10))


@pytest.mark.parametrize("B,S,H,F,E,K", [
    (2, 8, 32, 64, 4, 2),
    (1, 16, 64, 128, 8, 1),
])
def test_moe_layer_forward_shape_and_grad(B, S, H, F, E, K):
    torch.manual_seed(0)
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    layer = DistributedMoELayer(
        hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K, topology=topo,
    )
    x = torch.randn(B, S, H, requires_grad=True)
    y = layer(x)
    assert y.shape == (B, S, H)
    # Gradient must flow back into the input and the router gate.
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.router.gate_w.grad is not None
    assert torch.isfinite(layer.router.gate_w.grad).all()


def test_expert_to_rank_round_trip():
    topo = ParallelTopology(
        world_size=4, rank=1, dp_size=1, ep_size=4, mesh=None,
        device=torch.device("cpu"),
    )
    layer = DistributedMoELayer(
        hidden_dim=8, ffn_dim=16, num_experts=10, top_k=2, topology=topo,
    )
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    ranks = layer._expert_to_rank(ids)
    # Round-trip: every expert id should belong to its declared rank.
    for r in range(4):
        owned = ParallelTopology(
            world_size=4, rank=r, dp_size=1, ep_size=4, mesh=None,
            device=torch.device("cpu"),
        ).experts_on_this_rank(10)
        for e in owned:
            assert ranks[e].item() == r, (
                f"expert {e}: bucketize says rank {ranks[e].item()}, "
                f"expected {r}"
            )
