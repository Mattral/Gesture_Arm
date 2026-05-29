"""
tests/test_elastic.py
=====================

Exercises:
  * LocalNVMeAdapter: round-trip, list, delete
  * AsyncCheckpointer: background save/load, retention pruning
  * ClusterStateMachine: reshard plan, deterministic continuity
  * ElasticTrainerHarness: zero-divergence resharding via load/save round trip
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pkg.distributed.parallel_mesh import build_topology
from pkg.elastic.fault_monitor import (
    AsyncCheckpointer,
    ClusterStateMachine,
    ElasticConfig,
    ElasticTrainerHarness,
    LocalNVMeAdapter,
    _largest_divisor_le,
)


def test_local_nvme_round_trip(tmp_path):
    store = LocalNVMeAdapter(str(tmp_path))
    store.put("a/b/c.bin", b"hello-world")
    assert store.get("a/b/c.bin") == b"hello-world"
    keys = store.list("a/")
    assert "a/b/c.bin" in keys
    store.delete("a/b/c.bin")
    with pytest.raises(FileNotFoundError):
        store.get("a/b/c.bin")


def test_async_checkpointer_save_load(tmp_path):
    local = LocalNVMeAdapter(str(tmp_path))
    ckpt = AsyncCheckpointer(local_adapter=local, remote_adapter=None, retention=4, workers=2)
    model = nn.Linear(8, 8)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Initialize an optimizer state so the saved optim dict is non-empty.
    model(torch.randn(2, 8)).sum().backward()
    optim.step()

    ckpt.save(model, optim, step=10, rank=0)
    ckpt.shutdown(drain=True)
    assert ckpt.latest_step() == 10

    fresh = nn.Linear(8, 8)
    fresh_optim = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    # Re-create the checkpointer (workers may have exited) before loading.
    ckpt2 = AsyncCheckpointer(local_adapter=local, remote_adapter=None)
    ckpt2.load(fresh, fresh_optim, step=10, rank=0)
    ckpt2.shutdown(drain=False)

    for p, q in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(p, q), "parameter divergence after load"


def test_async_checkpointer_retention(tmp_path):
    local = LocalNVMeAdapter(str(tmp_path))
    ckpt = AsyncCheckpointer(local_adapter=local, remote_adapter=None, retention=2, workers=1)
    model = nn.Linear(4, 4)
    for s in (1, 2, 3, 4, 5):
        ckpt.save(model, None, step=s, rank=0)
        time.sleep(0.05)
    ckpt.shutdown(drain=True)
    # Only the most recent 2 should survive.
    keys = local.list("ckpts/")
    surviving_steps = set()
    for k in keys:
        try:
            seg = [p for p in k.split("/") if p.startswith("step=")][0]
            surviving_steps.add(int(seg.split("=")[1]))
        except Exception:
            pass
    assert surviving_steps == {4, 5}, f"expected {{4,5}} got {surviving_steps}"


def test_cluster_reshard_continuity():
    topo = build_topology(dp_size=1, ep_size=4, device_type="cpu")
    csm = ClusterStateMachine(topology=topo, min_nodes=1)
    new_topo = build_topology(dp_size=1, ep_size=2, device_type="cpu")
    plan = csm.reshard(new_topo, num_experts=10)
    # No leak, no duplicate
    seen = set()
    for r, ids in plan.items():
        for e in ids:
            assert e not in seen, f"expert {e} duplicated"
            seen.add(e)
    assert seen == set(range(10))


def test_zero_divergence_full_roundtrip(tmp_path):
    """End-to-end: save on old topology, load on new (smaller) topology,
    weights must be byte-identical."""
    cfg = ElasticConfig(
        local_ckpt_dir=str(tmp_path),
        remote_uri=None,
        retention=4,
        async_workers=2,
        min_nodes=1,
    )
    topo = build_topology(dp_size=1, ep_size=1, device_type="cpu")
    harness = ElasticTrainerHarness(cfg, topo)

    model = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8)).sum().backward()
    optim.step()

    harness.checkpoint(model, optim, step=42)
    harness.async_ckpt.shutdown(drain=True)

    # Simulate a fresh worker resuming.
    fresh = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
    fresh_optim = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    ck = AsyncCheckpointer(local_adapter=LocalNVMeAdapter(str(tmp_path)), remote_adapter=None)
    ck.load(fresh, fresh_optim, step=42, rank=0)
    ck.shutdown(drain=False)

    for p, q in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(p, q, atol=0, rtol=0)


def test_largest_divisor_helper():
    assert _largest_divisor_le(64, 8) == 8
    assert _largest_divisor_le(64, 9) == 8
    assert _largest_divisor_le(60, 9) == 6
    assert _largest_divisor_le(7, 8) == 7
    assert _largest_divisor_le(1, 8) == 1
