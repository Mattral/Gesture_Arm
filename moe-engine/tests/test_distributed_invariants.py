"""
Distributed invariants tests (Week 2)

- test_token_conservation_distributed: launches a 4-process Gloo world
  runs DistributedMoELayer across ranks and verifies total dispatched tokens == N*K

- test_distributed_backward_no_nan: runs a forward/backward and ensures
  no NaN gradients and that gradient norm is finite on all ranks.

These tests are designed to run on CPU via Gloo so CI can validate them.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pkg.distributed.parallel_mesh import DistributedMoELayer, ParallelTopology


def _run_worker(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # Build EP process-groups so all_to_all helpers can target the EP axis.
    ep_size = 2
    dp_size = world_size // ep_size

    # Create EP groups: ranks with same ep_rank are grouped together
    ep_groups = []
    for ep_idx in range(ep_size):
        ranks = [r for r in range(world_size) if (r % ep_size) == ep_idx]
        ep_groups.append(dist.new_group(ranks=ranks))

    class _SimpleMesh:
        def __init__(self, groups, ep_size, rank):
            self._groups = groups
            self._ep_size = ep_size
            self._rank = rank

        def __getitem__(self, key):
            if key == 'ep':
                return self
            raise KeyError(key)

        def get_group(self):
            return self._groups[self._rank % self._ep_size]

    mesh = _SimpleMesh(ep_groups, ep_size, rank)

    topo = ParallelTopology(world_size=world_size, rank=rank, dp_size=dp_size, ep_size=ep_size, mesh=mesh, device=torch.device('cpu'))

    # Small layer
    H = 64
    F = 128
    E = 8
    K = 2
    B = 2
    S = 4

    layer = DistributedMoELayer(hidden_dim=H, ffn_dim=F, num_experts=E, top_k=K, topology=topo, dtype=torch.float32)

    # Synthetic local input for this rank
    # Each rank has B*S tokens locally; total N_total = B*S*world_size
    local_B = B
    local_S = S
    tokens = torch.randn(local_B, local_S, H, dtype=torch.float32)

    # Forward
    out = layer(tokens)

    # Verify no NaNs in output
    assert not torch.isnan(out).any(), f"Rank {rank}: NaN in output"

    # Gather dispatch counts from router (best-effort): router returns dispatch cnt per local expert
    # Note: router is inside layer.router
    # For safety, call router on flattened tokens to extract dispatch counts
    flat = tokens.reshape(local_B * local_S, H)
    idx, w, dispatch_cnt = layer.router(flat)

    # Sum per-rank dispatched tokens
    local_dispatched = dispatch_cnt.sum().item()
    tensor_local_dispatched = torch.tensor([local_dispatched], dtype=torch.long)

    # All-reduce to compute total dispatched across ranks
    tensor_total = tensor_local_dispatched.clone()
    dist.all_reduce(tensor_total, op=dist.ReduceOp.SUM)

    total_dispatched = int(tensor_total.item())
    expected_total = (local_B * local_S) * K * world_size

    assert total_dispatched == expected_total, (
        f"Total dispatched tokens mismatch: {total_dispatched} != {expected_total}"
    )

    # Backward sanity: create a loss and backward; ensure no NaN gradients on parameters
    loss = out.abs().sum()
    loss.backward()

    # Check parameters for NaN gradients
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Rank {rank}: Non-finite grad in {name}"

    # cleanup
    dist.destroy_process_group()


def test_token_conservation_distributed():
    world_size = 4
    # use a port unlikely to be in use
    port = 29500
    mp.spawn(_run_worker, args=(world_size, port), nprocs=world_size, join=True)


def test_distributed_backward_no_nan():
    # Reuse the same worker function which performs forward+backward sanity
    world_size = 4
    port = 29501
    mp.spawn(_run_worker, args=(world_size, port), nprocs=world_size, join=True)
