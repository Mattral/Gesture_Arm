"""
pkg/distributed/parallel_mesh.py
================================

Multi-dimensional distributed topology for hyperscale MoE training.

Combines:

  * **PyTorch 2.5+ native `init_device_mesh`** to construct a 2D DeviceMesh
    `(dp, ep)`. Slots are reserved (and documented) for a future 4D mesh
    `(dp, tp, pp, ep)` once Tensor and Pipeline parallelism are wired in.
  * **FSDP2** (`torch.distributed._composable.fsdp.fully_shard`) for
    per-parameter sharded DTensor data-parallelism. No FlatParameter, no
    monolithic FSDP1 wrapper – pure DTensor sharding along the `dp` axis.
  * **Expert Parallelism** via non-blocking `all_to_all_single` collectives
    on a dedicated `torch.cuda.Stream`, ensuring routed-token transit is
    overlapped with the compute of preceding/following local-expert FFNs.

Token life-cycle in a single MoE layer
--------------------------------------
       (Router on every rank)
   tokens [N_local, H] --MoERouter--> (idx [N_local, K], weights [N_local, K])
   sort tokens by expert id            tokens_sorted [N_local*K, H]
                |
    all_to_all_dispatch (EP)
                v
   experts compute locally             expert_out  [N_local*K, H]
                |
    all_to_all_combine (EP)
                v
   scatter back to original positions  combined    [N_local, H]
                |
   weight by combine weights, sum over K slots

All shapes are documented inline at every transformation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

# ---- PyTorch 2.5+ distributed primitives. Imported lazily so the module is
# importable in single-process / CPU-only test environments. ---------------
try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    _HAS_DEVICE_MESH = True
except Exception:                                                        # pragma: no cover
    init_device_mesh = None                                              # type: ignore
    DeviceMesh = object                                                  # type: ignore
    _HAS_DEVICE_MESH = False

try:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    _HAS_FSDP2 = True
except Exception:                                                        # pragma: no cover
    fully_shard = None                                                   # type: ignore
    MixedPrecisionPolicy = None                                          # type: ignore
    _HAS_FSDP2 = False

from pkg.kernels.moe_router import MoERouter


# ==========================================================================
# Topology descriptor – immutable record of the current mesh slice. Recomputed
# (not mutated) by the elastic state-machine when nodes drop/rejoin.
# ==========================================================================
@dataclass(frozen=True)
class ParallelTopology:
    world_size: int
    rank: int
    dp_size: int
    ep_size: int
    tp_size: int = 1
    pp_size: int = 1
    mesh: Optional[DeviceMesh] = field(default=None, compare=False, repr=False)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def dp_rank(self) -> int:
        return (self.rank // self.ep_size) % self.dp_size

    @property
    def ep_rank(self) -> int:
        return self.rank % self.ep_size

    def experts_on_this_rank(self, total_experts: int) -> List[int]:
        """Return the list of *global* expert indices owned by this EP rank.

        Even-divides experts across the EP axis; the *remainder* experts are
        round-robin-assigned to the lowest EP ranks so resharding after a
        node drop never leaves experts orphaned.
        """
        per_rank = total_experts // self.ep_size
        rem = total_experts - per_rank * self.ep_size
        start = self.ep_rank * per_rank + min(self.ep_rank, rem)
        extra = 1 if self.ep_rank < rem else 0
        return list(range(start, start + per_rank + extra))


def build_topology(
    dp_size: int,
    ep_size: int,
    device_type: str = "cuda",
) -> ParallelTopology:
    """Initialize (or query) the process group and return a Topology.

    Falls back to a degenerate 1-rank topology on CPU-only systems so the
    rest of the code path (and the test suite) can run anywhere.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if world_size == 1 or not _HAS_DEVICE_MESH:
        # Degenerate path used by tests and CPU smoke runs.
        dev = torch.device(device_type if torch.cuda.is_available() and device_type == "cuda" else "cpu")
        return ParallelTopology(
            world_size=1, rank=0, dp_size=1, ep_size=1, mesh=None, device=dev,
        )

    assert dp_size * ep_size == world_size, (
        f"dp_size({dp_size}) * ep_size({ep_size}) must equal world_size({world_size})"
    )
    mesh = init_device_mesh(
        device_type,
        (dp_size, ep_size),
        mesh_dim_names=("dp", "ep"),
    )
    dev = torch.device(f"{device_type}:{rank % max(torch.cuda.device_count(), 1)}")
    return ParallelTopology(
        world_size=world_size,
        rank=rank,
        dp_size=dp_size,
        ep_size=ep_size,
        mesh=mesh,
        device=dev,
    )


# ==========================================================================
# Dedicated CUDA stream for EP collectives.
# ==========================================================================
class _CommStream:
    """Singleton-per-device stream used for all EP all-to-alls.

    Forward compute runs on the default stream. The router's output is the
    last thing computed on the default stream before the dispatch all-to-all
    is issued on this stream; an event records the dependency. While the
    collective is in flight, the default stream is free to run unrelated
    work (e.g. the *previous* layer's combine, or auxiliary loss
    computation). Combine is issued on this stream too and produces an
    event that the default stream waits on before consuming the result.
    """

    _streams: dict = {}

    @classmethod
    def get(cls, device: torch.device) -> "torch.cuda.Stream | None":
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        idx = device.index if device.index is not None else 0
        if idx not in cls._streams:
            cls._streams[idx] = torch.cuda.Stream(device=idx, priority=-1)
        return cls._streams[idx]


# ==========================================================================
# All-to-all helpers. These wrap `dist.all_to_all_single` so that:
#   * they no-op cleanly on a 1-rank world (test path),
#   * they always execute on the dedicated comm stream when CUDA is up,
#   * they return an event the caller can use to overlap downstream work.
# ==========================================================================
def all_to_all_dispatch(
    tokens_sorted: torch.Tensor,       # [N_local*K, H] sorted by expert id
    send_counts: torch.Tensor,         # [ep_size]      int64
    topology: ParallelTopology,
) -> Tuple[torch.Tensor, torch.Tensor, "torch.cuda.Event | None"]:
    """Dispatch sorted tokens to their assigned EP ranks.

    Returns
    -------
    received    : [N_recv, H]      tokens this rank now owns
    recv_counts : [ep_size] int64  counts per source rank
    event       : CUDA event recording completion of the collective
                  (None on CPU).
    """
    if topology.ep_size == 1 or not dist.is_initialized():
        return tokens_sorted, send_counts.clone(), None

    ep_group = topology.mesh["ep"].get_group() if topology.mesh is not None else None
    stream = _CommStream.get(topology.device)

    # Exchange send_counts -> recv_counts via a tiny all_to_all_single.
    recv_counts = torch.empty_like(send_counts)
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    else:
        dist.all_to_all_single(recv_counts, send_counts, group=ep_group)

    # Allocate the receive buffer now that we know the total inbound count.
    total_recv = int(recv_counts.sum().item())
    H = tokens_sorted.shape[1]
    received = torch.empty(
        (total_recv, H), dtype=tokens_sorted.dtype, device=tokens_sorted.device,
    )

    if stream is not None:
        with torch.cuda.stream(stream):
            dist.all_to_all_single(
                received, tokens_sorted,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts.tolist(),
                group=ep_group,
            )
            event = torch.cuda.Event()
            event.record(stream)
        return received, recv_counts, event

    dist.all_to_all_single(
        received, tokens_sorted,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=ep_group,
    )
    return received, recv_counts, None


def all_to_all_combine(
    expert_out: torch.Tensor,          # [N_recv, H]
    recv_counts: torch.Tensor,         # [ep_size] – from dispatch
    send_counts: torch.Tensor,         # [ep_size] – original send sizes
    topology: ParallelTopology,
) -> Tuple[torch.Tensor, "torch.cuda.Event | None"]:
    """Reverse permutation: send expert outputs back to their origin ranks."""
    if topology.ep_size == 1 or not dist.is_initialized():
        return expert_out, None

    ep_group = topology.mesh["ep"].get_group() if topology.mesh is not None else None
    stream = _CommStream.get(topology.device)
    total_send = int(send_counts.sum().item())
    H = expert_out.shape[1]
    out = torch.empty(
        (total_send, H), dtype=expert_out.dtype, device=expert_out.device,
    )

    if stream is not None:
        with torch.cuda.stream(stream):
            dist.all_to_all_single(
                out, expert_out,
                output_split_sizes=send_counts.tolist(),
                input_split_sizes=recv_counts.tolist(),
                group=ep_group,
            )
            event = torch.cuda.Event()
            event.record(stream)
        return out, event

    dist.all_to_all_single(
        out, expert_out,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
        group=ep_group,
    )
    return out, None


# ==========================================================================
# Local expert implementation: a 2-layer SwiGLU FFN. Compact, BF16-friendly.
# ==========================================================================
class _SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False, dtype=dtype)
        self.w_up   = nn.Linear(hidden_dim, ffn_dim, bias=False, dtype=dtype)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # [n, H] -> [n, H]
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


# ==========================================================================
# The headline DistributedMoELayer.
# ==========================================================================
class DistributedMoELayer(nn.Module):
    """A complete distributed MoE layer.

    Composed of:
      * a top-K router (custom Triton kernel),
      * `len(local_expert_ids)` local SwiGLU expert FFNs,
      * dispatch / combine all-to-all collectives on a dedicated stream.

    Parameters along the EP axis are *not* sharded across DP – each EP rank
    owns its experts in full. Parameters that ARE replicated across EP
    (router gate, non-MoE layers) are sharded along DP via FSDP2.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        topology: ParallelTopology,
        capacity_factor: float = 1.25,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.topology = topology

        self.router = MoERouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dtype=dtype,
        )

        local_ids = topology.experts_on_this_rank(num_experts)
        self.local_expert_ids: List[int] = local_ids
        self.experts = nn.ModuleList([
            _SwiGLUExpert(hidden_dim, ffn_dim, dtype=dtype) for _ in local_ids
        ])
        # Mapping global_expert_id -> local index. -1 means "not on this rank".
        self.register_buffer(
            "_global_to_local",
            torch.full((num_experts,), -1, dtype=torch.long),
            persistent=False,
        )
        for li, gi in enumerate(local_ids):
            self._global_to_local[gi] = li

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, S, H]   ->   [B, S, H] (after MoE FFN with residual NOT applied)."""
        B, S, H = x.shape
        N = B * S
        K = self.top_k
        ep = self.topology.ep_size
        device = x.device
        flat = x.reshape(N, H)                                           # [N, H]

        # 1. Route.
        idx, w, _ = self.router(flat)                                    # idx:[N,K] w:[N,K]
        # Replicate tokens K times along the slot axis: each token-slot pair
        # is one routing event.
        tokens_rep = flat.unsqueeze(1).expand(N, K, H).reshape(N * K, H) # [N*K, H]
        flat_idx = idx.reshape(N * K)                                    # [N*K]
        flat_w = w.reshape(N * K)                                        # [N*K]

        # 2. Sort by *target EP rank* (not by expert id, which would still
        #    work but produces unnecessary intra-rank reshuffles). Each
        #    expert lives on exactly one EP rank, computed via global_to_local
        #    on *every* rank by mirroring the mod-arithmetic of
        #    `experts_on_this_rank`. We pre-compute the mapping below.
        target_rank = self._expert_to_rank(flat_idx)                     # [N*K]
        sort_order = torch.argsort(target_rank, stable=True)
        tokens_sorted = tokens_rep[sort_order]                           # [N*K, H]
        idx_sorted = flat_idx[sort_order]                                # [N*K]
        w_sorted = flat_w[sort_order]                                    # [N*K]
        # Used at combine-time to restore the original order.
        inverse_order = torch.argsort(sort_order)

        # Per-EP-rank send counts.
        send_counts = torch.bincount(target_rank, minlength=ep).to(torch.int64)

        # 3. Dispatch tokens (overlapped with combine of previous layer in async loops).
        received, recv_counts, ev_disp = all_to_all_dispatch(
            tokens_sorted, send_counts, self.topology,
        )

        # 4. Exchange expert IDs alongside tokens (so receiving rank knows
        #    which local expert each token belongs to).
        # Note: Expert IDs are sent via a separate small all_to_all_single call.
        ids_to_send = idx_sorted.to(torch.int64)                          # [N*K]
        ids_recv = self._exchange_ids(ids_to_send, send_counts, recv_counts)

        # 4b. Compute expert outputs: route each token to its assigned local expert.
        # Verify ids_recv matches expected shape for safety.
        assert ids_recv.shape[0] == received.shape[0], (
            f"Expert ID mismatch: got {ids_recv.shape[0]} ids but "
            f"{received.shape[0]} tokens"
        )
        
        expert_out = torch.empty_like(received)
        for li, gi in enumerate(self.local_expert_ids):
            mask = ids_recv == gi
            if mask.any():
                sel = received[mask]
                expert_out[mask] = self.experts[li](sel)

        # 5. Combine (reverse all-to-all).
        if ev_disp is not None and torch.cuda.is_available():
            # Make sure expert compute (default stream) is finished before
            # the comm stream starts the combine.
            ev_local = torch.cuda.Event()
            ev_local.record()
            stream = _CommStream.get(device)
            if stream is not None:
                stream.wait_event(ev_local)

        combined, ev_comb = all_to_all_combine(
            expert_out, recv_counts, send_counts, self.topology,
        )

        if ev_comb is not None:
            torch.cuda.current_stream().wait_event(ev_comb)

        # 6. Un-sort and weight by combine weights.
        unsorted = combined[inverse_order]                               # [N*K, H]
        weighted = unsorted * w_sorted.unsqueeze(-1).to(unsorted.dtype)  # [N*K, H]
        out = weighted.view(N, K, H).sum(dim=1)                          # [N, H]
        
        # ===== TOKEN CONSERVATION INVARIANT (post-combine check) =====
        # After dispatch + compute + combine, we should have exactly [N, H] outputs.
        # If shape is wrong, tokens were lost or duplicated in all-to-all.
        assert out.shape == (N, H), (
            f"Combine shape mismatch: expected ({N}, {H}), got {out.shape}. "
            f"Indicates token loss/duplication in all-to-all collectives."
        )
        # Verify no NaN/Inf introduced by expert compute or combine.
        assert not torch.isnan(out).any(), (
            "NaN detected after combine; expert compute or collective may have failed."
        )
        
        return out.view(B, S, H)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _expert_to_rank(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized: global expert id -> owning EP rank.

        Mirrors `experts_on_this_rank`'s remainder-aware split.
        """
        E = self.num_experts
        ep = self.topology.ep_size
        per_rank = E // ep
        rem = E - per_rank * ep
        # boundaries[r] = start expert id of rank r. Length ep+1.
        boundaries = torch.tensor(
            [r * per_rank + min(r, rem) for r in range(ep + 1)],
            device=expert_ids.device, dtype=torch.long,
        )
        boundaries[-1] = E
        # We want rank r such that boundaries[r] <= expert_id < boundaries[r+1].
        # Using starts-only (boundaries[1:]) with right=True returns the first
        # index i s.t. starts[i] > expert_id, which is exactly the owning rank.
        ranks = torch.bucketize(expert_ids, boundaries[1:], right=True)
        return ranks.clamp_max(ep - 1)

    def _exchange_ids(
        self,
        ids_sorted: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Small companion all_to_all_single carrying the int64 expert ids."""
        if self.topology.ep_size == 1 or not dist.is_initialized():
            return ids_sorted
        ep_group = (
            self.topology.mesh["ep"].get_group()
            if self.topology.mesh is not None else None
        )
        total_recv = int(recv_counts.sum().item())
        out = torch.empty((total_recv,), dtype=torch.int64, device=ids_sorted.device)
        dist.all_to_all_single(
            out, ids_sorted.contiguous(),
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=ep_group,
        )
        return out


# ==========================================================================
# FSDP2 helper – applies `fully_shard` to every replicated submodule along
# the `dp` axis of the mesh. MoE expert weights are *intentionally* skipped:
# they are already partitioned across the `ep` axis.
# ==========================================================================
def apply_fsdp2(
    model: nn.Module,
    topology: ParallelTopology,
    mixed_precision_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> nn.Module:
    """Apply per-parameter DTensor sharding along the DP axis."""
    if not _HAS_FSDP2 or topology.mesh is None or topology.dp_size == 1:
        return model  # No-op for single-GPU / test runs.

    dp_mesh = topology.mesh["dp"]
    mp_policy = (
        MixedPrecisionPolicy(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=torch.float32,
        ) if mixed_precision_dtype is not None else None
    )
    # Apply leaf-first (inner) then to the root – the FSDP2 idiomatic order
    # so that each transformer block becomes its own communication unit.
    for name, module in model.named_modules():
        # Heuristic: shard every direct child of the model that is not the
        # full DistributedMoELayer (which contains EP-sharded experts we
        # must NOT touch with FSDP).
        if isinstance(module, DistributedMoELayer):
            # Shard only the router (replicated across EP) along DP.
            fully_shard(module.router, mesh=dp_mesh, mp_policy=mp_policy)
        elif isinstance(module, (nn.Linear, nn.LayerNorm)) and name:
            fully_shard(module, mesh=dp_mesh, mp_policy=mp_policy)
    # Finally shard the root for residual params and to enable root-level
    # FSDP hooks (gradient hooks, pre-forward, etc.).
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    return model
