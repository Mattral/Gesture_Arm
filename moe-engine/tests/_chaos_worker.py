"""
tests/_chaos_worker.py
======================

Worker entry point invoked by ``torchrun`` from the chaos test suite.
Deliberately *not* prefixed with ``test_`` so pytest's collector skips it.

Responsibilities
----------------
* Bootstrap a Gloo CPU process group from torchrun env vars.
* Drive a tiny deterministic optimizer loop that exercises:
    - ``pkg.elastic.fault_monitor.LocalNVMeAdapter`` + ``AsyncCheckpointer``
    - ``torch.distributed.all_reduce`` (so the PG is genuinely live)
* Emit per-rank, per-generation JSONL telemetry that the pytest driver
  parses to verify the three mathematical invariants:
    1. Monotonic step progression  (within each generation)
    2. Token conservation          (cum_tokens == total_steps * tokens_per_step)
    3. Checksum identity           (SHA256 + ``torch.testing.assert_close``)
* On Scenario A, the targeted local rank ``SIGKILL``s itself mid-run; the
  surrounding torchelastic agent restarts the cohort and the worker
  resumes from the latest sharded snapshot.
* On Scenario B, ``LocalNVMeAdapter.put`` is monkey-patched once to inject
  a multi-second latency spike at a configured step, validating the
  async checkpoint queue does not deadlock.

Environment contract (set by tests/test_chaos.py before torchrun spawn)
----------------------------------------------------------------------
``CHAOS_WORK_DIR``        absolute path to a per-test scratch dir
``CHAOS_SCENARIO``        ``"A"`` or ``"B"``
``CHAOS_TOTAL_STEPS``     int
``CHAOS_TOKENS_PER_STEP`` int
``CHAOS_KILL_STEP``       step index at which rank=KILL_RANK self-kills (A)
``CHAOS_KILL_RANK``       local rank to terminate (A)
``CHAOS_LATENCY_STEP``    step at which storage put() stalls (B)
``CHAOS_LATENCY_SECONDS`` float magnitude of the injected stall (B)
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkg.elastic.fault_monitor import AsyncCheckpointer, LocalNVMeAdapter  # noqa: E402


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------
def _model_sha256(model: nn.Module) -> str:
    """Stable SHA256 of a model's state-dict bytes (key-sorted, contiguous)."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k].detach().to("cpu").contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _state_dict_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Detached CPU clone of every parameter — for assert_close comparisons."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def _build_model() -> nn.Module:
    """Tiny deterministic model. Same seed across ranks/generations so an
    untouched cold-start would yield identical SHA256, but the optimizer
    loop quickly drifts each rank's copy and FSDP-style sync is achieved
    by all-reducing gradients.
    """
    torch.manual_seed(1234)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )


def _detect_generation(rank: int, work: Path) -> int:
    """Generation == number of prior crashes for THIS rank, derived from
    on-disk telemetry. Robust to whether torchelastic exports
    ``TORCHELASTIC_RESTART_COUNT``.
    """
    tele_dir = work / "telemetry"
    if not tele_dir.exists():
        return 0
    prior = list(tele_dir.glob(f"rank-{rank:02d}-gen-*.jsonl"))
    return len(prior)


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    work = Path(os.environ["CHAOS_WORK_DIR"]).resolve()
    work.mkdir(parents=True, exist_ok=True)

    scenario = os.environ.get("CHAOS_SCENARIO", "A").upper()
    total_steps = int(os.environ.get("CHAOS_TOTAL_STEPS", "10"))
    tokens_per_step = int(os.environ.get("CHAOS_TOKENS_PER_STEP", "8"))
    kill_step = int(os.environ.get("CHAOS_KILL_STEP", "3"))
    kill_rank = int(os.environ.get("CHAOS_KILL_RANK", "2"))
    latency_step = int(os.environ.get("CHAOS_LATENCY_STEP", "4"))
    latency_seconds = float(os.environ.get("CHAOS_LATENCY_SECONDS", "10.0"))

    generation = _detect_generation(rank, work)
    tele_dir = work / "telemetry"
    tele_dir.mkdir(parents=True, exist_ok=True)
    tele_path = tele_dir / f"rank-{rank:02d}-gen-{generation:02d}.jsonl"
    tele = tele_path.open("a", buffering=1)

    def emit(**rec: object) -> None:
        rec.update({
            "rank": rank,
            "world": world,
            "generation": generation,
            "ts": time.time(),
        })
        tele.write(json.dumps(rec, default=str) + "\n")
        tele.flush()
        try:
            os.fsync(tele.fileno())
        except OSError:
            pass

    emit(event="boot", local_rank=local_rank, scenario=scenario,
         pid=os.getpid(), torch_version=torch.__version__)

    # ----- process group ------------------------------------------------
    # Gloo can sometimes fail to connect transiently in containerized
    # or heavily-loaded CI environments. Retry a few times before
    # giving up so the chaos test exercises the recovery logic instead
    # of flaky infra failures.
    max_init_attempts = 6
    for attempt in range(1, max_init_attempts + 1):
        try:
            dist.init_process_group(backend="gloo", init_method="env://")
            emit(event="pg_ready")
            break
        except Exception as e:
            emit(event="pg_init_retry", attempt=attempt, error=str(e))
            if attempt == max_init_attempts:
                raise
            time.sleep(0.5 * attempt)

    # ----- model + optimizer + checkpointer ----------------------------
    model = _build_model()
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    ckpt_dir = work / "ckpts"
    local = LocalNVMeAdapter(str(ckpt_dir))

    # Scenario B: monkey-patch put() to stall once at latency_step. We do
    # this before constructing the AsyncCheckpointer so its worker thread
    # picks up the patched method via the adapter handle.
    if scenario == "B":
        original_put = local.put
        marker_dir = work / ".latency_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)
        target_seg = f"step={latency_step:06d}"

        def _stalling_put(key: str, payload: bytes) -> None:
            marker = marker_dir / f"rank-{rank:02d}-gen-{generation:02d}"
            if (target_seg in key) and (not marker.exists()):
                marker.touch()
                emit(event="latency_inject", step=latency_step,
                     seconds=latency_seconds, key=key)
                time.sleep(latency_seconds)
            original_put(key, payload)

        local.put = _stalling_put  # type: ignore[method-assign]

    ckpt = AsyncCheckpointer(local, None, retention=8, workers=1)

    # ----- optional resume ---------------------------------------------
    start_step = 0
    latest = ckpt.latest_step()
    if latest is not None:
        ckpt.load(model, optim, latest, rank=rank)

        # ---- Checksum Identity invariant on the resume boundary -------
        # Compare against the SHA256 emitted at save-time (recorded under
        # work/checksums/) AND additionally do a tensor-bitwise compare
        # via torch.testing.assert_close(rtol=0, atol=0).
        loaded_sha = _model_sha256(model)
        saved_meta_path = (
            work / "checksums" / f"step-{latest:06d}-rank-{rank:02d}.json"
        )
        match = False
        saved_sha = None
        if saved_meta_path.exists():
            saved = json.loads(saved_meta_path.read_text())
            saved_sha = saved["sha256"]
            match = (saved_sha == loaded_sha)
            # Bit-exact tensor equality on top of the SHA — belt & braces.
            saved_tensors_blob = saved.get("tensors_path")
            if saved_tensors_blob:
                blob = torch.load(saved_tensors_blob, map_location="cpu",
                                  weights_only=False)
                live = _state_dict_snapshot(model)
                for k in sorted(blob.keys()):
                    torch.testing.assert_close(
                        live[k], blob[k], rtol=0.0, atol=0.0,
                        msg=f"tensor mismatch at key={k} step={latest}",
                    )
        emit(event="checksum_verify", step=latest,
             saved_sha256=saved_sha, loaded_sha256=loaded_sha, match=match)
        assert match, (
            f"checksum identity broken on resume: "
            f"saved={saved_sha} loaded={loaded_sha}"
        )
        start_step = latest + 1
        emit(event="resume", from_step=latest, start_step=start_step)
    else:
        emit(event="cold_start")

    # ----- training loop -----------------------------------------------
    cum_tokens = 0
    # Replay any prior cum_tokens for this rank (across earlier generations)
    # so the conservation invariant aggregates correctly even after a kill.
    if start_step > 0:
        cum_tokens = start_step * tokens_per_step

    for step in range(start_step, total_steps):
        # Deterministic per-(step,rank) input so identical recovery yields
        # identical post-step weights — required for the checksum invariant.
        gen = torch.Generator().manual_seed(1000 * step + rank)
        x = torch.randn(tokens_per_step, 8, generator=gen)
        y = torch.randn(tokens_per_step, 8, generator=gen)

        optim.zero_grad(set_to_none=True)
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()

        # All-reduce gradients so the PG is genuinely exercised — a dead
        # rank here causes the others to error out and torchelastic to
        # restart the cohort, which is exactly the chaos we want.
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world)

        optim.step()
        cum_tokens += tokens_per_step

        # Save sharded snapshot. We block on the writer queue so that the
        # checksum we record reflects what is actually persisted on disk.
        ckpt.save(model, optim, step, rank=rank,
                  extra_meta={"world": world, "generation": generation})
        ckpt._q.join()

        sha = _model_sha256(model)
        ck_dir = work / "checksums"
        ck_dir.mkdir(parents=True, exist_ok=True)
        # Persist a tensor blob alongside the SHA so the resume path can
        # do a torch.testing.assert_close bit-exact comparison.
        tensors_path = ck_dir / f"step-{step:06d}-rank-{rank:02d}.pt"
        torch.save(_state_dict_snapshot(model), tensors_path)
        (ck_dir / f"step-{step:06d}-rank-{rank:02d}.json").write_text(
            json.dumps({
                "sha256": sha, "step": step, "rank": rank,
                "tensors_path": str(tensors_path),
            })
        )
        emit(event="step", step=step, loss=float(loss.item()),
             tokens=tokens_per_step, cum_tokens=cum_tokens, sha256=sha)

        # ---------- Scenario A: SIGKILL the targeted local rank ----------
        if (scenario == "A"
                and step == kill_step
                and local_rank == kill_rank
                and generation == 0):
            emit(event="self_sigkill", step=step, local_rank=local_rank)
            tele.flush()
            try:
                os.fsync(tele.fileno())
            except OSError:
                pass
            os.kill(os.getpid(), signal.SIGKILL)
            # not reached

    # ----- shutdown -----------------------------------------------------
    ckpt.shutdown(drain=True)
    if dist.is_initialized():
        dist.destroy_process_group()

    final_dir = work / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / f"rank-{rank:02d}-gen-{generation:02d}.json").write_text(
        json.dumps({
            "rank": rank,
            "generation": generation,
            "final_step": total_steps - 1,
            "cum_tokens": cum_tokens,
            "final_sha256": _model_sha256(model),
        })
    )
    emit(event="finish", final_step=total_steps - 1, cum_tokens=cum_tokens)
    tele.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(2)
