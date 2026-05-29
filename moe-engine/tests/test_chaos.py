"""
tests/test_chaos.py
===================

Production-grade chaos & fault-tolerance integration suite for moe-engine.

This pytest driver does **not** import torch.distributed itself. Instead, it
shells out to ``torchrun`` (CPU/Gloo) executing ``tests/_chaos_worker.py``,
injects controlled failures, and verifies — by parsing the JSONL telemetry
each rank emits to disk — three mathematical invariants:

  1. **Monotonic step progression** — within any single generation, the
     ``step`` indices an event stream emits are strictly increasing.
  2. **Token conservation** — across all generations of a given rank, the
     final ``cum_tokens`` equals ``total_steps * tokens_per_step`` exactly.
     Crashes mid-step never double-count and never lose tokens.
  3. **Checksum identity** — on every resume from a sharded snapshot the
     worker re-derives a SHA256 of the loaded weights AND runs
     ``torch.testing.assert_close(rtol=0, atol=0)`` against a saved tensor
     blob. If either differs, the worker exits non-zero, propagating to a
     test failure here.

Three scenarios are exercised:

  * ``test_chaos_baseline_no_fault`` — control / smoke: no faults, asserts
     the verification machinery is sound under the happy path.
  * ``test_chaos_scenario_a_sudden_node_failure`` — ``SIGKILL`` LOCAL_RANK=2
     at step 3 of a 4-rank cohort with ``--max-restarts=2`` and verify
     TorchElastic re-rendezvous + checksum-identical resume.
  * ``test_chaos_scenario_b_storage_stall`` — inject a 10-second stall into
     ``LocalNVMeAdapter.put`` at step 4 and verify the async checkpoint
     queue does not deadlock + a telemetry warning is emitted.

Process hygiene
---------------
Each ``torchrun`` launch is wrapped in a ``_torchrun_session`` context
manager that spawns the agent with ``start_new_session=True`` (so the
agent + every worker it forks share an isolated process group) and unconditionally issues
``os.killpg(pgid, SIGKILL)`` in a ``try/finally``. A second autouse fixture then
``pgrep``s for any leftover ``_chaos_worker.py`` processes — making zombie
bleed-over into subsequent tests an asserted impossibility.

The whole suite is gated behind ``@pytest.mark.chaos`` and is skipped
silently if ``torchrun`` is not on ``$PATH``.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest

pytestmark = pytest.mark.chaos

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve().parent / "_chaos_worker.py"

# Module-level skip: chaos suite requires torchrun on $PATH.
if shutil.which("torchrun") is None:                             # pragma: no cover
    pytest.skip("torchrun not available on $PATH", allow_module_level=True)


# ==========================================================================
# Bulletproof torchrun session.
# ==========================================================================
@contextlib.contextmanager
def _torchrun_session(
    *,
    nproc: int,
    max_restarts: int,
    env: Dict[str, str],
    timeout: float,
    monitor_interval: float = 1.0,
) -> Iterator[Tuple[subprocess.Popen, str]]:
    """Spawn ``torchrun`` in a fresh session, guarantee teardown.

    On scope exit — whether the wrapped block returned cleanly, raised, or
    timed out — we issue ``os.killpg(pgid, SIGKILL)`` followed by
    ``proc.wait()``. Combined with ``start_new_session=True`` this makes
    leaking a distributed-worker process group physically impossible.
    """
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        f"--max-restarts={max_restarts}",
        f"--monitor-interval={monitor_interval}",
        str(WORKER),
    ]

    # Strip torchelastic env that the parent shell might have set so we
    # don't accidentally rendezvous with somebody else's run.
    full_env = dict(os.environ)
    for var in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
    ):
        full_env.pop(var, None)
    full_env.update(env)
    full_env.setdefault("OMP_NUM_THREADS", "1")
    full_env.setdefault("MKL_NUM_THREADS", "1")
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    # Make sure the worker can `import pkg.*`.
    pp = full_env.get("PYTHONPATH", "")
    full_env["PYTHONPATH"] = (str(ROOT) + (os.pathsep + pp if pp else ""))

    proc = subprocess.Popen(
        cmd,
        env=full_env,
        cwd=str(ROOT),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    captured: bytes = b""
    try:
        try:
            captured, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Bubble up — finally block will reap the group.
            try:
                captured = (proc.stdout.read() if proc.stdout else b"") or b""
            except Exception:
                captured = b""
            raise
        yield proc, captured.decode("utf-8", errors="replace")
    finally:
        # Strict bulletproof process sanitization.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:                       # pragma: no cover
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


# ==========================================================================
# Telemetry parsing helpers.
# ==========================================================================
def _load_telemetry(work: Path) -> Dict[int, Dict[int, List[dict]]]:
    """Returns ``{rank: {generation: [event, ...]}}``.

    Tolerates a partial trailing JSONL line (a SIGKILL mid-write is normal
    and must not blind the parser to earlier complete lines).
    """
    out: Dict[int, Dict[int, List[dict]]] = {}
    tele_dir = work / "telemetry"
    if not tele_dir.exists():
        return out
    for f in sorted(tele_dir.glob("rank-*-gen-*.jsonl")):
        # filename layout: rank-XX-gen-YY.jsonl
        parts = f.stem.split("-")
        rank, gen = int(parts[1]), int(parts[3])
        events: List[dict] = []
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Partial line at SIGKILL boundary — earlier lines are valid.
                continue
        out.setdefault(rank, {})[gen] = events
    return out


def _load_finals(work: Path) -> Dict[int, Dict[int, dict]]:
    out: Dict[int, Dict[int, dict]] = {}
    fin = work / "final"
    if not fin.exists():
        return out
    for f in sorted(fin.glob("rank-*-gen-*.json")):
        parts = f.stem.split("-")
        rank, gen = int(parts[1]), int(parts[3])
        out.setdefault(rank, {})[gen] = json.loads(f.read_text())
    return out


def _assert_monotonic_steps(events: List[dict], where: str) -> None:
    """Within a single generation, ``step`` events must strictly increase."""
    last = -1
    for ev in events:
        if ev.get("event") == "step":
            s = int(ev["step"])
            assert s > last, f"{where}: non-monotonic step {s} after {last}"
            last = s


def _final_cum_tokens(events: List[dict]) -> int:
    """Last cum_tokens value seen in the stream (step or finish event)."""
    cum = 0
    for ev in events:
        if ev.get("event") in ("step", "finish") and "cum_tokens" in ev:
            cum = int(ev["cum_tokens"])
    return cum


# ==========================================================================
# Zombie sanity check (autouse) — belt & braces on top of killpg.
# ==========================================================================
@pytest.fixture(autouse=True)
def _no_zombie_workers_after_test() -> Iterator[None]:
    yield
    try:
        out = subprocess.check_output(["pgrep", "-af", str(WORKER)], text=True)
    except subprocess.CalledProcessError:
        out = ""  # pgrep exits 1 when nothing matches — desired state.
    leftovers = [ln for ln in out.splitlines() if ln.strip()]
    assert not leftovers, (
        "zombie chaos worker processes detected after test:\n  "
        + "\n  ".join(leftovers)
    )


# ==========================================================================
# Scenario: control / no-fault baseline.
# ==========================================================================
@pytest.mark.timeout(120)
def test_chaos_baseline_no_fault(tmp_path: Path) -> None:
    """Validates telemetry + invariant logic under a happy path. Without
    this control we cannot disambiguate test-machinery bugs from fault-
    injection bugs in the more complex scenarios."""
    total_steps, tokens_per_step, world = 6, 8, 2
    env = {
        "CHAOS_WORK_DIR": str(tmp_path),
        "CHAOS_SCENARIO": "A",
        "CHAOS_TOTAL_STEPS": str(total_steps),
        "CHAOS_TOKENS_PER_STEP": str(tokens_per_step),
        "CHAOS_KILL_STEP": "9999",   # unreachable: no kill happens
        "CHAOS_KILL_RANK": "999",
    }
    with _torchrun_session(
        nproc=world, max_restarts=0, env=env, timeout=120,
    ) as (proc, log):
        pass
    assert proc.returncode == 0, (
        f"baseline torchrun nonzero exit (rc={proc.returncode}):\n"
        f"{log[-4000:]}"
    )

    tele = _load_telemetry(tmp_path)
    finals = _load_finals(tmp_path)
    assert set(tele.keys()) == set(range(world)), (
        f"missing telemetry for some ranks: have {sorted(tele)}"
    )
    for r in range(world):
        # Exactly one generation in baseline (no restarts).
        assert list(tele[r].keys()) == [0], (
            f"rank {r}: expected only gen 0, got {list(tele[r])}"
        )
        events = tele[r][0]
        _assert_monotonic_steps(events, where=f"baseline rank={r} gen=0")
        # Token conservation.
        cum = _final_cum_tokens(events)
        assert cum == total_steps * tokens_per_step, (
            f"rank {r}: cum_tokens={cum} != "
            f"{total_steps * tokens_per_step} (token conservation broken)"
        )
        # Finish-event sidecar.
        assert r in finals and 0 in finals[r], (
            f"rank {r}: missing final/rank-{r:02d}-gen-00.json"
        )
        assert finals[r][0]["final_step"] == total_steps - 1
        assert finals[r][0]["cum_tokens"] == total_steps * tokens_per_step


# ==========================================================================
# Scenario A: sudden node failure (SIGKILL) + TorchElastic recovery.
# ==========================================================================
@pytest.mark.timeout(240)
def test_chaos_scenario_a_sudden_node_failure(tmp_path: Path) -> None:
    """SIGKILL local rank 2 mid-run. Assert TorchElastic re-rendezvous,
    resume from the last sharded snapshot, and bit-exact checksum identity
    of recovered weights (SHA256 + ``torch.testing.assert_close``)."""
    total_steps, tokens_per_step, world = 10, 8, 4
    kill_step, kill_rank = 3, 2
    env = {
        "CHAOS_WORK_DIR": str(tmp_path),
        "CHAOS_SCENARIO": "A",
        "CHAOS_TOTAL_STEPS": str(total_steps),
        "CHAOS_TOKENS_PER_STEP": str(tokens_per_step),
        "CHAOS_KILL_STEP": str(kill_step),
        "CHAOS_KILL_RANK": str(kill_rank),
    }
    with _torchrun_session(
        nproc=world, max_restarts=2, env=env, timeout=240,
    ) as (proc, log):
        pass
    assert proc.returncode == 0, (
        f"scenario A torchrun nonzero exit (rc={proc.returncode}):\n"
        f"{log[-6000:]}"
    )

    tele = _load_telemetry(tmp_path)
    finals = _load_finals(tmp_path)

    assert set(tele.keys()) == set(range(world)), (
        f"missing telemetry for some ranks: have {sorted(tele)}"
    )

    # The targeted rank must observe at least 2 generations: gen 0 (truncated)
    # and gen 1 (post-recovery).
    target_gens = sorted(tele[kill_rank].keys())
    assert len(target_gens) >= 2 and target_gens[0] == 0, (
        f"rank {kill_rank}: expected ≥2 generations starting at 0, "
        f"got {target_gens}"
    )

    # Gen 0 of the targeted rank MUST contain a self_sigkill event at the
    # configured step.
    g0 = tele[kill_rank][0]
    sigkill_evs = [
        ev for ev in g0
        if ev.get("event") == "self_sigkill" and ev.get("step") == kill_step
    ]
    assert sigkill_evs, (
        f"rank {kill_rank}: no self_sigkill event at step {kill_step} in gen 0"
    )

    # Checksum identity: every checksum_verify event across all ranks/gens
    # must report match=true. Worker also asserts via assert_close at load
    # time, so a divergence would crash the worker (rc != 0) but we double-
    # check the telemetry record here too.
    cv_count = 0
    for r, gens in tele.items():
        for g, events in gens.items():
            for ev in events:
                if ev.get("event") == "checksum_verify":
                    cv_count += 1
                    assert ev.get("match") is True, (
                        f"checksum identity broken on rank={r} gen={g} "
                        f"step={ev.get('step')}: "
                        f"saved={ev.get('saved_sha256')} "
                        f"loaded={ev.get('loaded_sha256')}"
                    )
    assert cv_count >= 1, (
        "no checksum_verify events emitted — recovery resume path "
        "did not exercise the SHA256 + assert_close invariant"
    )

    # Final generation of every rank must complete the full run with exact
    # token conservation across the crash boundary.
    for r in range(world):
        last_gen = max(tele[r].keys())
        events = tele[r][last_gen]
        _assert_monotonic_steps(events, where=f"scenarioA rank={r} gen={last_gen}")
        finish_evs = [e for e in events if e.get("event") == "finish"]
        assert finish_evs, (
            f"rank {r} gen {last_gen}: no finish event — run did not "
            f"complete cleanly after recovery"
        )
        finish = finish_evs[-1]
        assert finish["final_step"] == total_steps - 1, (
            f"rank {r} gen {last_gen}: final_step={finish['final_step']} "
            f"!= {total_steps - 1}"
        )
        assert finish["cum_tokens"] == total_steps * tokens_per_step, (
            f"rank {r} gen {last_gen}: cum_tokens={finish['cum_tokens']} "
            f"!= {total_steps * tokens_per_step} (token conservation broken)"
        )
        # Filesystem sidecar agrees with telemetry.
        assert r in finals and last_gen in finals[r], (
            f"rank {r}: missing final/rank-{r:02d}-gen-{last_gen:02d}.json"
        )
        assert finals[r][last_gen]["cum_tokens"] == total_steps * tokens_per_step


# ==========================================================================
# Scenario B: storage IO interruption (no deadlock, telemetry warning).
# ==========================================================================
@pytest.mark.timeout(180)
def test_chaos_scenario_b_storage_stall(tmp_path: Path) -> None:
    """Inject a 10-second stall into ``LocalNVMeAdapter.put`` at step 4.
    The async checkpoint queue must drain cleanly (no deadlock) and a
    ``latency_inject`` telemetry event must be emitted."""
    total_steps, tokens_per_step, world = 10, 8, 2
    latency_step, latency_seconds = 4, 10.0
    env = {
        "CHAOS_WORK_DIR": str(tmp_path),
        "CHAOS_SCENARIO": "B",
        "CHAOS_TOTAL_STEPS": str(total_steps),
        "CHAOS_TOKENS_PER_STEP": str(tokens_per_step),
        "CHAOS_LATENCY_STEP": str(latency_step),
        "CHAOS_LATENCY_SECONDS": str(latency_seconds),
        "CHAOS_KILL_STEP": "9999",   # unreachable
        "CHAOS_KILL_RANK": "999",
    }
    t0 = time.perf_counter()
    with _torchrun_session(
        nproc=world, max_restarts=0, env=env, timeout=180,
    ) as (proc, log):
        pass
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, (
        f"scenario B torchrun nonzero exit (rc={proc.returncode}):\n"
        f"{log[-6000:]}"
    )
    # Sanity: stall actually fired (run cannot have finished faster than
    # the injected stall, modulo a small slack for timer skew).
    assert elapsed >= latency_seconds * 0.8, (
        f"run finished in {elapsed:.1f}s — latency injection appears not "
        f"to have fired (expected ≥ {latency_seconds * 0.8:.1f}s)"
    )

    tele = _load_telemetry(tmp_path)
    finals = _load_finals(tmp_path)
    assert set(tele.keys()) == set(range(world))

    # No restarts in this scenario → exactly one generation per rank.
    for r in range(world):
        assert list(tele[r].keys()) == [0], (
            f"rank {r}: expected only gen 0 (no restart), got {list(tele[r])}"
        )
        events = tele[r][0]
        _assert_monotonic_steps(events, where=f"scenarioB rank={r} gen=0")
        cum = _final_cum_tokens(events)
        assert cum == total_steps * tokens_per_step, (
            f"rank {r}: cum_tokens={cum} != {total_steps * tokens_per_step}"
        )
        assert any(e.get("event") == "finish" for e in events), (
            f"rank {r}: no finish event — async queue may have deadlocked"
        )
        # Sidecar agrees.
        assert r in finals and 0 in finals[r]
        assert finals[r][0]["cum_tokens"] == total_steps * tokens_per_step

    # Telemetry warning emission: at least one latency_inject event, and it
    # must reference the configured step.
    inject_evs = [
        ev
        for r in tele
        for g in tele[r]
        for ev in tele[r][g]
        if ev.get("event") == "latency_inject"
    ]
    assert inject_evs, "no latency_inject event observed in telemetry"
    assert all(ev.get("step") == latency_step for ev in inject_evs), (
        f"latency_inject step mismatch: {[ev.get('step') for ev in inject_evs]}"
    )
