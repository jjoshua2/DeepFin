"""Mid-iteration revival of the broker and workers.

Ensuring the fleet only at the selfplay-phase boundary is what turned the
2026-07-24 broker crash into a ~50-minute outage: the ingest wait loop can run
for ``wait_timeout_s * 3`` with nothing checking whether anything is still
alive to produce games.  These tests pin the two halves of the fix — that the
wait loop calls back at all, and that the callback revives corpses without
disturbing anything healthy.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import chess_anti_engine.tune.distributed_runtime as distributed_runtime
from chess_anti_engine.ext_check import C_CORE_REFUSAL_EXIT_CODE, CCoreIncompleteError
from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.tune.distributed_runtime import (
    _ingest_distributed_selfplay,
    revive_dead_selfplay_processes,
)


class _FakeProc:
    """Stand-in for subprocess.Popen with a settable exit status."""

    def __init__(
        self,
        pid: int,
        returncode: int | None = None,
        launch_signature: str | None = None,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
      # Real workers carry this; _ensure_distributed_workers compares it
      # against the config to decide on a restart.
        self._cae_worker_launch_signature = launch_signature

    def poll(self) -> int | None:
        return self.returncode


def _run_starved_ingest(
    tmp_path: Path,
    *,
    on_poll: Callable[[], None],
    on_poll_interval_s: float,
    wait_timeout_s: float = 0.05,
) -> dict[str, Any]:
    """Run an ingest that can never meet its target, so the wait loop spins.

    Arguments are passed explicitly rather than splatted from a dict: a
    ``dict[str, object]`` splat defeats the type checker at every parameter
    and would need a blanket suppression on each call site.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    return _ingest_distributed_selfplay(
        buf=DiskReplayBuffer(
            256,
            shard_dir=tmp_path / "replay",
            rng=np.random.default_rng(0), read_only=False,
            shuffle_cap=64,
            shard_size=8,
        ),
        holdout_buf=ArrayReplayBuffer(8, rng=np.random.default_rng(1)),
        holdout_frac=0.0,
        holdout_frozen=False,
        inbox_dir=inbox,
        processed_dir=tmp_path / "processed",
      # Unreachable target so the loop actually spins and times out.
        target_games=10_000,
        accepted_model_shas={"sha-current"},
        wait_timeout_s=wait_timeout_s,
        poll_seconds=0.001,
        rng=np.random.default_rng(2),
        min_games_fraction=0.5,
        on_poll=on_poll,
        on_poll_interval_s=on_poll_interval_s,
    )


def test_wait_loop_invokes_on_poll_while_starved(tmp_path: Path) -> None:
    """The callback must fire during the wait, not only at the boundary.

    With zero matching games the soft deadline's ``min_games`` guard can never
    fire, so this loop runs to the hard ceiling. That is exactly the window in
    which a dead broker has to be noticed.
    """
    calls: list[int] = []

    _run_starved_ingest(
        tmp_path, on_poll=lambda: calls.append(1), on_poll_interval_s=0.0,
    )

    assert calls, "on_poll never ran during a fully starved ingest wait"


def test_on_poll_repeats_on_its_interval(tmp_path: Path) -> None:
    """The callback must keep firing on a cadence — not once, not every spin.

    Both bounds are load-bearing, and an earlier version of this test asserted
    only ``len(calls) <= 1`` against a 1h interval. That passes when the
    callback fires exactly ONCE for the whole 8100s wait, which is precisely
    the 2026-07-24 outage: broker healthy at the first poll, dead at t=5min,
    never noticed. A revive that does not repeat is not a revive.

    Starved, so the loop runs to the hard ceiling (wait_timeout_s * 3 = 1.2s)
    with a 0.1s interval => ~12 calls. The bounds are deliberately loose for
    slow CI while still killing both failure modes: one-shot gives 1, and a
    missing rate-limit gives ~1200 (the loop spins every poll_seconds=1ms).
    """
    calls: list[int] = []

    _run_starved_ingest(
        tmp_path,
        on_poll=lambda: calls.append(1),
        on_poll_interval_s=0.1,
        wait_timeout_s=0.4,
    )

    assert 4 <= len(calls) <= 60, (
        f"on_poll ran {len(calls)}x over a ~1.2s starved wait at a 0.1s "
        "interval; expected repeated firing (~12), not one-shot or every-spin"
    )


def test_first_on_poll_is_immediate(tmp_path: Path) -> None:
    """The first liveness check must not wait out a full interval.

    A broker can die during its own launch or in the gap before the first
    shard scan. With zero matching games the soft deadline can never fire, so
    that death costs the whole hard ceiling — and sleeping 60s before asking
    the first question adds a minute to an outage we already know how to end.

    A 1h interval means the ONLY call that can occur is the immediate one, so
    exactly 1 pins immediacy; scheduling the first poll one interval out would
    give 0 here.
    """
    calls: list[int] = []

    _run_starved_ingest(
        tmp_path, on_poll=lambda: calls.append(1), on_poll_interval_s=3600.0,
    )

    assert len(calls) == 1, (
        f"expected exactly one immediate poll, got {len(calls)}"
    )


def test_on_poll_failure_does_not_abort_the_ingest(tmp_path: Path) -> None:
    """A failing revive must not take down an ingest that is still working."""

    def _boom() -> None:
        raise RuntimeError("relaunch failed")

    summary = _run_starved_ingest(
        tmp_path, on_poll=_boom, on_poll_interval_s=0.0,
    )

    assert summary["matching_games"] == 0


def test_revive_relaunches_an_exited_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[str] = []
    replacement = _FakeProc(pid=222)
    monkeypatch.setattr(
        distributed_runtime, "_launch_inference_broker",
        lambda **_kw: (launched.append("broker"), replacement)[1],
    )

    box: list[object] = [_FakeProc(pid=111, returncode=1)]
    revived = revive_dead_selfplay_processes(
        config={"distributed_inference_broker_enabled": True},
        trial_id="t", trial_dir=tmp_path, publish_dir=tmp_path,
        broker_proc_box=box,  # pyright: ignore[reportArgumentType]
        worker_procs=[],
    )

    assert revived is True
    assert launched == ["broker"]
    assert box[0] is replacement, "caller must receive the new broker handle"


def test_revive_leaves_a_live_broker_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged-but-alive broker must NOT be duplicated.

    Two brokers bound to the same shared-memory slots would be far worse than
    one slow one, so the revive keys strictly off process exit.
    """
    monkeypatch.setattr(
        distributed_runtime, "_launch_inference_broker",
        lambda **_kw: pytest.fail("relaunched a broker that was still alive"),
    )

    alive = _FakeProc(pid=111, returncode=None)
    box: list[object] = [alive]
    revived = revive_dead_selfplay_processes(
        config={"distributed_inference_broker_enabled": True},
        trial_id="t", trial_dir=tmp_path, publish_dir=tmp_path,
        broker_proc_box=box,  # pyright: ignore[reportArgumentType]
        worker_procs=[],
    )

    assert revived is False
    assert box[0] is alive


def test_revive_relaunches_only_exited_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched_idx: list[int] = []

    def _fake_launch(*, worker_index: int, **_rest: object) -> _FakeProc:
        launched_idx.append(worker_index)
        return _FakeProc(pid=900 + worker_index)

    monkeypatch.setattr(distributed_runtime, "_launch_distributed_worker", _fake_launch)

    alive = _FakeProc(pid=1, returncode=None)
    dead = _FakeProc(pid=2, returncode=-9)
    workers: list[object] = [alive, dead]

    revived = revive_dead_selfplay_processes(
        config={"distributed_workers_per_trial": 2},
        trial_id="t", trial_dir=tmp_path, publish_dir=tmp_path,
        broker_proc_box=[None],
        worker_procs=workers,  # pyright: ignore[reportArgumentType]
    )

    assert revived is True
    assert launched_idx == [1], "only the exited worker should be relaunched"
    assert workers[0] is alive, "live worker must be untouched"
    assert workers[1] is not dead, "dead worker must be replaced in place"


def test_revive_does_not_restart_a_live_worker_whose_config_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PR #224 hazard, pinned.

    ``_ensure_distributed_workers`` restarts workers whose launch signature no
    longer matches the config, and the config is re-read from the live yaml
    every iteration. If the mid-iteration revive used that function, an
    unrelated yaml edit would tear down workers with games in flight — the
    exact waste PR #224 removed. Reviving a corpse is unambiguous; re-signing a
    live worker is a phase-boundary decision.
    """
    monkeypatch.setattr(
        distributed_runtime, "_launch_distributed_worker",
        lambda **_kw: pytest.fail("restarted a live worker mid-iteration"),
    )

    alive = _FakeProc(pid=1, returncode=None, launch_signature="stale-signature")

    revived = revive_dead_selfplay_processes(
      # A config that would certainly produce a different signature.
        config={"distributed_workers_per_trial": 1, "mcts_simulations": 4242},
        trial_id="t", trial_dir=tmp_path, publish_dir=tmp_path,
        broker_proc_box=[None],
        worker_procs=[alive],  # pyright: ignore[reportArgumentType]
    )

    assert revived is False


# ── C-core refusal exit code: fatal, never relaunched ─────────────────────

def test_revive_does_not_relaunch_a_worker_that_refused_on_the_c_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale build refuses on every launch; relaunching it would spin forever."""
    monkeypatch.setattr(
        distributed_runtime, "_launch_distributed_worker",
        lambda **_kw: pytest.fail("relaunched a worker that refused on the C core"),
    )
    refused = _FakeProc(pid=2, returncode=C_CORE_REFUSAL_EXIT_CODE)
    with pytest.raises(CCoreIncompleteError, match="refused to start"):
        revive_dead_selfplay_processes(
            config={"distributed_workers_per_trial": 1},
            trial_id="t", trial_dir=tmp_path, publish_dir=tmp_path,
            broker_proc_box=[None],
            worker_procs=[refused],  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize("code", [1, 2, -9, -15, 0])
def test_revive_still_relaunches_every_other_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int,
) -> None:
    launched: list[int] = []

    def _fake_launch(*, worker_index: int, **_rest: object) -> _FakeProc:
        launched.append(worker_index)
        return _FakeProc(pid=900 + worker_index)

    monkeypatch.setattr(distributed_runtime, "_launch_distributed_worker", _fake_launch)
    workers: list[object] = [_FakeProc(pid=2, returncode=code)]
    assert revive_dead_selfplay_processes(
        config={"distributed_workers_per_trial": 1},
        trial_id="t", trial_dir=tmp_path, publish_dir=tmp_path,
        broker_proc_box=[None],
        worker_procs=workers,  # pyright: ignore[reportArgumentType]
    ) is True
    assert launched == [0]


@pytest.mark.parametrize(
    ("code", "fatal"), [(C_CORE_REFUSAL_EXIT_CODE, True), (1, False), (-9, False)],
)
def test_ensure_workers_treats_only_the_refusal_code_as_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, fatal: bool,
) -> None:
    launched: list[int] = []

    def _fake_launch(*, worker_index: int, **_rest: object) -> _FakeProc:
        launched.append(worker_index)
        return _FakeProc(pid=900 + worker_index, launch_signature="sig")

    monkeypatch.setattr(distributed_runtime, "_launch_distributed_worker", _fake_launch)
    monkeypatch.setattr(distributed_runtime, "_worker_launch_signature", lambda **_kw: "sig")
    procs: list[object] = [_FakeProc(pid=2, returncode=code, launch_signature="sig")]
    def call() -> None:
        distributed_runtime._ensure_distributed_workers(
            config={"distributed_workers_per_trial": 1}, trial_dir=tmp_path, trial_id="t",
            procs=procs,  # pyright: ignore[reportArgumentType]
        )

    if fatal:
        with pytest.raises(CCoreIncompleteError):
            call()
        assert launched == []
    else:
        call()
        assert launched == [0]


def test_ensure_workers_raises_before_relaunching_an_earlier_dead_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """idx 0 died with 1, idx 1 refused (78): raise first, launch nothing.

    ``_ensure_distributed_workers`` works on a copy of the caller's list, so a
    relaunch of idx 0 before idx 1 raised would be an orphan nobody cleans up.
    """
    launched: list[int] = []

    def _fake_launch(*, worker_index: int, **_rest: object) -> _FakeProc:
        launched.append(worker_index)
        return _FakeProc(pid=900 + worker_index, launch_signature="sig")

    monkeypatch.setattr(distributed_runtime, "_launch_distributed_worker", _fake_launch)
    monkeypatch.setattr(distributed_runtime, "_worker_launch_signature", lambda **_kw: "sig")
    procs: list[object] = [
        _FakeProc(pid=2, returncode=1, launch_signature="sig"),
        _FakeProc(pid=3, returncode=C_CORE_REFUSAL_EXIT_CODE, launch_signature="sig"),
    ]
    with pytest.raises(CCoreIncompleteError, match="idx=1"):
        distributed_runtime._ensure_distributed_workers(
            config={"distributed_workers_per_trial": 2}, trial_dir=tmp_path, trial_id="t",
            procs=procs,  # pyright: ignore[reportArgumentType]
        )
    assert launched == []


def test_ensure_workers_stops_its_own_launches_when_a_worker_refuses_mid_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """idx 1 is alive at the up-front scan and exits 78 before the loop reaches it.

    idx 0 (dead, code 1) has already been relaunched by then; that process lives
    only in the function's local copy, so it must be stopped before the raise.
    """

    class _DiesDuringLoop(_FakeProc):
        def __init__(self) -> None:
            super().__init__(pid=3, returncode=None, launch_signature="sig")
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            if self.polls > 1:  # alive for the scan, refused afterwards
                self.returncode = C_CORE_REFUSAL_EXIT_CODE
            return self.returncode

    launched: list[_FakeProc] = []
    stopped: list[_FakeProc] = []

    def _fake_launch(*, worker_index: int, **_rest: object) -> _FakeProc:
        proc = _FakeProc(pid=900 + worker_index, launch_signature="sig")
        launched.append(proc)
        return proc

    monkeypatch.setattr(distributed_runtime, "_launch_distributed_worker", _fake_launch)
    monkeypatch.setattr(distributed_runtime, "_worker_launch_signature", lambda **_kw: "sig")
    monkeypatch.setattr(distributed_runtime, "_stop_process", stopped.append)
    swept: list[int] = []
    monkeypatch.setattr(
        distributed_runtime, "terminate_engines_owned_by",
        lambda pid: (swept.append(pid), [])[1],
    )
    procs: list[object] = [_FakeProc(pid=2, returncode=1, launch_signature="sig"), _DiesDuringLoop()]
    with pytest.raises(CCoreIncompleteError, match="idx=1"):
        distributed_runtime._ensure_distributed_workers(
            config={"distributed_workers_per_trial": 2}, trial_dir=tmp_path, trial_id="t",
            procs=procs,  # pyright: ignore[reportArgumentType]
        )
    assert len(launched) == 1  # idx 0 was relaunched before idx 1 refused
    assert stopped == launched  # ... and is not leaked
    # Stopped through _stop_worker_processes, which also sweeps the stopped
    # worker's orphaned Stockfish engines.
    assert swept == [900]


def test_a_failing_mid_loop_cleanup_does_not_replace_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator must read the C-core refusal, not a cleanup error."""

    class _DiesDuringLoop(_FakeProc):
        def __init__(self) -> None:
            super().__init__(pid=3, returncode=None, launch_signature="sig")
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            if self.polls > 1:
                self.returncode = C_CORE_REFUSAL_EXIT_CODE
            return self.returncode

    def _cleanup_fails(_procs: object) -> None:
        raise OSError("cleanup exploded")

    monkeypatch.setattr(
        distributed_runtime, "_launch_distributed_worker",
        lambda *, worker_index, **_kw: _FakeProc(pid=900 + worker_index, launch_signature="sig"),
    )
    monkeypatch.setattr(distributed_runtime, "_worker_launch_signature", lambda **_kw: "sig")
    monkeypatch.setattr(distributed_runtime, "_stop_worker_processes", _cleanup_fails)
    procs: list[object] = [_FakeProc(pid=2, returncode=1, launch_signature="sig"), _DiesDuringLoop()]
    with pytest.raises(CCoreIncompleteError, match="idx=1"):
        distributed_runtime._ensure_distributed_workers(
            config={"distributed_workers_per_trial": 2}, trial_dir=tmp_path, trial_id="t",
            procs=procs,  # pyright: ignore[reportArgumentType]
        )


def test_launch_stamps_the_worker_log_paths_the_refusal_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_raise_if_c_core_refusal`` reads ``_cae_worker_log_paths`` off the proc."""
    stub = _FakeProc(pid=4242)
    spawned: list[dict[str, object]] = []

    def _fake_spawn(**kwargs: object) -> _FakeProc:
        spawned.append(kwargs)
        return stub

    monkeypatch.setattr(distributed_runtime, "_spawn_with_reap", _fake_spawn)
    # The command line is not under test here (it needs a full server config).
    monkeypatch.setattr(distributed_runtime, "_build_distributed_worker_cmd", lambda **_kw: ["true"])
    proc = distributed_runtime._launch_distributed_worker(
        config={}, trial_dir=tmp_path, trial_id="t", worker_index=3,
    )
    assert proc is stub
    root = distributed_runtime._distributed_worker_artifacts_root(tmp_path) / "worker_03"
    assert getattr(proc, "_cae_worker_log_paths", None) == (root / "worker.out", root / "worker.log")
    assert spawned[0]["log_path"] == root / "worker.out"
    # ... and the refusal message actually names them.
    stub.returncode = C_CORE_REFUSAL_EXIT_CODE
    with pytest.raises(CCoreIncompleteError) as exc:
        distributed_runtime._raise_if_c_core_refusal(
            proc, idx=3, trial_id="t",
        )
    assert str(root / "worker.out") in str(exc.value)
    assert str(root / "worker.log") in str(exc.value)


def test_revive_raises_before_relaunching_the_broker_or_an_earlier_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[str] = []
    monkeypatch.setattr(
        distributed_runtime, "_launch_distributed_worker",
        lambda *, worker_index, **_kw: launched.append(f"worker{worker_index}"),
    )
    monkeypatch.setattr(
        distributed_runtime, "_ensure_inference_broker",
        lambda **_kw: launched.append("broker"),
    )
    with pytest.raises(CCoreIncompleteError, match="idx=1"):
        revive_dead_selfplay_processes(
            config={"distributed_workers_per_trial": 2},
            trial_id="t", trial_dir=tmp_path, publish_dir=tmp_path,
            broker_proc_box=[_FakeProc(pid=1, returncode=1)],  # pyright: ignore[reportArgumentType]
            worker_procs=[  # pyright: ignore[reportArgumentType]
                _FakeProc(pid=2, returncode=1),
                _FakeProc(pid=3, returncode=C_CORE_REFUSAL_EXIT_CODE),
            ],
        )
    assert launched == []


def test_ingest_wait_loop_lets_the_c_core_refusal_through(tmp_path: Path) -> None:
    """Other on_poll failures are swallowed (test above); this one ends the trial."""

    def _refused() -> None:
        raise CCoreIncompleteError("worker refused")

    with pytest.raises(CCoreIncompleteError, match="worker refused"):
        _run_starved_ingest(tmp_path, on_poll=_refused, on_poll_interval_s=0.0)
