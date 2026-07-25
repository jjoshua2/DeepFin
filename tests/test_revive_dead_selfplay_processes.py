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
            rng=np.random.default_rng(0),
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


def test_on_poll_is_rate_limited(tmp_path: Path) -> None:
    """A once-per-interval guard, so polling stays cheap in the hot loop."""
    calls: list[int] = []

    _run_starved_ingest(
        tmp_path,
        on_poll=lambda: calls.append(1),
      # Far longer than the whole wait: only the initial due-time may pass.
        on_poll_interval_s=3600.0,
    )

    assert len(calls) <= 1, f"on_poll ran {len(calls)}x despite a 1h interval"


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
