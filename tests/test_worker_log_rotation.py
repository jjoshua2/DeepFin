"""A worker (re)launch must move the previous log aside, never write over it.

Banked from the 2026-08-04 cold start. The primary stall -- the worker->server
upload path wedging at ~00:23 -- is STILL UNDIAGNOSED, and the reason is that
the evidence no longer exists: the 01:47:22 revive left
``worker_00/worker.log`` containing exactly ONE ``worker starting version=``
line, at 01:47:22, with the 00:19-01:47 window gone. The pause line
("stale backlog target reached: 1928/1870 games for model 2fffc555", 00:34:22)
and worker_00's ~4,264 drop lines from 00:37:20 were read live that session and
cannot be re-read now. An agent auditing the disk afterwards correctly found
zero drop lines: absence of evidence there is evidence DESTRUCTION.

⚑ What this fix does NOT claim. The truncating writer is NOT identified.
``logging.FileHandler`` defaults to append (``worker.py:378``), ``_spawn_with_reap``
opens ``worker.out`` with ``"ab"``, ``os.execv`` on self-update reuses the same
argv, and ``scripts/train.sh`` documents the append behaviour as the reason its
drain logic must truncate its READING at ``worker starting version=``. Yet the
file was truncated in place -- the artifact directory's mtime is still 00:15,
so the files were not recreated. Rotation is chosen precisely BECAUSE it does
not depend on winning that argument: the previous generation is renamed to a
different filename before the replacement process can open anything, so
whatever truncates ``worker.log`` afterwards truncates an empty new file.

That is also the honest limit of the fix: it preserves the log across every
launch that goes through ``_launch_distributed_worker`` (the revive path at
``distributed_runtime.py:1270`` and the phase-boundary ensure). An in-place
truncation by something that does NOT go through the launcher would still lose
the CURRENT generation -- but the prior one survives, which is the difference
between "undiagnosable" and "diagnosable".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import chess_anti_engine.tune.distributed_runtime as dr
from chess_anti_engine.tune.distributed_runtime import _launch_distributed_worker


class _StubProc:
    """Enough of a Popen for the launcher to stamp its signature onto."""

    returncode: int | None = None

    def poll(self) -> int | None:
        return None


def _launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, worker_index: int = 0,
) -> Path:
    """Run the real launcher with the spawn stubbed out.

    Only the process spawn and the command build are replaced -- the artifact
    path derivation and the rotation under test are the production code.
    """
    spawned: list[Path] = []

    def _fake_cmd(**_kwargs: Any) -> list[str]:
        return ["/bin/true"]

    def _fake_spawn(*, cmd: list[str], log_path: Path, **_kwargs: Any):
        del cmd
        spawned.append(log_path)
  # The real spawn opens the .out in append mode; mimic that so a test can
  # tell "rotated" from "never written".
        with log_path.open("ab"):
            pass
        return _StubProc()

    monkeypatch.setattr(dr, "_build_distributed_worker_cmd", _fake_cmd)
    monkeypatch.setattr(dr, "_spawn_with_reap", _fake_spawn)

    _launch_distributed_worker(
        config={},
        trial_dir=tmp_path,
        trial_id="t0",
        worker_index=worker_index,
    )
    assert spawned, "the launcher did not reach the spawn"
    return tmp_path / "distributed_workers" / f"worker_{worker_index:02d}"


def test_relaunch_preserves_the_previous_worker_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on origin/main: the incident window is simply gone after a revive."""
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)
    incident = (
        "2026-08-04 00:34:22,000 INFO chess_anti_engine.worker selfplay paused "
        "by server: stale backlog target reached: 1928/1870 games\n"
        "2026-08-04 00:37:20,000 WARNING chess_anti_engine.worker_buffer "
        "upload buffer at 5000 positions; dropping 37-position game batch\n"
    )
    (root / "worker.log").write_text(incident, encoding="utf-8")

    _launch(tmp_path, monkeypatch)

    rotated = root / "worker.log.1"
    assert rotated.exists(), (
        "the pre-relaunch log must survive somewhere findable; without this "
        "the revive destroys the only record of what wedged the fleet"
    )
    assert rotated.read_text(encoding="utf-8") == incident, (
        "the rotated generation must be the previous log VERBATIM"
    )


def test_the_live_log_starts_clean_after_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rotation must MOVE, not copy: the replacement gets a fresh file.

    A copy would leave the old lines in front of the new process's output and
    recreate the two-generations-in-one-file problem ``scripts/train.sh``
    already has to defend against when it truncates its reading at
    ``worker starting version=``.
    """
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)
    (root / "worker.log").write_text("old generation\n", encoding="utf-8")

    _launch(tmp_path, monkeypatch)

    live = root / "worker.log"
    assert not live.exists() or live.read_text(encoding="utf-8") == "", (
        "the live log must be empty for the replacement worker"
    )


def test_two_generations_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fleet that revives twice must not lose the FIRST failure.

    The 2026-08-04 stall began at ~00:23 and the revive fired at 01:47. A single
    generation is enough for one revive; a second revive before anyone reads the
    logs would take the original evidence with it.
    """
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)

    (root / "worker.log").write_text("gen-A\n", encoding="utf-8")
    _launch(tmp_path, monkeypatch)
    (root / "worker.log").write_text("gen-B\n", encoding="utf-8")
    _launch(tmp_path, monkeypatch)

    assert (root / "worker.log.1").read_text(encoding="utf-8") == "gen-B\n"
    assert (root / "worker.log.2").read_text(encoding="utf-8") == "gen-A\n", (
        "the older generation must age into .2 rather than being overwritten"
    )


def test_a_first_launch_creates_no_empty_rotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing to preserve on a cold launch; do not litter the artifact dir."""
    root = _launch(tmp_path, monkeypatch)

    assert not (root / "worker.log.1").exists()
    assert not (root / "worker.out.1").exists()


def test_an_empty_previous_log_is_not_rotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty file carries no evidence; rotating it would push a real
    generation out of the window for nothing."""
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)
    (root / "worker.log").write_text("", encoding="utf-8")

    _launch(tmp_path, monkeypatch)

    assert not (root / "worker.log.1").exists()


def test_the_stdout_capture_is_rotated_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``worker.out`` carries tracebacks the logger never sees.

    A worker that dies on an unhandled exception writes it to stdout, not
    through logging -- so for exactly the failures a revive responds to, the
    .out is the only record.
    """
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)
    (root / "worker.out").write_text("Traceback (most recent call last):\n", encoding="utf-8")

    _launch(tmp_path, monkeypatch)

    assert (root / "worker.out.1").read_text(encoding="utf-8") == (
        "Traceback (most recent call last):\n"
    )


def test_rotation_failure_does_not_block_the_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserving evidence must never cost the fleet a worker.

    The revive exists to bring a dead worker back; if rotation raises (read-only
    dir, races with an operator's tail), the launch must still happen. A guard
    that can take down the thing it guards is worse than the gap it closes.
    """
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)
    (root / "worker.log").write_text("some content\n", encoding="utf-8")

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "replace", _boom)

    root_out = _launch(tmp_path, monkeypatch)

    assert root_out.exists(), "the worker must still have been launched"


class _DeadProc:
    """A worker corpse: ``poll()`` returns an exit code, so the revive fires."""

    returncode: int | None = 1

    def poll(self) -> int | None:
        return 1


def test_the_revive_path_reaches_the_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect is in the REVIVE, so drive the REVIVE and read the disk.

    ``revive_dead_selfplay_processes`` is the site that destroyed the evidence
    on 2026-08-04; rotation living in ``_launch_distributed_worker`` only helps
    if the revive actually reaches it.

    ⚑ This was a ``inspect.getsource`` grep for ``"_launch_distributed_worker("``
    in a previous revision. That test could not fail for the reason it exists:
    the substring survives a revive that calls the launcher on the wrong path,
    passes the wrong worker index, or has the call sitting behind a branch that
    never runs -- and it stays green if the launcher stops rotating altogether.
    Drive the real function against a corpse and assert the file moved.
    """
    root = tmp_path / "distributed_workers" / "worker_03"
    root.mkdir(parents=True)
    (root / "worker.log").write_text("the 00:19-01:47 window\n", encoding="utf-8")

    def _fake_cmd(**_kwargs: Any) -> list[str]:
        return ["/bin/true"]

    def _fake_spawn(*, cmd: list[str], log_path: Path, **_kwargs: Any):
        del cmd, log_path
        return _StubProc()

    monkeypatch.setattr(dr, "_build_distributed_worker_cmd", _fake_cmd)
    monkeypatch.setattr(dr, "_spawn_with_reap", _fake_spawn)

  # Index 3 with a 4-long list: the corpse must be revived in its OWN slot, so
  # a revive that relaunches index 0 rotates the wrong directory and fails.
    worker_procs: list[Any] = [_StubProc(), _StubProc(), _StubProc(), _DeadProc()]

    revived = dr.revive_dead_selfplay_processes(
        config={},
        trial_id="t0",
        trial_dir=tmp_path,
        publish_dir=tmp_path / "publish",
        broker_proc_box=[None],
        worker_procs=worker_procs,
    )

    assert revived is True, "a corpse in the list must be reported as revived"
    assert (root / "worker.log.1").read_text(encoding="utf-8") == (
        "the 00:19-01:47 window\n"
    ), (
        "the revive must relaunch through the launcher that rotates logs; the "
        "previous generation is the evidence the 2026-08-04 revive destroyed"
    )
    assert not (root / "worker.log").exists() or (
        (root / "worker.log").read_text(encoding="utf-8") == ""
    ), "the live log must start empty after the rotation, not carry the old text"


def test_a_healthy_fleet_rotates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for the test above.

    ``revive_dead_selfplay_processes`` runs hundreds of times per iteration
    inside the ingest wait loop. If it rotated on every call, a healthy worker's
    log would be shredded across generations within one iteration and the two-
    generation window would cover seconds instead of a run. Without this case a
    mutation that rotates unconditionally passes the positive test.
    """
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)
    (root / "worker.log").write_text("healthy\n", encoding="utf-8")

    monkeypatch.setattr(dr, "_build_distributed_worker_cmd", lambda **_k: ["/bin/true"])
    monkeypatch.setattr(dr, "_spawn_with_reap", lambda **_k: _StubProc())

    worker_procs: list[Any] = [_StubProc()]

    revived = dr.revive_dead_selfplay_processes(
        config={},
        trial_id="t0",
        trial_dir=tmp_path,
        publish_dir=tmp_path / "publish",
        broker_proc_box=[None],
        worker_procs=worker_procs,
    )

    assert revived is False
    assert not (root / "worker.log.1").exists(), (
        "a live worker must not have its log rotated out from under it"
    )
    assert (root / "worker.log").read_text(encoding="utf-8") == "healthy\n"


def test_rotation_runs_before_the_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering is the whole fix.

    Rotating AFTER the spawn would race the replacement worker's first writes
    into the rotated file and leave the live log holding a fragment -- exactly
    the interleaving that made tonight's log unreadable.
    """
    root = tmp_path / "distributed_workers" / "worker_00"
    root.mkdir(parents=True)
    (root / "worker.log").write_text("pre-existing\n", encoding="utf-8")
    observed: dict[str, bool] = {}

    def _fake_cmd(**_kwargs: Any) -> list[str]:
        return ["/bin/true"]

    def _fake_spawn(*, cmd: list[str], log_path: Path, **_kwargs: Any):
        del cmd, log_path
        observed["rotated_before_spawn"] = (root / "worker.log.1").exists()
        return _StubProc()

    monkeypatch.setattr(dr, "_build_distributed_worker_cmd", _fake_cmd)
    monkeypatch.setattr(dr, "_spawn_with_reap", _fake_spawn)

    _launch_distributed_worker(
        config={}, trial_dir=tmp_path, trial_id="t0", worker_index=0,
    )

    assert observed.get("rotated_before_spawn") is True, (
        "the previous log must already be moved aside when the replacement "
        "process starts"
    )
