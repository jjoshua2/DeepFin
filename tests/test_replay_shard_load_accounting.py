"""Shard loads that fail must be accounted for, not swallowed.

``_load_refresh_chunks`` picks ``n_pick`` shards to refresh the hot shuffle
buffer with and used to wrap each load in a bare ``except Exception: pass``.
Nothing anywhere compared ``len(loaded)`` against ``n_pick``, so a refresh
that loaded three of five shards, or zero of five, looked exactly like one
that loaded all five.

Zero is the dangerous case and it is not hypothetical. ``_prefetch_worker``
only stores a NON-EMPTY result, so an empty refresh leaves the prefetch slot
unset, the next tick retries, and the loop refills "whenever the prefetched
slot is empty" — roughly every 0.1s. Production runs
``shuffle_refresh_interval: 1``. A persistent load fault therefore becomes a
hot retry loop that never logs, while training goes on sampling a shuffle
buffer that has silently stopped being refreshed.

Failures are classified by TRACKING rather than by exception type:
``_enforce_window`` pops a shard out of ``_shard_paths`` under the lock before
calling ``delete_shard_path``, so an untracked path was deliberately deleted
and losing it is expected. Type-based classification would be wrong here —
``zarr.open_group`` on a missing group raises ``GroupNotFoundError``, which is
a ``ValueError`` and not a ``FileNotFoundError`` — and fragile, since
``delete_shard_path`` uses ``shutil.rmtree(..., ignore_errors=True)`` and a
reader can catch a half-deleted group mid-flight.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay import disk_buffer as db
from chess_anti_engine.replay.disk_buffer import (
    _REFRESH_EMPTY_STREAK_ALARM,
    DiskReplayBuffer,
)


def _buffer(tmp_path: Path) -> DiskReplayBuffer:
    return DiskReplayBuffer(
        capacity=10_000,
        shard_dir=tmp_path / "shards",
        rng=np.random.default_rng(0),
        refresh_interval=0,  # no background prefetch thread during the test
        refresh_shards=4,
    )


def _arrays(n: int = 2) -> dict[str, np.ndarray]:
    policy = np.zeros((n, 4672), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.zeros((n,), dtype=np.int8),
        "priority": np.ones((n,), dtype=np.float32),
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def _track(buf: DiskReplayBuffer, paths: list[Path]) -> None:
    """Register paths as tracked shards, as ingest would."""
    buf._shard_paths = deque(paths)
    buf._shard_sizes = deque([2] * len(paths))
    buf._total_positions = 2 * len(paths)


def _fail_on(
    monkeypatch: pytest.MonkeyPatch, failing: set[Path], exc: Exception | None = None,
) -> None:
    """Make load_shard_arrays raise for *failing* paths and succeed elsewhere.

    Default is GroupNotFoundError's shape -- a ValueError -- specifically
    because a FileNotFoundError-based classifier would get this wrong.
    """
    boom = exc if exc is not None else ValueError("group not found: no .zgroup")

    def _fake(path: Path | str, **_kwargs: object) -> tuple[dict[str, np.ndarray], dict]:
        if Path(path) in failing:
            raise boom
        return _arrays(), {}

    monkeypatch.setattr(db, "load_shard_arrays", _fake)


def test_failure_on_a_tracked_shard_is_counted_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    buf = _buffer(tmp_path)
    paths = [tmp_path / f"s{i}.zarr" for i in range(4)]
    _track(buf, paths)
    _fail_on(monkeypatch, {paths[1]})

    loaded = buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=4, rng=np.random.default_rng(1),
    )

    assert len(loaded) == 3, "the three healthy shards must still load"
    assert buf._refresh_failed_total == 1
    assert buf._refresh_vanished_total == 0, (
        "a tracked shard is not the benign trim race"
    )
    out = capsys.readouterr().out
    assert "[disk_buf] WARNING" in out
    assert "s1.zarr" in out, "the line must name the shard that failed"


def test_failure_on_an_untracked_shard_is_the_benign_trim_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """_enforce_window pops before deleting, so untracked == deliberately gone.

    This is the routine case -- the prefetch thread snapshots the shard list,
    the trimmer deletes one of those shards, and the load then fails. Logging
    it as a fault would bury the real signal under expected noise.
    """
    buf = _buffer(tmp_path)
    paths = [tmp_path / f"s{i}.zarr" for i in range(4)]
  # Tracked set deliberately EXCLUDES paths[2]: the trimmer already popped it.
    _track(buf, [p for p in paths if p != paths[2]])
    _fail_on(monkeypatch, {paths[2]})

    loaded = buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=4, rng=np.random.default_rng(1),
    )

    assert len(loaded) == 3
    assert buf._refresh_vanished_total == 1
    assert buf._refresh_failed_total == 0
    assert "WARNING" not in capsys.readouterr().out


def test_a_filenotfound_on_a_tracked_shard_still_counts_as_a_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification is by tracking, so the exception type must not decide it.

    A FileNotFoundError on a shard the buffer still believes it owns means the
    file went missing WITHOUT the trimmer removing it -- external deletion,
    a lost mount, a disk-full truncation. That is a real fault even though the
    type looks benign, and the inverse (a ValueError from zarr for a shard the
    trimmer did remove) must not be.
    """
    buf = _buffer(tmp_path)
    paths = [tmp_path / f"s{i}.zarr" for i in range(2)]
    _track(buf, paths)
    _fail_on(monkeypatch, {paths[0]}, exc=FileNotFoundError("gone"))

    buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=2, rng=np.random.default_rng(1),
    )

    assert buf._refresh_failed_total == 1
    assert buf._refresh_vanished_total == 0


def test_a_fully_empty_refresh_streak_alarms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The case the old code made invisible: refresh returns nothing, forever."""
    buf = _buffer(tmp_path)
    paths = [tmp_path / f"s{i}.zarr" for i in range(3)]
    _track(buf, paths)
    _fail_on(monkeypatch, set(paths))

    for _ in range(_REFRESH_EMPTY_STREAK_ALARM - 1):
        assert buf._load_refresh_chunks(
            shard_paths=paths, refresh_shards=3, rng=np.random.default_rng(1),
        ) == []
    assert "no longer being" not in capsys.readouterr().out, (
        "must not cry wolf during a burst of window trimming"
    )

    buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=3, rng=np.random.default_rng(1),
    )
    out = capsys.readouterr().out
    assert "shuffle refresh has returned 0 of 3 requested shards" in out
    assert "sampling stale data" in out


def test_the_empty_streak_resets_on_any_successful_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    buf = _buffer(tmp_path)
    paths = [tmp_path / f"s{i}.zarr" for i in range(2)]
    _track(buf, paths)

    _fail_on(monkeypatch, set(paths))
    for _ in range(5):
        buf._load_refresh_chunks(
            shard_paths=paths, refresh_shards=2, rng=np.random.default_rng(1),
        )
    assert buf._refresh_empty_streak == 5

    _fail_on(monkeypatch, set())
    buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=2, rng=np.random.default_rng(1),
    )
    assert buf._refresh_empty_streak == 0


def test_the_tracked_failure_warning_is_throttled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Per-shard failures arrive in bursts; the counter still records each one."""
    buf = _buffer(tmp_path)
    paths = [tmp_path / f"s{i}.zarr" for i in range(4)]
    _track(buf, paths)
    _fail_on(monkeypatch, set(paths))

    buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=4, rng=np.random.default_rng(1),
    )

    lines = [ln for ln in capsys.readouterr().out.splitlines() if "failed to load" in ln]
    assert len(lines) == 1, f"expected one throttled line, got {len(lines)}"
    assert buf._refresh_failed_total == 4, "every failure is still counted"


def test_the_seed_load_path_shares_the_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """_scan_existing_shards had its own copy of the same silent swallow."""
    buf = _buffer(tmp_path)
    path = tmp_path / "seed.zarr"
    _track(buf, [path])
    _fail_on(monkeypatch, {path})

    assert buf._try_load_shard(path, context="shuffle seed") is None
    assert buf._refresh_failed_total == 1
    assert "shuffle seed failed to load a TRACKED shard" in capsys.readouterr().out
