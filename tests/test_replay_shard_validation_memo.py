"""``validate_arrays`` must run once per distinct shard content, not per read.

The shuffle refresh re-reads the SAME local shards for a whole run — at
``refresh_interval=5`` / ``refresh_shards=3`` over a 9.7k-step run each of
~650 shards is decoded and re-validated ~19 times — and ``validate_arrays``
is a pure function of the arrays it is handed. Measured on the production
path (the refresh runs on the trainer's prefetch WORKER thread, warm cache,
3 shards): **7.467s with validation, 1.022s without — 7.31x**.

⚑ THE MEMO KEY IS THE WHOLE DESIGN, because the obvious key does not work. A
``.zarr`` shard is a DIRECTORY of ~291 chunk files, so ``stat(shard)`` reports
the directory's mtime — which does NOT move when a chunk inside is rewritten
in place. A path+dir-mtime memo would therefore be a gate that cannot fire on
the one mutation that matters, which is this repo's signature defect wearing a
performance hat. ``_shard_validation_fingerprint`` walks the tree instead.

Every test below is written to FAIL if the memo either (a) stops skipping
(the win evaporates) or (b) starts skipping something it must not (the guard
hollows out). ``test_a_rewritten_chunk_is_revalidated_even_though_the_dir_mtime_did_not_move``
is the one that kills the naive key.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    local_shard_path,
    save_local_shard_arrays,
)

ROWS = 8
N_SHARDS = 3


def _shard_arrays(rank: int) -> dict[str, np.ndarray]:
    policy = np.zeros((ROWS, 4672), dtype=np.float32)
    policy[:, rank % 4672] = 1.0
    return {
        "x": np.full((ROWS, 146, 8, 8), float(rank), dtype=np.float32),
        "policy_target": policy,
        "priority": np.full((ROWS,), float(rank), dtype=np.float32),
        "wdl_target": np.arange(ROWS, dtype=np.int8) % 3,
        "has_policy": np.ones((ROWS,), dtype=np.uint8),
    }


def _write_window(tmp_path: Path, n: int = N_SHARDS) -> Path:
    shard_dir = tmp_path / "replay_shards"
    shard_dir.mkdir()
    for rank in range(n):
        save_local_shard_arrays(
            local_shard_path(shard_dir, rank),
            arrs=_shard_arrays(rank),
            meta=ShardMeta(positions=ROWS),
        )
    return shard_dir


def _buffer(
    shard_dir: Path, seed: int = 0, *, refresh_interval: int = 0,
) -> DiskReplayBuffer:
    """``refresh_interval=0`` by default: the memo tests drive
    ``_try_load_shard`` directly and must not race a refresh. The end-to-end
    identity test opts back in, because without refreshes no shard is ever
    RE-read and the memo could never hit at all.
    """
    return DiskReplayBuffer(
        capacity=10**9,
        shard_dir=shard_dir,
        rng=np.random.default_rng(seed),
        read_only=True,
        shuffle_cap=1200,
        refresh_interval=refresh_interval,
        refresh_shards=N_SHARDS,
        deterministic_refresh=True,
    )


def _poison_policy_rows(shard: Path) -> None:
    """Make ``validate_arrays`` reject this shard: a zero-sum policy row.

    Written through zarr rather than by byte-poking a chunk file so the shard
    stays a well-formed, decodable group — the point is to trip the CONTENT
    check, not the decoder.
    """
    g = zarr.open_group(str(shard), mode="r+")
    g["policy_target"][:] = np.zeros((ROWS, 4672), dtype=np.float32)


# --- the win: an unchanged shard is validated exactly once ------------------


def test_an_unchanged_shard_is_validated_once_then_skipped(tmp_path: Path) -> None:
    shard_dir = _write_window(tmp_path, n=1)
    buf = _buffer(shard_dir)
    shard = local_shard_path(shard_dir, 0)

    buf._validated_shards.clear()
    buf._shard_validations_run = 0
    buf._shard_validations_skipped = 0

    for _ in range(4):
        assert buf._try_load_shard(shard, context="test") is not None

    assert buf._shard_validations_run == 1, (
        "the first read must validate; a memo that never records is the no-op "
        "this change exists to avoid"
    )
    assert buf._shard_validations_skipped == 3, (
        "every later read of the SAME bytes must skip — this is the 7.31x"
    )


def test_the_memo_is_per_instance_and_never_persisted(tmp_path: Path) -> None:
    """A fresh buffer re-validates from scratch, which bounds the blast radius.

    ⚑ Construction itself seeds the pool through ``_load_refresh_chunks``, so a
    buffer has ALREADY validated and memoized its seed shards by the time a
    test gets hold of it. The per-instance claim is therefore read off the
    freshly-built object: every load it did was a first sight (``skipped == 0``)
    and it recorded them (``run >= 1``).
    """
    shard_dir = _write_window(tmp_path, n=1)
    shard = local_shard_path(shard_dir, 0)

    first = _buffer(shard_dir)
    assert first._shard_validations_run >= 1
    assert first._shard_validations_skipped == 0
    assert str(shard.resolve()) in first._validated_shards
    # ... and now the same instance skips it.
    first._try_load_shard(shard, context="test")
    assert first._shard_validations_skipped == 1

    second = _buffer(shard_dir)
    assert second._shard_validations_skipped == 0, (
        "a second buffer must not inherit the first's verdict"
    )
    assert second._shard_validations_run >= 1


# --- the guard: corruption is still caught ---------------------------------


def test_corruption_is_caught_on_a_first_load(tmp_path: Path) -> None:
    """The baseline the memo must not weaken: no memo entry, so validation runs."""
    shard_dir = _write_window(tmp_path, n=1)
    shard = local_shard_path(shard_dir, 0)
    _poison_policy_rows(shard)

    with pytest.raises(ValueError, match="non-positive sum"):
        load_shard_arrays(shard, lazy=False)

    buf = _buffer(shard_dir)
    assert buf._try_load_shard(shard, context="test") is None, (
        "a shard that fails validation must not be handed to the pool"
    )
    assert buf._validated_shards == {}, (
        "a shard that FAILED validation must never be memoized as good"
    )


def test_corruption_after_a_clean_load_is_still_caught(tmp_path: Path) -> None:
    """The coordinator's mutation: validate, then corrupt, then reload."""
    shard_dir = _write_window(tmp_path, n=1)
    shard = local_shard_path(shard_dir, 0)
    buf = _buffer(shard_dir)

    assert buf._try_load_shard(shard, context="test") is not None
    assert len(buf._validated_shards) == 1, "first load must memoize"

    _poison_policy_rows(shard)
    os.utime(shard)  # the coordinator's "+ touch its mtime"

    assert buf._try_load_shard(shard, context="test") is None, (
        "the fingerprint moved, so validation must re-run and reject"
    )


def test_a_rewritten_chunk_is_revalidated_even_though_the_dir_mtime_did_not_move(
    tmp_path: Path,
) -> None:
    """⚑ THE ONE THAT KILLS THE NAIVE KEY.

    Rewriting a chunk file inside the `.zarr` does not touch the `.zarr`
    DIRECTORY's own mtime. A memo keyed on ``stat(shard).st_mtime`` would skip
    validation here and admit a corrupt shard into the pool. The tree walk
    catches it because the chunk file's own mtime and the tree's total size
    move.
    """
    shard_dir = _write_window(tmp_path, n=1)
    shard = local_shard_path(shard_dir, 0)
    buf = _buffer(shard_dir)

    assert buf._try_load_shard(shard, context="test") is not None
    dir_stat_before = os.stat(shard).st_mtime_ns

    _poison_policy_rows(shard)

    assert os.stat(shard).st_mtime_ns == dir_stat_before, (
        "precondition: the DIRECTORY mtime must be unmoved, or this test is "
        "not exercising the failure it exists for"
    )
    assert buf._try_load_shard(shard, context="test") is None, (
        "the tree fingerprint must notice a rewritten chunk that the "
        "directory's own mtime cannot"
    )


def test_a_shard_swapped_for_different_content_is_revalidated(tmp_path: Path) -> None:
    """Same path, same row count, different bytes — the fingerprint must move."""
    shard_dir = _write_window(tmp_path, n=1)
    shard = local_shard_path(shard_dir, 0)
    buf = _buffer(shard_dir)

    assert buf._try_load_shard(shard, context="test") is not None
    buf._shard_validations_run = 0

    g = zarr.open_group(str(shard), mode="r+")
    g["x"][:] = np.full((ROWS, 146, 8, 8), 99.0, dtype=np.float32)

    assert buf._try_load_shard(shard, context="test") is not None
    assert buf._shard_validations_run == 1, (
        "different content at the same path must be re-validated"
    )


# --- the memo must not change what is loaded -------------------------------


def test_a_memoized_reload_returns_byte_identical_arrays(tmp_path: Path) -> None:
    """Skipping a pure CHECK must not change the arrays the check ran on."""
    shard_dir = _write_window(tmp_path, n=1)
    shard = local_shard_path(shard_dir, 0)
    buf = _buffer(shard_dir)

    validated = buf._try_load_shard(shard, context="test")
    memoized = buf._try_load_shard(shard, context="test")
    assert buf._shard_validations_skipped >= 1, "the second read must be a memo hit"

    assert validated is not None
    assert memoized is not None
    assert set(validated) == set(memoized)
    for key in validated:
        assert np.array_equal(validated[key], memoized[key]), (
            f"{key} differs between a validated and a memoized load"
        )


def test_the_batch_stream_is_identical_with_and_without_memo_hits(
    tmp_path: Path,
) -> None:
    """End to end: 50 draws, memo live vs memo defeated, byte for byte.

    The memo buffer keeps its cache; the control's cache is cleared before
    every load, so it re-validates every single time — today's behaviour. Both
    must produce the same batches in the same order.
    """
    shard_dir = _write_window(tmp_path, n=N_SHARDS)

    memo_buf = _buffer(shard_dir, seed=7, refresh_interval=4)
    control = _buffer(shard_dir, seed=7, refresh_interval=4)

    real_try_load = DiskReplayBuffer._try_load_shard

    def _never_memoized(self: DiskReplayBuffer, sp: Path, *, context: str):
        self._validated_shards.clear()
        return real_try_load(self, sp, context=context)

    memo_draws = [memo_buf.sample_batch_arrays(16) for _ in range(50)]

    control._try_load_shard = _never_memoized.__get__(control, DiskReplayBuffer)  # type: ignore[method-assign]
    control_draws = [control.sample_batch_arrays(16) for _ in range(50)]

    assert memo_buf._shard_validations_skipped > 0, (
        "the memo arm must actually have hit, or this proves nothing"
    )
    assert control._shard_validations_skipped == 0, (
        "the control arm must never skip, or it is not the baseline"
    )

    assert len(memo_draws) == len(control_draws) == 50
    for i, (a, b) in enumerate(zip(memo_draws, control_draws)):
        assert set(a) == set(b), f"draw {i}: key sets differ"
        for key in a:
            assert np.array_equal(a[key], b[key]), f"draw {i}: {key} differs"
