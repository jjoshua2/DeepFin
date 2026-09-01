"""Decoding a refresh's shards concurrently must not move a single row.

``_load_refresh_chunks`` picks its shards with one ``rng.choice`` and THEN
loads them. The draw is therefore already finished before any decoding starts,
so concurrency cannot reach the randomness. The one thing it can reach is the
ORDER the loaded chunks are appended to ``loaded`` — that is the order they
enter the shuffle pool, and the pool's layout is what ``sample_batch_arrays``
indexes into. Assemble as futures complete and the pool is permuted; assemble
in submission order and it is byte-identical.

⚑ Why the pool pays at all, given blosc already decompresses with 8 internal
threads: numcodecs turns that OFF on non-main threads, and the refresh has
always run on one (the trainer's prefetch worker). Measured, worker thread,
3 shards, memo warm: serial 5.409s → 4 workers 2.315s (2.34x).

``test_completion_order_does_not_leak_into_the_pool`` is the mutant's home:
it makes the LAST-submitted shard the FIRST to finish, so an implementation
that appends on completion cannot accidentally agree with the serial order.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay import disk_buffer as db
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta,
    local_shard_path,
    save_local_shard_arrays,
)

ROWS = 8
N_SHARDS = 6
REFRESH_SHARDS = 4
BATCH = 16
DRAWS = 50


def _shard_arrays(rank: int) -> dict[str, np.ndarray]:
    """Every row carries its source shard's rank in ``x`` and ``priority``."""
    policy = np.zeros((ROWS, 4672), dtype=np.float32)
    policy[:, rank % 4672] = 1.0
    return {
        "x": np.full((ROWS, 146, 8, 8), float(rank), dtype=np.float32),
        "policy_target": policy,
        "priority": np.full((ROWS,), float(rank), dtype=np.float32),
        "wdl_target": np.arange(ROWS, dtype=np.int8) % 3,
        "has_policy": np.ones((ROWS,), dtype=np.uint8),
    }


def _write_window(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "replay_shards"
    shard_dir.mkdir()
    for rank in range(N_SHARDS):
        save_local_shard_arrays(
            local_shard_path(shard_dir, rank),
            arrs=_shard_arrays(rank),
            meta=ShardMeta(positions=ROWS),
        )
    return shard_dir


def _buffer(shard_dir: Path, seed: int = 0) -> DiskReplayBuffer:
    return DiskReplayBuffer(
        capacity=10**9,
        shard_dir=shard_dir,
        rng=np.random.default_rng(seed),
        read_only=True,
        shuffle_cap=1200,
        refresh_interval=4,
        refresh_shards=REFRESH_SHARDS,
        deterministic_refresh=True,
    )


def _ranks(chunks: list[dict[str, np.ndarray]]) -> list[float]:
    """The source-shard rank of every chunk, in pool-insertion order."""
    return [float(np.asarray(c["x"]).reshape(-1)[0]) for c in chunks]


def _on_worker_thread(fn, *args, **kwargs):
    """Run *fn* off the main thread.

    ⚑ NOT A DETAIL — WITHOUT THIS EVERY TEST BELOW IS VACUOUS. The pool is
    deliberately gated to non-main-thread callers (blosc already decodes 8-way
    on the main thread, where an outer pool measured 0.82x), and pytest runs
    tests ON the main thread. Calling `_load_refresh_chunks` directly would
    therefore take the SERIAL branch, and "parallel is byte-identical to
    serial" would be comparing serial against serial.
    """
    with ThreadPoolExecutor(max_workers=1) as driver:
        return driver.submit(fn, *args, **kwargs).result()


def test_the_main_thread_does_not_get_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REVIEW F2. Main-thread callers keep blosc's own 8-way decode.

    Pinned by observing the THREAD the loads actually run on, not by reading
    the branch back: a gate asserted against its own condition cannot fail.
    """
    shard_dir = _write_window(tmp_path)
    buf = _buffer(shard_dir)
    paths = list(buf._shard_paths)
    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 4)

    real_load = DiskReplayBuffer._try_load_shard
    threads: list[str] = []

    def _record(self: DiskReplayBuffer, sp: Path, *, context: str) -> Any:
        threads.append(threading.current_thread().name)
        return real_load(self, sp, context=context)

    monkeypatch.setattr(DiskReplayBuffer, "_try_load_shard", _record)

    buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(2),
    )
    assert threads, "precondition: some shard must have been loaded"
    assert set(threads) == {threading.main_thread().name}, (
        "a main-thread refresh must decode inline, not on pool threads"
    )

    threads.clear()
    _on_worker_thread(
        buf._load_refresh_chunks,
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(2),
    )
    assert threads, "precondition: some shard must have been loaded"
    assert all(t.startswith("replay-refresh-load") for t in threads), (
        "a worker-thread refresh must use the decode pool"
    )


def test_parallel_refresh_decode_is_byte_identical_to_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same draw, both schedulers: same chunks, same order, same bytes."""
    shard_dir = _write_window(tmp_path)
    buf = _buffer(shard_dir)
    paths = list(buf._shard_paths)

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 1)
    serial = buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(11),
    )
    serial_idx: list[int] = []
    buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(11), chosen_out=serial_idx,
    )

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 4)
    parallel = _on_worker_thread(
        buf._load_refresh_chunks,
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(11),
    )
    parallel_idx: list[int] = []
    _on_worker_thread(
        buf._load_refresh_chunks,
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(11), chosen_out=parallel_idx,
    )

    assert len(parallel) == len(serial) == REFRESH_SHARDS
    assert _ranks(parallel) == _ranks(serial), (
        "chunks entered the pool in a different order than the serial load"
    )
    assert parallel_idx == serial_idx, "chosen_out must stay 1:1 with loaded"
    for i, (a, b) in enumerate(zip(serial, parallel)):
        assert set(a) == set(b)
        for key in a:
            assert np.array_equal(a[key], b[key]), f"chunk {i}: {key} differs"


def test_completion_order_does_not_leak_into_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE MUTANT'S HOME — finish the loads in REVERSE submission order.

    With a real thread pool the completion order is a race, so a test that
    merely ran the pool could pass against an append-on-completion
    implementation by luck. This one forces the last-submitted shard to return
    first, which makes the two orders provably different.
    """
    shard_dir = _write_window(tmp_path)
    buf = _buffer(shard_dir)
    paths = list(buf._shard_paths)

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 1)
    serial = buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(3),
    )

    real_load = DiskReplayBuffer._try_load_shard
    order: list[int] = []
    call = {"n": 0}

    def _staggered(self: DiskReplayBuffer, sp: Path, *, context: str) -> Any:
        # Submission i sleeps (N - i) ticks: the LAST submitted returns first.
        i = call["n"]
        call["n"] += 1
        time.sleep(0.05 * (REFRESH_SHARDS - i))
        order.append(i)
        return real_load(self, sp, context=context)

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", REFRESH_SHARDS)
    monkeypatch.setattr(DiskReplayBuffer, "_try_load_shard", _staggered)
    parallel = _on_worker_thread(
        buf._load_refresh_chunks,
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(3),
    )

    assert order != sorted(order), (
        "precondition: the loads must have COMPLETED out of submission order, "
        "or this test cannot tell the two assembly rules apart"
    )
    assert _ranks(parallel) == _ranks(serial), (
        "results must be assembled in SUBMISSION order, not completion order"
    )


def test_a_failed_shard_still_drops_out_without_shifting_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One load returns None; the survivors keep their order and their ranks."""
    shard_dir = _write_window(tmp_path)
    buf = _buffer(shard_dir)
    paths = list(buf._shard_paths)

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 1)
    baseline_idx: list[int] = []
    baseline = buf._load_refresh_chunks(
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(5), chosen_out=baseline_idx,
    )
    doomed = baseline_idx[1]

    real_load = DiskReplayBuffer._try_load_shard

    def _one_fails(self: DiskReplayBuffer, sp: Path, *, context: str) -> Any:
        if sp == paths[doomed]:
            return None
        return real_load(self, sp, context=context)

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 4)
    monkeypatch.setattr(DiskReplayBuffer, "_try_load_shard", _one_fails)
    got_idx: list[int] = []
    got = _on_worker_thread(
        buf._load_refresh_chunks,
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(5), chosen_out=got_idx,
    )

    assert len(got) == len(baseline) - 1
    assert got_idx == [i for i in baseline_idx if i != doomed]
    assert len(got_idx) == len(got), "chosen_out must stay 1:1 with loaded"
    assert _ranks(got) == [
        r for r, i in zip(_ranks(baseline), baseline_idx) if i != doomed
    ]


def test_a_refusing_executor_falls_back_to_serial_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REVIEW F4. At interpreter teardown `submit` refuses new futures.

    `_prefetch_loop` is a DAEMON thread, so it can still be inside a refresh
    when `threading._shutdown` has run, after which
    `ThreadPoolExecutor.submit` raises "cannot schedule new futures after
    interpreter shutdown". Nothing is wrong with the run at that point — the
    process is exiting — so the only visible effect of not catching it is a
    spurious traceback on a clean shutdown.

    Simulated by an executor that refuses, rather than by tearing down a real
    interpreter: the branch under test is "submit raised RuntimeError", and
    that is exactly what is injected.
    """
    shard_dir = _write_window(tmp_path)
    buf = _buffer(shard_dir)
    paths = list(buf._shard_paths)
    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 4)

    expected = _on_worker_thread(
        buf._load_refresh_chunks,
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(9),
    )

    class _RefusingExecutor:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def __enter__(self) -> _RefusingExecutor:
            return self

        def __exit__(self, *_a: Any) -> None:
            return None

        def submit(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError(
                "cannot schedule new futures after interpreter shutdown",
            )

    monkeypatch.setattr(db, "ThreadPoolExecutor", _RefusingExecutor)
    got = _on_worker_thread(
        buf._load_refresh_chunks,
        shard_paths=paths, refresh_shards=REFRESH_SHARDS,
        rng=np.random.default_rng(9),
    )

    assert _ranks(got) == _ranks(expected), (
        "the serial fallback must load the same shards in the same order"
    )


def test_the_sampled_batch_stream_is_unchanged_by_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: 50 draws, serial vs 4-way decode, byte for byte."""
    shard_dir = _write_window(tmp_path)

    def _draw(buf: DiskReplayBuffer) -> list[dict[str, np.ndarray]]:
        # On a worker thread, exactly as `_iter_prefetched_batches` samples --
        # and the only way the pool arm reaches the pool at all.
        return [buf.sample_batch_arrays(BATCH) for _ in range(DRAWS)]

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 1)
    serial_draws = _on_worker_thread(_draw, _buffer(shard_dir, seed=7))

    monkeypatch.setattr(db, "_REFRESH_LOAD_WORKERS", 4)
    par_draws = _on_worker_thread(_draw, _buffer(shard_dir, seed=7))

    assert len(serial_draws) == len(par_draws) == DRAWS
    for i, (a, b) in enumerate(zip(serial_draws, par_draws)):
        assert set(a) == set(b), f"draw {i}: key sets differ"
        for key in a:
            assert np.array_equal(a[key], b[key]), f"draw {i}: {key} differs"
