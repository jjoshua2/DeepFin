"""The prefetch queue must be bounded by bytes (audit A19).

`BackgroundShardPrefetcher` decodes every new inbox shard into RAM on a daemon
thread. The producer rate is "however fast shards land in the inbox"; the
consumer rate is "once per training iteration". Nothing related those two, and
nothing bounded the queue: no length cap, no byte budget, no backpressure.

A 2000-row shard decodes to ~102 MB resident (measured on live shards, trial
13a9f), so an undrained queue cost ~5.1 GB at 50 shards and ~10.2 GB at 100.

⚑ The regime that fills it fastest is also the one with the longest drain
interval: the cold-start upload burst, many workers uploading at once while the
trainer is still inside a long first iteration. On the live box -- 98 GB total,
~28 GB available beside the training process and the replay window -- that is an
OOM path, and silent until fatal. `distributed_prefetch_shards: true` is set in
the production config, so this is live code, not a dormant path.

⚑ Backpressure is DEFER, and the distinction from DROP is the thing to keep
straight while reading these tests: an over-budget scan leaves the shard
untouched on disk in inbox/. It is picked up either by a later scan once the
trainer has drained, or by the iter-time inbox-poll fallback in
`_ingest_distributed_selfplay`, which decodes it inline -- exactly where the
decode happened before this module existed. So no data is lost and the only
cost is latency.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.tune import prefetch as prefetch_mod
from chess_anti_engine.tune.prefetch import BackgroundShardPrefetcher

# ⚑ `DEFAULT_MAX_QUEUED_BYTES` and `_arrays_nbytes` are imported lazily inside
# the tests that need them. They do not exist on origin/main, and a
# module-scope import would turn every case in this file into one collection
# error there -- collapsing a per-test red measurement into a single
# uninformative failure.

# Deliberately not 102 MB: the tests must run on the CPU-only box in seconds.
# What matters is that a shard has a KNOWN decoded size so the accounting can
# be checked exactly, not that the size matches production.
_ROWS = 256
_BYTES_PER_SHARD = _ROWS * 1024 * 4  # one float32 (1024,) row per row


def _fake_arrays() -> dict[str, Any]:
    return {"x": np.zeros((_ROWS, 1024), dtype=np.float32)}


def _make(
    tmp_path: Path,
    *,
    n_shards: int,
    max_queued_bytes: int,
    monkeypatch: pytest.MonkeyPatch,
) -> BackgroundShardPrefetcher:
    """A prefetcher over `n_shards` synthetic paths with the decode stubbed.

    The decode is stubbed rather than writing real zarr shards because the
    subject is the QUEUE's accounting and bound, not zarr. Every decode returns
    an array of known size, so `queued_bytes()` is checkable against arithmetic
    rather than against the thing under test.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = [tmp_path / f"shard_{i:05d}.zarr" for i in range(n_shards)]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        prefetch_mod, "load_shard_arrays", lambda _p: (_fake_arrays(), {"rows": _ROWS}),
    )
    return BackgroundShardPrefetcher(
        tmp_path,
        poll_seconds=0.1,
        path_iter=lambda _d: sorted(paths),
        max_queued_bytes=max_queued_bytes,
    )


# --------------------------------------------------------------------------
# The bound itself
# --------------------------------------------------------------------------


def test_a_fast_producer_is_capped_by_the_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on origin/main, where 200 shards would all be decoded and held.

    200 synthetic shards against a 3-shard budget. On main the queue takes all
    200; with the bound it stops at the budget plus at most one shard.
    """
    budget = 3 * _BYTES_PER_SHARD
    pf = _make(tmp_path, n_shards=200, max_queued_bytes=budget, monkeypatch=monkeypatch)

    pf._scan_once()

    assert pf.queued_bytes() < budget + _BYTES_PER_SHARD, (
        "the queue must stop within one shard of the budget, not absorb the "
        "whole inbox"
    )
    assert len(pf._queue) == 3, (
        "the scan admits while UNDER budget and stops on reaching it, so an "
        "exact multiple lands exactly on the budget"
    )


def test_the_overshoot_is_at_most_one_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented ceiling is `budget + largest single shard`.

    The check happens BEFORE the decode, because decoding and then discarding
    would already have paid the resident spike this bound exists to avoid. The
    price of that ordering is that the queue can finish OVER budget: the last
    shard was admitted while under it. The excess is strictly less than one
    shard, which is the ceiling the module docstring claims.

    Both shapes are covered, because they differ: an exact multiple of the
    shard size lands exactly on the budget with no overshoot at all, so testing
    only that shape would never exercise the excess.
    """
    for budget_shards in (1, 2, 5):
        budget = budget_shards * _BYTES_PER_SHARD
        pf = _make(
            tmp_path / f"b{budget_shards}", n_shards=50,
            max_queued_bytes=budget, monkeypatch=monkeypatch,
        )
        pf._scan_once()
        assert len(pf._queue) == budget_shards
        assert pf.queued_bytes() == budget, "exact multiple: no overshoot"

  # A budget that is NOT a multiple of the shard size: 2.5 shards admits
  # three (the third starts while queued=2 shards < 2.5) and finishes over.
    budget = (5 * _BYTES_PER_SHARD) // 2
    pf = _make(
        tmp_path / "frac", n_shards=50, max_queued_bytes=budget,
        monkeypatch=monkeypatch,
    )
    pf._scan_once()
    assert pf.queued_bytes() > budget, "this shape must actually overshoot"
    assert pf.queued_bytes() < budget + _BYTES_PER_SHARD, (
        "but by strictly less than one shard -- the documented ceiling"
    )


def test_an_unbounded_producer_would_blow_past_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ NEGATIVE CONTROL for the test above -- the harness must be able to
    SHOW the unbounded behaviour, or "capped at 4" proves nothing about the
    cap and everything about the fixture.

    Same fixture, a budget large enough to be irrelevant: all 200 shards land.
    """
    pf = _make(
        tmp_path, n_shards=200, max_queued_bytes=10 * 1024 ** 3,
        monkeypatch=monkeypatch,
    )

    pf._scan_once()

    assert len(pf._queue) == 200, (
        "with an effectively infinite budget the same fixture queues "
        "everything -- which is what main does at any inbox depth"
    )
    assert pf.queued_bytes() == 200 * _BYTES_PER_SHARD


# --------------------------------------------------------------------------
# Byte accounting
# --------------------------------------------------------------------------


def test_queued_bytes_matches_the_arrays_actually_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counter must equal the real resident size, not a shard count.

    Computed from the queue's own arrays rather than from the expected
    constant, so this fails if the counter and the queue ever disagree about
    what has been counted.
    """
    pf = _make(
        tmp_path, n_shards=6, max_queued_bytes=10 * 1024 ** 3, monkeypatch=monkeypatch,
    )
    pf._scan_once()

    from chess_anti_engine.tune.prefetch import _arrays_nbytes

    from_queue = sum(_arrays_nbytes(arrs) for _p, arrs, _m in pf._queue)
    assert pf.queued_bytes() == from_queue
    assert pf.queued_bytes() == 6 * _BYTES_PER_SHARD


def test_draining_resets_the_byte_counter_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A counter that drifts up across drains would wedge prefetch permanently.

    That is the failure mode of maintaining it by subtraction; it is set to
    zero on drain precisely because the whole queue has left.
    """
    pf = _make(
        tmp_path, n_shards=4, max_queued_bytes=10 * 1024 ** 3, monkeypatch=monkeypatch,
    )
    pf._scan_once()
    assert pf.queued_bytes() > 0

    items = pf.drain()

    assert len(items) == 4
    assert pf.queued_bytes() == 0
    assert pf._queue == []


def test_the_budget_frees_up_again_after_a_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backpressure must be transient, not a latch.

    Fill to the cap, drain, and scan again: the second scan must make progress.
    A bound that never releases would silently disable prefetch for the rest of
    the run after one busy period -- a performance regression with no error.
    """
    budget = 2 * _BYTES_PER_SHARD
    pf = _make(tmp_path, n_shards=20, max_queued_bytes=budget, monkeypatch=monkeypatch)

    pf._scan_once()
    assert len(pf._queue) == 2

    pf.drain()
    pf._scan_once()

    assert len(pf._queue) == 2, "the freed budget must be usable again"


def test_arrays_nbytes_ignores_non_array_values() -> None:
    """`load_shard_arrays` attaches metadata into the same dict.

    An exception in the accounting runs on the producer thread, where `_run`'s
    handler would swallow it -- silently disabling prefetch for the rest of the
    run. Degrade instead.
    """
    from chess_anti_engine.tune.prefetch import _arrays_nbytes

    arrs = {
        "x": np.zeros((10, 4), dtype=np.float32),
        "policy_encoding": "lc0_1858",
        "nothing": None,
    }
    assert _arrays_nbytes(arrs) == 10 * 4 * 4


# --------------------------------------------------------------------------
# Deferral is not dropping
# --------------------------------------------------------------------------


def test_deferred_shards_stay_on_disk_for_the_iter_time_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The load-bearing distinction: defer, not drop.

    Over budget, the prefetcher must not consume, move, delete or otherwise
    claim the shards it declined. They stay in inbox/ so the iter-time
    inbox-poll fallback decodes them inline -- which is where that decode
    happened before this module existed, making the deferral a no-op for
    correctness.
    """
    pf = _make(
        tmp_path, n_shards=20, max_queued_bytes=2 * _BYTES_PER_SHARD,
        monkeypatch=monkeypatch,
    )
    before = sorted(p.name for p in tmp_path.iterdir())

    pf._scan_once()

    assert sorted(p.name for p in tmp_path.iterdir()) == before, (
        "a deferred shard must be untouched on disk; the prefetcher is a "
        "latency optimisation over the iter-time path, not the only route in"
    )
    queued = {p.name for p, _, _ in pf._queue}
    assert len(queued) == 2
    assert set(before) - queued, "the rest must still be available to ingest"


def test_a_deferral_is_announced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """"The trainer is draining slower than shards arrive" is a real signal.

    Silent backpressure would look identical to an idle inbox, so the one
    observation distinguishing "bound working" from "bound never engaged"
    would not exist.
    """
    import logging

    pf = _make(
        tmp_path, n_shards=20, max_queued_bytes=_BYTES_PER_SHARD,
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.WARNING, logger=prefetch_mod.__name__):
        pf._scan_once()

    assert "shard prefetch deferred" in caplog.text
    assert pf._deferred_scans == 1


def test_the_deferral_warning_is_rate_limited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The scan runs every `poll_seconds`, so an unconditional warning would
    emit ~once a second for the whole of a long iteration."""
    import logging

    pf = _make(
        tmp_path, n_shards=20, max_queued_bytes=_BYTES_PER_SHARD,
        monkeypatch=monkeypatch,
    )

    with caplog.at_level(logging.WARNING, logger=prefetch_mod.__name__):
        for _ in range(10):
            pf._scan_once()

    assert pf._deferred_scans == 10, "every deferral is counted"
    assert caplog.text.count("shard prefetch deferred") == 1, "but not every one logs"


# --------------------------------------------------------------------------
# Shutdown under backpressure
# --------------------------------------------------------------------------


def test_stop_returns_promptly_while_the_queue_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The deadlock this design avoids by not blocking the producer.

    Backpressure that parked the producer on a condition variable would need
    `stop()` to wake a waiter that might be mid-decode. Deferring means the
    thread is only ever in `wait(poll_seconds)` or scanning, so shutdown is
    unchanged from the unbounded version -- pinned here rather than assumed.
    """
    pf = _make(
        tmp_path, n_shards=200, max_queued_bytes=_BYTES_PER_SHARD,
        monkeypatch=monkeypatch,
    )
    pf.start()
    try:
        deadline = threading.Event()
        deadline.wait(0.5)
        assert pf.queued_bytes() > 0, "the producer must have run at all"
    finally:
        pf.stop(timeout=5.0)

    assert pf._thread is None
    assert not any(t.name == "ShardPrefetch" and t.is_alive() for t in threading.enumerate())


def test_a_drain_after_stop_still_returns_what_was_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown must not strand already-decoded shards.

    They were removed from nowhere -- still on disk too -- but a caller that
    drains after stop should get them rather than silently re-decoding.
    """
    pf = _make(
        tmp_path, n_shards=8, max_queued_bytes=10 * 1024 ** 3, monkeypatch=monkeypatch,
    )
    pf._scan_once()
    pf.start()
    pf.stop(timeout=5.0)

    items = pf.drain()

    assert len(items) >= 8
    assert pf.queued_bytes() == 0


def test_drain_wakes_a_parked_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drain happens at the START of ingest, so the freed budget should be
    usable immediately rather than up to a poll interval later.

    With a deliberately long poll interval, a drain that did not notify would
    leave the queue empty for the whole interval.
    """
    paths = [tmp_path / f"shard_{i:05d}.zarr" for i in range(30)]
    for p in paths:
        p.mkdir()
    monkeypatch.setattr(
        prefetch_mod, "load_shard_arrays", lambda _p: (_fake_arrays(), {}),
    )
    pf = BackgroundShardPrefetcher(
        tmp_path,
        poll_seconds=30.0,  # far longer than this test will wait
        path_iter=lambda _d: sorted(paths),
        max_queued_bytes=2 * _BYTES_PER_SHARD,
    )
    pf.start()
    try:
        threading.Event().wait(0.5)
        assert len(pf._queue) == 2, "filled to the cap on the first scan"

        pf.drain()
        threading.Event().wait(1.0)

        assert len(pf._queue) == 2, (
            "after the drain the producer must re-fill without waiting out the "
            "30s poll interval -- drain() notifies the condition it waits on"
        )
    finally:
        pf.stop(timeout=5.0)


# --------------------------------------------------------------------------
# The invariants this change must NOT disturb
# --------------------------------------------------------------------------


def test_the_deferred_registration_invariant_still_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The module's stated design goal: decode moves off the trainer thread,
    REGISTRATION does not.

    `buf.add_many_arrays` must still happen at iter time on the trainer thread,
    or the sampling distribution changes. The prefetcher holds no buffer, and
    `drain()` hands back raw arrays for the trainer to register itself. A byte
    budget is about how far ahead the decode may run, so it must not touch
    this -- checked by asserting the prefetcher has no route to a buffer at all.
    """
    pf = _make(
        tmp_path, n_shards=3, max_queued_bytes=10 * 1024 ** 3, monkeypatch=monkeypatch,
    )
    pf._scan_once()

    assert not hasattr(pf, "add_many_arrays"), (
        "the prefetcher must expose no registration entry point"
    )
    for name in vars(pf):
        assert "buf" not in name.lower(), (
            f"the prefetcher must not hold a buffer reference ({name!r}); "
            "registration belongs to the trainer thread"
        )
  # What drain() hands back is RAW arrays, so registration is necessarily the
  # caller's to do on its own thread. That is the invariant, and it is a
  # property of the returned value rather than of the source text -- an
  # earlier revision of this test grepped the module for "add_many_arrays",
  # which both tripped on the docstring and would have proved nothing.
    for _sp, arrs, _meta in pf.drain():
        assert isinstance(arrs, dict)
        assert all(hasattr(v, "nbytes") or isinstance(v, (str, type(None), int, float))
                   for v in arrs.values())


def test_drain_preserves_the_consumers_tuple_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`distributed_runtime` unpacks `for sp, arrs, meta in prefetcher.drain()`.

    The byte budget is tracked in a separate counter specifically so this shape
    did not have to change; a 4-tuple would raise at the consumer, which no
    test in this file would otherwise notice.
    """
    pf = _make(
        tmp_path, n_shards=2, max_queued_bytes=10 * 1024 ** 3, monkeypatch=monkeypatch,
    )
    pf._scan_once()

    for sp, arrs, meta in pf.drain():
        assert isinstance(sp, Path)
        assert isinstance(arrs, dict)
        assert isinstance(meta, dict)


def test_a_non_positive_budget_floors_rather_than_disabling_prefetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zeroing a "max" must not mean "never prefetch".

    A literal 0 would make `_queued_bytes >= 0` true on the first check, so the
    prefetcher would defer everything forever -- a silent throughput
    regression with no error, and the opposite of what setting a maximum to
    zero reads like.
    """
    for bad in (0, -1, -1024):
        pf = _make(
            tmp_path / f"n{bad}", n_shards=3, max_queued_bytes=bad,
            monkeypatch=monkeypatch,
        )
        pf._scan_once()
        assert len(pf._queue) >= 1, f"budget={bad} must still prefetch something"


# --------------------------------------------------------------------------
# The config key must reach the consumer
# --------------------------------------------------------------------------


def test_the_default_budget_is_the_documented_size() -> None:
    """768 MB, justified from the measured ~102 MB/shard against a 98 GB box."""
    from chess_anti_engine.tune.prefetch import DEFAULT_MAX_QUEUED_BYTES

    assert DEFAULT_MAX_QUEUED_BYTES == 768 * 1024 * 1024
    assert 6 <= DEFAULT_MAX_QUEUED_BYTES / (102 * 1024 * 1024) <= 8


def test_the_yaml_key_reaches_the_running_prefetcher(tmp_path: Path) -> None:
    """⚑ THE SIGNATURE DEFECT: a knob accepted and then never applied.

    Not a unit test of `TrialConfig`, and not one of the constructor -- the
    whole path, from a raw config dict through `TrialConfig.from_dict` and the
    real `_lazy_construct_iter_helpers` to the value `_scan_once` compares
    against. Asserting on `tc.distributed_prefetch_max_queued_mb` alone would
    pass just as happily with the constructor argument dropped.
    """
    from chess_anti_engine.tune.trainable import _lazy_construct_iter_helpers
    from chess_anti_engine.tune.trial_config import TrialConfig

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    tc = TrialConfig.from_dict({
        "distributed_prefetch_shards": True,
        "distributed_prefetch_max_queued_mb": 42,
    })
    assert tc.distributed_prefetch_max_queued_mb == 42

    pf, _ = _lazy_construct_iter_helpers(
        shard_prefetcher=None, async_test_eval=None, tc=tc,
        distributed_dirs={"inbox_dir": inbox}, iteration_idx=1,
    )
    try:
        assert pf is not None
        assert pf._max_queued_bytes == 42 * 1024 * 1024, (
            "the yaml key must reach the byte comparison in _scan_once, not "
            "just land on the TrialConfig"
        )
    finally:
        if pf is not None:
            pf.stop(timeout=5.0)


def test_an_unset_key_gets_the_default_through_the_same_path(
    tmp_path: Path,
) -> None:
    """The realistic case: production's yaml does not set this key.

    Paired with the test above so a constructor that ignored its argument and
    always used the default could not pass both.
    """
    from chess_anti_engine.tune.trainable import _lazy_construct_iter_helpers
    from chess_anti_engine.tune.trial_config import TrialConfig

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    tc = TrialConfig.from_dict({"distributed_prefetch_shards": True})

    pf, _ = _lazy_construct_iter_helpers(
        shard_prefetcher=None, async_test_eval=None, tc=tc,
        distributed_dirs={"inbox_dir": inbox}, iteration_idx=1,
    )
    try:
        from chess_anti_engine.tune.prefetch import DEFAULT_MAX_QUEUED_BYTES

        assert pf is not None
        assert pf._max_queued_bytes == DEFAULT_MAX_QUEUED_BYTES
    finally:
        if pf is not None:
            pf.stop(timeout=5.0)


def test_the_DEFAULT_bounds_a_fast_producer_with_no_configuration_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE red-green, and deliberately constructed with only the arguments
    `origin/main` also accepts.

    Every other case here passes `max_queued_bytes=`, a keyword main's
    constructor does not have, so on main they fail with a TypeError -- red,
    but red for a signature reason rather than for the defect. This one builds
    the prefetcher exactly as production does, drops 14 shards of 64 MB into
    the inbox, and asks how much the queue took:

    * `origin/main`: all 14, i.e. ~896 MB, and nothing would have stopped it at
      140 shards either -- that is the ~10.2 GB OOM path in miniature;
    * with the bound: 12, i.e. exactly the 768 MB default.

    64 MB of `np.zeros` is virtual until touched, so this costs pages, not RAM.
    """
    mb = 64
    n = 14
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    paths = [inbox / f"shard_{i:05d}.zarr" for i in range(n)]
    for sp in paths:
        sp.mkdir()

    monkeypatch.setattr(
        prefetch_mod,
        "load_shard_arrays",
        lambda _p: ({"x": np.zeros(mb * 1024 * 1024, dtype=np.uint8)}, {}),
    )

  # No max_queued_bytes= -- production does not set the yaml key either, so
  # this is the realized configuration on the live run.
    pf = BackgroundShardPrefetcher(
        inbox, poll_seconds=0.1, path_iter=lambda _d: sorted(paths),
    )
    pf._scan_once()

    held_mb = len(pf._queue) * mb
    assert len(pf._queue) < n, (
        f"the queue took all {n} shards ({held_mb} MB) -- unbounded, which at "
        "the live 102 MB/shard and a cold-start backlog is the OOM path"
    )
    assert held_mb == 768, "the 768 MB default admits exactly 12 x 64 MB"
