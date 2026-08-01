"""The shuffle-pool refresh must be seed-reproducible when asked to be.

``sample_batch_arrays`` refreshes the hot pool every ``refresh_interval``
draws, and it has TWO implementations of that refresh which it picks between by
who won a race: the background prefetch thread's chunk if one is ready,
otherwise a synchronous load. The two do not consume the same generator --
the prefetched chunk was drawn from ``_prefetch_rng`` and leaves ``self.rng``
untouched, while the synchronous path draws from ``self.rng`` and ADVANCES it.
So one lost race changes both which shards enter the pool and the state of the
generator that selects every subsequent row, permanently and irreversibly.

Whether the thread wins depends on machine load, so a seeded offline ruler was
not in fact a function of its seed. Measured 2026-07-31 on the held-out policy
probe: 7 identical invocations produced 3 distinct draw sequences spanning
0.0056 nats of "excess over floor", against a rig built to resolve 0.0126.

``deterministic_refresh=True`` suppresses the prefetch thread so every refresh
takes the synchronous path. What this file pins:

1. the flag actually removes the thread (the gate is wired at all), and the
   default still creates one -- so a change that made the flag a no-op fails
   here rather than silently restoring the race;
2. with the flag, two same-seed buffers draw byte-identical batches across a
   span that crosses many refreshes;
3. the flag does not change the DISTRIBUTION -- the recency profile of the
   rows it draws matches the default path's. The fix has to buy determinism
   without moving the ruler it makes reproducible.

Point 3 is the one that cannot be waved through: both paths call the same
``_load_refresh_chunks`` with the same ``refresh_shards``, so only the stream
differs, and the test holds them to that.

Counterfactual, run on this fixture 2026-07-31 before shipping: with
``deterministic_refresh=False``, 3 of 6 seeds drew DIFFERENT rows on two
identical constructions, 603-687 of 960 sampled rows apart -- i.e. when the
race is lost the sequence does not drift, it desynchronises outright, because
the synchronous path advances ``self.rng`` and the prefetched one does not.
With the flag, 0 of 6. Test 2 is therefore a check that can fail, not a
tautology; it is only stated with the flag on because the failure mode it
guards is a race and a test of a race is a flaky test.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta,
    local_shard_path,
    save_local_shard_arrays,
)

N_SHARDS = 60
ROWS_PER_SHARD = 40
REFRESH_INTERVAL = 4  # production value
REFRESH_SHARDS = 5  # production value
BATCH = 16
# 60 draws at refresh_interval 4 is 15 refreshes: every one of them is a race
# under the default, so a flag that only postponed the divergence would still
# be caught.
DRAWS = 60
POOL_CAP = 1200  # well under the window, so the pool really does churn
SEEDS = (0, 1, 2)


def _shard_arrays(rank: int) -> dict[str, np.ndarray]:
    """One shard's rows, every row tagged with its source shard's rank.

    The tag rides in ``priority``: under the default shaping knobs
    (``sf_gap_priority_weight=0.0``, ``fast_low_surprise_priority=1.0``)
    ``_shape_shuffle_priority`` is a no-op, so the value is stored verbatim and
    can be read back off the pool. ``x`` carries the same rank so a batch's
    identity is checkable from the batch alone.
    """
    n = ROWS_PER_SHARD
    policy = np.zeros((n, 4672), dtype=np.float32)
    policy[:, rank % 4672] = 1.0
    return {
        "x": np.full((n, 146, 8, 8), float(rank), dtype=np.float32),
        "policy_target": policy,
        "priority": np.full((n,), float(rank), dtype=np.float32),
        "wdl_target": np.arange(n, dtype=np.int8) % 3,
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def _write_window(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "replay_shards"
    shard_dir.mkdir()
    for rank in range(N_SHARDS):
        save_local_shard_arrays(
            local_shard_path(shard_dir, rank),
            arrs=_shard_arrays(rank),
            meta=ShardMeta(positions=ROWS_PER_SHARD),
        )
    return shard_dir


def _open(shard_dir: Path, seed: int, *, deterministic: bool) -> DiskReplayBuffer:
    return DiskReplayBuffer(
        capacity=10**9,
        shard_dir=shard_dir,
        rng=np.random.default_rng(seed),
        read_only=True,
        shuffle_cap=POOL_CAP,
        refresh_interval=REFRESH_INTERVAL,
        refresh_shards=REFRESH_SHARDS,
        deterministic_refresh=deterministic,
    )


def _draw_ranks(buf: DiskReplayBuffer) -> np.ndarray:
    """Source-shard rank of every row drawn, in draw order."""
    out = []
    for _ in range(DRAWS):
        arrs = buf.sample_batch_arrays(BATCH)
        # x is constant-valued at the shard rank, so [:, 0, 0, 0] IS the rank.
        out.append(np.asarray(arrs["x"], dtype=np.float64)[:, 0, 0, 0])
    return np.concatenate(out)


def _prefetch_threads_alive() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("replay-prefetch-"))


def test_deterministic_refresh_removes_the_prefetch_thread(tmp_path) -> None:
    """The gate is wired -- and the default is unchanged.

    Both halves matter. Without the second assertion a flag that had been
    turned into a no-op, or a default that had silently acquired the
    deterministic behaviour, would both read as a pass.
    """
    shard_dir = _write_window(tmp_path)
    before = _prefetch_threads_alive()

    buf = _open(shard_dir, 0, deterministic=True)
    try:
        _draw_ranks(buf)
        assert _prefetch_threads_alive() == before, (
            "deterministic_refresh=True still started a shuffle-refresh prefetch "
            "thread; the refresh can still be served from _prefetch_rng and the "
            "draw sequence is not a function of the seed"
        )
        assert buf._prefetched_refresh is None
    finally:
        buf.close()

    loud = _open(shard_dir, 0, deterministic=False)
    try:
        _draw_ranks(loud)
        assert _prefetch_threads_alive() > before, (
            "the default no longer starts a prefetch thread -- either the "
            "production path lost its async refresh, or deterministic_refresh "
            "has become unconditional"
        )
    finally:
        loud.close()


def test_deterministic_refresh_is_reproducible_across_buffers(tmp_path) -> None:
    """Same seed, same directory, same rows -- byte for byte, every seed."""
    shard_dir = _write_window(tmp_path)
    for seed in SEEDS:
        runs = []
        for _ in range(2):
            buf = _open(shard_dir, seed, deterministic=True)
            try:
                runs.append(_draw_ranks(buf))
            finally:
                buf.close()
        assert np.array_equal(runs[0], runs[1]), (
            f"seed {seed}: two identical deterministic-refresh buffers drew "
            f"different rows ({int((runs[0] != runs[1]).sum())} of "
            f"{runs[0].size} positions differ)"
        )

    # Different seeds must still differ, or "reproducible" would be satisfied
    # by a buffer that ignored its seed and always drew the same rows.
    a = _open(shard_dir, SEEDS[0], deterministic=True)
    b = _open(shard_dir, SEEDS[1], deterministic=True)
    try:
        assert not np.array_equal(_draw_ranks(a), _draw_ranks(b))
    finally:
        a.close()
        b.close()


def test_deterministic_refresh_preserves_the_draw_distribution(tmp_path) -> None:
    """Determinism must not come from a narrower draw.

    Both refresh paths call ``_load_refresh_chunks`` with the same
    ``refresh_shards`` and the same recency weighting, so the only difference
    is which generator supplies the shard indices. The observable consequence
    is that the recency profile of the drawn rows is the same; a "fix" that
    reached determinism by, say, pinning the refresh to the newest shards
    would pass both tests above and fail this one.

    Compared across seeds rather than within one: a single seed's mean is a
    draw from the same distribution in both arms, so agreement of two point
    estimates would prove nothing. The gate is on the pooled mean over
    ``N_SEED_DIST`` seeds against a tolerance set from the pooled sd.
    """
    shard_dir = _write_window(tmp_path)
    n_seed_dist = 12
    det: list[float] = []
    default: list[float] = []
    for seed in range(n_seed_dist):
        for deterministic, sink in ((True, det), (False, default)):
            buf = _open(shard_dir, seed, deterministic=deterministic)
            try:
                ranks = _draw_ranks(buf)
            finally:
                buf.close()
            sink.append(float(np.mean((ranks + 1.0) / N_SHARDS)))

    d_arr = np.asarray(det)
    b_arr = np.asarray(default)
    se = float(
        np.sqrt(d_arr.var(ddof=1) / d_arr.size + b_arr.var(ddof=1) / b_arr.size)
    )
    gap = abs(float(d_arr.mean()) - float(b_arr.mean()))
    # 4 se, and a floor so a degenerate-variance run cannot make the gate
    # unfailable-tight rather than unfailable-loose.
    assert gap <= max(4.0 * se, 0.02), (
        f"deterministic refresh drew from a different part of the window: "
        f"mean recency {d_arr.mean():.4f} vs default {b_arr.mean():.4f} "
        f"(gap {gap:.4f}, 4se {4.0 * se:.4f})"
    )
