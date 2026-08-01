"""The resume seed of the hot shuffle pool must use the steady-state draw.

``_scan_existing_shards`` seeds a freshly-constructed ``DiskReplayBuffer``'s
shuffle pool from disk. It used to take the NEWEST ``2 * refresh_shards``
shards — in production 10 of 834, i.e. 1.2% of the window — so every restart
began training on an almost-degenerate slice of the replay window. The pool is
memory-only and never checkpointed, so this recurred at every restart.

The steady-state refresh (``_load_refresh_chunks``) does not do that: it draws
shards with weight linear in recency, which puts the mean recency percentile of
sampled rows at 2/3 rather than ~1.0. Seeding through the same draw costs the
same 10 shard loads and zero extra I/O.

The measurable claim, and what this test pins: the mean recency percentile of
the SEEDED rows must land near 2/3, not near 1.0. Rows are tagged with their
source shard's rank through the ``priority`` column, which the append path
stores verbatim under the default (no-op) shaping knobs.

This test FAILS on the pre-fix code, which seeds a fixed slice — the newest 10
of the 100 shards below — for a statistic of 0.951, deterministically and for
every seed (a fixed slice does not consult the rng).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta,
    local_shard_path,
    save_local_shard_arrays,
)

N_SHARDS = 100
REFRESH_SHARDS = 5  # production value; the seed loads 2x this
N_SEEDS = 16
BIG = 1_000_000  # capacity / pool cap far above the window: never trims

# Recency percentile of a row, (rank + 1) / N_SHARDS with rank oldest-first.
# Steady-state refresh theory is 2/3; the newest-only seed is ~1.0. The gate
# sits between them, far from both: 0.85 is >8 sd above the 2/3 mean at this
# sample size and 0.095 below the pre-fix value.
MAX_MEAN_RECENCY = 0.85

# Agreement required between the reported mean_recency and the same statistic
# measured off the pool the constructor built.
RECENCY_TOL = 1e-3


def _rows_for(rank: int) -> int:
    """Rows in the shard at *rank* — deliberately NOT a constant.

    Equal-sized shards make a row-weighted mean and an unweighted mean over
    the drawn shards numerically identical, so a fixture with a fixed rows-
    per-shard cannot tell an implementation that honours the row weights from
    one that drops them: the guard would be unable to fire. Sizes here run
    1..49 and, since 17 is coprime to 49, hit every value in that range.

    The spread is deliberately UNCORRELATED with rank (period 49 across 100
    ranks) rather than monotonic in it. A monotonic size ramp would also
    discriminate the weighting, but it would additionally shift the
    row-weighted recency statistic itself (to 0.75 under the fix, for weights
    linear in recency), forcing MAX_MEAN_RECENCY to be retuned. Decorrelating
    size from rank buys the discrimination without moving any threshold.
    """
    return 1 + (rank * 17) % 49


def _shard_arrays(rank: int) -> dict[str, np.ndarray]:
    """One shard's rows, tagged with their source shard's rank."""
    n = _rows_for(rank)
    policy = np.zeros((n, 4672), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        # The tag. Default sf_gap_priority_weight=0.0 and
        # fast_low_surprise_priority=1.0 make _shape_shuffle_priority a no-op,
        # so this reaches _shuffle_priority_store unchanged.
        "priority": np.full((n,), float(rank), dtype=np.float32),
        # Mixed W/D/L so no balance path is degenerate; irrelevant to the tag.
        "wdl_target": np.full((n,), rank % 3, dtype=np.int8),
        "has_policy": np.ones((n,), dtype=np.uint8),
    }


def _write_window(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "replay_shards"
    shard_dir.mkdir()
    for rank in range(N_SHARDS):
        save_local_shard_arrays(
            local_shard_path(shard_dir, rank),
            arrs=_shard_arrays(rank),
            meta=ShardMeta(positions=_rows_for(rank)),
        )
    return shard_dir


def _open(shard_dir: Path, seed: int) -> DiskReplayBuffer:
    return DiskReplayBuffer(
        capacity=BIG,  # no window trimming
        shard_dir=shard_dir,
        rng=np.random.default_rng(seed),
        read_only=True,
        shuffle_cap=BIG,  # no pool trimming
        refresh_interval=0,  # no prefetch thread; the seed path is unaffected
        refresh_shards=REFRESH_SHARDS,
    )


def _pool_ranks(buf: DiskReplayBuffer) -> np.ndarray:
    """Source-shard rank of every row that landed in the hot pool."""
    return np.asarray(buf._active_shuffle_priority(), dtype=np.float64)


def test_seeded_pool_is_recency_weighted_not_newest_only(tmp_path) -> None:
    shard_dir = _write_window(tmp_path)

    per_open: list[float] = []
    for seed in range(N_SEEDS):
        buf = _open(shard_dir, seed)
        try:
            ranks = _pool_ranks(buf)
        finally:
            buf.close()
        assert ranks.size > 0, "seed loaded no rows at all"
        per_open.append(float(np.mean((ranks + 1.0) / N_SHARDS)))

        # I/O volume is unchanged: still at most 2 * refresh_shards shard
        # LOADS, each contributing its shard's rows exactly once. A "fix" that
        # widened the draw by reading more shards, or one that drew with
        # replacement and paid for the same shard twice, is a different change
        # with a different cost; these two are the guard against both.
        #
        # Stated for the record: this replaces a bound on the seeded ROW count
        # (<= 2 * refresh_shards * ROWS_PER_SHARD), which is not expressible
        # now that shards differ in size. It is not a relaxation -- it bounds
        # the shard loads directly, which is the quantity that costs I/O, and
        # it additionally catches the double-load the row bound could not see.
        drawn = np.unique(ranks).astype(np.int64)
        assert drawn.size <= 2 * REFRESH_SHARDS, (
            f"seed loaded {drawn.size} shards, more than 2 * refresh_shards"
        )
        expected_rows = sum(_rows_for(int(r)) for r in drawn)
        assert int(ranks.size) == expected_rows, (
            f"seed pooled {ranks.size} rows from shards {drawn.tolist()} but "
            f"those shards hold {expected_rows}: a shard was loaded twice"
        )

    mean_recency = float(np.mean(per_open))
    assert mean_recency < MAX_MEAN_RECENCY, (
        f"seeded pool mean recency percentile {mean_recency:.3f} >= "
        f"{MAX_MEAN_RECENCY}: the seed is drawing from the newest shards only, "
        f"not through the steady-state recency-weighted draw "
        f"(per-open: {[round(v, 3) for v in per_open]})"
    )


def test_seeded_pool_spans_more_than_the_newest_shards(tmp_path) -> None:
    """The complementary direction: seeded rows must reach BELOW the newest slice.

    The statistic above is a mean and a mean can be moved without widening the
    support. This asserts the support directly — across opens, the seed must
    touch shards older than the newest ``2 * refresh_shards``, which the
    pre-fix code cannot do by construction.
    """
    shard_dir = _write_window(tmp_path)
    newest_only_cutoff = N_SHARDS - 2 * REFRESH_SHARDS  # rank 90 and up

    touched: set[int] = set()
    for seed in range(N_SEEDS):
        buf = _open(shard_dir, seed)
        try:
            touched.update(np.asarray(buf._active_shuffle_priority(), dtype=np.int64).tolist())
        finally:
            buf.close()

    older = sorted(r for r in touched if r < newest_only_cutoff)
    assert older, (
        f"every seeded row came from the newest {2 * REFRESH_SHARDS} shards "
        f"(ranks touched: {sorted(touched)})"
    )


def test_seed_log_line_reports_the_pool_it_actually_built(tmp_path, capsys) -> None:
    """The open-time line must equal the statistic measured off the pool itself.

    A number that is printed but computed from something other than the rows
    that landed in the pool would be exactly the defect this line exists to
    detect, so it is checked against an independent measurement rather than
    against itself.

    The specific way it could go wrong and still look plausible is by averaging
    over the shards DRAWN instead of over the rows POOLED — the two agree
    exactly when shards are the same size, which is why ``_rows_for`` varies
    them. The final assertion is a positive control on that: it requires the
    unweighted statistic to actually diverge here, so a later fixture change
    that flattened the sizes would fail loudly rather than silently disarm the
    check above.
    """
    shard_dir = _write_window(tmp_path)

    gaps_vs_unweighted: list[float] = []
    for seed in range(N_SEEDS):
        buf = _open(shard_dir, seed)
        try:
            ranks = _pool_ranks(buf)
        finally:
            buf.close()

        lines = [
            ln for ln in capsys.readouterr().out.splitlines()
            if "[disk_buf] shuffle seed:" in ln
        ]
        assert len(lines) == 1, f"expected exactly one seed line, got {lines}"
        line = lines[0]

        match = re.search(r"mean_recency=([0-9.]+)", line)
        assert match is not None, f"no mean_recency in seed line: {line}"
        reported = float(match.group(1))

        pct = (ranks + 1.0) / N_SHARDS
        observed = float(np.mean(pct))  # row-weighted: every pooled row counts
        assert abs(reported - observed) < RECENCY_TOL, (
            f"seed line reports mean_recency={reported} but the pool it built "
            f"measures {observed:.3f} ({line})"
        )
        assert f"rows={ranks.size}" in line, line
        assert f"/{N_SHARDS} " in line, line

        unweighted = float(np.mean((np.unique(ranks) + 1.0) / N_SHARDS))
        gaps_vs_unweighted.append(abs(observed - unweighted))

    assert max(gaps_vs_unweighted) > 10 * RECENCY_TOL, (
        "shard sizes are too uniform for this test to distinguish a "
        "row-weighted mean from a per-shard mean, so the assertion above "
        f"cannot fail: max gap {max(gaps_vs_unweighted):.4f}"
    )
