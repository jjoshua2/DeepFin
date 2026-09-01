"""``derive_corpus_targets.py --workers N``: the parallel read emits one corpus.

⚑⚑ THE IDENTITY ASSERTED HERE IS OVER THE DATA, NOT OVER THE COMPRESSED FILES,
AND THAT IS NOT A WEAKENING -- IT IS THE STRONGEST CLAIM THAT IS TRUE.  The shard
writer goes through ``numcodecs`` Blosc, whose MULTI-THREADED encoder emits
different compressed bytes for identical input on every call, so
``--workers 1`` does not reproduce its own files either.
:func:`test_the_shard_files_are_not_byte_reproducible_even_sequentially` measures
exactly that, and it is in this file so nobody "tightens" the assertions below
into something that fails at random.  What every test here compares is what a
shard MEANS: each array's dtype, shape and decompressed bytes, the ``.zattrs``
stamps, the shard layout, and the summary.

The fixtures write the corpus's JSONL shards DIRECTLY rather than through
``corpus.ShardWriter``, and that is deliberate: the writer rotates only in
``end_game``, so it can only produce shards whose boundaries are game
boundaries -- the one input shape under which the handoff this file exists to
test is a no-op.  ⚑ The production corpus does NOT have that shape.  MEASURED on
``run02_snap_20260829``: its shards hold exactly 8,192 rows each and
``w00-00000`` ends on game 1176 while ``w00-00001`` opens on it.  So the rows
themselves are still built through the generator's own ``PhaseResult.as_row``
(via :mod:`tests.test_derive_corpus_targets`'s helpers); only the placement of
the shard cuts is this file's.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus
from tests.test_derive_corpus_targets import (
    CONFIG_SHA,
    FEN_B,
    FEN_W,
    corpus_row,
    full_width_phase,
    ramp,
    write_manifest,
)

# ── fixtures ─────────────────────────────────────────────────────────────────


def _row(
    *, worker_id: int, game_id: int, ply: int, result: float | None,
    played_move: Any = None,
) -> dict[str, Any]:
    """One corpus row carrying the d7/d8/d9 rungs every value scheme reads.

    The cps vary with ``(game_id, ply)`` so no two rows derive to the same
    policy: an identity assertion over rows that were all the same vector would
    pass under a partition that lost track of which row went where.
    """
    fen = FEN_W if ply % 2 == 0 else FEN_B
    moves = sorted(move.uci() for move in chess.Board(fen).legal_moves)
    best = moves[(game_id + ply) % len(moves)]
    base = 400.0 - 7.0 * ((game_id * 13 + ply * 5) % 21)
    kwargs: dict[str, Any] = {}
    if played_move is not None:
        kwargs["played_move"] = played_move
    return corpus_row(
        fen=fen,
        phases=[full_width_phase(fen, {
            7: ramp(fen, best, best_cp=base - 12.0),
            8: ramp(fen, best, best_cp=base - 5.0),
            9: ramp(fen, best, best_cp=base),
        })],
        result=result,
        game_id=game_id,
        ply=ply,
        worker_id=worker_id,
        **kwargs,
    )


def game(game_id: int, plies: int, *, result: float | None = None,
         worker_id: int = 0) -> list[dict[str, Any]]:
    """One game's contiguous run of rows."""
    outcome = [1.0, 0.0, -1.0][game_id % 3] if result is None else result
    return [
        _row(worker_id=worker_id, game_id=game_id, ply=ply, result=outcome)
        for ply in range(plies)
    ]


def write_split_corpus(
    tmp_path: Path, rows: list[dict[str, Any]], cuts: list[int],
    *, name: str = "corpus", claim: dict[int, int] | None = None,
) -> Path:
    """A ``manifest + progress`` corpus whose shard cuts are ``cuts``.

    ``claim`` overrides what the inventory CLAIMS a shard holds, which is how
    the claimed-vs-actual guard is exercised: the partition's global row indices
    are computed from the claims, so a wrong claim is a wrong ``--limit``.
    """
    out = tmp_path / name
    out.mkdir()
    module = corpus.zstandard_module()
    assert module is not None, "the corpus fixtures need the zstandard module"
    progress: list[dict[str, Any]] = []
    start = 0
    for index, size in enumerate(cuts):
        chunk = rows[start:start + size]
        # ⚑ A zero-row shard IS written when a cut asks for one: the inventory
        # can name one, and a lane whose preceding shard is empty must still
        # find the row before its range.
        path = out / f"w00-{index:05d}.jsonl.zst"
        with open(path, "wb") as binary:
            writer = module.ZstdCompressor().stream_writer(binary)
            text = io.TextIOWrapper(writer, encoding="utf-8")
            for row in chunk:
                text.write(json.dumps(row, sort_keys=True) + "\n")
            text.flush()
            text.detach()
            writer.close()
        progress.append({
            "codec": "zstd", "path": str(path),
            "rows": (claim or {}).get(index, len(chunk)),
        })
        start += size
    assert start == len(rows), f"the cuts cover {start} of {len(rows)} rows"
    (out / corpus.progress_name(0)).write_text(
        "".join(json.dumps(entry) + "\n" for entry in progress), encoding="utf-8",
    )
    write_manifest(out, config_sha=CONFIG_SHA)
    return out


def run(corpus_dir: Path, out_dir: Path, *extra: str, temp: float = 0.5,
        rows_per_shard: int = 37) -> dict[str, Any]:
    code = derive.main([
        "--corpus", str(corpus_dir), "--out", str(out_dir),
        "--scheme", "uniform-d9", "--temp", str(temp), "--seed", "0",
        "--rows-per-shard", str(rows_per_shard), *extra,
    ])
    assert code == 0
    return json.loads(
        (out_dir / derive.SUMMARY_NAME).read_text(encoding="utf-8"),
    )


def shard_content(out_dir: Path) -> dict[str, str]:
    """What the shards MEAN: every array's data, plus every stamp.

    Keyed ``shard/array`` so a mismatch names the shard and the column rather
    than a chunk file, and hashed over ``tobytes()`` after the dtype and shape
    are folded into the value -- so a float16 column that came back as float32
    holding the same numbers is a difference, and a NaN compares equal to
    itself (``np.array_equal`` would not).
    """
    content: dict[str, str] = {}
    for shard in sorted(out_dir.glob("shard_*.zarr")):
        group = zarr.open_group(str(shard), mode="r")
        content[f"{shard.name}::.zattrs"] = json.dumps(
            dict(group.attrs), sort_keys=True, default=str,
        )
        for name in sorted(group.array_keys()):
            array = np.asarray(group[name])
            content[f"{shard.name}::{name}"] = (
                f"{array.dtype.str}|{array.shape}|"
                + hashlib.sha256(
                    np.ascontiguousarray(array).tobytes(),
                ).hexdigest()
            )
    return content


def canonical_summary(path: Path) -> dict[str, Any]:
    """The summary with every NaN replaced by a sentinel, minus ``started_utc``.

    ⚑ EXPLICIT, not left to ``==``.  ``float('nan') != float('nan')``, so two
    summaries that both report an empty reading compare UNEQUAL -- except that
    CPython's ``json`` scanner hands out one shared NaN object and dict equality
    takes an identity shortcut, so today they happen to compare equal.  That is
    an implementation detail of the parser, and an assertion that rests on it is
    one interpreter release away from failing for a reason that has nothing to
    do with this flag.  A NaN reading is a REAL and expected value here (an
    unfloored run's recovered floor, a value-less arm's blend weight), so it is
    mapped to a sentinel that compares equal to itself and to nothing else.
    """
    def canon(value: object) -> object:
        if isinstance(value, dict):
            return {key: canon(sub) for key, sub in value.items()}
        if isinstance(value, list):
            return [canon(item) for item in value]
        if isinstance(value, float) and math.isnan(value):
            return "<nan>"
        return value

    out = json.loads(path.read_text(encoding="utf-8"))
    out.pop("started_utc")
    return canon(out)  # pyright: ignore[reportReturnType]


def assert_same_corpus(sequential: Path, parallel: Path) -> None:
    """The two output directories are one corpus, described one way."""
    want, got = shard_content(sequential), shard_content(parallel)
    assert sorted(want) == sorted(got), (
        f"shard/array layout differs: only sequential "
        f"{sorted(set(want) - set(got))[:6]}, only parallel "
        f"{sorted(set(got) - set(want))[:6]}"
    )
    differing = sorted(key for key in want if want[key] != got[key])
    assert not differing, f"{len(differing)} array(s) differ, first {differing[:6]}"
    assert canonical_summary(sequential / derive.SUMMARY_NAME) == (
        canonical_summary(parallel / derive.SUMMARY_NAME)
    )


#: Distinct output directories inside one test's ``tmp_path``.
_RUN_COUNTER = itertools.count()


def both_ways(
    tmp_path: Path, corpus_dir: Path, *extra: str, workers: int,
    rows_per_shard: int = 37,
) -> tuple[Path, Path]:
    """Derive the same corpus sequentially and in parallel; return both dirs."""
    tag = next(_RUN_COUNTER)
    sequential = tmp_path / f"seq_{tag}"
    parallel = tmp_path / f"par_{tag}"
    run(corpus_dir, sequential, *extra, rows_per_shard=rows_per_shard)
    run(corpus_dir, parallel, *extra, "--workers", str(workers),
        rows_per_shard=rows_per_shard)
    return sequential, parallel


# ── the property the whole file rests on ─────────────────────────────────────


def test_the_shard_files_are_not_byte_reproducible_even_sequentially(
    tmp_path: Path,
) -> None:
    """⚑⚑ WHY THE ASSERTIONS HERE ARE OVER ARRAYS AND NOT OVER FILES.

    Two SEQUENTIAL derivations of one corpus at one ``--seed`` write the same
    numbers and different bytes, because ``numcodecs`` Blosc's multi-threaded
    encoder is non-deterministic.  This test fails the day that stops being
    true, which is the day a byte-level assertion would become legitimate --
    and until then it is the reason not to write one.
    """
    rows = [row for gid in range(6) for row in game(gid, 9)]
    corpus_dir = write_split_corpus(tmp_path, rows, [20, 20, 14])
    first, second = tmp_path / "one", tmp_path / "two"
    run(corpus_dir, first)
    run(corpus_dir, second)

    def files(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != derive.SUMMARY_NAME
        }

    one, two = files(first), files(second)
    assert sorted(one) == sorted(two)
    differing = [key for key in one if one[key] != two[key]]
    assert differing, (
        "the shard files are byte-reproducible now, so the identity claim in "
        "this file and in derive_corpus_targets.py can be tightened from "
        "'every array' to 'every file' -- read the --workers section first"
    )
    # And the DATA is identical, which is the claim every other test makes.
    assert shard_content(first) == shard_content(second)


# ── the merge table cannot silently miss a counter ───────────────────────────


def test_the_merge_rules_name_every_derive_stats_field() -> None:
    """⚑⚑ THE ANTI-DRIFT GATE.

    A counter added to :class:`DeriveStats` without a merge rule would read 0 in
    every parallel run and nothing else would notice -- a value that is accepted
    and then silently ignored, which is this repo's signature defect aimed at
    the instrument that reports it.
    """
    declared = {spec.name for spec in fields(derive.DeriveStats)}
    assert declared == derive._MERGE_COVERAGE, (
        f"no merge rule for {sorted(declared - derive._MERGE_COVERAGE)}; "
        f"a rule names {sorted(derive._MERGE_COVERAGE - declared)} which is not "
        "a DeriveStats field"
    )


def test_every_ordered_stream_names_a_real_stats_method() -> None:
    for name, method in derive._ORDERED_STREAMS:
        assert callable(getattr(derive.DeriveStats(), method)), method
        assert name in derive._STREAM_OWNED_FIELDS


# ── the partition ────────────────────────────────────────────────────────────


def test_plan_ranges_tiles_the_shards_it_puts_in_play() -> None:
    counts = [10] * 12
    ranges, in_play, rows = derive.plan_ranges(counts, workers=5, limit=0)
    assert in_play == 12
    assert rows == 120
    assert ranges[0].lo == 0
    assert ranges[-1].hi == 12
    for left, right in itertools.pairwise(ranges):
        assert left.hi == right.lo


def test_plan_ranges_ignores_the_shards_the_limit_never_reaches() -> None:
    """A 25-row limit over 12 ten-row shards opens three shards, not twelve."""
    ranges, in_play, rows = derive.plan_ranges([10] * 12, workers=7, limit=25)
    assert (in_play, rows) == (3, 25)
    assert len(ranges) == 3
    assert ranges[-1].hi == 3


def test_plan_ranges_never_hands_a_lane_an_empty_range() -> None:
    ranges, in_play, _ = derive.plan_ranges([1000, 1, 1, 1], workers=4, limit=0)
    assert in_play == 4
    assert [(r.lo, r.hi) for r in ranges] == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_plan_ranges_refuses_a_corpus_the_limit_never_opens() -> None:
    with pytest.raises(derive.ParallelDeriveError, match="no corpus shard"):
        derive.plan_ranges([], workers=2, limit=5)


def test_plan_ranges_refuses_a_lane_count_below_one() -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        derive.plan_ranges([10], workers=0, limit=0)


# ── identity, scheme by scheme ───────────────────────────────────────────────


@pytest.mark.parametrize("value_scheme", ["search", "qz50", "qzphase", "qzsegment"])
@pytest.mark.parametrize("workers", [2, 3, 7])
def test_the_parallel_read_derives_the_sequential_corpus(
    tmp_path: Path, value_scheme: str, workers: int,
) -> None:
    """Every arm, every lane count, over shard cuts that fall INSIDE games."""
    rows = [row for gid in range(20) for row in game(gid, 11 + gid % 5)]
    # 40-row cuts against 11-15-row games: no cut lands on a game boundary.
    cuts = [40] * (len(rows) // 40)
    cuts.append(len(rows) - sum(cuts))
    corpus_dir = write_split_corpus(tmp_path, rows, [c for c in cuts if c])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", value_scheme, workers=workers,
    )
    assert_same_corpus(sequential, parallel)


def test_the_spill_directory_is_gone_when_the_run_succeeds(tmp_path: Path) -> None:
    rows = [row for gid in range(8) for row in game(gid, 9)]
    corpus_dir = write_split_corpus(tmp_path, rows, [30, 30, 12])
    out = tmp_path / "out"
    run(corpus_dir, out, "--workers", "3")
    assert not (out / derive.SPILL_DIR_NAME).exists()
    assert sorted(p.name for p in out.iterdir() if p.is_dir())


# ── the adversarial boundaries ───────────────────────────────────────────────


@pytest.mark.parametrize("value_scheme", ["qzphase", "qzsegment"])
def test_a_boundary_exactly_at_a_game_start_carries_nothing(
    tmp_path: Path, value_scheme: str,
) -> None:
    """The skip run is empty and the overflow stops on its first row."""
    rows = [row for gid in range(6) for row in game(gid, 10)]
    corpus_dir = write_split_corpus(tmp_path, rows, [20, 20, 20])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", value_scheme, workers=3,
    )
    assert_same_corpus(sequential, parallel)


@pytest.mark.parametrize("value_scheme", ["qzphase", "qzsegment"])
def test_a_game_that_swallows_a_whole_range_is_owned_by_one_lane(
    tmp_path: Path, value_scheme: str,
) -> None:
    """⚑ The middle lane skips every row it was given and must NOT overflow.

    Its carry-in key and its range-end key are the same game, so the game
    belongs to the lane before it -- which is already carrying through.  A lane
    that overflowed anyway would derive the following rows a second time.
    """
    rows = game(0, 6) + game(1, 40) + game(2, 6)
    corpus_dir = write_split_corpus(tmp_path, rows, [8, 8, 8, 8, 8, 8, 4])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", value_scheme, workers=7,
    )
    assert_same_corpus(sequential, parallel)


@pytest.mark.parametrize("value_scheme", ["qzphase", "qzsegment"])
def test_a_dropped_tail_at_the_boundary_is_not_lost(
    tmp_path: Path, value_scheme: str,
) -> None:
    """⚑⚑ THE CASE THAT KILLS "OVERFLOW ON THE LAST ROW I PROCESSED".

    The shard before the cut ENDS on rows of game 2 that every scheme drops (a
    null ``result``), so the lane's last SURVIVING row belongs to game 1 while
    the next lane's skip predicate -- read off the raw JSON -- names game 2.  An
    overflow anchored on the surviving row stops at the cut, the next lane skips
    game 2's rows in its own range, and those rows are derived by nobody.
    """
    rows = (
        game(0, 8)
        + game(1, 8)
        + [_row(worker_id=0, game_id=2, ply=ply, result=None) for ply in range(3)]
        + [_row(worker_id=0, game_id=2, ply=ply, result=1.0) for ply in range(3, 9)]
        + game(3, 8)
    )
    # Cut immediately after the three dropped rows: the boundary sits inside
    # game 2's run and the leading part of that run is entirely dropped.
    corpus_dir = write_split_corpus(tmp_path, rows, [19, 14])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", value_scheme, workers=2,
    )
    assert_same_corpus(sequential, parallel)
    summary = json.loads(
        (parallel / derive.SUMMARY_NAME).read_text(encoding="utf-8"),
    )
    assert summary["realized"]["rows_dropped_no_result"] == 3, (
        "the fixture stopped dropping rows at the boundary, so this test no "
        "longer exercises the raw-key rule it exists for"
    )


@pytest.mark.parametrize("value_scheme", ["search", "qzphase"])
def test_the_limit_cuts_at_the_same_global_row(
    tmp_path: Path, value_scheme: str,
) -> None:
    """A limit inside a lane's range, mid-game."""
    rows = [row for gid in range(12) for row in game(gid, 9)]
    corpus_dir = write_split_corpus(tmp_path, rows, [27] * 4)
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", value_scheme, "--limit", "70",
        workers=4,
    )
    assert_same_corpus(sequential, parallel)


def test_the_limit_lands_inside_a_lanes_overflow(tmp_path: Path) -> None:
    """The carried game runs past the cut and the budget stops it there.

    ⚑ Both the carrying lane and the skipping lane read the row at
    ``limit - 1``; only the carrying one has an open game, and its flush is the
    one that must be stamped ``cut_by_limit``.
    """
    rows = game(0, 10) + game(1, 20) + game(2, 10)
    corpus_dir = write_split_corpus(tmp_path, rows, [12, 12, 8, 8])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", "qzphase", "--limit", "18",
        workers=4,
    )
    assert_same_corpus(sequential, parallel)
    summary = json.loads(
        (parallel / derive.SUMMARY_NAME).read_text(encoding="utf-8"),
    )
    assert summary["realized"]["value_scheme_realized"]["games_cut_by_limit"] == 1


def test_the_limit_exactly_on_a_shard_boundary(tmp_path: Path) -> None:
    """⚑ ``rows_read >= limit`` is TRUE when the limit equals the rows read.

    The sequential path stamps ``cut_by_limit`` on its final flush whenever the
    budget was met -- including when it was met exactly at a shard's last row --
    so a lane that stopped because its range ended must still say so.
    """
    rows = [row for gid in range(9) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [24, 24, 24])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", "qzphase", "--limit", "48",
        workers=3,
    )
    assert_same_corpus(sequential, parallel)


def test_a_limit_smaller_than_one_range(tmp_path: Path) -> None:
    """Seven lanes were asked for and one shard is in play."""
    rows = [row for gid in range(9) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [24, 24, 24])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", "qzsegment", "--limit", "15",
        workers=7,
    )
    assert_same_corpus(sequential, parallel)


def test_a_boundary_inside_a_leading_drop_run(tmp_path: Path) -> None:
    """The first rows of the next lane's range are dropped rows of the carried
    game, so the carrying lane must read PAST them to find where the game ends.
    """
    rows = (
        game(0, 10)
        + [_row(worker_id=0, game_id=1, ply=ply, result=1.0) for ply in range(4)]
        + [_row(worker_id=0, game_id=1, ply=ply, result=None) for ply in range(4, 8)]
        + [_row(worker_id=0, game_id=1, ply=ply, result=1.0) for ply in range(8, 12)]
        + game(2, 10)
    )
    corpus_dir = write_split_corpus(tmp_path, rows, [14, 10, 8])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", "qzsegment", workers=3,
    )
    assert_same_corpus(sequential, parallel)


# ── the runtime guards ───────────────────────────────────────────────────────


def test_an_inventory_that_miscounts_a_shard_is_refused(tmp_path: Path) -> None:
    """⚑ The claims place ``--limit``; a wrong claim is a wrong cut.

    Refused after the lanes have spilled and BEFORE any shard is finalised, so
    the failure leaves no half-described corpus behind.
    """
    rows = [row for gid in range(8) for row in game(gid, 9)]
    corpus_dir = write_split_corpus(tmp_path, rows, [24, 24, 24], claim={1: 25})
    out = tmp_path / "out"
    with pytest.raises(derive.ParallelDeriveError, match="not the rows on disk"):
        run(corpus_dir, out, "--workers", "3")
    assert not list(out.glob("shard_*.zarr"))
    assert not (out / derive.SUMMARY_NAME).exists()


def test_the_envelope_tolerance_is_checked_on_the_whole_read(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE FAIL-OPEN A PARTITION CREATES, AND THE CHECK THAT CLOSES IT.

    ``--max-envelope-misses`` is a budget over the WHOLE read.  Split three ways,
    three misses can be one per lane -- under every lane's own copy of the
    sequential refusal -- and the run would succeed where ``--workers 1``
    refuses.  Two lanes' worth of rows carry a d9-less block here, so no single
    lane exceeds a budget of 1 and the merged count does.
    """
    def blind(game_id: int, ply: int) -> dict[str, Any]:
        fen = FEN_W if ply % 2 == 0 else FEN_B
        moves = sorted(move.uci() for move in chess.Board(fen).legal_moves)
        return corpus_row(
            fen=fen,
            # d5 only: --scheme uniform-d9 cannot be answered by this row.
            phases=[full_width_phase(fen, {5: ramp(fen, moves[0])})],
            result=1.0, game_id=game_id, ply=ply, worker_id=0,
        )

    rows = [
        *game(0, 9), blind(1, 0), *game(2, 9), blind(3, 0), *game(4, 9),
    ]
    corpus_dir = write_split_corpus(tmp_path, rows, [10, 10, 9])
    sequential = tmp_path / "seq"
    with pytest.raises(derive.CorpusIntegrityError, match="envelope miss"):
        run(corpus_dir, sequential, "--max-envelope-misses", "1")
    parallel = tmp_path / "par"
    with pytest.raises(derive.CorpusIntegrityError, match="envelope miss"):
        run(corpus_dir, parallel, "--max-envelope-misses", "1", "--workers", "3")
    assert not (parallel / derive.SUMMARY_NAME).exists()


def test_a_tolerated_envelope_miss_reports_the_same_first_examples(
    tmp_path: Path,
) -> None:
    """The examples are the first EIGHT IN READ ORDER, not per lane."""
    def blind(game_id: int, ply: int) -> dict[str, Any]:
        fen = FEN_W if ply % 2 == 0 else FEN_B
        moves = sorted(move.uci() for move in chess.Board(fen).legal_moves)
        return corpus_row(
            fen=fen, phases=[full_width_phase(fen, {5: ramp(fen, moves[0])})],
            result=1.0, game_id=game_id, ply=ply, worker_id=0,
        )

    # ⚑ TEN MISSES IN THE FIRST LANE AND THREE IN THE LAST, and both halves of
    # that matter. The first lane exceeds its OWN cap of 8, so the claim the cap
    # rests on -- a lane's ninth miss can never be in the global first eight --
    # is exercised rather than assumed. The last lane contributes candidates the
    # merge must then order BEHIND them; with every miss in one lane the merged
    # list is exactly 8 long and its first eight and last eight are the same
    # list, which is how an earlier version of this fixture let a mutant that
    # reported the LAST eight survive.
    rows: list[dict[str, Any]] = []
    for gid in range(5):
        rows.extend(game(gid, 3))
        rows.extend(blind(gid, ply) for ply in (3, 4))
    for gid in range(5, 10):
        rows.extend(game(gid, 5))
    for gid in range(10, 13):
        rows.extend(game(gid, 4))
        rows.append(blind(gid, 4))
    cuts = [25, 25, 15]
    corpus_dir = write_split_corpus(tmp_path, rows, cuts)
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--max-envelope-misses", "50", workers=3,
    )
    assert_same_corpus(sequential, parallel)
    got = json.loads(
        (parallel / derive.SUMMARY_NAME).read_text(encoding="utf-8"),
    )["realized"]
    assert len(got["envelope_miss_examples"]) == 8
    assert got["rows_dropped_envelope"] == 13
    # The global first eight by read order -- not a round-robin over the lanes,
    # and not the last eight.
    assert [text.split(":")[0] for text in got["envelope_miss_examples"]] == [
        f"game {gid} ply {ply}" for gid in range(4) for ply in (3, 4)
    ]


def test_the_default_path_is_the_sequential_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--workers 1`` must not construct the parallel driver.

    ⚑ Asserted by making the driver EXPLODE rather than by comparing output: a
    default that quietly routed through a one-lane coordinator would produce the
    right corpus and leave the identity claim with nothing to compare against,
    so no output assertion could tell the difference.
    """
    rows = [row for gid in range(4) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [16, 16])

    def refuse(**_: Any) -> dict[str, Any]:
        raise AssertionError("--workers 1 reached the parallel driver")

    monkeypatch.setattr(derive, "derive_parallel", refuse)
    run(corpus_dir, tmp_path / "out")
    run(corpus_dir, tmp_path / "out_one", "--workers", "1")


def test_workers_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="--workers must be >= 1"):
        derive.main([
            "--corpus", "x", "--out", "y", "--scheme", "uniform-d9",
            "--workers", "0",
        ])


def test_derive_parallel_refuses_to_be_the_one_lane_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="N > 1 path"):
        derive.derive_parallel(
            corpus_dir=tmp_path, out_dir=tmp_path / "o",
            options=derive.DeriveOptions(
                scheme=derive.parse_scheme("uniform-d9"), temp=1.0,
                cp_slope=1.0, cp_draw_width=1.0, limit=0, seed=0,
                rows_per_shard=8, max_envelope_misses=0,
            ),
            workers=1,
        )


# ── the order-dependent floats ───────────────────────────────────────────────


def _lane(index: int, tmp_path: Path, **streams: list[float]) -> Any:
    """A ``_WorkerResult`` carrying nothing but banked streams."""
    paths: dict[str, str] = {}
    for name, values in streams.items():
        path = tmp_path / f"lane{index}_{name}.npy"
        np.save(path, np.asarray(values, dtype=np.float64))
        paths[name] = str(path)
    return derive._WorkerResult(
        index=index, stats=derive.DeriveStats(), games=[], envelope=[],
        stream_paths=paths, chunk_rows=[], actual_rows={}, closed_keys=[],
        tt_carried=[], survivors=0,
    )


#: ⚑ CHOSEN SO THE TWO FOLDS DISAGREE.  Left to right,
#: ``((1.0 + 1e16) - 1e16) == 0.0`` -- the 1.0 is lost to the exponent gap.
#: Folded as two lane subtotals, ``1.0 + (1e16 - 1e16) == 1.0``.  Any pair of
#: values whose magnitudes straddle float64's 53-bit window does this; these are
#: just the smallest ones to write down.
_NON_ASSOCIATIVE = ([1.0], [1e16, -1e16])


def test_the_ordered_streams_fold_in_read_order_not_as_lane_subtotals(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE REASON THE VALUES ARE BANKED AT ALL.

    Adding each lane's subtotal is not the sequential sum, and on a corpus large
    enough for the exponents to spread it is not even close.  The merge is
    asserted against what ``DeriveStats`` ITSELF computes over the concatenated
    sequence, so this is a statement about agreeing with the sequential path
    rather than about a number this test invented.

    ⚑ A UNIT TEST AND NOT AN END-TO-END ONE, deliberately: a fixture corpus's
    lane subtotals are usually bit-identical to its sequential sum, so the
    identity tests above cannot be relied on to catch a merge that adds
    subtotals.  MEASURED -- that exact mutant survived all of them.
    """
    head, tail = _NON_ASSOCIATIVE
    for name, method in derive._ORDERED_STREAMS:
        lanes = [
            _lane(0, tmp_path, **{name: head}),
            _lane(1, tmp_path, **{name: tail}),
        ]
        streams = derive._concat_streams(lanes)
        assert streams[name] == [*head, *tail]
        merged = derive._merge_stats(lanes, streams=streams)
        reference = derive.DeriveStats()
        for value in streams[name]:
            getattr(reference, method)(value)
        for field_name in derive._STREAM_OWNED_FIELDS[name]:
            assert getattr(merged, field_name) == getattr(reference, field_name), (
                f"{name}.{field_name} disagrees with the sequential fold"
            )
        subtotal = sum(head) + sum(tail)
        assert getattr(merged, f"{name}_sum") != subtotal, (
            f"{name}: the fixture no longer distinguishes the two folds"
        )


def test_the_stream_concatenation_is_in_lane_order(tmp_path: Path) -> None:
    """Handed the lanes out of order, the concatenation still reads lane 0 first."""
    lanes = [
        _lane(1, tmp_path, value_delta=[3.0, 4.0]),
        _lane(0, tmp_path, value_delta=[1.0, 2.0]),
    ]
    assert derive._concat_streams(lanes)["value_delta"] == [1.0, 2.0, 3.0, 4.0]


def test_the_game_stream_is_replayed_through_note_game(tmp_path: Path) -> None:
    """``qz_game_rows_min`` has a first-game sentinel, so it cannot be merged
    with a plain ``min`` over lanes that assembled no game."""
    lanes = [_lane(0, tmp_path), _lane(1, tmp_path), _lane(2, tmp_path)]
    lanes[1].games = [(7, False), (3, True)]
    lanes[2].games = [(11, False)]
    merged = derive._merge_stats(lanes, streams={})
    reference = derive.DeriveStats()
    for rows, cut in [(7, False), (3, True), (11, False)]:
        reference.note_game(rows, cut_by_limit=cut)
    assert merged.qz_games == reference.qz_games == 3
    assert merged.qz_game_rows_min == reference.qz_game_rows_min == 3
    assert merged.qz_game_rows_max == reference.qz_game_rows_max == 11
    assert merged.qz_games_cut_by_limit == reference.qz_games_cut_by_limit == 1


def test_an_empty_shard_before_a_range_does_not_break_the_handoff(
    tmp_path: Path,
) -> None:
    """⚑ The carry-in is the last row BEFORE the range, not of the shard before.

    A zero-row shard has no last row.  A lane that read only ``lo - 1`` would
    get ``None``, skip nothing, and derive the rows the previous lane is
    carrying through a second time.  ⚑ MEASURED that ``_check_closed_keys`` is
    what fires on that (``worker 0 game 1 closed by lanes 0 and 2``), not the
    ``rows_read`` guard -- a crash rather than a wrong corpus either way, but a
    crash is not the answer, and naming the wrong instrument is how a reader
    later concludes the wrong thing is load-bearing.
    """
    rows = game(0, 10) + game(1, 12) + game(2, 10)
    corpus_dir = write_split_corpus(tmp_path, rows, [14, 0, 10, 8])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", "qzphase", workers=4,
    )
    assert_same_corpus(sequential, parallel)


def test_a_row_without_a_game_identity_is_refused_by_name(tmp_path: Path) -> None:
    """The parallel read refuses it saying the same thing the grouper says."""
    rows = game(0, 6)
    rows[3].pop("game_id")
    corpus_dir = write_split_corpus(tmp_path, rows, [3, 3])
    with pytest.raises(derive.CorpusIntegrityError, match="worker_id, game_id"):
        run(corpus_dir, tmp_path / "out", "--value-scheme", "qzphase",
            "--workers", "2")


def test_the_ungrouped_arms_do_not_require_a_worker_id(tmp_path: Path) -> None:
    """⚑ A DIFFERENCE IN WHAT THE FLAG ACCEPTS, WHICH NO OUTPUT DIFF WOULD SHOW.

    ``search`` and ``qz50`` assemble no game, so the sequential path never reads
    ``worker_id``.  A handoff that keyed every row regardless would refuse a
    corpus ``--workers 1`` derives happily.
    """
    rows = [row for gid in range(6) for row in game(gid, 8)]
    for row in rows:
        row.pop("worker_id")
    corpus_dir = write_split_corpus(tmp_path, rows, [16, 16, 16])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", "qz50", workers=3,
    )
    assert_same_corpus(sequential, parallel)


# ── the guards, each made to fire ────────────────────────────────────────────


def test_a_non_binding_limit_puts_every_shard_back_in_play(tmp_path: Path) -> None:
    """⚑⚑ THE GUARD WAS SCOPED BY THE THING IT VALIDATES.

    ``plan_ranges`` drops a shard from play on the strength of the inventory's
    claim, and ``_check_claimed_rows`` can only check a shard some lane opened --
    so a claim that UNDERSTATES could exclude a shard on a number nothing tests.
    Found by an independent review of this PR; before the fix, this corpus read
    24 rows at ``--workers 1`` and 16 above it, with no guard firing and a
    smaller corpus written and stamped.
    """
    rows = [row for gid in range(3) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [8, 8, 8], claim={2: 0})
    sequential = run(corpus_dir, tmp_path / "seq", "--limit", "30")
    assert sequential["realized"]["rows_read"] == 24
    out = tmp_path / "par"
    with pytest.raises(derive.ParallelDeriveError, match="not the rows on disk"):
        run(corpus_dir, out, "--limit", "30", "--workers", "2")
    assert not (out / derive.SUMMARY_NAME).exists()


def test_plan_ranges_only_drops_shards_when_the_limit_binds() -> None:
    _, in_play, rows = derive.plan_ranges([10, 10, 0], workers=2, limit=25)
    assert (in_play, rows) == (3, 20), "a limit past the claims drops nothing"
    _, in_play, rows = derive.plan_ranges([10, 10, 10], workers=2, limit=25)
    assert (in_play, rows) == (3, 25)
    _, in_play, rows = derive.plan_ranges([10, 10, 10], workers=2, limit=15)
    assert (in_play, rows) == (2, 15), "a binding limit still drops the tail"


def _result(index: int, **kwargs: Any) -> Any:
    base: dict[str, Any] = {
        "index": index, "stats": derive.DeriveStats(), "games": [], "envelope": [],
        "stream_paths": {}, "chunk_rows": [], "actual_rows": {}, "closed_keys": [],
        "tt_carried": [], "survivors": 0,
    }
    base.update(kwargs)
    return derive._WorkerResult(**base)


def test_two_lanes_closing_one_game_is_refused() -> None:
    """⚑ The split-game failure, which is the one that yields a NORMAL corpus.

    Unreachable from a corpus while the handoff is right -- so it is asserted on
    the check itself rather than left as a gate nobody has watched fire.
    """
    lanes = [
        _result(0, closed_keys=[(0, 1), (0, 2)]),
        _result(1, closed_keys=[(0, 2), (0, 3)]),
    ]
    with pytest.raises(derive.ParallelDeriveError, match="assembled by two lanes"):
        derive._check_closed_keys(lanes)
    derive._check_closed_keys([
        _result(0, closed_keys=[(0, 1)]), _result(1, closed_keys=[(0, 2)]),
    ])


def test_lanes_that_do_not_tile_the_prefix_are_refused() -> None:
    """The only thing that notices overlapping ranges on an UNGROUPED arm."""
    stats = derive.DeriveStats()
    stats.rows_read = 40
    with pytest.raises(derive.ParallelDeriveError, match="do not tile"):
        derive._check_rows_read([_result(0, stats=stats)], 30)
    derive._check_rows_read([_result(0, stats=stats)], 40)


def test_a_repack_that_loses_a_row_is_refused() -> None:
    with pytest.raises(derive.ParallelDeriveError, match="lost or duplicated"):
        derive._check_rows_written([{"path": "a", "rows": 9}], [10])
    assert derive._check_rows_written([{"path": "a", "rows": 10}], [4, 6]) == 10


def test_a_spill_file_that_lost_rows_is_refused(tmp_path: Path) -> None:
    """⚑ Read off the FILE, so the check is not the lane's word for itself."""
    from chess_anti_engine.replay.shard import ShardMeta, save_local_shard_arrays

    lane = _result(0, chunk_rows=[3], survivors=3)
    path = derive._spill_path(tmp_path, 0, 0)
    save_local_shard_arrays(
        path,
        arrs={
            "x": np.zeros((2, 4, 8, 8), dtype=np.float16),
            "policy_target": np.full(
                (2, COMPACT_POLICY_SIZE), 1.0 / COMPACT_POLICY_SIZE,
                dtype=np.float16,
            ),
            "wdl_target": np.zeros((2,), dtype=np.int8),
            "priority": np.ones((2,), dtype=np.float32),
            "has_policy": np.ones((2,), dtype=np.uint8),
        },
        meta=ShardMeta(run_id="t", policy_size=COMPACT_POLICY_SIZE, positions=2),
    )
    with pytest.raises(derive.ParallelDeriveError, match="the file holds 2"):
        derive._measure_spill(tmp_path, [lane])


def test_the_merge_rules_do_not_name_a_field_twice() -> None:
    """⚑ A frozenset cannot see a duplicate, and a duplicate is a DOUBLE COUNT.

    A stream-owned field also listed in ``_SUM_FIELDS`` would be summed off the
    partials and then replayed on top of that sum, and the coverage gate above
    would still pass because the union collapses the repeat.
    """
    named = (
        derive._SUM_FIELDS
        + derive._MAX_FIELDS
        + derive._DICT_SUM_FIELDS
        + derive._CONSTANT_FIELDS
        + derive._SENTINEL_MIN_FIELDS
        + derive._ORDERED_EXAMPLE_FIELDS
        + derive._REPACK_OWNED_FIELDS
        + derive._GAME_OWNED_FIELDS
        + tuple(n for owned in derive._STREAM_OWNED_FIELDS.values() for n in owned)
    )
    repeated = sorted({n for n in named if named.count(n) > 1})
    assert not repeated, f"{repeated} carry two merge rules and would double-count"
    assert len(named) == len(derive._MERGE_COVERAGE)
