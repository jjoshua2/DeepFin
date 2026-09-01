"""``derive_corpus_targets.py --workers N``: the parallel read emits one corpus.

⚑⚑ THE IDENTITY ASSERTED HERE IS OVER THE DATA, NOT OVER THE COMPRESSED FILES,
AND THAT IS NOT A WEAKENING -- IT IS THE STRONGEST CLAIM THAT IS TRUE HERE, and
:func:`codec_is_deterministic` measures whether "here" still applies rather than
hard-coding it: where the codec IS reproducible, :func:`assert_same_corpus`
compares the files too.  The shard
writer goes through ``numcodecs`` Blosc, whose MULTI-THREADED encoder emits
different compressed bytes for identical input on every call, so
``--workers 1`` does not reproduce its own files either.
the test below measures
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

import functools
import hashlib
import io
import itertools
import json
import multiprocessing
import math
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zarr
from numcodecs.blosc import Blosc

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


#: The buffers ``save_local_shard_arrays`` actually hands Blosc.  ``_local_chunks``
#: caps the leading dimension at 512, so these are the LARGEST chunks this tool
#: ever writes -- and the smallest matter too, because Blosc's threading only
#: engages above a block-size threshold and a small enough buffer compresses
#: repeatably even here.  Both the widest (``x``) and the narrowest (``search_wdl``)
#: are probed, at the writer's own dtype and codec parameters.
_PROBE_SHAPES: tuple[tuple[int, ...], ...] = (
    (512, 175, 8, 8), (512, COMPACT_POLICY_SIZE), (512, 3),
)


def _probe_digests() -> list[str]:
    """Compress each probe shape three times; return the digests, in order."""
    codec = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)
    out: list[str] = []
    for shape in _PROBE_SHAPES:
        count = int(np.prod(shape))
        array = (np.arange(count, dtype=np.float32) % 7.0).astype(np.float16)
        array = array.reshape(shape)
        out.extend(
            hashlib.sha256(codec.encode(array)).hexdigest() for _ in range(3)
        )
    return out


@functools.lru_cache(maxsize=1)
def codec_is_deterministic() -> bool:
    """Would ``--workers 1`` and the repack write the same BYTES for one array?

    ⚑⚑ PROBED, IN BOTH PROCESS ROLES, because the two paths do not use the same
    encoder.  ``numcodecs`` disables Blosc's thread pool outside the main
    process: ``--workers 1`` writes in the MAIN process (threaded) and the
    repack writes in a SPAWNED lane (context encoder), and the two emit
    different bytes for identical input.  A probe taken only in the parent
    answers "is this process repeatable", which is a different question from the
    one :func:`assert_same_corpus` needs -- and today the difference is masked
    only because the threaded encoder is not repeatable either.  If a future
    Blosc made it repeatable, a parent-only probe would return True and every
    identity test would start demanding byte equality between a threaded write
    and a child's -- red for a reason that has nothing to do with ``--workers``.

    ⚑ ALSO NOT ASSUMED IN THE OTHER DIRECTION.  Today it returns False, which is
    why the assertions in this file are over decompressed arrays.  That is a
    property of the installed ``numcodecs`` and of
    ``numcodecs.blosc.use_threads``, either of which can change under us, so the
    regime is measured and every assertion states the strongest thing true in it.
    ``test_whether_the_shard_files_are_byte_reproducible_is_measured`` checks
    this verdict against what two real derivations actually wrote, so a probe
    that stops matching reality is itself a failure.
    """
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        child = pool.apply(_probe_digests)
    parent = _probe_digests()
    per_shape = len(parent) // len(_PROBE_SHAPES)
    return all(
        len({*parent[i:i + per_shape], *child[i:i + per_shape]}) == 1
        for i in range(0, len(parent), per_shape)
    )


def file_bytes(root: Path) -> dict[str, str]:
    """Every file under the output dir but the summary, by sha256."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != derive.SUMMARY_NAME
    }


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
    if codec_is_deterministic():
        # ⚑ THE ASSERTION TIGHTENS ITSELF.  Where the codec is reproducible
        # there is no reason to settle for array equality, and requiring it here
        # means the day someone pins `use_threads = False` (or numcodecs stops
        # varying) every test in this file starts checking the bytes without
        # anyone having to notice.
        one, two = file_bytes(sequential), file_bytes(parallel)
        assert sorted(one) == sorted(two)
        by_bytes = sorted(key for key in one if one[key] != two[key])
        assert not by_bytes, (
            f"the codec is deterministic here and {len(by_bytes)} file(s) still "
            f"differ, first {by_bytes[:6]}"
        )
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


def test_whether_the_shard_files_are_byte_reproducible_is_measured(
    tmp_path: Path,
) -> None:
    """⚑⚑ WHY THE ASSERTIONS HERE ARE OVER ARRAYS AND NOT OVER FILES.

    Two SEQUENTIAL derivations of one corpus at one ``--seed`` write the same
    numbers.  Whether they write the same BYTES is a property of the installed
    Blosc, not of this tool: the multi-threaded encoder's output varies call to
    call.  ⚑ MEASURED 2026-08-31 on ``numcodecs 0.13.1`` -- one process
    compressing one array four times gave four digests, and two back-to-back
    ``--workers 1`` runs differed in 11 of 324 files, every one an ``x`` or
    ``policy_target`` chunk (the arrays big enough for the threading to engage).

    ⚑ THE ASSERTION IS NOT "THE FILES MUST DIFFER".  That would be a test of the
    codec's flakiness, and it would fail the day the tool became reproducible --
    exactly when it should instead start demanding more.  Both regimes are
    stated here, and :func:`assert_same_corpus` follows the same rule, so a
    deterministic environment silently upgrades every identity test in this file
    from array equality to byte equality.
    """
    rows = [row for gid in range(6) for row in game(gid, 9)]
    corpus_dir = write_split_corpus(tmp_path, rows, [20, 20, 14])
    first, second = tmp_path / "one", tmp_path / "two"
    run(corpus_dir, first)
    run(corpus_dir, second)

    # The DATA is identical either way, and that is what every other test here
    # compares.
    assert shard_content(first) == shard_content(second)

    one, two = file_bytes(first), file_bytes(second)
    assert sorted(one) == sorted(two)
    differing = sorted(key for key in one if one[key] != two[key])

    # ⚑⚑ THE PROBE IS CHECKED AGAINST WHAT TWO REAL DERIVATIONS WROTE, and this
    # is the assertion that keeps the whole scheme honest. `codec_is_deterministic`
    # decides whether `assert_same_corpus` compares bytes; if it ever disagreed
    # with reality -- it probes buffers up to 11.5 MB while the tool's smallest
    # chunk is a few KB, and Blosc's threading engages on SIZE -- the identity
    # tests would silently sit at array-only comparison for good, or go red for
    # a reason unrelated to --workers. An earlier cut of this test asserted
    # `all(... for key in differing)`, which `all([])` satisfies vacuously and
    # so could not notice either.
    assert bool(differing) != codec_is_deterministic(), (
        f"the codec probe says deterministic={codec_is_deterministic()} and two "
        f"identical sequential derivations {'differ in ' + str(len(differing)) + ' file(s)' if differing else 'wrote identical files'}. "
        "The probe no longer describes what this tool writes, so the byte "
        "comparison in assert_same_corpus is gated on the wrong answer."
    )
    if differing:
        # Only Blosc's own compressed chunks may vary; a `.zarray`, a `.zattrs`
        # or a chunk of a small column moving would be a real difference.
        assert all(key.endswith(("/0.0", "/0.0.0.0")) for key in differing), (
            f"something other than a compressed chunk differs: {differing[:6]}"
        )


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
        # ⚑ RAW float64, the form the lanes append -- headerless, so a stream
        # can be flushed in pieces without a header to keep consistent.
        path = tmp_path / f"lane{index}_{name}.f64"
        np.asarray(values, dtype=np.float64).tofile(path)
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
        streams = derive._stream_files(lanes)
        assert list(derive._stream_values(streams[name])) == [*head, *tail]
        merged = derive._merge_stats(lanes, streams=streams)
        reference = derive.DeriveStats()
        for value in derive._stream_values(streams[name]):
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
    """Handed the lanes out of order, the fold still reads lane 0 first."""
    lanes = [
        _lane(1, tmp_path, value_delta=[3.0, 4.0]),
        _lane(0, tmp_path, value_delta=[1.0, 2.0]),
    ]
    files = derive._stream_files(lanes)["value_delta"]
    assert [Path(path).name for path in files] == [
        "lane0_value_delta.f64", "lane1_value_delta.f64",
    ]
    assert list(derive._stream_values(files)) == [1.0, 2.0, 3.0, 4.0]


def test_the_fold_never_materialises_the_whole_stream(tmp_path: Path) -> None:
    """⚑ The lanes drain to disk to bound memory; the coordinator must not undo it.

    ``_stream_files`` hands back PATHS. Returning the concatenated floats would
    move ~1 GB from seven lanes into one coordinator, where adding lanes cannot
    reduce it -- the same gigabyte the drain exists to avoid, in the worse place.

    ⚑ ``isinstance(..., Iterator)`` IS NOT THE OBSERVATION, and the first cut of
    this test made exactly that mistake.  A generator that builds the whole list
    and then ``yield from``\\ s it is an ``Iterator`` too, and passed -- a mutant
    doing precisely that SURVIVED.  What bounds the peak is that a value arrives
    before the NEXT file is opened, so that is what is measured: the second path
    does not exist, and the first file's values still come out.  A materialising
    fold raises before yielding anything.
    """
    lanes = [_lane(0, tmp_path, value_delta=[1.0, 2.0])]
    files = derive._stream_files(lanes)["value_delta"]
    assert all(isinstance(path, str) for path in files)
    probe = derive._stream_values([*files, str(tmp_path / "never_written.f64")])
    assert isinstance(probe, Iterator)
    assert [next(probe), next(probe)] == [1.0, 2.0]
    with pytest.raises(FileNotFoundError):
        next(probe)


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


def test_a_drained_stream_is_the_values_in_order_however_often_it_flushed(
    tmp_path: Path,
) -> None:
    """⚑ The bank is a BUFFER; the file is the record, and it is appended to.

    A lane drains at every spill cut, so a stream reaches disk in pieces. Two
    ways for that to go wrong silently: a drain that does not clear its buffer
    re-banks everything it already wrote, and a drain that truncates keeps only
    the last piece. Either changes the sequence the coordinator folds, and the
    fold is the summary.
    """
    bank = derive._BankingStats(tmp_path)
    for value in (1.0, 2.0, 3.0):
        bank.note_value_delta(value)
    bank.drain()
    for value in (4.0, 5.0):
        bank.note_value_delta(value)
    bank.drain()
    bank.drain()  # nothing pending; must not disturb the file
    stored = np.fromfile(bank.stream_path("value_delta"), dtype=np.float64)
    assert stored.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    # And the counters the drain does NOT own are still the sequential ones.
    reference = derive.DeriveStats()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        reference.note_value_delta(value)
    assert bank.value_delta_n == reference.value_delta_n
    assert bank.value_delta_sum == reference.value_delta_sum


# ── does the flag take effect at all? ────────────────────────────────────────


def test_the_lane_count_reaches_the_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ EVERY IDENTITY TEST ABOVE PASSES IF ``--workers`` IS SILENTLY IGNORED.

    That is the whole point of them: the output is supposed to be the same. So
    a driver that parsed ``--workers 3``, ran one lane, and wrote the right
    corpus would satisfy all of them -- a value accepted and then ignored, which
    is this repo's signature defect wearing the disguise of a passing suite.

    The observation that separates "took effect" from "was accepted" is the
    SPILL: N lanes means N lane directories, each holding rows it derived. The
    spill is removed on success, so the removal is stubbed out to look at it --
    and the stub is in the coordinator's own process, which is where the removal
    happens.
    """
    monkeypatch.setattr(derive, "_remove_spill", lambda spill_dir: None)
    rows = [row for gid in range(9) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [12] * 6)
    out = tmp_path / "out"
    run(corpus_dir, out, "--workers", "3")

    spill = out / derive.SPILL_DIR_NAME
    lanes = sorted(path.name for path in spill.iterdir() if path.is_dir())
    assert lanes == ["w000", "w001", "w002"], (
        f"--workers 3 produced {lanes}; the flag reached the partition but not "
        "the lanes"
    )
    for lane in lanes:
        chunks = sorted((spill / lane).glob("chunk_*.zarr"))
        assert chunks, f"lane {lane} derived nothing"
    # And the rows really are split: no lane holds all of them.
    per_lane = {
        lane: sum(
            int(np.asarray(zarr.open_group(str(chunk), mode="r")["x"]).shape[0])
            for chunk in (spill / lane).glob("chunk_*.zarr")
        )
        for lane in lanes
    }
    assert sum(per_lane.values()) == 72, per_lane
    assert max(per_lane.values()) < 72, (
        f"one lane derived every row: {per_lane}"
    )


# ── the multi-chunk spill and repack path ────────────────────────────────────


@pytest.mark.parametrize("value_scheme", ["search", "qzsegment"])
def test_many_spill_chunks_per_lane_derive_the_same_corpus(
    tmp_path: Path, value_scheme: str,
) -> None:
    """⚑⚑ THE PATH PRODUCTION ALWAYS TAKES AND NO TEST EVER DID.

    A lane cuts a spill file every ``--spill-chunk-rows`` surviving rows, so a
    5.5M-row derivation writes thousands per lane -- while every fixture corpus
    here is small enough that each lane wrote exactly ONE, leaving the
    multi-chunk spill, the cross-chunk repack slice and the repeated drain with
    no coverage at all. MEASURED by an independent review of PR #493: with the
    chunk size at its default, deleting the final ``drain`` broke nothing and a
    ``drain`` that truncated instead of appending passed all twelve identity
    tests above.

    Seven rows per chunk against ~60 surviving rows a lane: every lane cuts
    several, most output shards are stitched from more than one, and the drain
    runs many times per lane.
    """
    rows = [row for gid in range(16) for row in game(gid, 11 + gid % 4)]
    cuts = [30] * (len(rows) // 30)
    cuts.append(len(rows) - sum(cuts))
    corpus_dir = write_split_corpus(tmp_path, rows, [c for c in cuts if c])
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--value-scheme", value_scheme,
        "--spill-chunk-rows", "7", workers=3, rows_per_shard=11,
    )
    assert_same_corpus(sequential, parallel)


def test_the_spill_chunk_size_reaches_the_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The knob is only worth having if it reaches the code that spills.

    The lanes are spawned children that re-import the module, so this is not a
    thing a monkeypatch could ever have done -- it travels in ``_WorkerTask``.
    Asserted on the spill itself: at 7 rows a chunk every lane must cut several,
    and none may hold more than 7 rows.
    """
    monkeypatch.setattr(derive, "_remove_spill", lambda spill_dir: None)
    rows = [row for gid in range(12) for row in game(gid, 10)]
    corpus_dir = write_split_corpus(tmp_path, rows, [30] * 4)
    out = tmp_path / "out"
    run(corpus_dir, out, "--workers", "2", "--spill-chunk-rows", "7")

    sizes: list[int] = []
    for lane in sorted((out / derive.SPILL_DIR_NAME).iterdir()):
        if not lane.is_dir():
            continue
        chunks = sorted(lane.glob("chunk_*.zarr"))
        assert len(chunks) > 1, f"{lane.name} cut {len(chunks)} chunk(s) at 7 rows"
        sizes.extend(
            int(np.asarray(zarr.open_group(str(chunk), mode="r")["x"]).shape[0])
            for chunk in chunks
        )
    assert max(sizes) == 7, f"a chunk exceeded the requested size: {sizes}"
    assert sum(sizes) == 120


def test_plan_repack_tiles_the_survivors_across_chunks_and_lanes() -> None:
    """Every surviving row lands in exactly one output shard, in order."""
    survivors = [10, 7, 3]
    chunks = [[4, 4, 2], [4, 3], [3]]
    plans = derive._plan_repack(survivors, chunks, 8)
    assert [sum(part.hi - part.lo for part in plan) for plan in plans] == [8, 8, 4]
    # Walk the plan back into a global row order and require 0..19 exactly once.
    offsets: dict[tuple[int, int], int] = {}
    cursor = 0
    for lane, lane_chunks in enumerate(chunks):
        for chunk, rows in enumerate(lane_chunks):
            offsets[(lane, chunk)] = cursor
            cursor += rows
    seen: list[int] = []
    for plan in plans:
        for part in plan:
            base = offsets[(part.worker, part.chunk)]
            seen.extend(range(base + part.lo, base + part.hi))
    assert seen == list(range(20))


def test_plan_repack_refuses_a_spill_that_does_not_match_the_survivors() -> None:
    with pytest.raises(derive.ParallelDeriveError, match="the same rows"):
        derive._plan_repack([10], [[4, 4]], 8)


def test_a_mid_corpus_zero_claim_is_refused_once_a_lane_bases_on_it(
    tmp_path: Path,
) -> None:
    """⚑ ``_check_rows_read`` end to end: two lanes computing the same base.

    A shard claiming 0 rows in the MIDDLE leaves the shards after it with global
    bases that are 8 too low. With three lanes, lane 1 (shard 1) and lane 2
    (shard 2) both start at row 8 and both read the rows the limit allows, so
    the lanes between them read 16 rows for a partition that covers 12. Neither
    `_check_claimed_rows` (shard 1 is never read to EOF, so its claim is never
    checked) nor the closed-key check (this arm assembles no game) sees it.
    This is the guard that does, and it had no end-to-end test.
    """
    rows = [row for gid in range(3) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [8, 8, 8], claim={1: 0})
    assert run(corpus_dir, tmp_path / "seq", "--limit", "12")[
        "realized"]["rows_read"] == 12
    out = tmp_path / "par"
    with pytest.raises(derive.ParallelDeriveError, match="do not tile"):
        run(corpus_dir, out, "--limit", "12", "--workers", "3")
    assert not (out / derive.SUMMARY_NAME).exists()


def test_the_same_zero_claim_is_harmless_when_the_limit_stops_first(
    tmp_path: Path,
) -> None:
    """⚑ AND IT IS NOT REFUSED WHEN IT CANNOT MATTER, which is worth pinning.

    The same corpus at two lanes puts shards 1 and 2 in ONE lane, which reads
    them in order and stops at the limit inside shard 1 -- exactly where the
    sequential read stops. The mis-based shard is never reached, so the run is
    correct and succeeds. A guard that refused here would be refusing a corpus
    `--workers 1` derives, on the strength of a claim that changed nothing.
    """
    rows = [row for gid in range(3) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [8, 8, 8], claim={1: 0})
    sequential, parallel = both_ways(
        tmp_path, corpus_dir, "--limit", "12", workers=2,
    )
    assert_same_corpus(sequential, parallel)


def test_plan_ranges_at_a_limit_exactly_equal_to_the_claimed_total() -> None:
    """The boundary the binding predicate turns on, which `<` and `>` miss."""
    _, in_play, rows = derive.plan_ranges([10, 10, 10], workers=2, limit=30)
    assert (in_play, rows) == (3, 30)
    _, in_play, rows = derive.plan_ranges([10, 10, 10], workers=2, limit=29)
    assert (in_play, rows) == (3, 29)
    _, in_play, rows = derive.plan_ranges([10, 10, 10], workers=2, limit=31)
    assert (in_play, rows) == (3, 30)


def test_an_overstated_claim_refuses_rather_than_deriving_a_short_corpus(
    tmp_path: Path,
) -> None:
    """⚑ STATED, because it IS a difference in what the two paths accept.

    A shard holding fewer rows than the inventory claims derives fine at
    `--workers 1` and is REFUSED above it. That is the fail-closed direction --
    the partition's global row indices come from those claims, so believing them
    would cut `--limit` somewhere the sequential read does not -- but it is the
    same category as the ungrouped-arm test above, so it is a test rather than a
    surprise.
    """
    rows = [row for gid in range(3) for row in game(gid, 8)]
    corpus_dir = write_split_corpus(tmp_path, rows, [8, 8, 8], claim={1: 9})
    run(corpus_dir, tmp_path / "seq")
    with pytest.raises(derive.ParallelDeriveError, match="not the rows on disk"):
        run(corpus_dir, tmp_path / "par", "--workers", "2")
