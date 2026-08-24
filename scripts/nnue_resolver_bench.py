"""What check resolution costs, and what the qsearch arm costs on top of it.

⚑ THE OVERHEAD IS NOT ONE NUMBER, and quoting it as one hides the only thing it
depends on. A resolver call on a quiet position is a ``cboard_in_check`` plus a
terminal test and then the same evaluation as before; a call on a position IN
CHECK searches the evasions recursively. So the cost is

    blended = (1 - f) * quiet_cost + f * in_check_cost

and ``f``, the in-check rate of the position stream, is a property of the corpus
generator, not of the resolver. This script measures the two costs SEPARATELY
and reports the blend at the ``f`` the sampler itself observed — a rate that
comes off the sampler's own counter rather than an assumption.

⚑ THE STRATIFIED POOL HAS NO IN-CHECK POSITIONS IN IT. ``sample_fens`` excludes
them by construction, because the raw evaluator (rightly) refuses them and the
parity gate needs positions it can score. That exclusion is exactly why the
in-check pool here is collected separately, from the SAME playout generator with
the SAME seed and capture bias, so the two pools come from one distribution and
``f`` is the fraction the generator actually produced.

Three arms are timed against the same pools:

    nnue          the raw evaluator; quiet positions only, since it refuses check
    nnue-static   recursive check resolution + a static NNUE leaf
    nnue-qsearch  the same resolution + tactical quiescence beyond it

⚑ The box is shared with live training. Run it nice'd, and do not read a
contended number as a ceiling.

Usage::

    PYTHONPATH=. nice -n 19 python3 scripts/nnue_resolver_bench.py \\
        --pack big.pack --n 2000 --in-check 500
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import chess

from scripts.nnue_fens import _playout, sample_fens

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import CBoard


def collect_in_check(
    count: int,
    seed: int,
    capture_bias: float,
    max_playouts: int = 20_000,
) -> tuple[list[str], int, int]:
    """In-check FENs from the same playout generator the stratified sampler uses.

    Returns (fens, in_check_seen, positions_considered). The last two are the
    measured in-check RATE of the stream, and they are returned rather than
    printed so the caller reports one number it can trace.
    """
    rng = random.Random(seed)
    out: list[str] = []
    seen_states: set[str] = set()
    in_check_seen = 0
    considered = 0
    for _ in range(max_playouts):
        for board in _playout(rng, capture_bias):
            considered += 1
            if not board.is_check():
                continue
            in_check_seen += 1
            if len(out) >= count:
                continue
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                continue
            key = f"{board.board_fen()} {'w' if board.turn else 'b'}"
            if key in seen_states:
                continue
            seen_states.add(key)
            out.append(board.fen())
        # ⚑ Keep sampling the RATE after the pool is full: stopping at the first
        # `count` in-check positions would make f a function of --in-check, i.e.
        # of a flag, which is the sort of self-fulfilling measurement this repo
        # keeps paying for. The loop runs a fixed number of playouts either way.
        if considered >= 200_000:
            break
    return out, in_check_seen, considered


def _time_arm(
    arm: str, pack: str, boards: list[CBoard], repeats: int
) -> tuple[float, int, dict[str, int]]:
    """(seconds, evals, stats) for one arm over `boards`, repeated."""
    from chess_anti_engine.nnue import _nnue_ext

    _nnue_ext.arm_eval(arm, pack, boards[: min(64, len(boards))])  # warm the mapping
    t0 = time.perf_counter()
    stats: dict[str, int] = {}
    for _ in range(repeats):
        _values, stats = _nnue_ext.arm_eval(arm, pack, boards)
    return time.perf_counter() - t0, len(boards) * repeats, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000, help="quiet (stratified) positions")
    ap.add_argument("--in-check", type=int, default=500, help="in-check positions")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--capture-bias", type=float, default=0.55)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args(argv)

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.nnue import _nnue_ext

    pack = str(args.pack)
    quiet_fens, sample_stats = sample_fens(args.n, seed=args.seed, capture_bias=args.capture_bias)
    check_fens, in_check_seen, considered = collect_in_check(
        args.in_check, args.seed, args.capture_bias
    )
    if not quiet_fens or not check_fens:
        print("empty position pool; raise --n / --in-check", file=sys.stderr)
        return 2

    f_sampler = sample_stats.in_check_fraction
    f_stream = in_check_seen / considered if considered else 0.0

    print(f"pack            : {args.pack}")
    print(f"net sha256      : {_nnue_ext.source_sha256(_nnue_ext.load(pack))}")
    print(f"kernel          : {'avx2' if _nnue_ext.simd_active() else 'scalar'}")
    print(f"quiet pool      : {len(quiet_fens):,} stratified, not in check")
    print(f"in-check pool   : {len(check_fens):,}")
    print(
        f"in-check rate f : {f_sampler:.4f} (stratified sampler: "
        f"{sample_stats.in_check_excluded:,}/{sample_stats.considered:,})"
        f"  |  {f_stream:.4f} (this script: {in_check_seen:,}/{considered:,})"
    )
    print(f"repeats         : {args.repeats}")
    print()

    quiet = [CBoard.from_board(chess.Board(f)) for f in quiet_fens]
    checks = [CBoard.from_board(chess.Board(f)) for f in check_fens]

    # Baseline: the raw evaluator, quiet positions only. Same batch, same GIL
    # discipline (one release around the whole run) as arm_eval, so the
    # difference is the resolver and not the harness.
    handle = _nnue_ext.load(pack)
    _nnue_ext.benchmark(handle, quiet[: min(64, len(quiet))], 1, 1)
    evals, seconds, _checksum = _nnue_ext.benchmark(handle, quiet, args.repeats, 1)
    raw_rate = evals / seconds if seconds > 0 else float("nan")
    print(f"{'nnue (raw)':<14} {'quiet':<9} {evals:>9,} evals {seconds:7.3f}s "
          f"= {raw_rate:>11,.0f} evals/s")

    rates: dict[tuple[str, str], float] = {}
    for arm in ("nnue-static", "nnue-qsearch"):
        for label, boards in (("quiet", quiet), ("in-check", checks)):
            seconds, evals, stats = _time_arm(arm, pack, boards, args.repeats)
            rate = evals / seconds if seconds > 0 else float("nan")
            rates[(arm, label)] = rate
            leaves = stats["resolved_leaves"] or 1
            print(
                f"{arm:<14} {label:<9} {evals:>9,} evals {seconds:7.3f}s "
                f"= {rate:>11,.0f} evals/s  "
                f"nodes/leaf={stats['nodes'] / leaves:5.2f} "
                f"maxdepth={stats['max_depth_seen']:<2} "
                f"cutoffs={stats['depth_cutoffs']} "
                f"mate={stats['terminal_mate']:,} draw={stats['terminal_draw']:,} "
                f"qnodes={stats['qnodes']:,} qcut={stats['qply_cutoffs']:,} "
                f"in_check_frac={stats['calls_in_check'] / max(stats['calls'], 1):.3f}"
            )
            # ⚑ A depth cutoff means a line was neither resolved nor terminal and
            # was scored 0. Defensible as a backstop, indefensible as a silent
            # one — so the bench SAYS SO rather than leaving it in a column.
            if stats["depth_cutoffs"]:
                print(
                    f"  ⚑ {stats['depth_cutoffs']:,} depth cutoffs on {arm}/{label}: "
                    f"the recursion cap bound. Those values are floors, not evaluations.",
                    file=sys.stderr,
                )
    print()

    # ⚑ TWO REFERENCE POINTS, AND ONLY ONE OF THEM IS SOUND.
    #
    # "% below the raw evaluator" crosses two harnesses: the baseline runs
    # through _nnue_ext.benchmark() and the arms through arm_eval(). They do the
    # same evaluations but not the same bookkeeping, and the gap between them is
    # not zero — the static arm has measured FASTER than the raw baseline on the
    # quiet pool, which is a harness difference and certainly not a speedup from
    # adding work. Quote it only as an order of magnitude.
    #
    # The defensible number is the arm against ITS OWN quiet rate: same harness,
    # same code path, the only difference being how many of the positions were in
    # check. That isolates what check resolution costs.
    for arm in ("nnue-static", "nnue-qsearch"):
        q, c = rates[(arm, "quiet")], rates[(arm, "in-check")]
        if q <= 0 or c <= 0:
            continue
        print(f"{arm:<14} in-check eval costs {q / c:4.2f}x a quiet one (same harness)")
        for name, f in (("sampler", f_sampler), ("stream", f_stream)):
            blended = 1.0 / ((1.0 - f) / q + f / c)
            print(
                f"{arm:<14} blended at f={f:.4f} ({name:<7}) = {blended:>11,.0f} evals/s"
                f"   [{100.0 * (q - blended) / q:5.1f}% below this arm's own quiet rate"
                f"; {100.0 * (raw_rate - blended) / raw_rate:6.1f}% vs raw, cross-harness]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
