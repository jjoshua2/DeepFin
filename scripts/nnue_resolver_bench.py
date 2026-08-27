"""What check resolution costs, and what the qsearch arm costs on top of it.

⚑ THE OVERHEAD IS NOT ONE NUMBER, and quoting it as one hides the only thing it
depends on. A resolver call on a quiet position is a ``cboard_in_check`` plus a
terminal test and then the same evaluation as before; a call on a position IN
CHECK searches the evasions recursively. So the cost is

    blended = (1 - f) * quiet_cost + f * in_check_cost

and ``f``, the in-check rate of the position stream, is a property of the corpus
generator, not of the resolver. This script measures the two costs separately
and reports the blend at the ``f`` the same stream produced.

⚑⚑ THE TWO CONDITIONS ARE SAMPLED FROM ONE STREAM, WITH MATCHED WEIGHTS. An
earlier version blended the rate of the STRATIFIED pool (round-robin over
(bucket, threat-bin) cells, which deliberately over-represents thin cells) with
the rate of a first-N in-check pool taken from raw playouts. Weighting two
differently-drawn conditional samples by the natural in-check fraction estimates
neither distribution: it is the composition of a population nobody generated. So
both pools are now RESERVOIR-SAMPLED over the same fixed set of playouts, which
makes each a uniform draw from its own condition in one stream, and ``f`` is
measured over that same stream.

The stratified pool is still reported, because it is what ``nnue_bench.py`` uses
and it spans all eight layer stacks — but it is labelled COVERAGE and is never
blended.

⚑ POSITIONS CARRY THEIR MOVE HISTORY. Rebuilding a board from its FEN throws away
the move stack, and the resolver CONSULTS that history: two-fold repetition is
what makes a perpetual check terminate. A FEN-only pool silently measures a
resolver that can never see a repetition, so the boards here are snapshotted with
their stacks and converted whole.

⚑ The box is shared with live training. Run it nice'd, and do not read a
contended number as a ceiling.

Usage::

    PYTHONPATH=. nice -n 19 python3 scripts/nnue_resolver_bench.py \\
        --pack big.pack --quiet-n 2000 --check-n 500 --bank runs/resolver_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chess

from scripts.nnue_fens import sample_fens

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import CBoard

BANK_SCHEMA = 2


@dataclass
class Sample:
    """One position, with everything needed to re-stratify it later."""

    board: chess.Board          # WITH its move stack
    fen: str
    game_key: str
    ply: int
    in_check: bool


def resolver_key(board: chess.Board) -> tuple[object, ...]:
    """A dedup key over everything the RESOLVER can tell apart.

    ⚑ NOT placement+side-to-move. That is the key the parity sampler uses, and it
    is right there because the raw evaluator reads only the bitboards and the
    turn. The resolver reads more: castling rights and the en-passant square
    change the evasions, the halfmove clock decides the fifty-move terminal, and
    the repetition history decides the two-fold one. Two positions equal under
    the evaluator's key can be completely different resolver inputs, and
    collapsing them would drop real work from the measurement.

    Over-specific is safe here (it merely dedups less), so the key is the full
    FEN plus the transposition keys of the current reversible run — the window a
    repetition can possibly be found in.
    """
    run: list[object] = []
    probe = board.copy(stack=True)
    run.append(probe._transposition_key())
    while probe.move_stack:
        move = probe.pop()
        if probe.is_irreversible(move):
            break
        run.append(probe._transposition_key())
    return (board.fen(), tuple(run))


def sample_stream(
    seed: int,
    capture_bias: float,
    playouts: int,
    want_quiet: int,
    want_check: int,
    max_plies: int = 300,
) -> tuple[list[Sample], list[Sample], int, int]:
    """Reservoir-sample both conditions over one fixed set of playouts.

    Returns (quiet, in_check, considered, in_check_seen). Reservoir rather than
    first-N: the quiet condition is ~95% of the stream, so a first-N quiet pool
    would come from the opening plies of a handful of games while a first-N
    in-check pool spanned all of them. Both pools are uniform over their
    condition, drawn from the same games, which is what makes the blend below
    describe a population that exists.
    """
    rng = random.Random(seed)
    quiet: list[Sample] = []
    checks: list[Sample] = []
    seen_quiet: dict[tuple[object, ...], int] = {}
    seen_check: dict[tuple[object, ...], int] = {}
    n_quiet = n_check = 0          # distinct positions SEEN per condition
    considered = in_check_seen = 0

    for game in range(playouts):
        board = chess.Board()
        game_key = f"s{seed}g{game}"
        for ply in range(max_plies):
            if board.is_game_over(claim_draw=False):
                break
            moves = list(board.legal_moves)
            captures = [m for m in moves if board.is_capture(m)]
            pool = captures if (captures and rng.random() < capture_bias) else moves
            board.push(rng.choice(pool))
            considered += 1
            in_check = board.is_check()
            if in_check:
                in_check_seen += 1
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                continue

            key = resolver_key(board)
            reservoir, seen, want = (
                (checks, seen_check, want_check) if in_check else (quiet, seen_quiet, want_quiet)
            )
            if key in seen:
                continue
            seen[key] = 1
            if in_check:
                n_check += 1
                index = n_check
            else:
                n_quiet += 1
                index = n_quiet

            sample = Sample(board.copy(stack=True), board.fen(), game_key, ply, in_check)
            if len(reservoir) < want:
                reservoir.append(sample)
            else:
                # Standard reservoir replacement, over DISTINCT positions.
                j = rng.randrange(index)
                if j < want:
                    reservoir[j] = sample

    return quiet, checks, considered, in_check_seen


def _time_arm(
    arm: str, pack: str, boards: list[CBoard], repeats: int
) -> tuple[float, int, dict[str, int], list[int]]:
    """(seconds, evals, stats, values) for one arm over `boards`, repeated.

    ⚑ ONE LONG-LIVED CONTEXT ACROSS ALL REPEATS. arm_eval() builds and drops a
    context per call, so with --repeats > 1 each pass RESET the counters while the
    elapsed time and the eval count kept accumulating — every counter in the table
    was understated by the repeat factor, and nothing in the output said so. A
    handle accumulates, which is also how a corpus generator would use this.

    The warm-up runs on a SEPARATE handle so its work is not in the reported
    counters; the weight mapping is cached by file identity, so the timed handle
    still starts with the pages resident.
    """
    from chess_anti_engine.nnue import _nnue_ext

    warm = _nnue_ext.arm_open(arm, pack)
    _nnue_ext.arm_handle_eval(warm, boards[: min(64, len(boards))])
    del warm

    handle = _nnue_ext.arm_open(arm, pack)
    values: list[int] = []
    t0 = time.perf_counter()
    for _ in range(repeats):
        values = _nnue_ext.arm_handle_eval(handle, boards)
    elapsed = time.perf_counter() - t0
    return elapsed, len(boards) * repeats, _nnue_ext.arm_stats(handle), values


def bank_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the per-position bank atomically.

    ⚑ THE AGGREGATE IS NOT THE MEASUREMENT. An evals/s number cannot be
    re-stratified, so any later question — is the in-check cost concentrated in a
    few deep chains? does it track piece count? — forces a rerun on a box whose
    load has moved on. The rows carry the FEN, the value, the arm's settings as
    the CONTEXT reported them, and the game/ply cluster key, so a re-analysis is
    a groupby.

    Staged and os.replace'd, so a killed run leaves no half-written bank that a
    later read would take for a complete one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    with staged.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(staged, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--quiet-n", type=int, default=2000)
    ap.add_argument("--check-n", type=int, default=500)
    ap.add_argument("--playouts", type=int, default=400, help="games in the shared stream")
    ap.add_argument("--stratified-n", type=int, default=2000,
                    help="coverage-only pool; 0 to skip")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--capture-bias", type=float, default=0.55)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bank", type=Path, help="per-position JSONL output")
    args = ap.parse_args(argv)

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.nnue import _nnue_ext

    pack = str(args.pack)
    quiet_s, check_s, considered, in_check_seen = sample_stream(
        args.seed, args.capture_bias, args.playouts, args.quiet_n, args.check_n
    )
    if not quiet_s or not check_s:
        print("empty position pool; raise --playouts", file=sys.stderr)
        return 2
    f_stream = in_check_seen / considered if considered else 0.0

    handle = _nnue_ext.load(pack)
    sha = _nnue_ext.source_sha256(handle)
    print(f"pack            : {args.pack}")
    print(f"net sha256      : {sha}")
    print(f"kernel          : {'avx2' if _nnue_ext.simd_active() else 'scalar'}")
    print(f"stream          : {args.playouts} playouts, {considered:,} positions")
    print(f"in-check rate f : {f_stream:.4f} ({in_check_seen:,}/{considered:,}), "
          "measured on the stream both pools are drawn from")
    print(f"matched pools   : {len(quiet_s):,} quiet / {len(check_s):,} in-check, "
          "reservoir-sampled, history carried, resolver-complete dedup")
    print(f"repeats         : {args.repeats} (one context across all passes)")
    print()

    conditions: list[tuple[str, list[Sample]]] = [
        ("quiet", quiet_s),
        ("in-check", check_s),
    ]
    boards = {
        label: [CBoard.from_board(s.board) for s in samples] for label, samples in conditions
    }

    # Baseline: the raw evaluator, quiet positions only — it refuses check.
    _nnue_ext.benchmark(handle, boards["quiet"][:64], 1, 1)
    evals, seconds, _checksum = _nnue_ext.benchmark(handle, boards["quiet"], args.repeats, 1)
    raw_rate = evals / seconds if seconds > 0 else float("nan")
    print(f"{'nnue (raw)':<14} {'quiet':<9} {evals:>9,} evals {seconds:7.3f}s "
          f"= {raw_rate:>11,.0f} evals/s")

    rates: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []
    for arm in ("nnue-static", "nnue-qsearch"):
        for label, samples in conditions:
            seconds, evals, stats, values = _time_arm(arm, pack, boards[label], args.repeats)
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
                f"qnodes={stats['qnodes']:,} qmaxply={stats['qmax_ply_seen']} "
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
            if args.bank:
                for sample, value in zip(samples, values, strict=True):
                    rows.append({
                        "schema": BANK_SCHEMA,
                        "arm": arm,
                        "pool": f"natural-{label}",
                        "in_check": sample.in_check,
                        "fen": sample.fen,
                        "value": value,
                        "game_key": sample.game_key,
                        "ply": sample.ply,
                        "pack_sha256": sha,
                        "seed": args.seed,
                        # From the CONTEXT that ran, not from the module globals.
                        "resolver_max_depth": stats["resolver_max_depth"],
                        "qsearch_max_ply": stats["qsearch_max_ply"],
                        "qsearch_check_plies": stats["qsearch_check_plies"],
                    })
    print()

    # ⚑ TWO REFERENCE POINTS, AND ONLY ONE OF THEM IS SOUND.
    #
    # "% below the raw evaluator" crosses two harnesses: the baseline runs through
    # _nnue_ext.benchmark() and the arms through the arm handle. They do the same
    # evaluations but not the same bookkeeping, and the gap between them is not
    # zero — the static arm has measured FASTER than the raw baseline on the quiet
    # pool, which is a harness difference and certainly not a speedup from adding
    # work. Quote it only as an order of magnitude.
    #
    # The defensible number is the arm against ITS OWN quiet rate: same harness,
    # same code path, the only difference being how many positions were in check.
    for arm in ("nnue-static", "nnue-qsearch"):
        q, c = rates[(arm, "quiet")], rates[(arm, "in-check")]
        if q <= 0 or c <= 0:
            continue
        blended = 1.0 / ((1.0 - f_stream) / q + f_stream / c)
        print(f"{arm:<14} in-check eval costs {q / c:4.2f}x a quiet one (same harness)")
        print(
            f"{arm:<14} blended at f={f_stream:.4f} = {blended:>11,.0f} evals/s"
            f"   [{100.0 * (q - blended) / q:5.1f}% below this arm's own quiet rate"
            f"; {100.0 * (raw_rate - blended) / raw_rate:6.1f}% vs raw, cross-harness]"
        )

    # Coverage only. ⚑ NEVER blended with the above: the stratified sampler draws
    # round-robin across (bucket, threat-bin) cells on purpose, so its rate
    # describes a population the generator does not emit.
    if args.stratified_n:
        strat_fens, _stats = sample_fens(
            args.stratified_n, seed=args.seed, capture_bias=args.capture_bias
        )
        strat = [CBoard.from_board(chess.Board(f)) for f in strat_fens]
        seconds, evals, stats, _values = _time_arm("nnue-static", pack, strat, args.repeats)
        print()
        print(f"{'COVERAGE':<14} {'stratified':<9} {evals:>9,} evals {seconds:7.3f}s "
              f"= {evals / seconds:>11,.0f} evals/s  (nnue-static; NOT blended — "
              "round-robin over cells, so this is not the generator's mix)")
        print("  ⚑ FEN-only, so these boards carry no history and the resolver "
              "cannot see a repetition in them. Coverage, not a cost estimate.")

    if args.bank and rows:
        bank_rows(args.bank, rows)
        print(f"\nbanked {len(rows):,} rows -> {args.bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
