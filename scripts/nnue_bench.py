"""Throughput benchmark for the native NNUE evaluator.

⚑ THIS NUMBER INCLUDES FEATURE-INDEX COMPUTATION. The scoping projection that
sized the native-generator bet measured the accumulator gather ALONE and said so:
"excludes feature-INDEX computation (attack graph, unmeasured, plausibly small)".
Closing that caveat is the point of this script, so every eval here runs the
whole chain from a real CBoard — attack graph, threat relations, index mapping,
accumulator refresh, transform, layer stack.

Positions are the same stratified, not-in-check sample the parity gate uses, so
the piece-count and threat-density mix is the one the corpus generator will
actually see rather than a start-position micro-benchmark.

⚑ The box is shared with live training. Run this nice'd, report the thread count
you asked for AND the wall clock, and do not read a contended single-thread
number as a ceiling.

Usage::

    PYTHONPATH=. nice -n 19 python3 scripts/nnue_bench.py --pack big.pack --n 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

from scripts.nnue_fens import sample_fens


def _load_fens(args: argparse.Namespace) -> list[str]:
    if args.fens_in:
        return [ln.strip() for ln in args.fens_in.read_text().splitlines() if ln.strip()][: args.n]
    fens, _stats = sample_fens(args.n, seed=args.seed)
    return fens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000, help="distinct positions to cycle over")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--fens-in", type=Path)
    ap.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 8],
        help="thread counts to measure (default: 1 and 8)",
    )
    ap.add_argument("--repeats", type=int, default=8, help="passes over the position list")
    args = ap.parse_args(argv)

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.nnue import _nnue_ext

    fens = _load_fens(args)
    # An empty --fens-in or --n 0 otherwise reaches the mean-piece-count divide
    # below and dies on ZeroDivisionError, which reports a bug in the benchmark
    # rather than the input error it actually is.
    if not fens:
        print(
            "no FENs to benchmark (empty --fens-in, or --n 0)",
            file=sys.stderr,
        )
        return 2
    py_boards = [chess.Board(f) for f in fens]

    # ⚑ Two position sets, because one number here is a lie in both directions.
    # The STRATIFIED set spans all eight layer stacks, but random play puts most
    # of its mass in sparse endgames where there is little to gather; the
    # MIDDLEGAME set (>= 24 pieces) is the regime a game-playing corpus generator
    # actually spends its time in, and it is the slower one. Report both.
    sets: list[tuple[str, list[chess.Board]]] = [("stratified", py_boards)]
    middlegame = [b for b in py_boards if bin(b.occupied).count("1") >= 24]
    if middlegame:
        sets.append(("middlegame(>=24p)", middlegame))

    handle = _nnue_ext.load(str(args.pack))
    print(f"pack           : {args.pack}")
    print(f"net sha256     : {_nnue_ext.source_sha256(handle)}")
    print(f"avx2 compiled  : {bool(_nnue_ext.HAVE_AVX2)}")
    # ⚑ The reuse factor is part of the reading, not a knob footnote. Every pass
    # after the first re-evaluates the SAME positions, so their weight rows and
    # feature tables are already hot; a high-repeat number is a warm-cache
    # number. --repeats 1 evaluates each position exactly once and is the
    # conservative figure to quote when the concern is working-set reuse.
    print(f"repeats        : {args.repeats} pass(es) over each set "
          f"(position reuse factor {args.repeats}x)")
    for label, group in sets:
        counts = [bin(b.occupied).count("1") for b in group]
        threats = [
            len(_nnue_ext.active_features(CBoard.from_board(b), 0)[1]) for b in group[:512]
        ]
        print(
            f"  {label:<18} n={len(group):>7,}  mean pieces={sum(counts) / len(counts):5.1f}"
            f"  mean active threats={sum(threats) / len(threats):5.1f} (first {len(threats)})"
        )
    print()

    for label, group in sets:
        boards = [CBoard.from_board(b) for b in group]
        for simd in (True, False):
            try:
                _nnue_ext.set_simd(simd)
            except ValueError:
                continue
            kernels = "avx2" if _nnue_ext.simd_active() else "scalar"
            # One untimed pass so the weight pages are resident: a cold mmap
            # would measure page faults on a 111 MB file, not the evaluator.
            _nnue_ext.benchmark(handle, boards[: min(64, len(boards))], 1, 1)
            for threads in args.threads:
                evals, seconds, checksum = _nnue_ext.benchmark(
                    handle, boards, args.repeats, threads
                )
                rate = evals / seconds if seconds > 0 else float("nan")
                print(
                    f"{label:<18} {kernels:>6} threads={threads:<3} {evals:>10,} evals in "
                    f"{seconds:7.3f}s = {rate:>12,.0f} evals/s   (checksum {checksum})"
                )
        print()

    # Restore the build's default kernel. ⚑ Not an unconditional set_simd(True):
    # on a portable build AVX2 is not compiled in, that call raises, and an
    # otherwise complete benchmark would exit nonzero on its last line.
    _nnue_ext.set_simd(bool(_nnue_ext.HAVE_AVX2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
