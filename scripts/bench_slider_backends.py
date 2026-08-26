#!/usr/bin/env python3
"""Measure what the table-backed sliders bought, per stage.

⚑ THIS SCRIPT MEASURES ONE BUILD. The comparison is across BUILDS (a `.so`
carries exactly one slider backend, chosen by the preprocessor), so it cannot
alternate arms inside a process the way `scripts/nnue_incremental_bench.py` does.
The caller alternates at process level instead, and must: interleave the two
builds rather than running one arm to completion, and take a MINIMUM over
repeats rather than a mean. Wall-clock noise on a shared box is one-sided —
interference only ever makes a run slower — so the minimum is the estimator that
converges on the machine's actual capability, while a mean mostly measures who
else was running.

    for i in 1 2 3 4 5 6; do
      for tree in $ORDER; do   # ORDER flipped on odd/even i
        PYTHONPATH=$tree nice -n 5 python scripts/bench_slider_backends.py \
            --pack "$PACK" --json
      done
    done

Stages, and why each is here:
  movegen  — `CBoard.legal_move_indices()`, the ~18% of qsearch wall the profile
             in docs/nnue_speed_plan.md §1 attributes to move generation. This
             is the stage the change targets directly, so it is the upper bound
             on what any of the others can show.
  qsearch  — the `nnue-qsearch` arm end to end. THE DECIDING NUMBER: §3 predicts
             an 8-14% reduction in qsearch wall and pre-commits "below 5%
             measured => report as a miss against the prediction, investigate
             before merging."
  fastq    — the `nnue-fastq` arm, which reaches the sliders through both
             movegen and `cae_see_capture`'s x-ray swap loop. Its corpus is
             structurally deduplicated before timing because FastQ's canonical
             DAG persists across rows of one handle; otherwise each game's
             repeated start/opening prefix becomes a cache-hit benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding import _lc0_ext
from chess_anti_engine.nnue import _nnue_ext


def corpus(games: int, plies: int, seed: int) -> list[chess.Board]:
    """Deterministic self-play positions, sorted-move sampling for stability."""
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    for _ in range(games):
        board = chess.Board()
        for _ in range(plies):
            if board.is_game_over():
                break
            boards.append(board.copy())
            board.push(rng.choice(sorted(board.legal_moves, key=lambda m: m.uci())))
    return boards


def dedupe_structural_positions(boards: list[chess.Board]) -> list[chess.Board]:
    """Keep the first occurrence of each structural chess position.

    The key deliberately omits move clocks and history, matching the canonical
    NNUE DAG's position identity. `en_passant="legal"` prevents a nominal EP
    square that cannot produce a legal capture from splitting one structure
    into two benchmark rows.
    """
    seen: set[str] = set()
    out: list[chess.Board] = []
    for board in boards:
        key = " ".join(board.fen(en_passant="legal").split()[:4])
        if key in seen:
            continue
        seen.add(key)
        out.append(board)
    return out


def bench_movegen(boards: list[chess.Board], repeats: int) -> dict[str, float]:
    cboards = [CBoard.from_board(b) for b in boards]
    best = float("inf")
    moves = 0
    for _ in range(repeats):
        start = time.perf_counter()
        moves = sum(len(cb.legal_move_indices()) for cb in cboards)
        best = min(best, time.perf_counter() - start)
    return {"positions": len(cboards), "moves": moves, "seconds": best,
            "moves_per_s": moves / best}


def bench_arm(arm: str, pack: str, boards: list[chess.Board], repeats: int) -> dict[str, float]:
    """⚑ FRESH HANDLE PER REPEAT. The arms carry a position DAG that survives
    across `arm_handle_eval` calls on one handle, so re-timing the same corpus
    on the same handle measures cache hits, not search. Reusing a handle here
    inflated FastQ from 8.3k rows/s to 226k — a 27x "speedup" that is entirely
    the second pass finding the first pass's results. The one-time costs a
    warm-up would normally hide (weight mmap, table init) are paid by a
    throwaway handle before the timed loop; `arm_open` shares the mapped weights
    across handles, so that cost is paid once and not per repeat.
    """
    cboards = [CBoard.from_board(b) for b in boards]
    _nnue_ext.arm_handle_eval(
        _nnue_ext.arm_open(arm, pack), cboards[: min(32, len(cboards))]
    )
    best = float("inf")
    for _ in range(repeats):
        handle = _nnue_ext.arm_open(arm, pack)
        start = time.perf_counter()
        _nnue_ext.arm_handle_eval(handle, cboards)
        best = min(best, time.perf_counter() - start)
    return {"rows": len(cboards), "seconds": best, "rows_per_s": len(cboards) / best}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", help="NNUE pack; omit to run movegen only")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--movegen-positions", type=int, default=20000)
    parser.add_argument("--qsearch-positions", type=int, default=400)
    parser.add_argument("--fastq-positions", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pool = corpus(games=400, plies=60, seed=args.seed)
    fastq_pool = dedupe_structural_positions(pool)
    # ⚑ REPORT WHICH BINARIES WERE ACTUALLY TIMED. This script is normally run
    # from one checkout with PYTHONPATH pointing at another, and an editable
    # install or a cwd that happens to contain `chess_anti_engine/` will shadow
    # PYTHONPATH without a word — at which case both "arms" are the same build
    # and the comparison silently reads zero. The .so paths are the evidence
    # that two different objects were measured; keep them in the output.
    out: dict[str, object] = {
        "loaded_from": {
            "_lc0_ext": _lc0_ext.__file__,
            "_nnue_ext": _nnue_ext.__file__,
        },
        "slider_backend": {
            "_lc0_ext": getattr(_lc0_ext, "SLIDER_BACKEND", "unknown(pre-change build)"),
            "_nnue_ext": getattr(_nnue_ext, "SLIDER_BACKEND", "unknown(pre-change build)"),
        },
        "pool": len(pool),
        "fastq_structural_unique_pool": len(fastq_pool),
    }
    out["movegen"] = bench_movegen(pool[: args.movegen_positions], args.repeats)
    if args.pack:
        if not Path(args.pack).is_file():
            parser.error(f"--pack does not exist: {args.pack}")
        out["qsearch"] = bench_arm("nnue-qsearch", args.pack,
                                   pool[: args.qsearch_positions], args.repeats)
        out["fastq"] = bench_arm("nnue-fastq", args.pack,
                                 fastq_pool[: args.fastq_positions], args.repeats)

    if args.json:
        print(json.dumps(out))
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
