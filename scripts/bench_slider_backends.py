#!/usr/bin/env python3
"""Measure slider speed and enforce the per-node speed-plan regression gate.

This script measures ONE build. Compare builds by interleaving separate process
runs; a `.so` carries exactly one slider backend. Wall-clock interference is
one-sided, so each process takes the minimum of repeated identical runs.

The S2 acceptance gate in ``docs/nnue_speed_plan.md`` is executable here:
``--baseline-json`` requires identical deterministic work (move count, qsearch
row count and qnode count) and fails if qsearch microseconds/qnode regresses by
more than 3%. A faster result passes; this is a regression tripwire, not a claim
that two noisy wall measurements differ significantly.

Typical use from otherwise identical main/PR checkouts::

    python scripts/bench_slider_backends.py --pack "$PACK" --json > main.json
    python scripts/bench_slider_backends.py --pack "$PACK" --json \
        --baseline-json main.json > pr.json

For stronger speed attribution, interleave several independent process runs of
the two builds as documented in the PR rather than timing one build to completion
before the other.

Stages:
  movegen  — CBoard legal move generation, the direct target;
  qsearch  — nnue-qsearch end-to-end, the preregistered deciding S2 stage;
  fastq    — nnue-fastq on a structurally deduplicated corpus so persistent-DAG
             hits across repeated game prefixes cannot masquerade as slider speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import chess

from chess_anti_engine.encoding import _lc0_ext
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext


REGRESSION_LIMIT = 0.03


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
    """Keep the first occurrence of each canonical-DAG structural position."""
    seen: set[str] = set()
    out: list[chess.Board] = []
    for board in boards:
        # First four FEN fields are pieces/STM/castling/EP. python-chess's
        # ``legal`` EP spelling drops a nominal EP square when no legal EP
        # capture exists, matching the structural identity used by the DAG.
        key = " ".join(board.fen(en_passant="legal").split()[:4])
        if key in seen:
            continue
        seen.add(key)
        out.append(board)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bench_movegen(boards: list[chess.Board], repeats: int) -> dict[str, float | int]:
    cboards = [CBoard.from_board(b) for b in boards]
    best = float("inf")
    moves = 0
    for _ in range(repeats):
        start = time.perf_counter()
        moves = sum(len(cb.legal_move_indices()) for cb in cboards)
        best = min(best, time.perf_counter() - start)
    return {
        "positions": len(cboards),
        "moves": moves,
        "seconds": best,
        "moves_per_s": moves / best,
    }


def _arm_work(arm: str, handle: object) -> tuple[int, int]:
    if arm == "nnue-fastq":
        stats = _nnue_ext.fastq_stats(handle)
        return int(stats["nodes"]), int(stats["nnue_evals"])
    stats = _nnue_ext.arm_stats(handle)
    return int(stats["qnodes"]), int(stats["nnue_evals"])


def bench_arm(
    arm: str, pack: str, boards: list[chess.Board], repeats: int,
) -> dict[str, float | int]:
    """Time identical cold semantic contexts; retain deterministic work counts.

    The one-time mmap/table-init costs are paid by a throwaway warm-up handle.
    Every timed repeat gets a fresh handle because DAG-backed arms retain
    canonical nodes between calls. Work counters must match across repeats;
    otherwise taking the fastest wall sample would silently select a different
    amount of search work.
    """
    cboards = [CBoard.from_board(b) for b in boards]
    _nnue_ext.arm_handle_eval(
        _nnue_ext.arm_open(arm, pack), cboards[: min(32, len(cboards))]
    )
    best = float("inf")
    expected_work: tuple[int, int] | None = None
    for _ in range(repeats):
        handle = _nnue_ext.arm_open(arm, pack)
        start = time.perf_counter()
        _nnue_ext.arm_handle_eval(handle, cboards)
        elapsed = time.perf_counter() - start
        work = _arm_work(arm, handle)
        if expected_work is None:
            expected_work = work
        elif work != expected_work:
            raise RuntimeError(
                f"{arm} work changed across identical repeats: "
                f"{expected_work} vs {work}",
            )
        best = min(best, elapsed)
    if expected_work is None:
        raise RuntimeError("repeats must be >= 1")
    nodes, nnue_evals = expected_work
    return {
        "rows": len(cboards),
        "seconds": best,
        "rows_per_s": len(cboards) / best,
        "nodes": nodes,
        "nnue_evals": nnue_evals,
        "us_per_node": best * 1.0e6 / max(1, nodes),
    }


def regression_gate(
    current: dict[str, Any], baseline: dict[str, Any],
    *, limit: float = REGRESSION_LIMIT,
) -> list[str]:
    """Return gate failures; an empty list is a pass.

    Work equality is exact. Time is compared only after that equality is proved,
    and only for the qsearch stage the speed plan registered. The pack hash is
    part of the gate so a net swap cannot be called a per-node speed change.
    """
    failures: list[str] = []
    if current.get("pack_sha256") != baseline.get("pack_sha256"):
        failures.append("NNUE pack SHA differs between baseline and current")

    for section, keys in (
        ("movegen", ("positions", "moves")),
        ("qsearch", ("rows", "nodes", "nnue_evals")),
    ):
        cur = current.get(section)
        base = baseline.get(section)
        if not isinstance(cur, dict) or not isinstance(base, dict):
            failures.append(f"missing {section} section in baseline/current")
            continue
        for key in keys:
            if cur.get(key) != base.get(key):
                failures.append(
                    f"{section}.{key} changed: baseline={base.get(key)!r}, "
                    f"current={cur.get(key)!r}",
                )

    cur_q = current.get("qsearch")
    base_q = baseline.get("qsearch")
    if isinstance(cur_q, dict) and isinstance(base_q, dict):
        cur_us = float(cur_q["us_per_node"])
        base_us = float(base_q["us_per_node"])
        ceiling = base_us * (1.0 + limit)
        if cur_us > ceiling:
            failures.append(
                f"qsearch us/node regressed {((cur_us / base_us) - 1.0) * 100:.2f}% "
                f"(baseline={base_us:.6f}, current={cur_us:.6f}, "
                f"limit={limit * 100:.1f}%)",
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", help="NNUE pack; omit to run movegen only")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--movegen-positions", type=int, default=20000)
    parser.add_argument("--qsearch-positions", type=int, default=400)
    parser.add_argument("--fastq-positions", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--baseline-json", type=Path,
        help="fail on deterministic-work changes or >3%% qsearch us/node regression",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    pool = corpus(games=400, plies=60, seed=args.seed)
    fastq_pool = dedupe_structural_positions(pool)
    out: dict[str, Any] = {
        "loaded_from": {
            "_lc0_ext": _lc0_ext.__file__,
            "_nnue_ext": _nnue_ext.__file__,
        },
        "slider_backend": {
            "_lc0_ext": getattr(_lc0_ext, "SLIDER_BACKEND", "unknown(pre-change build)"),
            "_nnue_ext": getattr(_nnue_ext, "SLIDER_BACKEND", "unknown(pre-change build)"),
        },
        "seed": args.seed,
        "pool": len(pool),
        "fastq_structural_unique_pool": len(fastq_pool),
    }
    out["movegen"] = bench_movegen(pool[: args.movegen_positions], args.repeats)
    if args.pack:
        pack = Path(args.pack)
        if not pack.is_file():
            parser.error(f"--pack does not exist: {pack}")
        out["pack_sha256"] = sha256_file(pack)
        out["qsearch"] = bench_arm(
            "nnue-qsearch", str(pack), pool[: args.qsearch_positions], args.repeats,
        )
        out["fastq"] = bench_arm(
            "nnue-fastq", str(pack), fastq_pool[: args.fastq_positions], args.repeats,
        )

    failures: list[str] = []
    if args.baseline_json is not None:
        if not args.pack:
            parser.error("--baseline-json requires --pack so qsearch can be gated")
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        failures = regression_gate(out, baseline)
        out["regression_gate"] = {
            "baseline": str(args.baseline_json),
            "limit": REGRESSION_LIMIT,
            "passed": not failures,
            "failures": failures,
        }

    text = json.dumps(out) if args.json else json.dumps(out, indent=2)
    print(text)
    if failures:
        for failure in failures:
            print(f"REGRESSION GATE FAILED: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
