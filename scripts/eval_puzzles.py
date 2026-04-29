#!/usr/bin/env python3
"""Puzzle evaluation, LC0-blog style — policy + value head bake-off.

Defaults to the persistent 2200-2800 / 1000-per-bucket suite at
`data/puzzles/lichess_2200_2800_n3000.csv` so across-checkpoint
comparisons stay apples-to-apples. Each run appends a row to
`data/puzzles/eval_log.csv` so the leaderboard accumulates over time.

Examples:
    # Default: full persistent suite, policy + value modes.
    PYTHONPATH=. python3 scripts/eval_puzzles.py --checkpoint runs/.../trainer.pt

    # Override puzzle source / restrict modes.
    PYTHONPATH=. python3 scripts/eval_puzzles.py --checkpoint X.pt --mode policy
    PYTHONPATH=. python3 scripts/eval_puzzles.py --checkpoint X.pt --puzzle-epd data/wac.epd
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import time
from pathlib import Path

from chess_anti_engine.eval import (
    load_epd,
    load_lichess_csv,
    run_policy_sequence_eval,
    run_value_head_puzzle_eval,
)
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

DEFAULT_PUZZLE_CSV = "data/puzzles/lichess_2200_2800_n3000.csv"
DEFAULT_BUCKETS: tuple[tuple[int, int], ...] = ((2200, 2400), (2400, 2600), (2600, 2800))
DEFAULT_LOG = "data/puzzles/eval_log.csv"


def _parse_buckets(spec: str) -> tuple[tuple[int, int], ...]:
    """Parse "2200,2400,2600,2800" → ((2200,2400), (2400,2600), (2600,2800))."""
    edges = [int(x) for x in spec.split(",") if x.strip()]
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("--rating-buckets needs at least two edges")
    return tuple(zip(edges[:-1], edges[1:]))


def _parse_modes(spec: str) -> tuple[str, ...]:
    valid = {"policy", "value", "search"}
    parts = tuple(p.strip() for p in spec.split(",") if p.strip())
    bad = [p for p in parts if p not in valid]
    if not parts or bad:
        raise argparse.ArgumentTypeError(
            f"--mode must be comma-separated subset of {sorted(valid)}; got {spec!r}"
        )
    return parts


def _print(name: str, result, dt: float, *, sims_label: str) -> None:
    print()
    print(f"[{name}] {result.correct}/{result.total} correct = {result.accuracy:.4f}"
          f"  ({dt:.0f}s, {dt/max(1,result.total)*1000:.0f} ms/puzzle)")
    if result.by_rating:
        print(f"[{name}] per-rating-bucket accuracy ({sims_label}):")
        print(f"  {'bucket':>14}  {'n':>6}  {'correct':>7}  {'acc':>7}")
        for low, high, total, correct, acc in result.by_rating:
            print(f"  {f'{low}-{high}':>14}  {total:>6}  {correct:>7}  {acc:>7.4f}")


def _append_log(
    log_path: Path,
    *,
    checkpoint: str,
    suite_name: str,
    mode: str,
    result,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    by_rating = {f"{low}-{high}": acc for low, high, _, _, acc in result.by_rating}
    new_file = not log_path.exists()
    fieldnames = ["timestamp", "checkpoint", "suite", "mode", "n", "correct", "accuracy"]
    fieldnames += [f"acc_{k}" for k in sorted(by_rating)]
    with log_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        row = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "checkpoint": checkpoint,
            "suite": suite_name,
            "mode": mode,
            "n": result.total,
            "correct": result.correct,
            "accuracy": f"{result.accuracy:.4f}",
        }
        for k, v in by_rating.items():
            row[f"acc_{k}"] = f"{v:.4f}"
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="trainer.pt or checkpoint dir")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--puzzle-csv",
        default=DEFAULT_PUZZLE_CSV,
        help=f"Lichess puzzle CSV (default {DEFAULT_PUZZLE_CSV})",
    )
    src.add_argument("--puzzle-epd", help="EPD with `bm` opcode (overrides --puzzle-csv)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-puzzles", type=int, default=None,
                   help="Cap on puzzles loaded; default uses the whole suite")
    p.add_argument("--min-rating", type=int, default=None)
    p.add_argument("--max-rating", type=int, default=None)
    p.add_argument(
        "--themes",
        default="",
        help="Comma-separated Lichess themes; puzzle kept if any theme matches",
    )
    p.add_argument(
        "--rating-buckets",
        type=_parse_buckets,
        default=DEFAULT_BUCKETS,
        help='Comma-separated bucket edges, default 2200,2400,2600,2800',
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--mode",
        type=_parse_modes,
        default=("policy", "value"),
        help="Comma-separated modes from {policy,value,search}; default policy,value",
    )
    p.add_argument(
        "--log-csv",
        default=DEFAULT_LOG,
        help=f"Append per-mode results here (default {DEFAULT_LOG}); pass empty string to disable",
    )
    args = p.parse_args()

    print(f"[puzzle] loading model: {args.checkpoint}")
    t0 = time.time()
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    print(f"[puzzle] loaded in {time.time()-t0:.1f}s")

    if args.puzzle_epd:
        print(f"[puzzle] loading EPD: {args.puzzle_epd}")
        suite = load_epd(args.puzzle_epd)
        if args.max_puzzles is not None:
            suite.puzzles = suite.puzzles[: args.max_puzzles]
    else:
        themes = tuple(t for t in args.themes.split(",") if t)
        print(f"[puzzle] loading Lichess CSV: {args.puzzle_csv}")
        suite = load_lichess_csv(
            args.puzzle_csv,
            max_puzzles=args.max_puzzles,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            themes_filter=themes,
        )

    print(f"[puzzle] {len(suite)} puzzles loaded from suite '{suite.name}'")
    if len(suite) == 0:
        print("[puzzle] no puzzles after filtering — nothing to do.")
        return

    log_path = Path(args.log_csv).expanduser() if args.log_csv else None
    for mode in args.mode:
        t0 = time.time()
        if mode == "policy":
            result = run_policy_sequence_eval(
                model, suite,
                device=args.device,
                batch_size=args.batch_size,
                rating_buckets=args.rating_buckets,
            )
            sims_label = "policy-only argmax"
        elif mode == "value":
            result = run_value_head_puzzle_eval(
                model, suite,
                device=args.device,
                batch_size=args.batch_size,
                rating_buckets=args.rating_buckets,
            )
            sims_label = "value-only push-eval"
        else:  # search — kept for back-compat; not in default
            from chess_anti_engine.eval import run_puzzle_eval  # local: avoids MCTS import on the common path
            import numpy as np
            rng = np.random.default_rng(42)
            result = run_puzzle_eval(
                model, suite,
                device=args.device,
                mcts_simulations=200,
                batch_size=args.batch_size,
                rng=rng,
                rating_buckets=args.rating_buckets,
            )
            sims_label = "200 sims"
        _print(f"puzzle:{mode}", result, time.time() - t0, sims_label=sims_label)
        if log_path is not None:
            _append_log(
                log_path,
                checkpoint=args.checkpoint,
                suite_name=suite.name,
                mode=mode,
                result=result,
            )

    if log_path is not None:
        print(f"\n[puzzle] appended results to {log_path}")


if __name__ == "__main__":
    main()
