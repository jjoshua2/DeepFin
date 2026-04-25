#!/usr/bin/env python3
"""Ad-hoc puzzle evaluation, LC0-blog style.

Reports top-1 accuracy on a Lichess puzzle CSV (or EPD), bucketed by rating
when annotations are present. Reference:
https://lczero.org/blog/2024/02/how-well-do-lc0-networks-compare-to-the-greatest-transformer-network-from-deepmind/

Examples:
    # Lichess CSV (download lichess_db_puzzle.csv.zst from
    # https://database.lichess.org/#puzzles, decompress, optionally shuffle).
    PYTHONPATH=. python3 scripts/eval_puzzles.py \\
        --checkpoint runs/.../trainer.pt \\
        --puzzle-csv data/lichess_db_puzzle.csv \\
        --max-puzzles 5000 --simulations 200

    # WAC EPD (small smoke set, no rating buckets).
    PYTHONPATH=. python3 scripts/eval_puzzles.py \\
        --checkpoint runs/.../trainer.pt \\
        --puzzle-epd data/wac.epd --simulations 200
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields

import numpy as np

from chess_anti_engine.eval import load_epd, load_lichess_csv, run_puzzle_eval
from chess_anti_engine.eval.puzzles import DEFAULT_RATING_BUCKETS
from chess_anti_engine.model import ModelConfig
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def _cfg_from_params_json(params_path: str) -> ModelConfig:
    with open(params_path) as fh:
        params = json.load(fh)
    valid = {f.name for f in fields(ModelConfig)}
    filtered = {k: v for k, v in params.items() if k in valid}
    filtered.setdefault("kind", str(params.get("model", "transformer")))
    if "no_smolgen" in params and "use_smolgen" not in filtered:
        filtered["use_smolgen"] = not bool(params["no_smolgen"])
    return ModelConfig(**filtered)


def _parse_buckets(spec: str) -> tuple[tuple[int, int], ...]:
    """Parse "0,1000,1500,2000,2500,3000,9999" → ((0,1000), (1000,1500), ...)."""
    edges = [int(x) for x in spec.split(",") if x.strip()]
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("--rating-buckets needs at least two edges")
    return tuple(zip(edges[:-1], edges[1:]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="trainer.pt or checkpoint dir")
    p.add_argument("--params", default=None, help="override params.json")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--puzzle-csv", help="Lichess puzzle CSV path")
    src.add_argument("--puzzle-epd", help="EPD with `bm` opcode")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-puzzles", type=int, default=None)
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
        default=DEFAULT_RATING_BUCKETS,
        help='Comma-separated rating-bucket edges, e.g. "0,1500,2000,2500,3000,9999"',
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = _cfg_from_params_json(args.params) if args.params else None

    print(f"[puzzle] loading model: {args.checkpoint}")
    t0 = time.time()
    model = load_model_from_checkpoint(args.checkpoint, device=args.device, model_config=cfg)
    print(f"[puzzle] loaded in {time.time()-t0:.1f}s")

    if args.puzzle_csv:
        themes = tuple(t for t in args.themes.split(",") if t)
        print(f"[puzzle] loading Lichess CSV: {args.puzzle_csv}")
        suite = load_lichess_csv(
            args.puzzle_csv,
            max_puzzles=args.max_puzzles,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            themes_filter=themes,
        )
    else:
        print(f"[puzzle] loading EPD: {args.puzzle_epd}")
        suite = load_epd(args.puzzle_epd)
        if args.max_puzzles is not None:
            suite.puzzles = suite.puzzles[: args.max_puzzles]

    print(f"[puzzle] {len(suite)} puzzles loaded from suite '{suite.name}'")
    if len(suite) == 0:
        print("[puzzle] no puzzles after filtering — nothing to do.")
        return

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    result = run_puzzle_eval(
        model, suite,
        device=args.device,
        mcts_simulations=args.simulations,
        batch_size=args.batch_size,
        rng=rng,
        rating_buckets=args.rating_buckets,
    )
    dt = time.time() - t0

    print()
    print(f"[puzzle] {result.correct}/{result.total} correct = {result.accuracy:.4f}"
          f"  ({dt:.0f}s, {dt/max(1,result.total)*1000:.0f} ms/puzzle)")

    if result.by_rating:
        print()
        print(f"[puzzle] per-rating-bucket accuracy ({args.simulations} sims):")
        print(f"  {'bucket':>14}  {'n':>6}  {'correct':>7}  {'acc':>7}")
        for low, high, total, correct, acc in result.by_rating:
            label = f"{low}-{high}"
            print(f"  {label:>14}  {total:>6}  {correct:>7}  {acc:>7.4f}")


if __name__ == "__main__":
    main()
