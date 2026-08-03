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
import hashlib
import time
from pathlib import Path
from typing import cast

import torch

from chess_anti_engine.eval import (
    load_epd,
    load_lichess_csv,
    run_policy_sequence_eval,
    run_value_head_puzzle_eval,
)
from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
import itertools

DEFAULT_PUZZLE_CSV = "data/puzzles/lichess_2200_2800_n3000.csv"
DEFAULT_BUCKETS: tuple[tuple[int, int], ...] = ((2200, 2400), (2400, 2600), (2600, 2800))
DEFAULT_LOG = "data/puzzles/eval_log.csv"


def _parse_buckets(spec: str) -> tuple[tuple[int, int], ...]:
    """Parse "2200,2400,2600,2800" → ((2200,2400), (2400,2600), (2600,2800))."""
    edges = [int(x) for x in spec.split(",") if x.strip()]
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("--rating-buckets needs at least two edges")
    return tuple(itertools.pairwise(edges))


def _parse_modes(spec: str) -> tuple[str, ...]:
    valid = {"policy", "value", "search", "gumbel"}
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


def _resolve_log_target(log_path: Path, fieldnames: list[str]) -> tuple[Path, bool]:
    """Pick the CSV to append to and whether to write a header first.

    Never append wider rows under a narrower header: if the existing log's
    header already matches ``fieldnames`` we append in place, but on any schema
    drift — upgrading to the new ``search`` column, or a different rating-bucket
    set — we route to a schema-versioned sibling (``<stem>.<hash><suffix>``)
    instead of corrupting the persistent leaderboard.
    """
    if not log_path.exists():
        return log_path, True
    with log_path.open(newline="") as fh:
        existing = next(csv.reader(fh), [])
    if existing == fieldnames:
        return log_path, False
    tag = hashlib.sha1(",".join(fieldnames).encode()).hexdigest()[:8]
    versioned = log_path.with_name(f"{log_path.stem}.{tag}{log_path.suffix}")
    return versioned, not versioned.exists()


def _append_log(
    log_path: Path,
    *,
    checkpoint: str,
    suite_name: str,
    mode: str,
    result,
    search_label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    by_rating = {f"{low}-{high}": acc for low, high, _, _, acc in result.by_rating}
  # Record the full search config per row so the accumulating leaderboard stays
  # comparable across tuning versions — a number generated under the new root-log
  # play search must not be silently compared to a legacy-config row.
    fieldnames = ["timestamp", "checkpoint", "suite", "mode", "search", "n", "correct", "accuracy"]
    fieldnames += [f"acc_{k}" for k in sorted(by_rating)]
    target, write_header = _resolve_log_target(log_path, fieldnames)
    with target.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        row = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "checkpoint": checkpoint,
            "suite": suite_name,
            "mode": mode,
            "search": search_label,
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
    p.add_argument("--gpu-mem-fraction", type=float, default=None,
                   help="Cap this process to a fraction of total GPU memory "
                        "(torch.cuda.set_per_process_memory_fraction). Use when running "
                        "CONCURRENT with a live trainer: it makes the eval fail-fast with an "
                        "OOM inside its own process instead of faulting the shared GPU context "
                        "and killing the trainer's inference broker. e.g. 0.4 on a 32GB card.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model forward (speeds up high-sim search; "
                        "~30-60s warmup). Run at most ONE compiled eval per GPU concurrently "
                        "— concurrent compiles contend on the inductor cache / GPU.")
    # search/gumbel knobs (used by --mode search|gumbel)
    p.add_argument("--sims", type=int, default=200, help="MCTS simulations per puzzle (search/gumbel modes)")
    p.add_argument("--gumbel-c-scale", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale"], help="gumbel value-transform scale (mode gumbel; production-tuned 0.025)")
    p.add_argument("--gumbel-c-visit", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit"], help="gumbel c_visit depth-ramp base (mode gumbel)")
    p.add_argument("--gumbel-topk", type=int, default=PLAY_SEARCH_DEFAULTS["topk"], help="gumbel root candidates (mode gumbel)")
  # --gumbel-c-puct / --gumbel-fpu removed 2026-08-03: both are inert in a
  # Gumbel search (mcts.gumbel.INERT_GUMBEL_KNOBS; play-path audit F2), so a
  # sweep over them returned a perfectly reproducible null that read as a
  # measurement.
    p.add_argument("--gumbel-qexp", type=float, default=1.0,
                   help="gumbel q_visit_exp: exponent on max_visit in q_scale=c_scale*(c_visit+max_visit^exp). "
                        "1.0=linear (default); <1=sublinear (less sim-count-dependent optimum)")
    p.add_argument("--gumbel-global-scale", action="store_true",
                   help="gumbel descent scales q_scale by the ROOT max_visit (global) instead of "
                        "per-node local max_visit; pairs with --gumbel-qexp<1 for sim-invariance")
    p.add_argument("--gumbel-qfloor", type=float, default=-1.0,
                   help="gumbel decoupled (additive) value-transform floor: when >=0, "
                        "q_scale=qfloor+c_scale*max_visit^qexp (floor independent of c_scale). "
                        "<0 (default) = legacy coupled floor c_scale*(c_visit+max_visit^qexp)")
    p.add_argument("--gumbel-halving-div", type=int, default=2,
                   help="gumbel sequential-halving divisor: each round keeps ceil(n/div). "
                        "2 (default) = standard halving (top half); 3/4 = more aggressive "
                        "elimination (fewer rounds, visits concentrate on survivors sooner)")
    p.add_argument("--gumbel-cvisit-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit_root"],
                   help="gumbel root-halving c_visit override (value-transform floor at the "
                        "root site only; descent keeps --gumbel-c-visit). 900 (default) = the "
                        "root/descent SPLIT that fixes scaling; <0 = use c_visit at both (legacy)")
    p.add_argument("--gumbel-cscale-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale_root"],
                   help="gumbel ROOT-ONLY c_scale (descent keeps --gumbel-c-scale). Pairs with "
                        "--gumbel-qexp-root<0 for a LOG root q_scale=c_scale_root*log1p(c_visit_root"
                        "+max_visit), which needs a large c_scale (~7) vs the tiny descent (~0.025). "
                        "<0 = use c_scale at the root too (legacy linear)")
    p.add_argument("--gumbel-qexp-root", type=float, default=PLAY_SEARCH_DEFAULTS["q_visit_exp_root"],
                   help="gumbel ROOT-ONLY value-transform exponent (descent keeps --gumbel-qexp). "
                        "-1 (default) = LOG slow-growth: linear root q_scale explodes at high sims "
                        "and saturates sigma(q); log stays ~100 from 256 to millions of nodes "
                        "(sim-invariant). >=90 = use --gumbel-qexp at the root too")
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

    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        # `.index or 0` maps device "cuda" (index None) -> 0 while preserving an
        # explicit "cuda:N"; avoids the `is not None` check torch's stub flags.
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_mem_fraction), torch.device(args.device).index or 0)
        print(f"[puzzle] GPU memory capped at fraction {args.gpu_mem_fraction} "
              "(fail-fast OOM instead of faulting a concurrent trainer)")

    print(f"[puzzle] loading model: {args.checkpoint}")
    t0 = time.time()
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    if args.compile:
        _dev = torch.device(args.device)
        if _dev.type == "cuda":
            # bootstrap cudagraph TLS on this proc; set_device needs an index.
            # torch's stub types device.index as int, but it IS None for a
            # device with no explicit index (e.g. "cuda"), so the guard is real.
            dev_index = _dev.index
            torch.cuda.set_device(dev_index if dev_index is not None else 0)  # pyright: ignore[reportUnnecessaryComparison]
        model = cast("torch.nn.Module", torch.compile(model))
        print("[puzzle] torch.compile enabled (forward); first batch pays warmup")
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
        else:  # "search" (PUCT) or "gumbel" — MCTS move-selection accuracy
            from chess_anti_engine.eval import run_puzzle_eval  # local: avoids MCTS import on the common path
            import numpy as np
            rng = np.random.default_rng(42)
            gcfg = None
            if mode == "gumbel":
                from chess_anti_engine.mcts.gumbel import GumbelConfig
                gcfg = GumbelConfig(
                    topk=args.gumbel_topk, c_scale=args.gumbel_c_scale,
                    c_visit=args.gumbel_c_visit, q_visit_exp=args.gumbel_qexp,
                    q_global_scale=args.gumbel_global_scale,
                    q_visit_floor=args.gumbel_qfloor,
                    halving_div=args.gumbel_halving_div,
                    c_visit_root=args.gumbel_cvisit_root,
                    c_scale_root=args.gumbel_cscale_root,
                    q_visit_exp_root=args.gumbel_qexp_root,
                )
            result = run_puzzle_eval(
                model, suite,
                device=args.device,
                mcts_simulations=args.sims,
                batch_size=args.batch_size,
                rng=rng,
                rating_buckets=args.rating_buckets,
                gumbel_cfg=gcfg,
            )
            if mode == "gumbel":
                sims_label = (
                    f"{args.sims} sims gumbel c_scale={args.gumbel_c_scale} "
                    f"c_visit={args.gumbel_c_visit} c_visit_root={args.gumbel_cvisit_root} "
                    f"c_scale_root={args.gumbel_cscale_root} "
                    f"q_visit_exp_root={args.gumbel_qexp_root} topk={args.gumbel_topk}"
                )
            else:
                sims_label = f"{args.sims} sims puct"
        _print(f"puzzle:{mode}", result, time.time() - t0, sims_label=sims_label)
        if log_path is not None:
            _append_log(
                log_path,
                checkpoint=args.checkpoint,
                suite_name=suite.name,
                mode=mode,
                result=result,
                search_label=sims_label,
            )

    if log_path is not None:
        print(f"\n[puzzle] appended results to {log_path}")


if __name__ == "__main__":
    main()
