#!/usr/bin/env python3
"""Search exchange-rate curve: Elo vs MCTS sims for ONE checkpoint.

Arenas the same checkpoint against itself at adjacent sims rungs
(default 32, 64, 128, 256, 512) using the standardized paired-opening arena
(scripts/arena_standard.py, matched_sims mode). The output table — Elo of
each rung vs the previous one, with a pentanomial 95% CI — is the exchange
rate between search budget and strength: it prices what a throughput
regression (e.g. a 20% slower architecture) costs in Elo.

Each rung's result is also appended to runs/arena_results.jsonl with a
``label`` of the form ``elo_vs_sims:64v32``.

Usage::

    PYTHONPATH=. python3 scripts/elo_vs_sims.py \\
        --checkpoint runs/live/trainer.pt --games-per-rung 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.arena_standard import add_common_args, default_openings_path, run_arena
import itertools


def _parse_sims(spec: str) -> list[int]:
    sims = [int(s) for s in spec.split(",") if s.strip()]
    if len(sims) < 2:
        raise SystemExit("--sims needs at least two comma-separated rungs")
    if any(s <= 0 for s in sims) or sims != sorted(set(sims)):
        raise SystemExit("--sims rungs must be positive, unique, and ascending")
    return sims


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint to arena against itself (trainer.pt or dir)")
    p.add_argument("--sims", default="32,64,128,256,512",
                   help="comma-separated ascending sims rungs (default: 32,64,128,256,512)")
    p.add_argument("--games-per-rung", type=int, default=400,
                   help="games per adjacent-rung arena; games/2 opening pairs (default: 400)")
    add_common_args(p)
    args = p.parse_args()

    sims = _parse_sims(args.sims)
    openings_path: Path = (
        args.openings if args.openings is not None else default_openings_path()
    )

    rows: list[tuple[int, dict]] = []
    for lo, hi in itertools.pairwise(sims):
        print(f"\n[elo_vs_sims] === {hi} sims vs {lo} sims ===")
        record = run_arena(
            candidate=args.checkpoint,
            reference=args.checkpoint,
            games=args.games_per_rung,
            openings_path=openings_path,
            opening_plies=args.opening_plies,
            mode="matched_sims",
            sims_candidate=hi,
            sims_reference=lo,
            ms_per_move=0,
            max_plies=args.max_plies,
            temperature=args.temperature,
            gumbel_add_noise=not args.no_gumbel_noise,
            device=args.device,
            seed=args.seed,
            out_path=args.out,
            label=f"elo_vs_sims:{hi}v{lo}",
        )
        rows.append((hi, record))

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:+.1f}"

    print("\n[elo_vs_sims] search exchange rate")
    print(f"{'sims':>6}  {'Elo vs prev rung':>18}  {'95% CI':>20}  {'pairs':>6}")
    print(f"{sims[0]:>6}  {'(baseline)':>18}  {'':>20}  {'':>6}")
    for hi, record in rows:
        ci_lo, ci_hi = record["elo_ci95"]
        ci = f"[{fmt(ci_lo)}, {fmt(ci_hi)}]"
        print(f"{hi:>6}  {fmt(record['elo']):>18}  {ci:>20}  {record['pairs']:>6}")


if __name__ == "__main__":
    main()
