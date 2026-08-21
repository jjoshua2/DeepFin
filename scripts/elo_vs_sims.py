#!/usr/bin/env python3
"""Search exchange-rate curve: Elo vs MCTS sims for ONE checkpoint.

Arenas the same checkpoint against itself at adjacent sims rungs
(default 32, 64, 128, 256, 512) using the standardized paired-opening arena
(scripts/arena_standard.py, matched_sims mode). The output table — Elo of
each rung vs the previous one, with a pentanomial 95% CI — is the exchange
rate between search budget and strength: it prices what a throughput
regression (e.g. a 20% slower architecture) costs in Elo.

Each rung's result is also appended to runs/arena_results.jsonl with a
``label`` of the form ``elo_vs_sims:64v32``, and each rung keeps its own
per-game log named from that label. ``--resume`` therefore replays only the
rung that was interrupted. To run the SAME ladder again on purpose, pass
``--label`` — without it the second ladder writes to the first one's game logs
and is refused.

Usage::

    PYTHONPATH=. python3 scripts/elo_vs_sims.py \\
        --checkpoint runs/live/trainer.pt --games-per-rung 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.arena_standard import (
    SEARCH_SHAPES,
    add_common_args,
    apply_search_overrides,
    default_openings_path,
    resolve_search_shape,
    run_arena,
)
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
    p.add_argument("--label", default=None,
                   help="suffix for every rung's label, i.e. "
                        "'elo_vs_sims:64v32:<label>'. Each rung's per-game log "
                        "is named from its label, so this is what lets a "
                        "DELIBERATE repeat of the same ladder start instead of "
                        "hitting the 'game log already exists' refusal — "
                        "arena_standard's --games-out is per-arena and cannot "
                        "name a whole ladder. Without it the labels are exactly "
                        "what they have always been.")
    add_common_args(p)
    args = p.parse_args()

    sims = _parse_sims(args.sims)
    if args.search_shape is None:
        # A sims ladder is the run the silent shape hurt most: the play shape's
        # topk 32 doubles root breadth between the 32- and 256-sim rungs, so the
        # ladder varies breadth AND depth and cannot be read as a search-scaling
        # curve at all (docs/experiment_ledger.md 2026-07-28).
        raise SystemExit(
            f"--search-shape is required ({'|'.join(SEARCH_SHAPES)}); a sims "
            "ladder is only interpretable at a pinned shape."
        )
    side = apply_search_overrides(resolve_search_shape(args.search_shape))
    openings_path: Path = (
        args.openings if args.openings is not None else default_openings_path()
    )

    suffix = f":{args.label}" if args.label else ""
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
            label=f"elo_vs_sims:{hi}v{lo}{suffix}",
            search_candidate=side,
            search_reference=side,
            # Each rung has its own game log (the label and the sims are both
            # in the default path), so --resume replays only the rung that was
            # interrupted: a completed rung reloads its pairs and plays nothing.
            resume=bool(args.resume),
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
