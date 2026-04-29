"""Walk-forward backtest of regret-lever inverse fit at varying half-lives.

Reads progress.csv from a Ray trial dir and replays the (regret, raw_wr, se)
sequence through ``_fit_inverse_lever`` for several recency_half_life values.
Reports per-hl prediction stats so we can pick a setting that tracks the plant
without overreacting to single-iter noise.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from chess_anti_engine.stockfish.pid import _fit_inverse_lever  # noqa: E402

TARGET_WR = 0.58
HALF_LIVES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def load_trajectory(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            wins = int(float(r.get("win", 0) or 0))
            draws = int(float(r.get("draw", 0) or 0))
            losses = int(float(r.get("loss", 0) or 0))
            games = wins + draws + losses
            if games <= 0:
                continue
            raw_wr = (wins + 0.5 * draws) / games
            se = math.sqrt(max(raw_wr * (1.0 - raw_wr) / games, 1e-9))
            regret_used = float(r["opponent_wdl_regret_limit"])
            regret_chosen_for_next = float(r["opponent_wdl_regret_limit_next"])
            rows.append({
                "iter": int(float(r["iter"])),
                "regret": regret_used,
                "raw_wr": raw_wr,
                "se": se,
                "games": games,
                "next_regret_actual": regret_chosen_for_next,
            })
    return rows


def backtest(rows: list[dict[str, float]], hl: float) -> dict[str, float]:
    """Walk forward; at each iter compute fit's regret prediction for target_wr.

    Returns aggregate stats across the trajectory.
    """
    history: list[tuple[float, float, float]] = []
    predictions: list[float | None] = []
    actual_chosen: list[float] = []
    for row in rows:
        history.append((row["regret"], row["raw_wr"], row["se"]))
        pred = _fit_inverse_lever(
            history,
            target_wr=TARGET_WR,
            expected_slope_sign=+1,  # regret: more regret → more wins
            recency_half_life=hl,
        )
        predictions.append(pred)
        actual_chosen.append(row["next_regret_actual"])

    valid_preds = [p for p in predictions if p is not None]
    if len(valid_preds) < 2:
        return {"hl": hl, "n_valid": len(valid_preds)}

    pred_diffs = [
        abs(p2 - p1) for p1, p2 in zip(valid_preds[:-1], valid_preds[1:])
    ]
    chosen_diffs = [
        abs(actual_chosen[i] - actual_chosen[i - 1]) for i in range(1, len(actual_chosen))
    ]

    raw_wrs = [r["raw_wr"] for r in rows]

    fit_pinned_clipped = sum(
        1 for p in valid_preds if p > 0.50 - 1e-9 or p < 0.01 + 1e-9
    )

    return {
        "hl": hl,
        "n_valid": len(valid_preds),
        "pred_mean": mean(valid_preds),
        "pred_std": stdev(valid_preds) if len(valid_preds) > 1 else 0.0,
        "pred_jitter": mean(pred_diffs) if pred_diffs else 0.0,
        "chosen_jitter": mean(chosen_diffs) if chosen_diffs else 0.0,
        "raw_wr_mean": mean(raw_wrs),
        "raw_wr_std": stdev(raw_wrs) if len(raw_wrs) > 1 else 0.0,
        "n_pinned": fit_pinned_clipped,
    }


def main() -> None:
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        trials = sorted(
            (REPO / "runs" / "pbt2_small" / "tune").glob("train_trial_*/progress.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if not trials:
            sys.exit("no progress.csv found")
        csv_path = trials[-1]
    print(f"trial: {csv_path}")
    rows = load_trajectory(csv_path)
    print(f"iters loaded: {len(rows)}")
    print(f"target_wr: {TARGET_WR}")
    print()
    print(f"{'hl':>4}  {'n_valid':>7}  {'pred_mean':>10}  {'pred_std':>9}  "
          f"{'pred_jitter':>11}  {'n_pinned':>9}")
    print("-" * 64)
    for hl in HALF_LIVES:
        s = backtest(rows, hl)
        if s.get("n_valid", 0) < 2:
            print(f"{hl:>4.1f}  {'<2 valid':>7}")
            continue
        print(
            f"{hl:>4.1f}  {s['n_valid']:>7}  {s['pred_mean']:>10.4f}  "
            f"{s['pred_std']:>9.4f}  {s['pred_jitter']:>11.4f}  "
            f"{s['n_pinned']:>9}"
        )

    print()
    print("legend:")
    print("  pred_mean    = mean of fit's predicted regret-for-target across iters")
    print("  pred_std     = std-dev of those predictions")
    print("  pred_jitter  = mean |pred_t - pred_{t-1}| (how much the prediction swings)")
    print("  n_pinned     = # iters where fit prediction hit min/max bounds (degenerate fit)")


if __name__ == "__main__":
    main()
