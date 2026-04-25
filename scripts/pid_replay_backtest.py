"""Replay historical PID observations against the current pid.py logic and
sweep (ema_alpha, deadband_sigma) on practical-failure-mode metrics.

The original FP/FN framing assumed a forward-window mean as ground truth
for "should the controller have acted?". On 450-iter live data, that proxy
is dominated by real per-iter swings that aren't noise. We score instead
on the failures the controller is actually supposed to prevent:

  - airbag_fires: iters with raw_wr < safety_floor (the catastrophic
    "controller waited too long" signal — what triggered the iter 448
    investigation in the first place).
  - drift_iters: iters where ema is >2pp from target (chronic miss).
  - regret_swings: count of regret direction reversals, weighted by
    magnitude (proxy for limit-cycle behavior).

Lower is better on all three. The half-step-on-disagreement fix changes
how the controller acts when it does, so we expect different α/σ tradeoffs
than the prior backtest.

Caveat: the historical (W, D, L) per iter were generated under a SPECIFIC
regret trajectory that depended on the live alpha/sigma. Replaying with
different (alpha, sigma) is counterfactual w.r.t. the feedback loop — what
this measures is decision quality of the controller given identical
observations, NOT end-to-end training outcome.

Usage:
    PYTHONPATH=. python3 scripts/pid_replay_backtest.py
"""
from __future__ import annotations

import json
from pathlib import Path

from chess_anti_engine.stockfish.pid import DifficultyPID

TRIAL_DIR = Path("runs/pbt2_small/tune/train_trial_ba920_00000_0_lr=0.0003_2026-04-22_17-53-42")
RESULT_PATH = TRIAL_DIR / "result.json"

# Match yaml constants — only alpha and sigma sweep.
TARGET = 0.57
WDL_REGRET_START = 0.20
WDL_REGRET_MIN = 0.01
WDL_REGRET_MAX = 0.40
INVERSE_WINDOW = 20
INVERSE_MAX_STEP = 0.02
INVERSE_MAX_STEP_FRAC = 0.12
INVERSE_SAFETY_FLOOR = 0.50
INVERSE_EMERGENCY_EASE = 0.03
INVERSE_RECENCY_HL = 3.0
MIN_NODES = 5000
MAX_NODES = 5000  # locked min==max in current yaml; nodes-stage gate stays inactive

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
SIGMAS = [0.5, 1.0, 1.5, 2.0]

# An iter is "drifting" if ema is more than this far from target.
DRIFT_THRESHOLD = 0.02


def load_iters() -> list[dict]:
    rows: list[dict] = []
    with open(RESULT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_raw_wr(row: dict) -> float:
    win = int(row.get("win", 0) or 0)
    draw = int(row.get("draw", 0) or 0)
    loss = int(row.get("loss", 0) or 0)
    total = win + draw + loss
    return (win + 0.5 * draw) / total if total > 0 else 0.0


def replay(rows: list[dict], *, alpha: float, sigma: float) -> dict:
    pid = DifficultyPID(
        initial_nodes=MIN_NODES,
        target_winrate=TARGET,
        ema_alpha=alpha,
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES,
        initial_wdl_regret=WDL_REGRET_START,
        wdl_regret_min=WDL_REGRET_MIN,
        wdl_regret_max=WDL_REGRET_MAX,
        wdl_regret_stage_end=-1.0,  # disabled — keep stage 1 active
        inverse_regret_window=INVERSE_WINDOW,
        inverse_regret_max_step=INVERSE_MAX_STEP,
        inverse_regret_max_step_frac=INVERSE_MAX_STEP_FRAC,
        inverse_regret_safety_floor=INVERSE_SAFETY_FLOOR,
        inverse_regret_emergency_ease_step=INVERSE_EMERGENCY_EASE,
        inverse_regret_recency_half_life=INVERSE_RECENCY_HL,
        inverse_regret_target_deadband_sigma=sigma,
        min_games_between_adjust=0,  # always permit per-iter step
    )
    regret_trail: list[float] = []
    acted_per_iter: list[bool] = []
    drift_holds = 0  # held this iter despite raw and ema both being far from target
    held_through_floor_drop = 0  # held in iter t-1 even though raw[t] hit airbag
    for idx, row in enumerate(rows):
        wins = int(row.get("win", 0) or 0)
        draws = int(row.get("draw", 0) or 0)
        losses = int(row.get("loss", 0) or 0)
        if wins + draws + losses <= 0:
            acted_per_iter.append(False)
            regret_trail.append(float(pid.wdl_regret))
            continue
        raw_wr = (wins + 0.5 * draws) / (wins + draws + losses)
        update = pid.observe(wins=wins, draws=draws, losses=losses, force=True)
        acted = bool(update.wdl_regret_changed)
        acted_per_iter.append(acted)
        regret_trail.append(float(pid.wdl_regret))
        if (
            abs(raw_wr - TARGET) >= DRIFT_THRESHOLD
            and abs(update.ema_winrate - TARGET) >= DRIFT_THRESHOLD
            and not acted
        ):
            drift_holds += 1
        # Catastrophic-delay detector: did the controller hold in iter t-1
        # while iter t's raw_wr was about to crash below the safety floor?
        if (
            idx + 1 < len(rows)
            and not acted
        ):
            next_row = rows[idx + 1]
            n2 = next_row.get("win", 0) + next_row.get("draw", 0) + next_row.get("loss", 0)
            if n2 > 0:
                next_raw = (next_row["win"] + 0.5 * next_row["draw"]) / n2
                if next_raw < INVERSE_SAFETY_FLOOR:
                    held_through_floor_drop += 1

    # Regret oscillation: count direction reversals weighted by step size.
    # Lower = smoother trajectory; higher = limit-cycling.
    swing_score = 0.0
    if len(regret_trail) >= 3:
        prev_dir = 0
        for i in range(1, len(regret_trail)):
            diff = regret_trail[i] - regret_trail[i - 1]
            if abs(diff) < 1e-9:
                continue
            cur_dir = 1 if diff > 0 else -1
            if prev_dir != 0 and cur_dir != prev_dir:
                swing_score += abs(diff)
            prev_dir = cur_dir

    # Longest consecutive no-action streak.
    max_hold = 0
    cur_hold = 0
    for acted in acted_per_iter:
        if acted:
            cur_hold = 0
        else:
            cur_hold += 1
            max_hold = max(max_hold, cur_hold)

    return {
        "alpha": alpha,
        "sigma": sigma,
        "acts": sum(acted_per_iter),
        "drift_holds": drift_holds,
        "floor_drops": held_through_floor_drop,
        "max_hold": max_hold,
        "swing": swing_score,
    }


def main() -> None:
    rows = load_iters()
    print(f"Loaded {len(rows)} iters (1..{rows[-1]['iter']}) from {RESULT_PATH.name}")
    print(f"drift_threshold={DRIFT_THRESHOLD}  safety_floor={INVERSE_SAFETY_FLOOR}\n")

    results = []
    for alpha in ALPHAS:
        for sigma in SIGMAS:
            results.append(replay(rows, alpha=alpha, sigma=sigma))

    print(
        f"{'alpha':>6}  {'sigma':>5}  {'acts':>5}  {'driftH':>6}  {'floorH':>6}  "
        f"{'maxH':>4}  {'swing':>6}"
    )
    for r in results:
        print(
            f"{r['alpha']:>6.2f}  {r['sigma']:>5.2f}  {r['acts']:>5}  {r['drift_holds']:>6}  "
            f"{r['floor_drops']:>6}  {r['max_hold']:>4}  {r['swing']:>6.3f}"
        )
    print()
    print("driftH = held when both raw and ema were ≥drift_threshold from target")
    print("floorH = held in iter t-1 while iter t's raw was about to drop below safety_floor")
    print("maxH   = longest consecutive no-action streak (the iter-448 failure mode)")
    print("swing  = sum of |Δregret| at direction reversals (limit-cycle proxy)")
    print()
    # Lower is better on driftH, floorH, maxH, swing.
    best = min(results, key=lambda r: (r["floor_drops"], r["drift_holds"], r["max_hold"], r["swing"]))
    print(
        f"Best (lex: floorH, driftH, maxH, swing): "
        f"α={best['alpha']} σ={best['sigma']}  "
        f"driftH={best['drift_holds']} floorH={best['floor_drops']} maxH={best['max_hold']} "
        f"swing={best['swing']:.3f}"
    )
    prod = next((r for r in results if r["alpha"] == 0.10 and r["sigma"] == 1.5), None)
    if prod is not None:
        print(
            f"Current prod (α=0.10, σ=1.5): driftH={prod['drift_holds']} floorH={prod['floor_drops']} "
            f"maxH={prod['max_hold']} swing={prod['swing']:.3f}"
        )


if __name__ == "__main__":
    main()
