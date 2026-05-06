"""Replay stage-2 nodes-lever observations and sweep (ema_alpha, sigma).

Companion to ``pid_replay_backtest.py`` (regret-side). The regret backtest
disables the stage gate; this one does the opposite — pins regret at the
floor so the nodes lever is the active controller from iter 0 — then
replays the historical stage-2 W/D/L observations through alternative
``(α, σ)`` settings.

Caveat (same as the regret-side script): the historical W/D/L per iter
were generated under the live (α, σ) and the live nodes trajectory.
Replaying with different settings is counterfactual w.r.t. the feedback
loop. It measures decision quality of the controller given identical
observations, not end-to-end winrate. Useful for ranking, not for
prediction.

Stage-2 history is small (~38 iters as of 2026-04-26 on trial 2b03f), so
treat the leaderboard as directional.

Usage:
    PYTHONPATH=. python3 scripts/pid_replay_backtest_nodes.py
    PYTHONPATH=. python3 scripts/pid_replay_backtest_nodes.py --trial-result PATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_anti_engine.stockfish.pid import DifficultyPID

# Yaml constants — fixed across the sweep. Sweep dims are α and σ.
TARGET = 0.60
MIN_NODES = 5_000
MAX_NODES = 20_000
NODES_WINDOW = 20
NODES_MAX_STEP_FRAC = 0.05  # current yaml value (2026-04-26 cut from 0.10)
NODES_RECENCY_HL = 3.0
NODES_SAFETY_FLOOR = 0.50
NODES_EMERGENCY_EASE = 1000.0

ALPHAS = [0.05, 0.10, 0.15, 0.20]
SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# An iter is "drifting" if ema is more than this far from target.
DRIFT_THRESHOLD = 0.02
# Stage 2 is "active" when regret is at floor AND nodes have started moving.
# Use the live `wdl_regret` field as the gate.
STAGE2_REGRET_THRESHOLD = 0.0105


def load_iters(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_result_path(root: Path) -> Path:
    candidates = sorted(
        root.glob("runs/pbt2_small/tune/train_trial_*/result.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no result.json files under {root / 'runs' / 'pbt2_small' / 'tune'}; "
            "pass --trial-result"
        )
    return candidates[-1]


def stage2_rows(rows: list[dict]) -> list[dict]:
    """Slice to iters where the live trajectory had regret at floor."""
    return [r for r in rows if float(r.get("wdl_regret", 1.0)) <= STAGE2_REGRET_THRESHOLD]


def replay(rows: list[dict], *, alpha: float, sigma: float, initial_nodes: int) -> dict:
    pid = DifficultyPID(
        initial_nodes=initial_nodes,
        target_winrate=TARGET,
        ema_alpha=alpha,
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES,
        # Pin regret at floor + set stage_end = floor so the gate flips True
        # at construction → nodes lever is active from iter 0 of replay.
        initial_wdl_regret=0.01,
        wdl_regret_min=0.01,
        wdl_regret_max=0.40,
        wdl_regret_stage_end=0.01,
        regret_window=20,
        regret_max_step=0.01,
        regret_max_step_frac=0.0,
        regret_safety_floor=0.50,
        regret_emergency_ease_step=0.01,
        regret_recency_half_life=0.0,
        regret_deadband_sigma=1.0,
        nodes_window=NODES_WINDOW,
        nodes_max_step_frac=NODES_MAX_STEP_FRAC,
        nodes_safety_floor=NODES_SAFETY_FLOOR,
        nodes_emergency_ease_step=NODES_EMERGENCY_EASE,
        nodes_recency_half_life=NODES_RECENCY_HL,
        nodes_deadband_sigma=sigma,
        min_games_between_adjust=0,
    )
    nodes_trail: list[float] = []
    acted_per_iter: list[bool] = []
    drift_holds = 0
    held_through_floor_drop = 0
    airbag_fires = 0

    for idx, row in enumerate(rows):
        wins = int(row.get("win", 0) or 0)
        draws = int(row.get("draw", 0) or 0)
        losses = int(row.get("loss", 0) or 0)
        if wins + draws + losses <= 0:
            acted_per_iter.append(False)
            nodes_trail.append(float(pid.nodes))
            continue
        raw_wr = (wins + 0.5 * draws) / (wins + draws + losses)
        nodes_before = int(pid.nodes)
        update = pid.observe(wins=wins, draws=draws, losses=losses, force=True)
        nodes_after = int(pid.nodes)
        acted = nodes_after != nodes_before
        acted_per_iter.append(acted)
        nodes_trail.append(float(nodes_after))

        if raw_wr < NODES_SAFETY_FLOOR:
            airbag_fires += 1
        if (
            abs(raw_wr - TARGET) >= DRIFT_THRESHOLD
            and abs(update.ema_winrate - TARGET) >= DRIFT_THRESHOLD
            and not acted
        ):
            drift_holds += 1
        if idx + 1 < len(rows) and not acted:
            nxt = rows[idx + 1]
            n2 = int(nxt.get("win", 0) or 0) + int(nxt.get("draw", 0) or 0) + int(nxt.get("loss", 0) or 0)
            if n2 > 0:
                next_raw = (int(nxt["win"]) + 0.5 * int(nxt["draw"])) / n2
                if next_raw < NODES_SAFETY_FLOOR:
                    held_through_floor_drop += 1

    swing_score = 0.0
    if len(nodes_trail) >= 3:
        prev_dir = 0
        for i in range(1, len(nodes_trail)):
            diff = nodes_trail[i] - nodes_trail[i - 1]
            if abs(diff) < 1.0:  # nodes are integer-ish; ignore tiny float noise
                continue
            cur_dir = 1 if diff > 0 else -1
            if prev_dir != 0 and cur_dir != prev_dir:
                swing_score += abs(diff)
            prev_dir = cur_dir

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
        "airbag_fires": airbag_fires,
        "max_hold": max_hold,
        "swing": swing_score,
        "n_iters": len(nodes_trail),
        "final_nodes": int(nodes_trail[-1]) if nodes_trail else MIN_NODES,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    p.add_argument("--trial-result", type=Path, default=None)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()

    path = (
        args.trial_result.expanduser().resolve()
        if args.trial_result is not None
        else _latest_result_path(args.root.expanduser().resolve())
    )
    all_rows = load_iters(path)
    rows = stage2_rows(all_rows)
    if not rows:
        raise SystemExit(f"No stage-2 iters (regret <= {STAGE2_REGRET_THRESHOLD}) in {path}")
    initial_nodes = int(rows[0].get("sf_nodes", MIN_NODES))
    print(f"Loaded {len(all_rows)} iters; stage-2 slice = {len(rows)} iters "
          f"(iter {rows[0].get('iter')}..{rows[-1].get('iter')})")
    print(f"replay starts nodes={initial_nodes}, target_wr={TARGET}, "
          f"max_step_frac={NODES_MAX_STEP_FRAC}, drift_threshold={DRIFT_THRESHOLD}\n")

    results = []
    for alpha in ALPHAS:
        for sigma in SIGMAS:
            results.append(replay(rows, alpha=alpha, sigma=sigma, initial_nodes=initial_nodes))

    print(
        f"{'alpha':>6}  {'sigma':>5}  {'acts':>5}  {'drift':>6}  {'floor':>6}  "
        f"{'airbag':>6}  {'maxH':>4}  {'swing':>10}  {'final':>6}"
    )
    for r in results:
        print(
            f"{r['alpha']:>6.2f}  {r['sigma']:>5.2f}  {r['acts']:>5}  "
            f"{r['drift_holds']:>6}  {r['floor_drops']:>6}  "
            f"{r['airbag_fires']:>6}  {r['max_hold']:>4}  "
            f"{r['swing']:>10.0f}  {r['final_nodes']:>6}"
        )
    print()
    print("acts   = iters where the nodes lever changed value")
    print("drift  = held when both raw and ema were ≥drift_threshold from target")
    print("floor  = held in iter t-1 while iter t's raw was about to drop below safety_floor")
    print("airbag = iters where raw_wr < safety_floor (ground truth catastrophic dip)")
    print("maxH   = longest consecutive no-action streak")
    print("swing  = Σ |Δnodes| at direction reversals (limit-cycle proxy; lower = smoother)")
    print()
    best_smooth = min(results, key=lambda r: (r["floor_drops"], r["swing"], r["drift_holds"]))
    print(
        f"Best smooth (lex: floor, swing, drift): "
        f"α={best_smooth['alpha']} σ={best_smooth['sigma']}  "
        f"swing={best_smooth['swing']:.0f} drift={best_smooth['drift_holds']} "
        f"floor={best_smooth['floor_drops']} maxH={best_smooth['max_hold']}"
    )
    prod = next((r for r in results if r["alpha"] == 0.10 and r["sigma"] == 0.5), None)
    if prod is not None:
        print(
            f"Current prod (α=0.10, σ=0.5):     swing={prod['swing']:.0f} "
            f"drift={prod['drift_holds']} floor={prod['floor_drops']} maxH={prod['max_hold']}"
        )


if __name__ == "__main__":
    main()
