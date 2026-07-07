"""Loop-health invariant monitor over a trial's result.json (CPU-only).

Every check guards a failure mode the loop has actually had; a green run
prints one compact line per iteration and exits 0, so this is cron/CI safe.

ALERT (exit 1) — needs action:
  - replay_has_policy_frac < 0.9: value-only rows flooding the window
    (record_fast_ply_value flipped on / live-yaml revert — the 2026-07-02
    branch-checkout incident shipped exactly this silently).
  - replay_pmass_gap_share / _fast_share > 0: a KILLED priority-shaping knob
    (#104 family) is live again.
  - opening_fenlist_games == 0 for 3+ consecutive iters while --expect-fenlist:
    blind-spot FEN seeding stopped delivering (dose 0.02 => a burst of zeros
    is drift, a single zero is Poisson noise).
  - pid_ema_winrate outside [0.35, 0.75]: airbag territory (curriculum
    starvation misfire, 2026-07-05).
  - train_steps_used == 0 for 2+ consecutive iters: ingest drought.
  - games_generated < 100: selfplay collapse.

NOTE (informational): distributed_stale_games > 0 (benign post-restart
winrate spike marker), wdl_regret easing >40% in one step (airbag fired),
iteration wall-time > 2x the scanned-range median.

Usage:
  PYTHONPATH=. python3 scripts/loop_health.py                    # newest trial, last 20 iters
  PYTHONPATH=. python3 scripts/loop_health.py --last 50 --no-expect-fenlist
  PYTHONPATH=. python3 scripts/loop_health.py --result-json PATH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_outcome_stats(raw: str) -> dict[str, int]:
    """outcome_stats is a pipe-delimited 'key=value|key=value' string."""
    out: dict[str, int] = {}
    for pair in raw.split("|"):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            continue
    return out


def check_row(
    row: dict,
    prev: dict | None,
    median_iter_s: float,
    fen_zero_streak: int,
    steps_zero_streak: int,
) -> tuple[list[str], list[str]]:
    """Pure invariant logic for one iteration -> (alerts, notes)."""
    alerts: list[str] = []
    notes: list[str] = []

    hp = row.get("replay_has_policy_frac")
    if hp is not None and float(hp) < 0.9:
        alerts.append(f"has_policy_frac={float(hp):.3f} <0.9 — value-only rows in window "
                      "(record_fast_ply_value on? live-yaml revert?)")
    for key, label in (("replay_pmass_gap_share", "gap"), ("replay_pmass_fast_share", "fast")):
        share = row.get(key)
        if share is not None and float(share) > 1e-3:
            alerts.append(f"{key}={float(share):.4f} >0 — KILLED {label}-priority knob live again")
    if fen_zero_streak >= 3:
        alerts.append(f"opening_fenlist_games=0 for {fen_zero_streak} consecutive iters — "
                      "FEN seeding stopped delivering")
    wr = row.get("pid_ema_winrate")
    if wr is not None and not 0.35 <= float(wr) <= 0.75:
        alerts.append(f"pid_ema_winrate={float(wr):.3f} outside [0.35, 0.75] — airbag territory")
    if steps_zero_streak >= 2:
        alerts.append(f"train_steps_used=0 for {steps_zero_streak} consecutive iters — ingest drought")
    games = row.get("games_generated")
    if games is not None and int(games) < 100:
        alerts.append(f"games_generated={int(games)} <100 — selfplay collapse")

    stale = row.get("distributed_stale_games")
    if stale is not None and int(stale) > 0:
        notes.append(f"stale_games={int(stale)} — expect a benign winrate spike (restart artifact)")
    if prev is not None:
        r0, r1 = prev.get("wdl_regret"), row.get("wdl_regret")
        if r0 and r1 and float(r1) > 1.4 * float(r0):
            notes.append(f"wdl_regret eased {float(r0):.4f}->{float(r1):.4f} (>40% step) — airbag fired?")
    t = row.get("time_this_iter_s")
    if t is not None and median_iter_s > 0 and float(t) > 2.0 * median_iter_s:
        notes.append(f"iter took {float(t):.0f}s > 2x median {median_iter_s:.0f}s")
    return alerts, notes


def newest_result_json() -> Path:
    work_dir = Path(os.environ.get("TRAIN_WORK_DIR", "runs/pbt2_small"))
    candidates = sorted(
        work_dir.glob("tune/train_trial_*/result.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        sys.exit(f"no result.json under {work_dir}/tune/train_trial_*/")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-json", type=Path, default=None,
                    help="trial result.json (default: newest under $TRAIN_WORK_DIR/tune)")
    ap.add_argument("--last", type=int, default=20, help="iterations to scan (default 20)")
    ap.add_argument("--expect-fenlist", action=argparse.BooleanOptionalAction, default=True,
                    help="alert when FEN-seeding delivery flatlines (on while #108 is active)")
    args = ap.parse_args()

    path = args.result_json or newest_result_json()
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[-args.last:]
    if not rows:
        sys.exit(f"no iterations in {path}")

    times = sorted(float(r["time_this_iter_s"]) for r in rows if r.get("time_this_iter_s"))
    median_iter_s = times[len(times) // 2] if times else 0.0

    any_alert = False
    fen_zero_streak = 0
    steps_zero_streak = 0
    prev: dict | None = None
    for row in rows:
        stats = parse_outcome_stats(str(row.get("outcome_stats") or ""))
        fen_games = stats.get("opening_fenlist_games", 0)
        fen_zero_streak = 0 if (fen_games > 0 or not args.expect_fenlist) else fen_zero_streak + 1
        steps_zero_streak = 0 if int(row.get("train_steps_used") or 0) > 0 else steps_zero_streak + 1

        alerts, notes = check_row(row, prev, median_iter_s, fen_zero_streak, steps_zero_streak)
        it = row.get("training_iteration", "?")
        print(f"iter {it}: wr_ema={float(row.get('pid_ema_winrate') or 0):.3f} "
              f"regret={float(row.get('wdl_regret') or 0):.4f} "
              f"games={int(row.get('games_generated') or 0)} "
              f"fen={fen_games} steps={int(row.get('train_steps_used') or 0)} "
              f"window={int(row.get('replay') or 0)} "
              f"t={float(row.get('time_this_iter_s') or 0):.0f}s")
        for a in alerts:
            any_alert = True
            print(f"  ALERT: {a}")
        for n in notes:
            print(f"  NOTE:  {n}")
        prev = row

    print(f"\n{'ALERTS PRESENT' if any_alert else 'all invariants green'} "
          f"({len(rows)} iters scanned, {path})")
    sys.exit(1 if any_alert else 0)


if __name__ == "__main__":
    main()
