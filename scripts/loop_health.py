"""Loop-health invariant monitor over a trial's result.json (CPU-only).

Every check guards a failure mode the loop has actually had; a green run
prints one compact line per iteration and exits 0, so this is cron/CI safe.

ALERT (exit 1) — needs action:
  - replay_has_policy_frac < 0.9 (only on iters that actually ingested rows):
    value-only rows entering the window (record_fast_ply_value flipped on /
    live-yaml revert — the 2026-07-02 branch-checkout incident shipped exactly
    this silently). The frac is a per-ITERATION ingest metric, so this catches
    the inflow; rows already in the window persist ~a day after a revert.
  - replay_pmass_gap_share > 0: the KILLED #104 gap-priority sampling knob is
    live again.
  - pid_ema_winrate < 0.35: airbag territory / weak opponent (curriculum
    starvation misfire, 2026-07-05). The HIGH side is a NOTE, not an alert —
    a >0.75 spike is the benign post-restart transient.
  - FEN delivery flatline (only when the window shows FEN seeding is active):
    opening_fenlist_games == 0 for 3+ consecutive iters (total delivery
    stopped), or selfplay_fenlist_games == 0 for 3+ iters while total delivery
    continues (the SF-labelled value-injection sub-stream stopped). Dose 0.02
    => a burst of zeros is drift, a single zero is Poisson noise.
  - train_steps_used == 0 for 2+ consecutive iters: a written 0-step iteration.
    NOTE: the canonical no-games drought suppresses the result.json row
    entirely (the trainable short-circuits and writes nothing), so a row-scan
    cannot see it — use --max-age-min to alert on a stale newest row instead.
  - matching_games < 100: selfplay collapse (demoted to a NOTE on a benign
    post-restart iteration, which the stale_games marker identifies).
  - --max-age-min M (opt-in): the newest row's timestamp is older than M
    minutes — result.json has stopped advancing (stall / no-games drought).

NOTE (informational): pid_ema_winrate > 0.75 (benign restart spike / difficulty
miscalibration), distributed_stale_games > 0 (benign post-restart marker),
wdl_regret easing >40% in one step (airbag fired), iteration wall-time > 2x
the scanned-range median.

Usage:
  PYTHONPATH=. python3 scripts/loop_health.py                    # newest trial, last 20 iters
  PYTHONPATH=. python3 scripts/loop_health.py --last 50 --no-expect-fenlist
  PYTHONPATH=. python3 scripts/loop_health.py --result-json PATH
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from chess_anti_engine.tune.result_keys import row_counter, row_counter_opt
from scripts.trial_paths import latest_result_path


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
    *,
    fen_total_alert: bool,
    fen_selfplay_alert: bool,
    steps_zero_streak: int,
) -> tuple[list[str], list[str]]:
    """Pure invariant logic for one iteration -> (alerts, notes).

    FEN alerts are decided in main() (they need window-level state — whether
    seeding is active at all) and passed in as booleans, so this stays a pure,
    testable function of one row plus the caller's precomputed streak flags.
    """
    alerts: list[str] = []
    notes: list[str] = []

    stale = row.get("distributed_stale_games")
    stale_n = int(stale) if stale is not None else 0
    games = row_counter_opt(row, "matching_games")
    games_n = games if games is not None else 0
    # "Stale" used to be treated as PROOF of a restart, which is why a frozen
    # worker fleet hid here for days (2026-07-24): every iteration looked like
    # a restart and got the benign reading.
    #
    # A single iteration cannot tell the two apart — a real restart also strands
    # in-flight games under the previous sha while matching is still ramping.
    # What separates them is PERSISTENCE: a restart is one-off, whereas workers
    # pinned to an old model_sha keep outproducing the matching count every
    # iteration. So stale>matching twice running is the alert; once is a note.
    stale_outruns = stale_n > 0 and games_n > 0 and stale_n > games_n
    prev_stale = int(prev.get("distributed_stale_games") or 0) if prev else 0
    prev_games = row_counter(prev, "matching_games") if prev else 0
    prev_outruns = prev_stale > 0 and prev_games > 0 and prev_stale > prev_games
    workers_frozen = stale_outruns and prev_outruns
    is_restart_iter = stale_n > 0 and not workers_frozen

    # Value-only-row inflow. Guarded on actual ingest AND on a strictly-positive
    # frac: the producer emits exactly 0.0 both for an empty has_policy
    # denominator (shards missing the flag) and a genuine but total flood; a
    # real record_fast_ply_value flood is ~0.25 (has SOME full plies), so
    # `0 < frac < 0.9` catches the flood without the missing-field false alarm.
    ingested = row.get("replay_positions_ingested")
    if ingested is None:
        ingested = row_counter_opt(row, "matching_positions")
    hp = row.get("replay_has_policy_frac")
    if hp is not None and ingested is not None and int(ingested) > 0 and 0.0 < float(hp) < 0.9:
        alerts.append(f"has_policy_frac={float(hp):.3f} <0.9 on {int(ingested)} ingested rows — "
                      "value-only rows entering the window "
                      "(record_fast_ply_value on? live-yaml revert?)")
    gap = row.get("replay_pmass_gap_share")
    if gap is not None and float(gap) > 1e-3:
        alerts.append(f"replay_pmass_gap_share={float(gap):.4f} >0 — "
                      "KILLED #104 gap-priority sampling knob live again")
    if fen_total_alert:
        alerts.append("opening_fenlist_games=0 for 3+ consecutive iters — "
                      "FEN seeding stopped delivering entirely")
    if fen_selfplay_alert:
        alerts.append("selfplay_fenlist_games=0 for 3+ iters — "
                      "the SF-labelled value-injection sub-stream stopped")
    wr = row.get("pid_ema_winrate")
    # 0.0 is the reported fallback when the PID controller is inactive, not a
    # real winrate — require strictly positive so a PID-disabled run isn't red.
    if wr is not None and 0.0 < float(wr) < 0.35:
        alerts.append(f"pid_ema_winrate={float(wr):.3f} <0.35 — airbag territory / weak opponent")
    if steps_zero_streak >= 2:
        alerts.append(f"train_steps_used=0 for {steps_zero_streak} consecutive iters — ingest drought")
    if games is not None and games_n < 100:
        # Workers are still spinning up on the first post-restart iteration, so a
        # low count there is benign (the tool's own stale-games NOTE) — demote.
        if is_restart_iter:
            notes.append(f"matching_games={games_n} <100 on a restart iter — "
                         "benign (workers spinning up)")
        else:
            alerts.append(f"matching_games={games_n} <100 — selfplay collapse")

    if wr is not None and float(wr) > 0.75:
        notes.append(f"pid_ema_winrate={float(wr):.3f} >0.75 — likely benign "
                     "(post-restart spike / difficulty miscalibration), not airbag")
    if is_restart_iter:
        notes.append(f"stale_games={stale_n} — expect a benign winrate spike (restart artifact)")
    if workers_frozen:
        alerts.append(
            f"stale_games={stale_n} > matching_games={games_n} for 2+ iters — "
            "workers frozen on an old model_sha; most selfplay is being "
            "discarded (check worker logs for 'model tag STALE')"
        )
    elif stale_outruns:
        notes.append(
            f"stale_games={stale_n} > matching_games={games_n} — benign if this "
            "follows a restart; an alert if it repeats next iteration"
        )
    if prev is not None:
        r0, r1 = prev.get("wdl_regret"), row.get("wdl_regret")
        if r0 is not None and r1 is not None and float(r0) > 0.0 and float(r1) > 1.4 * float(r0):
            notes.append(f"wdl_regret eased {float(r0):.4f}->{float(r1):.4f} (>40% step) — airbag fired?")
    t = row.get("time_this_iter_s")
    if t is not None and median_iter_s > 0 and float(t) > 2.0 * median_iter_s:
        notes.append(f"iter took {float(t):.0f}s > 2x median {median_iter_s:.0f}s")
    return alerts, notes


def load_rows(path: Path, last: int) -> list[dict]:
    """Parse the last ``last`` rows of result.json, tolerating a torn tail line.

    Streams the file through a bounded deque so a long-running result.json is
    not materialized in full; a cron read can land while Ray is mid-append, so
    an unparseable line is skipped with a stderr note rather than crashing. A
    small buffer over ``last`` covers a torn/blank tail line without dropping a
    real row from the requested window.
    """
    from collections import deque

    with open(path, encoding="utf-8") as f:
        tail = deque(f, maxlen=last + 4)
    rows: list[dict] = []
    n = len(tail)
    for i, line in enumerate(tail):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == n - 1:
                print(f"[loop_health] skipped a torn trailing line in {path} "
                      "(live append in progress)", file=sys.stderr)
            else:
                print(f"[loop_health] skipped an unparseable line in {path}",
                      file=sys.stderr)
    return rows[-last:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-json", type=Path, default=None,
                    help="trial result.json (default: newest under $TRAIN_WORK_DIR/tune)")
    ap.add_argument("--last", type=int, default=20, help="iterations to scan (default 20, min 1)")
    ap.add_argument("--expect-fenlist", action=argparse.BooleanOptionalAction, default=None,
                    help="force the FEN-delivery check on/off; default auto-detects from whether "
                         "the scanned window ever delivered a FEN game (so non-FEN trials are green)")
    ap.add_argument("--max-age-min", type=float, default=None,
                    help="alert if the newest row's timestamp is older than this many minutes "
                         "(catches the no-games drought / stall, which writes no result row). "
                         "Off by default — set it to the expected iteration cadence x ~2.")
    args = ap.parse_args()
    if args.last < 1:
        sys.exit("--last must be >= 1")

    path = args.result_json or latest_result_path(required=True)
    assert path is not None  # required=True raises rather than returning None
    rows = load_rows(path, args.last)
    if not rows:
        sys.exit(f"no iterations in {path}")

    per_iter_stats = [parse_outcome_stats(str(r.get("outcome_stats") or "")) for r in rows]

    # FEN check is armed only when seeding is actually active in this window
    # (--expect-fenlist forces it either way). A non-FEN trial never delivers,
    # so it stays green instead of false-alerting.
    # Both seed kinds count as delivery (plain + blame-backed decision-point
    # seeds emit their own per-source keys).
    def _fen_total(s: dict, prefix: str) -> int:
        return int(s.get(f"{prefix}_fenlist_games", 0)) + int(
            s.get(f"{prefix}_fenlist_backed_games", 0))

    seen_total_fen = any(_fen_total(s, "opening") > 0 for s in per_iter_stats)
    # --expect-fenlist forces the check on/off; otherwise auto-detect from
    # whether the window ever delivered. When forced on, the alerts must fire
    # even over a window that never delivered (that IS the outage the override
    # exists to catch), so fen_expected is the ONLY gate below — the seen_*
    # signals are not ANDed into the alert conditions.
    fen_expected = args.expect_fenlist if args.expect_fenlist is not None else seen_total_fen

    times = sorted(float(r["time_this_iter_s"]) for r in rows if r.get("time_this_iter_s"))
    median_iter_s = statistics.median(times) if times else 0.0

    any_alert = False
    total_fen_zero = 0
    selfplay_fen_zero = 0
    steps_zero_streak = 0
    prev: dict | None = None
    for row, stats in zip(rows, per_iter_stats, strict=True):
        total_fen = _fen_total(stats, "opening")
        selfplay_fen = _fen_total(stats, "selfplay")
        total_fen_zero = 0 if total_fen > 0 else total_fen_zero + 1
        # Count the selfplay-substream streak purely on selfplay delivery — do
        # NOT reset it on a total-zero iter, or a bursty total stream with
        # isolated zeros would forever re-arm the counter and mask a dead
        # substream (the substream is a subset of total, so a real total outage
        # trips this too, which is fine).
        selfplay_fen_zero = 0 if selfplay_fen > 0 else selfplay_fen_zero + 1
        sv = row.get("train_steps_used")
        steps_zero_streak = steps_zero_streak + 1 if (sv is not None and int(sv) == 0) else 0

        fen_total_alert = fen_expected and total_fen_zero >= 3
        fen_selfplay_alert = fen_expected and selfplay_fen_zero >= 3

        alerts, notes = check_row(
            row, prev, median_iter_s,
            fen_total_alert=fen_total_alert, fen_selfplay_alert=fen_selfplay_alert,
            steps_zero_streak=steps_zero_streak,
        )
        it = row.get("training_iteration", "?")
        print(f"iter {it}: wr_ema={float(row.get('pid_ema_winrate') or 0):.3f} "
              f"regret={float(row.get('wdl_regret') or 0):.4f} "
              f"games={row_counter(row, 'matching_games')} "
              f"fen={total_fen}(self {selfplay_fen}) "
              f"steps={int(row.get('train_steps_used') or 0)} "
              f"window={int(row.get('replay') or 0)} "
              f"t={float(row.get('time_this_iter_s') or 0):.0f}s")
        for a in alerts:
            any_alert = True
            print(f"  ALERT: {a}")
        for n in notes:
            print(f"  NOTE:  {n}")
        prev = row

    # Opt-in staleness: the no-games drought writes no new row, so it shows up
    # as the newest row's timestamp going stale rather than as a scanned value.
    if args.max_age_min is not None:
        ts = rows[-1].get("timestamp")
        if ts is not None:
            age_min = (time.time() - float(ts)) / 60.0
            if age_min > args.max_age_min:
                any_alert = True
                print(f"  ALERT: newest row is {age_min:.0f} min old "
                      f"(> {args.max_age_min:.0f}) — result.json not advancing "
                      "(stall / no-games drought)")

    print(f"\n{'ALERTS PRESENT' if any_alert else 'all invariants green'} "
          f"({len(rows)} iters scanned, fen_check={'on' if fen_expected else 'off'}, {path})")
    sys.exit(1 if any_alert else 0)


if __name__ == "__main__":
    main()
