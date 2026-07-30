#!/usr/bin/env python3
"""The promotion gate's shadow-window readout: promote to enforce, hold, or kill.

WHAT IT READS
    A Ray ``progress.csv`` from the live tune dir. It uses the PER-ITERATION
    columns ``gate_sample_games_cur`` / ``gate_sample_games_prev`` /
    ``gate_sample_delta_elo``, plus ``time_this_iter_s`` for the cadence
    adjustment and ``pid_curriculum_w/d/l`` for the pooled-count identity.
    It never reads ``gate_delta_elo`` or the other window aggregates:
    consecutive windows overlap ~95%, so their sd understates the
    per-iteration sd ~10x and a rule keyed to them cannot fail.

    ``gate_sample_confound_elo`` comes along too, when present: it is the
    PID-lag offset each sample is predicted to carry, in the same Elo units as
    the delta beside it, and the readout regresses one on the other. See
    ``--verbose`` and "THE PID LAG DOES NOT CANCEL" in
    ``chess_anti_engine/tune/promotion_gate.py``.

    Nothing else is touched. Read-only, CPU only, no GPU, no model load --
    safe to run against a live run.

THE COLUMN TO WATCH IN SHADOW MODE IS ``gate_would_demote``
    ``gate_decision == 1`` is NOT "the gate is happy". It is emitted for four
    different situations: nothing fired (``promote_no_regression``), shadow
    mode suppressed a window demote (``shadow_would_demote``), shadow mode
    suppressed a single-iteration STEP demote
    (``shadow_would_demote_step``), and the ``gate_max_hold_iters`` cap
    yielded (``hold_expired``). In shadow mode the whole point is to see the
    gate WANT to fire, and that used to be legible only as
    ``gate_reason_code == 6`` -- a number documented in no yaml and in no
    script.

    ``gate_would_demote`` is 1.0 on exactly the reasons that mean "the demote
    rule fired, whatever the mode then did with it". Chart that column. The
    reason code still distinguishes WHICH leg fired: 5 window-demote,
    6 shadow-suppressed window, 7 hold cap yielded, 8 step-demote,
    9 shadow-suppressed step.

WHAT IT PRINTS
    One verdict line: the verdict, the rows read and how many were usable,
    the mean cur/prev game counts and their share, the window cadence and the
    prev_share that cadence implies, the mean and sd of the per-iteration
    anchored delta in Elo, and -- on a failure -- every leg that fired, by
    name. ``--verbose`` adds the per-leg reference values.

WHAT THE VERDICT MEANS (and the exit code)
    promote_to_enforce (0)  the in-loop split reproduces the offline
                            reconstruction; the instrument is sound. This is
                            a verdict about the INSTRUMENT, never about
                            whether the model improved -- the loop moves
                            ~0.02 Elo/iteration and no window here can see it.
    hold_in_shadow     (1)  every leg passed but the window cannot promote
                            yet. Read ``HOLD:`` -- it is shorter than the
                            pre-registered 40 iterations, or the anchored
                            offset is larger than expected, or the anchored
                            delta is provably tracking the PID's difficulty
                            step rather than the model. Extend the window.
    kill               (2)  at least one leg failed. Read ``FAILED:`` -- a
                            cadence leg means "your cadence moved", NOT "your
                            attribution is broken".
    not run            (3)  the csv has no gate_sample_* columns at all, so
                            the window never ran. Reported separately because
                            "did not run" reading as "kill" is the exact
                            confusion this whole module exists to remove.
    no such file       (4)  the path does not exist. Distinct from 3 for the
                            same reason 3 is distinct from 2: it used to share
                            ``kill``'s exit code, which re-created the very
                            did-not-run/verdict confusion 3 was added to fix,
                            for anything branching on the exit status.

    PYTHONPATH=. python3 scripts/gate_shadow_readout.py runs/pbt2_small/<trial>/progress.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chess_anti_engine.tune.promotion_gate import (
    CONFOUND_SLOPE_MAX,
    CONFOUND_Z,
    OFFLINE,
    READOUT_HOLD,
    READOUT_KILL,
    READOUT_PROMOTE,
    shadow_readout_from_csv,
)

_EXIT = {READOUT_PROMOTE: 0, READOUT_HOLD: 1, READOUT_KILL: 2}
_SAMPLE_COLUMNS = ("gate_sample_games_cur", "gate_sample_games_prev",
                   "gate_sample_delta_elo")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("progress_csv", help="path to the trial's progress.csv")
    ap.add_argument(
        "--last-n", type=int, default=40,
        help="iterations to read, newest first (default: 40)",
    )
    ap.add_argument(
        "--min-games-per-side", type=int, default=15,
        help="games a side an iteration needs to be usable (default: 15, the "
             "shipped gate_min_games_per_side)",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="also print the offline reference each leg is measured against",
    )
    args = ap.parse_args()

    path = Path(args.progress_csv)
    if not path.is_file():
        print(f"no such file: {path}\n"
              "  (a git worktree has no runs/; point this at the live checkout, "
              "e.g. /home/josh/projects/chess/runs/pbt2_small/<trial>/progress.csv)")
        return 4

    # "The gate never ran" must not print as "kill". A csv written before this
    # module shipped has no gate_sample_* columns at all, which the readout
    # sees as an empty window and reports as a failed rule -- the same
    # did-not-run/failed confusion the old ``gate_passed: 1`` carried.
    with path.open(newline="") as fh:
        header = csv.DictReader(fh).fieldnames or []
    missing = [c for c in _SAMPLE_COLUMNS if c not in header]
    if missing:
        print(f"NOT RUN  {path} has no {', '.join(missing)} column(s), so the "
              "shadow window never ran on it. This is not a kill: the gate "
              "emits these at gate_mode: off, so a csv without them predates "
              "the gate or comes from a different trial.")
        return 3

    r = shadow_readout_from_csv(
        path, min_games_per_side=args.min_games_per_side, last_n=args.last_n,
    )
    if args.verbose:
        print(f"reference (offline, live iters 163-219, n={OFFLINE.n_usable}): "
              f"games_cur={OFFLINE.mean_games_cur} games_prev={OFFLINE.mean_games_prev} "
              f"prev_share={OFFLINE.prev_share} "
              f"refresh_lag={OFFLINE.refresh_lag_seconds:.1f}s "
              f"games_per_second={OFFLINE.games_per_second:.4f} "
              f"cadence={OFFLINE.mean_iter_seconds}s "
              f"delta mean={OFFLINE.mean_delta_elo} sd={OFFLINE.sd_delta_elo}")
        print(f"confound leg: holds when the OLS slope of "
              f"gate_sample_delta_elo on gate_sample_confound_elo is "
              f"significantly above {CONFOUND_SLOPE_MAX} (one-sided, z="
              f"{CONFOUND_Z}). At 40 rows se(slope) is ~0.60, so this leg "
              f"cannot decide anything on the pre-registered window -- read "
              f"'needs ~N rows' and pass --last-n once N exist.")
    print(r)
    return _EXIT[r.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
