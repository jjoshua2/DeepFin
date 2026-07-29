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

    Nothing else is touched. Read-only, CPU only, no GPU, no model load --
    safe to run against a live run.

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
    hold_in_shadow     (1)  every leg passed but the anchored offset is
                            larger than expected; extend the window.
    kill               (2)  at least one leg failed. Read ``FAILED:`` -- a
                            cadence leg means "your cadence moved", NOT "your
                            attribution is broken".
    not run            (3)  the csv has no gate_sample_* columns at all, so
                            the window never ran. Reported separately because
                            "did not run" reading as "kill" is the exact
                            confusion this whole module exists to remove.

    PYTHONPATH=. python3 scripts/gate_shadow_readout.py runs/pbt2_small/<trial>/progress.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chess_anti_engine.tune.promotion_gate import (
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
        return 2

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
    print(r)
    return _EXIT[r.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
