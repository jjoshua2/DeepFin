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

⚑ READ THE PER-LEG TABLE, NOT THE EXIT CODE
    An exit code is an OR over axes. This command prints one line per AXIS
    with the state that axis is in -- PASS, FAIL, HOLD, SKIPPED or
    UNMEASURED -- and the pre-registered success criterion is a statement
    about those states, not about the process's return value.

    The trap is concrete and current (ledger, audit wave 3, K1 x G3-2): the
    attribution axis false-killed on the production fleet, and fixing that axis
    ALONE would have made the same command exit 0 -- the pre-registered signal
    to set ``gate_mode: enforce`` -- while the PID-confound axis, the deciding
    KILL rule the ledger registered for exactly this promotion, measured
    nothing at all. ``gate_sample_confound_elo`` is NaN on 109 of 109 live rows
    because the server's upload compactor rebuilds ``ShardMeta`` without
    ``opponent_wdl_regret_limit``. A window whose confound axis has fewer than
    3 measurements therefore CANNOT exit 0 or 1; it gets exit 5.

⚑ THE REFERENCE MUST BE RE-DERIVED BEFORE ``gate_mode`` LEAVES ``off``
    Every leg is stated relative to ``OfflineReference``, measured over live
    iterations 163-219 (2026-06). The ``prev`` arm is the shards still tagged
    with the previous sha because the worker had not picked up the new
    manifest yet, so it is a model-refresh LAG -- roughly constant in SECONDS,
    not as a share -- and the axis is reported in seconds for that reason.

    On the current trial the two available readings of that lag DISAGREE:
    ``progress.csv`` implies ~166 s over the last 40 rows, while an
    independent shard re-derivation over the same trial implies ~113 s, which
    is essentially the calibration value. A moved lag and a mis-attributed
    split are the same number in ``progress.csv`` and cannot be separated from
    it, so the disagreement is exactly what has to be resolved before any
    constant moves.

    ``--rederive-reference SHARD_ROOT`` is the independent side, read from the
    shard ``.zattrs`` the reference was originally built from and NOT from the
    gate's own columns (a control conditioned on its own outcome cannot fail).
    It prints constants; it applies nothing. Record them in
    ``docs/experiment_ledger.md`` and paste them into ``OfflineReference`` in
    the same change, at restart prep -- never mid-window, and never after
    reading a verdict you did not like.

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

    In ENFORCE mode the column that says whether the brake actually engaged is
    ``gate_hold_effective``, not ``gate_decision``: a hold with no anchor on
    disk publishes the demoted net anyway and is otherwise indistinguishable
    in the csv. ``gate_fallback_missing`` counts those.

WHAT IT PRINTS
    One verdict line then the per-leg table. The verdict line carries the
    verdict, the rows read and how many were usable, the mean cur/prev game counts and their share, the
    window cadence, the prev_share that cadence implies, the measured refresh
    lag against the reference lag, and the mean and sd of the per-iteration
    anchored delta in Elo. ``--verbose`` adds the offline reference itself.

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
                            attribution is broken", and a refresh_lag leg
                            means one of those two and cannot say which.
    not run            (3)  the csv has no gate_sample_* columns at all, so
                            the window never ran. Reported separately because
                            "did not run" reading as "kill" is the exact
                            confusion this whole module exists to remove.
    no such file       (4)  the path does not exist. Distinct from 3 for the
                            same reason 3 is distinct from 2: it used to share
                            ``kill``'s exit code, which re-created the very
                            did-not-run/verdict confusion 3 was added to fix,
                            for anything branching on the exit status.
    confound unmeasured (5) every other axis may have passed, but the
                            PID-confound axis has fewer than 3 measurements,
                            so the deciding KILL rule the ledger pre-registered
                            for enabling this gate was evaluated over an empty
                            set. NOT a kill: nothing is broken in this window.
                            The producer is -- see ``ShardMeta``.

    PYTHONPATH=. python3 scripts/gate_shadow_readout.py runs/pbt2_small/<trial>/progress.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from chess_anti_engine.tune.promotion_gate import (
    CONFOUND_SLOPE_MAX,
    CONFOUND_Z,
    OFFLINE,
    READOUT_EXIT_CONFOUND_UNMEASURED,
    read_iteration_bins,
    read_shard_arms,
    readout_exit_code,
    rederive_reference_from_shards,
    shadow_readout_from_csv,
)

_NOT_RUN, _NO_FILE = 3, 4
_SAMPLE_COLUMNS = ("gate_sample_games_cur", "gate_sample_games_prev",
                   "gate_sample_delta_elo")


def _rederive(progress_csv: Path, shard_root: Path) -> int:
    """Print the reference constants this shard window implies. Applies nothing."""
    if not shard_root.is_dir():
        print(f"no such shard directory: {shard_root}")
        return _NO_FILE
    bins = read_iteration_bins(progress_csv)
    shards = read_shard_arms(shard_root)
    r = rederive_reference_from_shards(bins, shards)
    print(
        f"re-derived from {r.n_shards} shards under {shard_root} binned against "
        f"{r.n_iterations} iterations of {progress_csv}\n"
        f"  usable bins (both arms non-empty): {r.n_usable}\n"
        f"  mean_games_cur    {r.mean_games_cur:.1f}   "
        f"(shipped {OFFLINE.mean_games_cur})\n"
        f"  mean_games_prev   {r.mean_games_prev:.1f}   "
        f"(shipped {OFFLINE.mean_games_prev})\n"
        f"  prev_share        {r.prev_share:.4f}   "
        f"(shipped {OFFLINE.prev_share})\n"
        f"  mean_iter_seconds {r.mean_iter_seconds:.1f}   "
        f"(shipped {OFFLINE.mean_iter_seconds})\n"
        f"  refresh_lag       {r.refresh_lag_seconds:.1f}s  "
        f"(shipped {OFFLINE.refresh_lag_seconds:.1f}s)\n"
        f"  mean_delta_elo    {r.mean_delta_elo:.2f}   "
        f"(shipped {OFFLINE.mean_delta_elo})\n"
        f"  sd_delta_elo      {r.sd_delta_elo:.2f}   "
        f"(shipped {OFFLINE.sd_delta_elo})\n"
        "\nOfflineReference body these imply:\n"
        f"{r.as_offline_reference_source()}"
        "\nNOTHING WAS APPLIED. These are read from shard .zattrs -- an "
        "independent\nreconstruction that never touches the gate's own split, "
        "which is what makes\nthe attribution axis a control at all. Record "
        "them in docs/experiment_ledger.md\nand edit OfflineReference in the "
        "same change, at restart prep."
    )
    return 0


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
        "--refresh-lag-seconds", type=float, default=None,
        help="evaluate the attribution axis against THIS fleet refresh lag "
             f"instead of the shipped reference's {OFFLINE.refresh_lag_seconds:.1f} s. "
             "Use only with a lag re-derived from shards "
             "(--rederive-reference) and recorded in the ledger: it moves a "
             "pre-registered leg.",
    )
    ap.add_argument(
        "--rederive-reference", metavar="SHARD_ROOT", default=None,
        help="do not judge anything; re-derive the OfflineReference constants "
             "from the shard .zattrs under SHARD_ROOT (e.g. "
             "runs/pbt2_small/server/trials/<trial>/processed), binned against "
             "the given progress.csv. Prints constants, applies nothing.",
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
        return _NO_FILE

    if args.rederive_reference is not None:
        return _rederive(path, Path(args.rederive_reference))

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
        return _NOT_RUN

    ref = OFFLINE
    if args.refresh_lag_seconds is not None:
        # ``refresh_lag_seconds`` is derived (share x cadence), so the override
        # lands on the share the reference was measured at.
        ref = replace(
            OFFLINE,
            prev_share=float(args.refresh_lag_seconds) / OFFLINE.mean_iter_seconds,
        )
        print(
            f"⚑ attribution axis re-based: refresh lag "
            f"{args.refresh_lag_seconds:.1f}s instead of the shipped "
            f"{OFFLINE.refresh_lag_seconds:.1f}s (prev_share reference "
            f"{OFFLINE.prev_share} -> {ref.prev_share:.4f}). The COUNT legs "
            "still use the shipped constants; a full re-derivation moves all "
            "of them."
        )

    r = shadow_readout_from_csv(
        path, min_games_per_side=args.min_games_per_side, last_n=args.last_n,
        ref=ref,
    )
    if args.verbose:
        print(f"reference (offline, live iters 163-219, n={ref.n_usable}): "
              f"games_cur={ref.mean_games_cur} games_prev={ref.mean_games_prev} "
              f"prev_share={ref.prev_share} "
              f"refresh_lag={ref.refresh_lag_seconds:.1f}s "
              f"games_per_second={ref.games_per_second:.4f} "
              f"cadence={ref.mean_iter_seconds}s "
              f"delta mean={ref.mean_delta_elo} sd={ref.sd_delta_elo}")
        print(f"confound leg: holds when the OLS slope of "
              f"gate_sample_delta_elo on gate_sample_confound_elo is "
              f"significantly above {CONFOUND_SLOPE_MAX} (one-sided, z="
              f"{CONFOUND_Z}). At 40 rows se(slope) is ~0.60, so this leg "
              f"cannot decide anything on the pre-registered window -- read "
              f"'needs ~N rows' and pass --last-n once N exist.")
    # The verdict line first (it is what every existing consumer greps for),
    # then the per-axis table, which is what the rule is actually about.
    print(r)
    print("legs:")
    print(r.per_leg_report())
    code = readout_exit_code(r)
    if code == READOUT_EXIT_CONFOUND_UNMEASURED:
        # THE HARD ASSERTION. Not a kill and not a hold: the window is fine and
        # an axis of the rule has no data behind it. Saying so in its own exit
        # code is the whole point -- "promote" and "half the instrument is
        # unplumbed" must not be the same observation.
        print(
            f"\nCONFOUND UNMEASURED (exit {READOUT_EXIT_CONFOUND_UNMEASURED}): "
            f"n_confound={r.n_confound} < 3 over {r.n_usable} usable rows.\n"
            f"  The verdict above ({r.verdict}) was computed WITHOUT the "
            "PID-confound axis, which\n"
            "  docs/experiment_ledger.md pre-registers as a deciding KILL rule "
            "for enabling\n"
            "  this gate (|corr(predicted confound, measured delta)| >= 0.5 -> "
            "do NOT enable).\n"
            "  gate_sample_confound_elo is NaN whenever the shards behind a row "
            "carry no\n"
            "  ShardMeta.opponent_wdl_regret_limit. Fix the producer, then "
            "re-run this."
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
