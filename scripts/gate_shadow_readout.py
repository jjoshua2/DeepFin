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
    nothing at all. ``gate_sample_confound_elo`` is NaN on 109 of 109 rows
    written so far, because the server's upload compactor rebuilt ``ShardMeta``
    without ``opponent_wdl_regret_limit``. That producer is fixed (#323), but
    the fix is DEPLOY-GATED ON A SERVER RESTART -- the compactor runs in the
    server process, so a running server keeps the old module -- and the axis
    stays unmeasured until the server restarts and a full window of new shards
    lands. A window whose confound axis has fewer than 3 measurements CANNOT
    exit 0 or 1; it gets exit 5.

⚑ THE REFERENCE MUST BE RE-DERIVED BEFORE ``gate_mode`` LEAVES ``off``
    Every leg is stated relative to ``OfflineReference``, measured over live
    iterations 163-219 (2026-06). The ``prev`` arm is the shards still tagged
    with the previous sha because the worker had not picked up the new
    manifest yet, so it is a model-refresh LAG -- roughly constant in SECONDS,
    not as a share -- and the axis is reported in seconds for that reason.

    On the current trial ``progress.csv`` implies ~166 s over the last 40 rows
    against the reference's 117.5 s. A moved lag and a mis-attributed split are
    the same number in ``progress.csv`` and cannot be separated from it, so
    that disagreement is exactly what has to be resolved before any constant
    moves -- and it can only be resolved from the SHARD side.

    ``--rederive-reference SHARD_ROOT`` is that independent side, read from the
    shard ``.zattrs`` the reference was originally built from and NOT from the
    gate's own columns (a control conditioned on its own outcome cannot fail).
    ⚑ IT CURRENTLY REFUSES, AND THE REFUSAL IS THE FINDING. Shards carry the
    SERVER's flush stamp while the loop attributes at INGEST, so binning them
    against iteration boundaries has a free phase, and on the live trial a
    quarter-iteration shift moves the reconstructed ``prev_share`` further than
    the attribution leg's whole tolerance. The command therefore sweeps the
    phase, prints every field as a BAND, and declines to emit constants (exit
    7) rather than hand back a number a bin edge chose. An earlier ledger entry
    read one phase of this reconstruction as a measurement and stated a
    conclusion from it; that sentence is retracted in the ledger.

    ⚑ THE REFUSAL IS STRUCTURAL: RE-RUNNING THIS LATER CANNOT FIX IT. The free
    phase belongs to the input, not to the window -- any fleet whose refresh lag
    is a real fraction of an iteration reads as unstable here, so waiting for a
    cleaner window buys nothing. The resolution is to attribute shards at the
    loop's own INGEST event and re-derive from THAT.

    When such a reconstruction exists it prints constants and applies nothing:
    record them in ``docs/experiment_ledger.md`` and paste them into
    ``OfflineReference`` in the same change, at restart prep -- never mid-window,
    and never after reading a verdict you did not like.

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
    identity not evaluated (6) the rows carry no ``pid_curriculum_w/d/l``, so
                            the exact-integer pooled identity -- the one leg
                            with no statistics in it -- was skipped. Same
                            OR-over-axes rule as 5, one axis over. Reachable on
                            any csv rotated from an earlier report schema.
    rederive unresolved (7) ``--rederive-reference`` only, and no verdict on
                            any window: the reconstruction's ``prev_share``
                            moved further across the bin-edge phase sweep than
                            the attribution leg's own tolerance, so the tool
                            emitted a refusal instead of constants. A caller
                            that captures the output must not paste it.

    PYTHONPATH=. python3 scripts/gate_shadow_readout.py runs/pbt2_small/<trial>/progress.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path

from chess_anti_engine.tune.promotion_gate import (
    CONFOUND_SLOPE_MAX,
    CONFOUND_Z,
    OFFLINE,
    READOUT_EXIT_CONFOUND_UNMEASURED,
    READOUT_EXIT_IDENTITY_UNEVALUATED,
    _REDERIVE_MIN_USABLE_FRACTION as _MIN_USABLE_FRACTION,
    read_iteration_bins,
    read_shard_arms,
    readout_exit_code,
    rederive_reference_with_phase_sweep,
    shadow_readout_from_csv,
)

_NOT_RUN, _NO_FILE = 3, 4
# --rederive-reference only. Deliberately outside the verdict range 0..2 and
# the axis codes 5..6: it is not a verdict on a training window at all, and a
# caller that maps "non-zero" onto "the gate says kill" would be wrong twice.
_REDERIVE_UNRESOLVED = 7


def _positive_seconds(raw: str) -> float:
    """A refresh lag argparse will accept: strictly positive, finite.

    The module carries an explicit invariant that no input may make the
    deciding command RAISE instead of deciding, and ``--refresh-lag-seconds 0``
    used to divide by zero inside the leg's own message (review N1). Refused
    here as well as guarded there, because a zero reference lag is not a lag.
    """
    val = float(raw)
    if not math.isfinite(val) or val <= 0.0:
        raise argparse.ArgumentTypeError(
            f"refresh lag must be a positive number of seconds, got {raw!r}"
        )
    return val


_SAMPLE_COLUMNS = ("gate_sample_games_cur", "gate_sample_games_prev",
                   "gate_sample_delta_elo")


_FIELD_LABELS = (
    ("mean_games_cur", "mean_games_cur", "{:.1f}", "mean_games_cur"),
    ("mean_games_prev", "mean_games_prev", "{:.1f}", "mean_games_prev"),
    ("prev_share", "prev_share", "{:.4f}", "prev_share"),
    ("mean_iter_seconds", "mean_iter_seconds", "{:.1f}", "mean_iter_seconds"),
    ("refresh_lag_seconds", "refresh_lag", "{:.1f}", "refresh_lag_seconds"),
    ("mean_delta_elo", "mean_delta_elo", "{:.2f}", "mean_delta_elo"),
    ("sd_delta_elo", "sd_delta_elo", "{:.2f}", "sd_delta_elo"),
)


def _rederive(progress_csv: Path, shard_root: Path) -> int:
    """Print the reference constants this shard window implies. Applies nothing."""
    if not shard_root.is_dir():
        print(f"no such shard directory: {shard_root}")
        return _NO_FILE
    bins = read_iteration_bins(progress_csv)
    shards, subtrees = read_shard_arms(shard_root)
    r = rederive_reference_with_phase_sweep(bins, shards)
    read_note = ", ".join(f"{k}={v}" for k, v in sorted(subtrees.items())) or "nothing"
    lines = [
        f"re-derived from {r.n_shards} shards under {shard_root} binned against "
        f"{r.n_iterations} iterations of {progress_csv}",
        f"  subtrees read: {read_note}",
        f"  usable bins (both arms non-empty): {r.n_usable} at zero shift",
        "",
        # ⚑ A BAND ENDPOINT IS ONLY AS GOOD AS THE n IT CAME FROM (review B2).
        # The live sweep's widest point kept 13 of 68 bins; printing the counts
        # is what lets a reader see that before reading the band.
        "  usable bins per shift (a shift below "
        f"{int(100 * _MIN_USABLE_FRACTION)}% of the best is DEGENERATE and "
        "does not set the band):",
        "    " + "  ".join(
            f"{shift:+.2f}: {n}{' DEGENERATE' if degen else ''}"
            for shift, n, degen in r.shift_usable
        ),
        "",
        "  field                point estimate   shipped     phase band "
        f"({r.n_band_shifts} non-degenerate of the bin-edge shifts "
        f"{list(r.shifts)} of an iteration)",
    ]
    for attr, label, fmt, ref_attr in _FIELD_LABELS:
        lo, hi = r.band(attr)
        ref = getattr(OFFLINE, ref_attr, float("nan"))
        flag = ""
        if attr == "prev_share" and not r.prev_share_is_phase_stable:
            flag = "  <-- UNRESOLVED, spans the leg's own tolerance"
        lines.append(
            f"  {label:<18} {fmt.format(getattr(r, attr)):>14}   "
            f"{fmt.format(float(ref)):>9}   "
            f"[{fmt.format(lo)}, {fmt.format(hi)}]{flag}"
        )
    lines += [
        "",
        "⚑ THE BIN EDGES ARE A CHOICE. `generated_at_unix` on a _compacted shard",
        "  is the SERVER's flush stamp; the loop attributes at INGEST, and a shard",
        "  flushed at the end of iteration N is ingested in N+1 where its sha is",
        "  `prev`. Nothing in this reconstruction can pin the alignment, so every",
        "  number above is reported as a band, not a measurement -- INCLUDING",
        "  `mean_iter_seconds`, whose seconds come from progress.csv but whose",
        "  mean is taken over the USABLE bins, and usability is what the phase",
        "  moves. The cadence reading that does not depend on this alignment is",
        "  the readout's own `cadence` leg, which never bins a shard.",
        "",
        "⚑ RE-RUNNING THIS COMMAND CANNOT RESOLVE THE REFUSAL. The free phase is",
        "  a property of the input, not of this window: any fleet whose refresh",
        "  lag is a real fraction of an iteration reads as unstable here. The",
        "  resolution is to attribute shards at the loop's own INGEST event and",
        "  re-derive from THAT -- not to run this again later.",
        "",
        "OfflineReference body these imply:",
        r.as_offline_reference_source(),
        "NOTHING WAS APPLIED. These are read from shard .zattrs -- an independent",
        "reconstruction that never touches the gate's own split, which is what",
        "makes the attribution axis a control at all. Record them in",
        "docs/experiment_ledger.md and edit OfflineReference in the same change,",
        "at restart prep.",
    ]
    print("\n".join(lines))
    # ⚑ A REFUSAL THAT EXITS 0 IS A SILENT ACCEPT. This mode judges no window,
    # so 0 would be the natural "it ran" -- but the one thing a caller does
    # with this subcommand is capture its output to paste into
    # ``OfflineReference``, and a wrapper that checks the status would be told
    # the constants are good precisely when the tool declined to emit any.
    # Same rule as the readout's own axes: assert the AXIS STATE.
    return 0 if r.prev_share_is_phase_stable else _REDERIVE_UNRESOLVED


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
        "--refresh-lag-seconds", type=_positive_seconds, default=None,
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
    if code == READOUT_EXIT_IDENTITY_UNEVALUATED:
        print(
            f"\nPOOLED IDENTITY NOT EVALUATED (exit "
            f"{READOUT_EXIT_IDENTITY_UNEVALUATED}): the rows carry no "
            "pid_curriculum_w/d/l,\n"
            "  so `gate_sample_games_cur + prev == pid_curriculum_w+d+l` -- the "
            "one leg with\n"
            "  no statistics in it, and the only one that catches shard loss or "
            "an\n"
            "  unrecognised sha outright -- was never checked. That is the same "
            "OR-over-axes\n"
            "  trap as the confound axis, so it does not share an exit code with "
            "promote."
        )
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
