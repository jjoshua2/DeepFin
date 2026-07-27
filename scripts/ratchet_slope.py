#!/usr/bin/env python3
"""Fit the strength-ratchet slope against CUMULATIVE OPTIMIZER STEPS, not dates.

WHY THIS EXISTS. The deciding yardstick for the 2026-07-27 bundle is "the ratchet
SLOPE across >=4 daily ``vs_boot512`` rows, CI excluding 0". Collected daily --
but **steps/day is not constant**. It DOUBLED at iter 124 (224 -> ~500 optimizer
steps/hour, measured). The two rows banked before that were earned in the old
regime; everything after is earned in the new one.

A slope fitted against calendar dates therefore mixes two throughput regimes and
would credit a throughput change as learning -- or, if flat, hide a real per-step
gain behind a halved cost-per-Elo. The independent variable has to be the thing
that actually causes learning: **optimizer steps**.

This is the same defect that produced several wrong readings during the
2026-07-27 audit (per-iteration grad-norm drift that vanished once normalised per
step; a `train_time_s` bar set while tripling the step count; a clip RATE read as
a clip EFFECT). Here it is fixed in the instrument instead of in a comment.

CUMULATIVE STEPS ACROSS THE CSV ROTATION. `progress.csv` is rotated (PR #262), so
the live file starts at iter 81 and `progress.<epoch>.csv` holds 1-80. Both are
read and merged, later files winning on duplicate iterations, so the cumulative
sum spans the whole trial rather than silently restarting at the rotation.

WEIGHTING. Each ratchet row carries its own 95% CI, and they differ a lot (the
first row is +-49 Elo on 200 games, later ones +-33 on 314). An unweighted fit
would treat them as equally informative. This does weighted least squares with
w = 1/se^2, se = (ci_hi - ci_lo) / (2 * 1.96).

Read-only. Touches nothing but CSVs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
Z95 = 1.959963985


def load_cumulative_steps(run_dir: Path) -> dict[int, float]:
    """iteration -> cumulative trainer_steps_done, spanning CSV rotations."""
    trial_dirs = sorted(glob.glob(str(run_dir / "tune" / "train_trial_*")))
    if not trial_dirs:
        raise SystemExit(f"no trial dirs under {run_dir / 'tune'}")
    per_iter: dict[int, float] = {}
    for tdir in trial_dirs:
        # Rotated files first, live `progress.csv` last, so the live file wins on
        # any duplicated iteration (a resume can re-emit rows).
        files = sorted(glob.glob(os.path.join(tdir, "progress.*.csv")))
        files.append(os.path.join(tdir, "progress.csv"))
        for f in files:
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    try:
                        it = int(float(row["training_iteration"]))
                        st = float(row["trainer_steps_done"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    per_iter[it] = st
    if not per_iter:
        raise SystemExit("no (training_iteration, trainer_steps_done) rows found")
    cum: dict[int, float] = {}
    total = 0.0
    for it in sorted(per_iter):
        total += per_iter[it]
        cum[it] = total
    return cum


def weighted_slope(x: np.ndarray, y: np.ndarray, se: np.ndarray) -> tuple[float, float, float, float]:
    """WLS slope of y on x. Returns (slope, se_slope, lo95, hi95)."""
    w = 1.0 / np.maximum(se, 1e-9) ** 2
    sw = w.sum()
    xbar = float((w * x).sum() / sw)
    ybar = float((w * y).sum() / sw)
    sxx = float((w * (x - xbar) ** 2).sum())
    if sxx <= 0:
        raise SystemExit("all ratchet rows share the same x — cannot fit a slope")
    slope = float((w * (x - xbar) * (y - ybar)).sum() / sxx)
    resid = y - (ybar + slope * (x - xbar))
    n = len(x)
    if n > 2:
        # Scale by the weighted residual spread so an over-optimistic set of CIs
        # cannot manufacture a tight slope CI on its own.
        chi2 = float((w * resid**2).sum()) / (n - 2)
        var = chi2 / sxx
    else:
        var = 1.0 / sxx
    se_slope = float(np.sqrt(max(var, 0.0)))
    return slope, se_slope, slope - Z95 * se_slope, slope + Z95 * se_slope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratchet-csv", type=Path, default=ROOT / "data/ratchet/ratchet.csv")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs/pbt2_small")
    ap.add_argument("--series", default="vs_boot512",
                    help="frozen-anchor series; vs_prev is a moving reference and is NOT the yardstick")
    ap.add_argument("--min-rows", type=int, default=4,
                    help="pre-committed minimum before the slope is a verdict")
    args = ap.parse_args()

    cum = load_cumulative_steps(args.run_dir)
    with open(args.ratchet_csv) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("series") == args.series]
    if not rows:
        raise SystemExit(f"no rows with series={args.series} in {args.ratchet_csv}")

    xs, ys, ses, meta = [], [], [], []
    missing = []
    for r in rows:
        it = int(float(r["iter"]))
        if it not in cum:
            missing.append(it)
            continue
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        se = (hi - lo) / (2 * Z95)
        xs.append(cum[it])
        ys.append(float(r["elo"]))
        ses.append(se)
        meta.append((r["date"], it, float(r["elo"]), lo, hi, int(float(r["games"])), cum[it]))

    print(f"series: {args.series}   rows usable: {len(xs)}   (need >= {args.min_rows} for a verdict)")
    if missing:
        print(f"  ! {len(missing)} row(s) dropped — iteration not found in any progress CSV: {missing}")
    print()
    print(f"  {'date':12s} {'iter':>6s} {'cum_steps':>11s} {'elo':>8s} {'95% CI':>20s} {'games':>6s}")
    for d, it, e, lo, hi, g, cs in meta:
        print(f"  {d:12s} {it:6d} {cs:11,.0f} {e:8.1f}  [{lo:+7.1f},{hi:+7.1f}] {g:6d}")

    if len(xs) < 2:
        print("\nNot enough rows to fit. No verdict.")
        return

    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    se = np.asarray(ses, float)
    slope, _, lo, hi = weighted_slope(x, y, se)
    per1k = 1000.0
    print()
    print(f"  WLS slope: {slope*per1k:+.4f} Elo per 1000 optimizer steps"
          f"   95% CI [{lo*per1k:+.4f}, {hi*per1k:+.4f}]")
    span = x.max() - x.min()
    print(f"  fitted over {span:,.0f} steps ({x.min():,.0f} -> {x.max():,.0f})")
    print(f"  implied Elo across that span: {slope*span:+.1f}")

    print()
    if len(xs) < args.min_rows:
        print(f"  VERDICT: NONE — {len(xs)}/{args.min_rows} rows. Interim only; explicitly NOT a verdict.")
    elif slope > 0 and lo > 0:
        print("  VERDICT: SUCCESS — slope > 0 with CI excluding 0.")
    elif slope <= 0 and hi < 0:
        print("  VERDICT: KILL/PIVOT — slope < 0 with CI excluding 0. Gradient volume is not the constraint.")
    else:
        print("  VERDICT: NULL — CI includes 0. By the pre-committed rule this is KILL/PIVOT")
        print("           (SUCCESS required the CI to exclude 0), not 'needs more time'.")


if __name__ == "__main__":
    main()
