"""Fit and compare the masked vs unmasked probe_gap TRENDS.

Reads probe_scores.jsonl (checkpoint-level means) and the per-row dumps
(for the row-bootstrap CI on the slope RATIO, which is what the pre-committed
kill rule is stated in).

Two lineages are fitted SEPARATELY and never pooled: trial 379f6 (the trunk,
steps 639 -> 76286) and trial 5ce02, which branched from the 379f6 iter-672
salvage at step 57999 with a different search config. A single regression over
both would fit a step that is a lineage change, not a trend.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
ARMS = ("unmasked", "mech", "aggr")
# step -> lineage.  5ce02 branched at 57999; its own steps are 65441+.
BRANCH_PATHS = ("5ce02", "anchors_20260811")


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """slope, stderr(slope), r2, df."""
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    a = ym - b * xm
    resid = y - (a + b * x)
    df = n - 2
    s2 = float((resid ** 2).sum() / df)
    se = float(np.sqrt(s2 / sxx))
    sst = float(((y - ym) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else float("nan")
    return b, se, r2, df


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def main() -> None:
    recs = [json.loads(l) for l in open(os.path.join(D, "probe_scores.jsonl"))]
    recs.sort(key=lambda r: r["step"])
    dumps = {}
    for f in glob.glob(os.path.join(D, "dumps", "probe_rows_*.npz")):
        step = int(os.path.basename(f).split("_")[2])
        dumps.setdefault(step, f)

    print("## Checkpoint table (gap = era - inwindow, x1e3)\n")
    hdr = f"{'lineage':8s} {'step':>6s} {'unmasked':>9s} {'mech':>9s} {'aggr':>9s}  {'p@max era':>9s} {'p@max inw':>9s}  path"
    print(hdr)
    for r in recs:
        lin = "5ce02" if any(b in r["path"] for b in BRANCH_PATHS) else "379f6"
        m = r["means"]
        print(f"{lin:8s} {r['step']:6d} {r['gap_unmasked']*1e3:9.4f} {r['gap_mech']*1e3:9.4f} "
              f"{r['gap_aggr']*1e3:9.4f}  {m['era']['p_at_max']:9.5f} {m['inwindow']['p_at_max']:9.5f}  {r['path']}")

    for lin in ("379f6", "5ce02"):
        sel = [r for r in recs
               if (any(b in r["path"] for b in BRANCH_PATHS)) == (lin == "5ce02")]
        if len(sel) < 3:
            print(f"\n## {lin}: only {len(sel)} points, no fit")
            continue
        x = np.array([r["step"] for r in sel], dtype=float)
        print(f"\n## Trend fit — lineage {lin}, n={len(sel)}, steps {int(x.min())}..{int(x.max())}")
        print(f"{'arm':10s} {'slope/10k steps (x1e3)':>24s} {'95% CI':>26s} {'r2':>7s}")
        slopes = {}
        for a in ARMS:
            y = np.array([r[f"gap_{a}"] for r in sel], dtype=float)
            b, se, r2, df = ols(x, y)
            t = T95.get(df, 1.96)
            slopes[a] = b
            print(f"{a:10s} {b*1e4*1e3:24.4f} {'[%.4f, %.4f]' % ((b-t*se)*1e7, (b+t*se)*1e7):>26s} {r2:7.3f}")
        for a in ("mech", "aggr"):
            ratio = slopes["unmasked"] / slopes[a] if slopes[a] != 0 else float("inf")
            same = (slopes["unmasked"] > 0) == (slopes[a] > 0)
            print(f"  ratio unmasked/{a} = {ratio:+.3f}   same sign: {same}   "
                  f"|ratio|>2 or <0.5: {(abs(ratio) > 2 or abs(ratio) < 0.5) or not same}")

        # row bootstrap on the ratio (paired: same resample across checkpoints)
        steps = [r["step"] for r in sel]
        if all(s in dumps for s in steps):
            data = {s: np.load(dumps[s]) for s in steps}
            n_era = int(data[steps[0]]["era_keep"].sum())
            n_inw = int(data[steps[0]]["inwindow_keep"].sum())
            rng = np.random.default_rng(0)
            out = {a: [] for a in ARMS}
            ratios_m, ratios_a = [], []
            for _ in range(2000):
                ie = rng.integers(0, n_era, n_era)
                ii = rng.integers(0, n_inw, n_inw)
                sl = {}
                for a in ARMS:
                    y = []
                    for s in steps:
                        z = data[s]
                        ke, ki = z["era_keep"], z["inwindow_keep"]
                        e = z[f"era_{a}"][ke][ie].mean()
                        w = z[f"inwindow_{a}"][ki][ii].mean()
                        y.append(e - w)
                    b, _, _, _ = ols(x, np.array(y))
                    sl[a] = b
                    out[a].append(b)
                ratios_m.append(sl["unmasked"] / sl["mech"] if sl["mech"] else np.nan)
                ratios_a.append(sl["unmasked"] / sl["aggr"] if sl["aggr"] else np.nan)
            print(f"  row-bootstrap (2000, paired) slope/10k steps x1e3:")
            for a in ARMS:
                v = np.array(out[a]) * 1e7
                print(f"    {a:9s} {v.mean():8.4f}  [{np.percentile(v,2.5):8.4f}, {np.percentile(v,97.5):8.4f}]  "
                      f"P(slope>0)={float((v>0).mean()):.3f}")
            for nm, rr in (("mech", ratios_m), ("aggr", ratios_a)):
                v = np.array(rr)
                v = v[np.isfinite(v)]
                print(f"    ratio unmasked/{nm}: {np.median(v):+.3f} "
                      f"[{np.percentile(v,2.5):+.3f}, {np.percentile(v,97.5):+.3f}]")

    # the one banked probe_gap number: +83% over 379f6 514 -> 735
    by_step = {r["step"]: r for r in recs}
    if 44613 in by_step and 63897 in by_step:
        print("\n## Re-derivation of the banked claim `probe_gap_policy_eregret +83% over 514->735`")
        for a in ARMS:
            lo = by_step[44613][f"gap_{a}"]
            hi = by_step[63897][f"gap_{a}"]
            print(f"  {a:9s} 514: {lo*1e3:8.4f}e-3   735: {hi*1e3:8.4f}e-3   change: {100*(hi/lo-1):+7.1f}%")


if __name__ == "__main__":
    main()
