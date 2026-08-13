"""Task #198 readout: does value_regret predict arena Elo across the lineage?

PRE-COMMITTED before the three missing reads exist (2026-08-13):

  POPULATION: the six checkpoints with a DIRECT paired arena vs boot512 at 200
  pairs, sims 32, --search-shape training — 477, 514, 672, 735, 768, 862.
  iter400 (play shape) and iter346 (shape unverifiable) are excluded.

  STATISTIC: Spearman rho between value_regret OVERALL (cp, lower better) and
  arena Elo vs boot512 (higher better). Spearman, not Pearson: n=6 and we care
  whether the ruler ORDERS checkpoints, not whether it is linear in Elo.

  VERDICT RULE, fixed now:
    rho <= -0.80  -> USABLE SCREEN. value_regret may screen candidates before
                     spending an arena. It still may not PROMOTE anything —
                     a screen decides what to arena, never what wins.
    -0.80 < rho <= -0.40 -> WEAK. Report as such; no procedural change.
    rho > -0.40 (incl. any positive) -> THE VALUE RULER DOES NOT TRACK PLAY
                     STRENGTH over this lineage. That is the more consequential
                     outcome: every "value improved" verdict in the ledger
                     becomes a statement about deep-SF agreement only, and must
                     stop being read as progress toward Elo.

  POWER, computed before the fact: with n=6, |rho| >= 0.83 is needed for p<0.05
  two-sided. So this design can only certify a STRONG relationship. A null here
  is "not demonstrated at n=6", NOT "no relationship" — the honest next step in
  that case is more banked checkpoints, not a re-reading of these six.

  CONFOUND, stated: value_regret grades against deep SF; we are training an
  ANTI-engine whose target is exploiting SF, so a weak correlation has an
  innocent explanation. It is still decision-relevant: a ruler that cannot
  order our own lineage cannot screen our own experiments.
"""
from __future__ import annotations

import glob
import os
import re

ARENA_ELO = {  # direct paired vs boot512, 200 pairs, sims 32, training shape
    477: -54.6, 514: 115.2, 672: 87.8, 735: 86.9, 768: 67.7, 862: 11.3,
}
BANKED_VR = {514: 60.5, 672: 66.0, 735: 64.1}  # scratchpad/valreg_ladder_20260811


def _overall(path: str) -> float | None:
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return None
    hits = re.findall(r"OVERALL[^\n]*?([0-9]+\.[0-9]+)", txt)
    return float(hits[-1]) if hits else None


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    vr = dict(BANKED_VR)
    for p in glob.glob(os.path.dirname(os.path.abspath(__file__)) + "/iter*.log"):
        m = re.search(r"iter(\d+)", os.path.basename(p))
        val = _overall(p)
        if m and val is not None:
            vr[int(m.group(1))] = val

    common = sorted(set(ARENA_ELO) & set(vr))
    print(f"{'iter':>5} {'value_regret cp':>16} {'arena Elo':>10}")
    for k in common:
        print(f"{k:5d} {vr[k]:16.1f} {ARENA_ELO[k]:+10.1f}")
    if len(common) < 4:
        print(f"\nn={len(common)} — run run_missing_reads.sh in a pause window first.")
        return
    rho = spearman([vr[k] for k in common], [ARENA_ELO[k] for k in common])
    print(f"\nSpearman rho(value_regret, Elo) = {rho:+.3f}  (n={len(common)})")
    if rho <= -0.80:
        print("VERDICT: USABLE SCREEN (may decide what to arena; never what wins).")
    elif rho <= -0.40:
        print("VERDICT: WEAK. No procedural change.")
    else:
        print("VERDICT: THE VALUE RULER DOES NOT TRACK PLAY STRENGTH over this lineage.")
        print("  => 'value improved' verdicts are deep-SF agreement statements only.")
    print("n=6 certifies only |rho|>=0.83 at p<0.05; a null is 'not demonstrated', not 'no effect'.")


if __name__ == "__main__":
    main()
