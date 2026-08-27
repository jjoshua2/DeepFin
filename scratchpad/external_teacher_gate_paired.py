"""Re-run the external-teacher gate with `ours` = iter595 instead of the stale iter190.

Resolves the confound the 2026-08-26 ledger entry states about itself. Reports iter190
alongside as a REPRODUCTION check: if this script cannot reproduce the banked 52.5%, the
iter595 number it produces is not comparable to it either.
"""
from __future__ import annotations

import json
import math
import sys


def load(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    return rows


AUDIT = load("data/audit_set_v1.jsonl")
BT4 = load("data/lc0/bt4_audit_cache_topk256_20260817.jsonl")
ARMS = {
    "iter190": load("data/lc0/ours_iter190_audit_cache_topk256_20260817.jsonl"),
    "iter595": load(sys.argv[1]),
}

keys = sorted(set(AUDIT) & set(BT4) & set(ARMS["iter190"]) & set(ARMS["iter595"]))
print(f"aligned on all four sources: {len(keys)} / audit {len(AUDIT)}")

# SF's exact best, and the full zero-regret set (ties are common: 645 decisive rows above
# were mate scores, where several moves share the top score).
sf_best = {k: AUDIT[k]["bestmove"] for k in keys}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.959963985
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


print()
print("| arm | ours top-1 disagree w/ SF | ours err>50cp (n) | BT4 = SF best | BT4 zero-regret |")
print("|---|---|---|---|---|")
sets: dict[str, set[str]] = {}
for name, ours in ARMS.items():
    disagree = sum(1 for k in keys if ours[k]["best_move"] != sf_best[k])
    bad = [k for k in keys if float(ours[k]["top1_regret"]) > 50.0]
    sets[name] = set(bad)
    n = len(bad)
    exact = sum(1 for k in bad if BT4[k]["best_move"] == sf_best[k])
    zero = sum(1 for k in bad if float(BT4[k]["top1_regret"]) == 0.0)
    lo, hi = wilson(exact, n)
    zlo, zhi = wilson(zero, n)
    print(
        f"| {name} | {disagree / len(keys):.1%} | {n} | "
        f"**{exact / n:.1%}** [{lo:.1%},{hi:.1%}] | {zero / n:.1%} [{zlo:.1%},{zhi:.1%}] |"
    )

a, b = sets["iter190"], sets["iter595"]
print()
print(f"error-set churn: iter190 {len(a)}, iter595 {len(b)}, "
      f"intersection {len(a & b)}, Jaccard {len(a & b) / len(a | b):.3f}")
print(f"  fixed by iter595 (in 190, not 595): {len(a - b)}")
print(f"  broken by iter595 (in 595, not 190): {len(b - a)}")

# The paired question the marginal rate cannot answer: on the rows iter595 STILL gets
# wrong, is BT4 still right?  A gate that only holds on rows we already fixed is useless.
still = sorted(a & b)
ex = sum(1 for k in still if BT4[k]["best_move"] == sf_best[k])
lo, hi = wilson(ex, len(still))
print(f"  on rows BOTH get wrong (n={len(still)}): BT4 = SF best {ex / len(still):.1%} [{lo:.1%},{hi:.1%}]")
only595 = sorted(b - a)
if only595:
    ex2 = sum(1 for k in only595 if BT4[k]["best_move"] == sf_best[k])
    print(f"  on rows only iter595 gets wrong (n={len(only595)}): BT4 = SF best {ex2 / len(only595):.1%}")

# Overall policy quality, both arms, on the identical aligned set.
print()
for name, ours in ARMS.items():
    tot = sum(float(ours[k]["top1_regret"]) for k in keys) / len(keys)
    exp = sum(float(ours[k]["exp_regret"]) for k in keys) / len(keys)
    print(f"{name}: mean top1_regret {tot:.2f} cp   mean exp_regret {exp:.2f} cp")

# ---- PAIRED tests. The marginal rates above are two independent-looking numbers over the
# SAME 4000 positions; only the paired form can say whether 405 iterations did anything.
import statistics

o190, o595 = ARMS["iter190"], ARMS["iter595"]
b = len(sets["iter190"] - sets["iter595"])
c = len(sets["iter595"] - sets["iter190"])
chi2 = (abs(b - c) - 1) ** 2 / (b + c)          # McNemar, continuity-corrected
print()
print(f"McNemar on the >50cp blunder indicator: b={b} c={c} chi2={chi2:.4f}  (crit 3.841 @ .05)")

d = [float(o595[k]["top1_regret"]) - float(o190[k]["top1_regret"]) for k in keys]
m, sd = statistics.fmean(d), statistics.stdev(d)
se = sd / math.sqrt(len(d))
print(f"paired Δ top1_regret (595−190): {m:+.3f} cp  95% CI [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]  "
      f"sd {sd:.1f}  n {len(d)}")
nz = [x for x in d if x != 0.0]
print(f"  positions where the top move CHANGED AT ALL: {len(nz)} / {len(d)} = {len(nz) / len(d):.1%}")

de = [float(o595[k]["exp_regret"]) - float(o190[k]["exp_regret"]) for k in keys]
me, sde = statistics.fmean(de), statistics.stdev(de)
see = sde / math.sqrt(len(de))
print(f"paired Δ exp_regret  (595−190): {me:+.3f} cp  95% CI [{me - 1.96 * see:+.3f}, {me + 1.96 * see:+.3f}]")
