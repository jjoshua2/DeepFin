"""How much target/teacher mass does the DEEP ruler IMPUTE rather than measure?

`eval/audit.py:265-270` gives every legal move the deep MultiPV did not list the
regret of the WORST LISTED line -- an explicit lower bound, documented as
"biased optimistic for bad distributions". With ~27 legal moves and MultiPV >= 10
that is ~17 imputed entries per position.

`S_R = E_t0[r] - E_q[r]`, so what matters is not the COUNT of imputed moves but
the MASS each distribution places on them, and whether the two distributions
place it symmetrically. `q` (the SF soft teacher) is built from SF's own top
lines and should be almost entirely listed; `t0` (the search visit target) can
put mass on moves SF never ranked. If that asymmetry is real, the imputation
biases `E_t0[r]` down more than `E_q[r]`, and `S_R` is UNDERSTATED.

Read-only, no SF, no GPU. Uses only banked artifacts.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/home/josh/projects/chess")

from chess_anti_engine.eval.audit import load_audit_set

SRC = "scratchpad/repairgate_prodteacher.jsonl"
AUDIT = "data/audit_set_v1.jsonl"


def main() -> None:
    aset = {p.key: p for p in load_audit_set(AUDIT)}
    stats: dict[str, list[float]] = {"train": [], "sf_soft": []}
    n_legal, n_listed, rows = [], [], 0

    for line in open(SRC):
        r = json.loads(line)
        pos = aset.get(r["key"])
        if pos is None:
            continue
        listed = set(pos.move_cp)
        if len(listed) < 2:
            continue
        ok = True
        cur = {}
        for cand in ("train", "sf_soft"):
            probs = (r["cand"].get(cand) or {}).get("probs")
            if not probs:
                ok = False
                break
            tot = sum(probs.values())
            if tot <= 0:
                ok = False
                break
            cur[cand] = sum(v for m, v in probs.items() if m not in listed) / tot
        if not ok:
            continue
        for k, v in cur.items():
            stats[k].append(v)
        n_legal.append(float(r["n_legal"]))
        n_listed.append(float(len(listed)))
        rows += 1

    print(f"positions {rows}   mean n_legal {np.mean(n_legal):.1f}   "
          f"mean deep-listed {np.mean(n_listed):.1f}   "
          f"=> imputed entries/pos {np.mean(n_legal) - np.mean(n_listed):.1f}")
    print(f"\n{'distribution':>12} {'mean unlisted mass':>19} {'median':>9} "
          f"{'p90':>8} {'frac>1%':>9} {'frac>5%':>9}")
    for k in ("train", "sf_soft"):
        v = np.asarray(stats[k])
        print(f"{k:>12} {v.mean():19.5f} {np.median(v):9.5f} "
              f"{np.percentile(v, 90):8.5f} {(v > 0.01).mean():9.3f} {(v > 0.05).mean():9.3f}")
    a = np.asarray(stats["train"])
    b = np.asarray(stats["sf_soft"])
    d = a - b
    boot = np.array([d[np.random.default_rng(i).integers(0, d.size, d.size)].mean()
                     for i in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nASYMMETRY  mean(train - sf_soft) unlisted mass = {d.mean():+.5f} "
          f"[{lo:+.5f}, {hi:+.5f}]")
    print("positive => t0 has MORE imputed mass than q => E_t0[r] biased down more"
          "\n           => S_R = E_t0[r] - E_q[r] is UNDERSTATED by the ruler")


if __name__ == "__main__":
    main()
