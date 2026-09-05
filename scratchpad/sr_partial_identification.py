"""Sharp identification bounds on `S_R` given the deep ruler's unlisted-move floor.

The audit assigns every unlisted legal move the worst-LISTED regret `f_i`, so its
true (capped) regret is `r_m = f_i + delta_m` with `0 <= delta_m <= D_i = C - f_i`.
Writing `a_m = t_m - q_m` over the unlisted set `U_i`,

    S_true - S_audit = SUM_i SUM_{m in U_i} a_m * delta_m

Each `delta_m` is free in `[0, D_i]` independently, so the extremes are attained by
raising ONLY the moves whose coefficient has the wanted sign:

    Delta_i^max = D_i * SUM_{m in U_i, a_m > 0} a_m      (>= 0)
    Delta_i^min = D_i * SUM_{m in U_i, a_m < 0} a_m      (<= 0)

⚑ This supersedes the "+21.53 cp worst case" recorded earlier, which was
`D_i * SUM_U a_m` — all unlisted moves at the cap. That nets the positive and
negative coefficients against each other, so it is neither bound: it understates
the maximum and ignores the negative branch entirely.

Read-only. No Stockfish, no GPU. Banked artifacts only.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/home/josh/projects/chess")

from chess_anti_engine.eval.audit import AUDIT_REGRET_CAP_CP, load_audit_set

SRC = "scratchpad/repairgate_prodteacher.jsonl"
AUDIT = "data/audit_set_v1.jsonl"


def norm(d: dict[str, float]) -> dict[str, float]:
    t = sum(d.values())
    return {} if t <= 0 else {k: v / t for k, v in d.items()}


def main() -> None:
    aset = {p.key: p for p in load_audit_set(AUDIT)}
    lo_l, hi_l, naive_l, dcap_l = [], [], [], []

    with open(SRC) as fh:
        lines = fh.readlines()
    for line in lines:
        r = json.loads(line)
        pos = aset.get(r["key"])
        if pos is None or len(pos.move_cp) < 2:
            continue
        t = norm((r["cand"].get("train") or {}).get("probs") or {})
        q = norm((r["cand"].get("sf_soft") or {}).get("probs") or {})
        if not t or not q:
            continue
        listed = set(pos.move_cp)
        floor = float(np.clip(pos.best_cp - min(pos.move_cp.values()),
                              0.0, AUDIT_REGRET_CAP_CP))
        d_i = AUDIT_REGRET_CAP_CP - floor
        pos_sum = neg_sum = 0.0
        for m in set(t) | set(q):
            if m in listed:
                continue
            a = t.get(m, 0.0) - q.get(m, 0.0)
            if a > 0:
                pos_sum += a
            else:
                neg_sum += a
        hi_l.append(d_i * pos_sum)
        lo_l.append(d_i * neg_sum)
        naive_l.append(d_i * (pos_sum + neg_sum))
        dcap_l.append(d_i)

    lo, hi, naive = np.asarray(lo_l), np.asarray(hi_l), np.asarray(naive_l)
    print(f"rows {lo.size}   mean headroom D_i = C - floor: {np.mean(dcap_l):.1f} cp")
    print("\nPER-ROW CORRECTION TO S_R (cp), mean over rows:")
    print(f"  Delta^min (only q>t unlisted moves at the cap)  {lo.mean():+9.2f}")
    print(f"  Delta^max (only t>q unlisted moves at the cap)  {hi.mean():+9.2f}")
    print(f"  [superseded 'worst case' = all at cap]          {naive.mean():+9.2f}")
    print(f"\nS_R identification interval, anchor S_audit = 20.0 cp:")
    print(f"  [{20.0 + lo.mean():+8.2f}, {20.0 + hi.mean():+8.2f}]   "
          f"width {hi.mean() - lo.mean():.2f} cp")
    print(f"\nsign of the correction is DETERMINED? "
          f"{'YES' if lo.mean() > 0 or hi.mean() < 0 else 'NO — the interval straddles 0'}")
    frac_pos = float((np.asarray(naive_l) > 0).mean())
    print(f"rows where the naive net correction is positive: {frac_pos:.3f}")


if __name__ == "__main__":
    main()
