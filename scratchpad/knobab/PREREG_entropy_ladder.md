# PREREG — is the two-knob effect ENTROPY-CONDITIONAL?

Written 2026-08-16 BEFORE the three new checkpoints were scored. Ledger `307f61455`
recorded the hypothesis from n=2; this pins the prediction numerically so the read
cannot be reinterpreted after the fact.

## Hypothesis
`gumbel_target_max_visit_cap: 5` + `gumbel_target_untempered_prior: true` SHARPEN the
stored target. Sharpening helps a target that is too SOFT and hurts one already too
SHARP. So their benefit should be a decreasing function of the checkpoint's baseline
target entropy (measured with the knobs OFF, `policy_temp` 1.5).

## The two points already banked
| ckpt | baseline entropy (knobs OFF) | accuracy delta (ON − OFF) |
|---|---|---|
| B `bt4heads_iter100` (production) | 0.692 | **−0.875 pp** [−1.63, −0.10] |
| A `ANCHOR_20260727_preC17` | 0.976 | **+0.825 pp** [+0.05, +1.60] |

## Arms added (same protocol: n=4000, seed 42, audit_set_v1, paired, both arms at policy_temp 1.5)
`ck_2026-07-29_iter218.pt` · `ck_2026-08-09_iter514.pt` (banked peak) · `ck_resume_iter672.pt`

## PRE-COMMITTED PREDICTION
1. **MONOTONE:** rank the 5 checkpoints by baseline entropy; the accuracy delta is
   non-increasing in entropy... **no — non-DEcreasing**: higher entropy ⇒ more positive
   delta. Spearman rho(entropy, accuracy delta) over n=5 must be **> 0**.
2. **SIGN CROSSING:** every checkpoint with baseline entropy < 0.75 has accuracy
   delta < 0; every one with entropy > 0.92 has delta > 0.
3. **CONFIRMED** requires BOTH. **KILLED** if rho <= 0, or if any checkpoint violates (2)
   with a CI excluding zero on the wrong side.
4. **INCONCLUSIVE** if rho > 0 but the three new deltas all have CIs spanning 0 — the
   effect would then be real only at the extremes and too small to act on.

## What this CANNOT establish
Entropy is CORRELATED with training time here, so "entropy" and "how late in the run"
are confounded across these five; a positive result narrows the mechanism, it does not
isolate entropy as the cause. It also says nothing about Elo — these knobs are
target-only and an arena is blind to them by construction.
