# PREREG — break the entropy/vintage collinearity on ONE checkpoint

Written 2026-08-16 BEFORE the sweep was run. Follows the INCONCLUSIVE entropy-ladder
readout (ledger `957b8ade1`), which said in advance that entropy and training vintage
are perfectly collinear across the five checkpoints and named the discriminating design.

## The design
`policy_temp` changes the PRIOR temperature and therefore the stored target's entropy,
**at fixed weights**. Measured already on the production checkpoint: pt 1.0 -> entropy
0.537, pt 1.5 -> 0.692. So sweeping pt on ONE checkpoint varies entropy with vintage,
lineage, weights and data all held exactly constant. That is the confound removed, not
merely acknowledged.

**Arms:** checkpoint `bt4heads_iter100_20260815` (production weights) only.
`policy_temp` in {1.0, 2.0, 3.0} × knobs {ON = (5, True), OFF = (0, False)}. Six runs.
n=4000, seed 42, `audit_set_v1`, paired on FEN, 20k bootstrap, row (d) = stored target.
(pt 1.5 both ways is already banked: entropy 0.692, accuracy delta −0.875 pp.)

## PRE-COMMITTED PREDICTION
The hypothesis is: these knobs SHARPEN, sharpening helps a too-soft target and hurts an
already-sharp one. At fixed weights it predicts

1. **MONOTONE IN ENTROPY:** across the four pt levels {1.0, 1.5, 2.0, 3.0}, the accuracy
   delta (ON − OFF) is non-decreasing in the OFF-arm baseline entropy. Spearman
   rho(entropy, accuracy delta) over the four points must be **> 0**.
2. **SIGN:** at pt 1.0 (entropy ~0.54, the sharpest) the delta is <= the pt 1.5 delta of
   −0.875 pp; at the softest arm reached (whatever entropy pt 3.0 gives, expected > 0.95)
   the delta is **> −0.875 pp**, i.e. strictly less negative.
3. **CONFIRMED** = both hold. **KILLED** = rho <= 0, or the softest arm is MORE negative
   than pt 1.5. **INCONCLUSIVE** = rho > 0 but every pairwise difference's CI spans 0.

## What a CONFIRMED result would and would not mean
WOULD: the knob effect tracks target ENTROPY specifically, not checkpoint vintage — the
five-checkpoint ladder is then explained by entropy rather than by "later checkpoint".
WOULD NOT: say anything about Elo. These knobs are target-only and an arena is blind to
them by construction. Nor does it license a production change on its own; `policy_temp`
is itself a live production key (1.5) and moving it is a separate, data-affecting decision.

⚑ Note the sweep deliberately includes pt values production does NOT run. That is the
point — it is a mechanism probe on a frozen set, not a config recommendation.
