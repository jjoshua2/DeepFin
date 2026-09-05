# PREREG — era-probe masked-vs-unmasked recheck (2026-08-15)

Registered BEFORE any number was produced by `probe_masked.py`.

## Kill rule (pre-committed by the parent task, restated verbatim)

> if the masked and unmasked `probe_gap` trends have **opposite sign**, or differ by
> **more than 2x in slope**, then every forgetting-hinge verdict read off `probe_gap`
> needs re-deriving. Otherwise the ruler is salvaged as-is.

## Definitions

- unmasked (the LIVE ruler, `era_probe.py:592`):
  `E = sum_{m legal} p(m) r(m)`, `p = softmax(logits | legal_mask)`
- masked (this experiment): `E_c = sum_{m in C} p(m) r(m) / sum_{m in C} p(m)`
  where `C` = legal moves SF actually scored (<= 6 at `sf_multipv: 6`).
- `probe_gap = era - inwindow` for each.

## Covered-set detection (stated before running)

Fabricated entries all carry the identical default `d = (worst_covered + 1)/2`,
which is strictly greater than every covered regret whenever `worst_covered < 1`.
Row rule: let `dmax = max_{legal} r`, `s = max_{legal, r < dmax - 1e-6} r`.
Declare fabrication iff `s` exists and `|dmax - (s+1)/2| <= 1e-4`; then
`C = {legal : r < dmax - 1e-6}`. Otherwise `C = all legal` (no fabrication
identifiable — masked == unmasked on that row). Cap-saturated rows (`d = 1.0`)
are conservatively NOT split, which biases the masked metric TOWARD the unmasked
one, i.e. against firing the kill rule.

## Predictions

- **P1 (GATE).** Unmasked, `checkpoint_000218` of trial `5ce02`, 2048 rows/set:
  era **0.069640**, inwindow **0.067193**, gap **+0.002447**, matching live
  `progress.csv` iter 218 to 6 dp. Anything else => harness wrong, stop.
- **P2.** Masked at iter 218, predicted from the banked decomposition
  (real / (1 - p_uncov)): era `0.040895/(1-0.0653) = 0.04374`,
  inwindow `0.037623/(1-0.0732) = 0.04059`, **masked gap ~ +0.00315**,
  i.e. ~1.29x the unmasked gap. Tolerance +-0.0004 (the renormaliser is applied
  per row, not to the aggregate, so this is an approximation, not an identity).
- **P3.** `|C| == 6` on >= 95% of rows with `L > 6`; `|C| == L` on rows with
  `L <= 6`.
- **P4 (the one that matters).** `gap_unmasked = gap_masked_real + gap_fab`, and
  `gap_fab` is NEGATIVE (-0.000825 at iter 218) and shrinks toward 0 as the net
  sharpens (mass leaves the uncovered tail). So the unmasked series carries an
  artefactual UPWARD slope component. Predicted: **both slopes positive, same
  sign, with `slope_unmasked > slope_masked`, ratio in [1.0, 2.0]** — i.e. the
  kill rule does NOT fire, but it is close on the 2x arm.
- **P5.** Mean net probability mass on uncovered moves DECREASES monotonically-ish
  across the lineage (the sharpening claim). If it does not, P4's mechanism is
  wrong and the direction of the artefact is not established.

## Checkpoints (pinned by path AND step, chosen before scoring)

Lineage TRUNK = trial `379f6` (2026-08-06 23:51 -> 2026-08-10), continued as
trial `5ce02` (branched from step 57999 = 379f6 iter 672). Verified by
steps/iter ~86: `(69836 - 57999)/138 = 85.8`.
Every entry's `step` is read out of the checkpoint itself, never from its filename.
