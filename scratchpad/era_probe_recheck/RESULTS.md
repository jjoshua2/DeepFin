# era-probe masked-vs-unmasked recheck — 2026-08-15

CPU only, no GPU. Training was stopped throughout; nothing was started.
Predictions registered in `probe_PREREG.md` before any number was produced.

## Kill rule (pre-committed, restated)

> if the masked and unmasked `probe_gap` trends have **opposite sign**, or differ by
> **more than 2x in slope**, then every forgetting-hinge verdict read off `probe_gap`
> needs re-deriving. Otherwise the ruler is salvaged as-is.

## 1. Reproduction gate — PASS

`checkpoint_000218` of `train_trial_5ce02_...` (step 79861), 2048 rows/set, CPU fp32:

| | reproduced | live `progress.csv` iter 218 | |Δ| |
|---|---|---|---|
| `probe_era_policy_eregret` | 0.069639894 | 0.069639878 | 1.6e-8 |
| `probe_inwindow_policy_eregret` | 0.067193168 | 0.067193150 | 1.8e-8 |
| `probe_gap_policy_eregret` | 0.002446726 | 0.002446729 | 3e-9 |

Agreement to 7-8 significant figures. Not literally bit-identical — the live probe runs
the same fp32 no-autocast arithmetic on GPU, so the reduction order differs. P1 met.

## 2. ⚑⚑ THE PREMISE IS FALSE: the frozen probe sets are MultiPV-**40** data

MEASURED, three independent ways:

1. **Provenance.** Both sets were cut `2026-08-04T04:12/04:13Z` from the `13a9f`
   lineage (`data/era_probe/*.provenance.json`).
2. **Config history.** `git log -L` on `sf_multipv` in `configs/pbt2_small.yaml`:
   **40** from 2026-04-29 (`02c64f700`) until 2026-08-06 (`ed9de8ee9`). The 40 -> 6
   change lands **two days AFTER** the sets were frozen.
3. **Structure of the stored vectors.** Mean **21.8** distinct regret values per row
   over mean 27.7 legal moves (a MultiPV-6 row can have at most 7). Only rows with
   L > 40 can carry a fabricated block: 15.0% (era) / 14.4% (inwindow). On those rows
   the number of legal moves tied at the row maximum equals **exactly L - 40** on
   **284/308** and **265/295** rows, and on **zero** rows is it smaller. The remaining
   24/30 tie higher because the `(worst+1)/2` default collided with the 1000cp cap.

⇒ **True fabricated probability mass at iter 218: 0.00126 (era) / 0.00237 (inwindow)** —
against the banked 0.0653 / 0.0732. A **51.6x / 30.9x over-count**.
True fabricated share of the ruler's LEVEL: **1.51% / 3.00%**, not 41.3% / 44.0%.

**Where the banked number came from.** `sfreg_probe_share.py:64` classifies
`is_fab = legal & (r >= max_legal(r))` — every move tied at the row maximum. On
MultiPV-40 data that block is overwhelmingly *real* Stockfish evaluations clipped at
`SF_OWN_REGRET_CAP_CP = 1000`: mean **2.08** legal moves per row sit at exactly 1.0, and
**25.2%** of rows have their maximum at exactly 1.0. The classifier is correct for
MultiPV-6 shards and wrong for these frozen sets.

## 3. The pre-committed experiment, run anyway

Three arms, one forward pass each, 13 checkpoints:

- `unmasked` — the live ruler verbatim.
- `mech` — mechanism-correct mask: drop the tied-at-max block on rows with L > 40 only,
  renormalise `p` over the rest.
- `aggr` — the banked classifier: drop the tied-at-max block on **every** row. Upper
  bound on the correction; discards real cap-saturated SF evaluations.

Two lineages, fitted separately and never pooled: trial `379f6` (trunk, steps
639 -> 76286) and trial `5ce02`, which branched from the `379f6` iter-672 salvage at
step 57999 (verified: `(69836 - 57999)/138 = 85.8` steps/iter).

### Checkpoint table (`probe_gap` x 1e3; step read from the checkpoint, not the filename)

| lineage | step | path | unmasked | mech | aggr |
|---|---|---|---|---|---|
| 379f6 | 639 | `data/ratchet/snapshots/ck_2026-08-07_iter7.pt` | 2.7558 | 3.4419 | 4.2990 |
| 379f6 | 21924 | `data/ratchet/snapshots/ck_2026-08-08_iter249.pt` | 1.7083 | 2.5173 | 3.5046 |
| 379f6 | 35288 | `data/ratchet/snapshots/ck_2026-08-08_iter399.pt` | 0.7925 | 1.6211 | 3.3844 |
| 379f6 | 44613 | `data/ratchet/snapshots/ck_2026-08-09_iter514.pt` | 1.6241 | 2.4842 | 3.6842 |
| 379f6 | 57999 | `data/ratchet/snapshots/ck_resume_iter672.pt` | 1.9269 | 2.7843 | 4.1909 |
| 379f6 | 63897 | `data/ratchet/snapshots/ck_2026-08-09_iter735_FINAL.pt` | 3.6361 | 4.4446 | 4.4385 |
| 5ce02 | 65441 | `scratchpad/anchors_20260811/ckpt_5ce02_iter086` | 2.4280 | 3.3309 | 3.7079 |
| 379f6 | 67105 | `data/ratchet/snapshots/ck_2026-08-10_iter768.pt` | 4.3634 | 4.8825 | 4.8965 |
| 5ce02 | 69836 | `data/ratchet/snapshots/ck_2026-08-11_5ce02_iter138` | 1.7287 | 2.7471 | 3.2709 |
| 379f6 | 72938 | `data/salvage/pre_mainmerge_20260810/seeds/slot_000` | 5.1720 | 5.4731 | 5.3690 |
| 379f6 | 76286 | `data/ratchet/snapshots/ck_2026-08-10_iter862_postmerge` | 4.9688 | 5.2230 | 4.7245 |
| 5ce02 | 79213 | `runs/.../train_trial_5ce02_.../checkpoint_000213` | 1.9770 | 3.0433 | 4.1876 |
| 5ce02 | 79861 | `data/ratchet/snapshots/ck_2026-08-12_5ce02_iter218` | 2.4467 | 3.4131 | 3.7223 |

### Trends

Trunk `379f6`, n=9, slope per 10k steps (x1e-3), 95% t-CI:

| arm | slope | 95% CI | r2 | bootstrap slope [95%] | P(slope>0) |
|---|---|---|---|---|---|
| unmasked | +0.4080 | [-0.0166, +0.8327] | 0.425 | +0.4084 [-0.1254, +0.9708] | 0.931 |
| mech | +0.3570 | [-0.0106, +0.7245] | 0.430 | +0.3588 [-0.1799, +0.9047] | 0.905 |
| aggr | +0.1643 | [-0.0167, +0.3453] | 0.397 | +0.1599 [-0.1205, +0.4366] | 0.866 |

**ratio unmasked/mech = +1.143** (bootstrap median +1.109 [-0.149, +2.513]), same sign.
ratio unmasked/aggr = +2.484.

Branch `5ce02`, n=4, span 14.4k steps: unmasked +0.0135, mech +0.0719, aggr +0.2924.
**Neither slope is distinguishable from zero** (r2 0.001 / 0.028, P(slope>0) = 0.50 /
0.53, CIs +-1.5e-3). Point ratio 0.187; row-bootstrap ratio **+0.991 [-0.294, +2.302]**.
A ratio of two slopes that are each consistent with zero has no resolution — this fit
cannot fire or clear the rule (`compute_instrument_resolution_before_the_threshold`).

Pooled 13-point sensitivity (cross-lineage, NOT a deciding fit): ratio unmasked/mech
**+1.022**.

### Shape-free agreement — the strongest read

The trunk series is V-shaped (falls 639 -> 35288, rises after), so an OLS slope is a weak
summary of it (method rule 16). The ordering statistic does not depend on shape:

- Pearson r(unmasked, mech) across all 13 nets = **0.9931** (trunk only 0.9967);
  regression `mech = 0.844 * unmasked + 1.186e-3`.
- **Pairwise ordering agreement over all 78 net pairs: 78/78 = 100%.**
  Every "net A has a wider gap than net B" statement is identical under both rulers.
- Even the wrong `aggr` classifier agrees on 68/78 (r = 0.902).

## 4. VERDICT — the kill rule does NOT fire

Same sign on every fit; ratio 1.14 on the deciding trunk fit, 1.02 pooled, and 100%
ordering agreement. The `5ce02` point-ratio of 0.19 is reported for completeness and is
not a firing: both of its slopes are zero within noise.

⚑ Note the rule was well designed and would have fired had the banked decomposition been
right: the `aggr` arm — the ledger's own classifier — gives ratio **2.484**. The
experiment cleared because the input measurement was wrong, not because the threshold was
loose.

## 5. Which banked verdicts are in question — NONE from fabrication

- **The 2026-08-02 forgetting-hinge proof (`40fcd2ddc`,
  `scratchpad/forget_curve_20260802/`) is fabrication-free BY CONSTRUCTION.**
  `build_sets.py:106` selects `full = d["n_cov"] == d["n_legal"]` — only rows where SF
  covered every legal move. Confirmed in the artefacts: `n_legal` maxes at exactly 40 in
  all five set files, and no row carries an uncovered entry. It never touched a
  fabricated value.
- **The one banked numeric `probe_gap` claim** (ledger line 41244, "the pre-736 slow
  drift, `probe_gap_policy_eregret` +83% ... over 514 -> 735") survives under both
  rulers: re-derived from the archived checkpoints, **+123.9% unmasked / +78.9% masked**.
  MEASURED. The source series for the banked +83% is no longer on disk (the `379f6`
  trial dir retains only iters 835-862), so the +83% vs +124% difference cannot be
  reconciled directly; INFERRED explanation — a two-point ratio on a series whose
  per-iteration scatter is +-20-25% (live `5ce02` iters 216/217/218 read
  0.001626 / 0.001964 / 0.002447). The qualitative claim is unaffected; the fragility of
  a two-point read on this column is worth recording.

## 6. ⚑ SEPARATE AND LARGER: the "in-window" leg has not been in-window since 2026-08-04

Not what this experiment was chartered to test, found while establishing provenance.

- MEASURED: `era_probe_inwindow_path` has pointed at `inwindow_20260804.npz` since
  2026-08-04 (`configs/pbt2_small.yaml:825`); it is the only in-window set on disk, and
  the paths are CONSTRUCTION-ONLY, so a rebuild is a manual pre-restart step.
- MEASURED: two restarts have happened since (`379f6` 2026-08-06, `5ce02` 2026-08-11)
  and neither rebuilt it. The instrument entry at ledger line 31597 describes this leg as
  "re-cut from the NEWEST shards, rebuilt at each restart"; that has not been true since
  the set was first cut.
- MEASURED: the window is pinned at 1.5M positions and evicted ~32k rows/iter on
  2026-08-04 (ledger, 2026-08-04 ~17:40), i.e. full turnover in ~47 iterations.
- INFERRED: the in-window rows left the replay window during 2026-08-04/05, **before the
  `379f6` run began**. So for the entire span over which `probe_gap` has been read, it
  has been the difference between two OLD-era sets cut 20 minutes apart, not
  era-vs-in-window. Its two legs track each other at **r = 0.987** across the 13 nets and
  the gap is **3.6% of the level** (range 1.1-6.1%), which is what that degeneracy looks
  like.

This is a much bigger threat to a forgetting-hinge readout than the fabrication ever was,
and it is untested here.

## Files

- `probe_PREREG.md` — predictions, registered first. P1 HIT. **P2/P3 MISSED** — both
  assumed MultiPV-6 content; the miss is what uncovered section 2. P4's mechanism is
  wrong in magnitude (the artefactual tailwind exists but is ~1/30 of the claimed size);
  its ordinal prediction (`slope_unmasked > slope_masked`, ratio in [1,2]) HIT at 1.143.
  P5 not supported: true fabricated mass does not fall monotonically (0.00179 -> 0.00126
  with excursions to 0.00242).
- `probe_masked.py` — the three-arm scorer.
- `probe_setstruct.py` — the MultiPV-40 structural proof.
- `probe_trend.py` — fits, bootstrap, re-derivation of the banked +83%.
- `probe_scores.jsonl`, `probe_trend_out.txt` — outputs.
- `dumps/probe_rows_<step>_<tag>.npz` — per-row values for all three arms on all 13
  checkpoints, so every CI here recomputes with no GPU and no re-scoring.
