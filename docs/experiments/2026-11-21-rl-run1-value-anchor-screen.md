# 2026-11-21: RL run 1 — value-anchor screen (`--kl-wdl` to the SL base) on the pilot's collection (pre-registration)

**Status: registered, not launched.** It launches only after review and merge of this
record, AND after the follow-up chain releases the GPU (expected ≈ 21:30). It trains new
cells on the pilot's frozen collection, plays no new collection games, and changes nothing
live.

## Why

**The owner's request.**
- On 2026-11-19 the owner asked for the KL penalty on policy AND value. It was never run:
  - no banked run has `kl_wdl` or `kl_policy` non-zero;
  - the 11-20 plan made it a conditional branch (`2026-11-20-rl-run1-multi-generation-plan.md` ≈ L462);
  - the amendment set both to 0 (`2026-11-21-rl-run1-amendment-games-not-rows.md` ≈ L323: "nothing below is called anchored").
- His words today: a "kl penalty from last known best" for the value, because the value
  loss is "always clearly worse", not noise.

**What was already measured.** I grepped `docs/experiments/2026-11-21-rl-loop-evidence-log.md`
and `~/chess-artifacts/operations/rl-claude-20261119/INDEX.md` for `kl-wdl` / `kl_wdl` /
`kl-policy`, value dose (`--w-wdl` < 1) and `--sl-mix-frac`:
- **No measurement of any of the three exists.** The only hits are the plan and amendment
  lines cited above.
- The closest evidence is the pilot itself, all at `--kl-wdl 0`:

  | Cell | vs base |
  |---|---|
  | S (value on search WDL only) | −20.0 [−36.8, −3.3] |
  | Zq (half outcome, half search) | −10.0 [−23.2, +3.2] |
  | P0 (no value loss), as a single net | +2.2 [−4.2, +8.6] (match 5) |

- The value diagnosis found that S's value loss broke sibling ordering, and that
  recalibration does not repair it. It lives in `rl-claude-20261119/pilot-value-diagnosis/`:
  current README sha256 `ae5e0866…` (the v1 copy `README.v1-20261121T1323.md` is `0bdac6a5…`),
  Addendum 1 `PREREG.addendum1.md` `2a07543c…`.

## What `--kl-wdl` computes (read at the pin, `0c795fb7a`)

- **Code path.** `scripts/simple_rl_loop.py` → `TrainSettings.kl_wdl` →
  `learn.ReferenceAnchor(reference, policy_coef=kl_policy, wdl_coef=kl_wdl)`, installed by
  `attach_anchor` on the Trainer's penalty hook (`train/trainer.py` ≈ L5916). The penalty is
  added to the training loss before `backward()`.
- **Direction.** It is **KL(reference ‖ student)** on the WDL head:
  `Σ q · (log q − log p)`, where q = softmax(reference WDL) and p = softmax(student WDL),
  computed per row. This is the same direction as the diagnosis's "KL(base ‖ net)".
- **Rows.** Rows with `is_network_turn` AND `has_policy` (`frozen_policy_anchor.eligible_legal`).
  The pilot collection is self-play at `--full-prob 1.0`, so every training row is eligible,
  and the anchor sees the same rows as the WDL loss.
- **Weighting.** The total is `losses.total + kl_wdl · mean(KL)`, where the WDL term
  inside `losses.total` is `w_wdl · CE(target, p)`.
  - The gradient of KL(q ‖ p) with respect to the logits equals that of CE(q, p).
  - So the head is pulled toward the blended target **t_λ = (w_wdl · t + λ · q) / (w_wdl + λ)**.
  - With w_wdl = 1, the anchored fixed point is a = 1 / (1 + λ) of the way from the base to
    the target.
- **Reference.** `--kl-reference` defaults to the run's generation-0 model, which is the
  init checkpoint, the SL base. Every pilot cell lost to the base, so the base is "the last
  known best". **This record passes `--kl-reference $SL` explicitly anyway.**
- **`--value-rows-per-game`** does not exist at the pin. On main, the anchor's rows are the
  eligible rows, NOT the value-row subset. The pilot did not use it, and neither does this
  screen.
- **Aurora and `--optimizer-hparams-source init-checkpoint`: no special interaction.**
  - The anchor's gradient is part of the same total loss, so it reaches whatever parameter
    groups own the value head and trunk, at their realised rates. Those are the SL base's
    Aurora split: matrix group 6e-5, AdamW groups 3e-6. They are recorded in
    `config.realised.optimizer`.
  - **Consequence: both optimizers are scale-invariant per update.** Aurora orthogonalises,
    and Adam divides by √v. So for parameters that ONLY the value terms reach (the value
    head), multiplying the value loss by a constant barely changes the step.
  - So `--w-wdl` < 1 mainly lowers the value's share in the SHARED trunk, not the head's
    movement.
  - The anchor is different: it changes the gradient's DIRECTION (toward t_λ), which a
    scale-invariant step does not undo.
- **Guards at the pin.**
  - The anchor refuses `accum_steps` > 1, prefetch and compile. The simple loop builds the
    Trainer with `prefetch_batches: False` and `use_compile: False`.
  - It refuses exact-epoch replay. The pilot's `--presentations 1` uses `ScheduledSampler`,
    which sets `stored_row_quota`, not `exact_without_replacement`, so the anchor is
    accepted.
  - No banked or e2e run has used `--kl-wdl` through `run_train`. A refusal is loud: it
    raises before training.

### Proof that the anchor took effect

**The code-level proof is the pre-launch test, and it is a launch precondition.**
`tests/test_simple_rl_learn.py::test_kl_anchor_gradients_are_nonzero_and_reach_the_consumer`
must pass at the pin, alongside `tests/test_simple_rl_accum_steps.py`. With `w_wdl = 0` the
test trains the value head ONLY through the anchor and asserts a non-zero value-head gradient.

That test is the only check that kills a no-op anchor. The review's mutant computed the penalty
but did not add it to the total (`plan-review-1475/diag_under_mutant.py`). Under it,
`summary.anchor` was still fully populated (`rows`, `reference_forwards`, `mean_wdl_kl` > 0)
while the value-head gradient was 0.0. So `summary.anchor` is bookkeeping, not proof. The
run-level gates below are what can fail on a real run.

**Per-cell field checks** (`stage_a/train_check.py`; a mismatch writes `arm_<cell>/VOID`):
- `summary.realised_w_wdl` == the cell's w:
  - 1.0 for the anchored S/Zq cells and the truncated-S run;
  - w for the dose cells;
  - 0.0 for the P0-anchor cell.
- `summary.realised_search_wdl_frac` == 1.0 for S-target cells, 0.5 for Zq cells, and 0.0
  for P0-anchor.
- `summary.updates` == 1412 and `summary.drawn_rows` == 722,742, the pilot's values.
- `presented.npz` sha256 == `63e227ac…`, byte-identical to the pilot's S/Zq/P0. This proves,
  at the consumer, that every cell trained on the same batches.
- **Anchored cells:** `summary.anchor` must show:
  - `wdl_coef` == λ and `policy_coef` == 0;
  - `reference_forwards` == `updates`;
  - `rows` == `drawn_rows`;
  - `mean_wdl_kl` > 0.

  The manifest's `inputs.kl_reference.sha256` must equal the SL base's (`a39f900c…`).
- **Unanchored cells:** `summary.anchor` must be null.
- **Truncated-S run:** `summary.snapshot_updates` == [353, 635, 988], with complete
  `snapshots/step_<N>/` directories.

**Gate (i), at training time.** Within each target, `summary.anchor.mean_wdl_kl` must be
strictly decreasing in λ. If it fails, the target is void (`TARGET_<T>_VOID`) and gets no
Stage B. **The gate is strict:** `train_check.py` also voids the whole target if ANY of its
anchored cells is VOID, because a missing λ leaves the monotonicity untested.

**Gate (ii), in Stage A.** An anchored cell is "anchored" only if both hold:
- the 95% CI upper bound of its hold-out KL(base ‖ cell) lies below the unanchored cell's
  lower bound (S 0.01573, Zq 0.02642);
- its point estimate is below that of every smaller-λ cell of its target.

The run-to-run noise scale for the second clause is reported, not used as a margin:
|KL(base ‖ truncated-S final model) − 0.0167|. The truncated-S final model is an S replicate.

A cell failing gate (ii) is "not anchored" and is skipped like an inert cell, so the next λ
is selected.

**Rerun rule.**
- **Training.** A crashed cell (exit ≠ 0) has its run dir renamed `arm_<cell>.crash1` and is
  rerun ONCE, whole, BEFORE `train_check.py` runs. `train_check.py` runs exactly once, after
  every rerun, so a second crash is VOID. A field mismatch is not a crash: it voids the cell,
  with no rerun. The command block implements this order. It renames only if no
  `arm_<cell>.crash1*` exists; if one does, the rerun is spent, nothing is renamed, and the cell
  is VOID.
- **Stage A.** The operator never renames anything: `run_stage_a.sh` (v3) enforces the budget
  itself. It refuses in each of these cases:
  - any `CONTROL_FAIL` exists anywhere under `offline/`, whatever its directory is called (a
    mismatch is never rerun);
  - the mode's output dir has already completed (`DONE` or `CONTROL_PASS`);
  - any sibling `<dir>.*` exists, whatever it was renamed to: the one rerun is spent.

  Otherwise, if the output dir exists (a crash), the script renames it to `<dir>.fail1` itself
  and runs again. That is the one rerun. The reviewer's guard test (third pass) and extra cases
  were run against v3 and are banked in `stage_a/guard_tests_v3/`.
- Stage A and Stage B refuse any cell with a `VOID` file and any target with a
  `TARGET_<T>_VOID` file.

## Pin

**The pilot's `$PIN` `0c795fb7a`, from the existing worktree
`~/projects/chess-worktrees/rl-run1-pilot-0c795fb7a`.**
- `--kl-wdl` works there: the anchor code is identical between the pin and main (918785204).
  `git diff 0c795fb7a 918785204` touches neither `ReferenceAnchor` / `attach_anchor` nor
  `train/trainer.py`. It changes only `--value-rows-per-game` in `learn.py` and the
  loop, and adds a `losses.py` change that is not on this path.
- At the pin, the cells train exactly as S / Zq / P0 did.
- The pilot's C0 → PIN guards (a) and (b) already passed for this collection at this pin, so
  no guard needs restating.
- The matches run at the same pin as every pilot and follow-up match, so the within-run
  comparison rule holds.
- **`--snapshot-updates` exists at the pin** (`scripts/simple_rl_loop.py` ≈ L735, exact-once
  schedule only, which `--presentations 1` is). After the listed optimizer update counts it
  writes `train/snapshots/step_<N>/`, containing a weights-only `model.pt`, `metrics.json` and
  `snapshot.json`. Its help states that training is unchanged (bit-identical on CPU).

## Strengths: derived from the measured scales and the optimizer model

**Measured on the pilot's hold-out rows** (39,237 rows, 416 games), from
`pilot-value-diagnosis/out/holdout_logits.npz`. The review reproduced these
(`plan-review-1475/recompute_g_lambda.out`):

| Cell | KL(base ‖ cell), unanchored [95% CI] | KL(target ‖ base) | Fraction moved toward its target, g |
|---|---|---|---|
| S | 0.0167 [0.01573, 0.01771] | 0.0625 | **0.2774** |
| Zq | 0.0275 [0.02642, 0.02863] | 0.219 | **0.1164** |

Notes on the columns:
- g is the no-intercept regression slope of the cell's W−L move on the gap (target − base W−L).
- KL(target ‖ base) is used because it needs no clipping: 7.6% of `search_wdl` entries are
  exactly 0, which makes KL(base ‖ target) depend on the clip.
- Zq's target contains the one-hot outcome, so it is farther from the base, and Zq covered
  less of it.

**Which model sets λ.** The value head is in an AdamW group at 3e-6, and its steps are
scale-invariant. So the head heads for t_λ at roughly the unanchored speed, and its
displacement ≈ min(g, a) of the gap, where a = 1 / (1 + λ). The anchor binds only once a < g.
**λ is set from this min model.**

The linear (gradient-flow) model, a · (1 − (1 − g)^{1+λ}) / g, is shown beside it. The two
models agree at the strong end and diverge by up to 0.15 at the weak end. The rev-1 claim that
they "agree within 0.05" was false. Under the min model, the rev-1 weak cells (S λ 2.6,
Zq λ 7.6) would not have bound at all: holds 1.000 and 0.999.

**Targets:** min-model holds of 0.70, 0.45 and 0.25 of the unanchored displacement.

| Cell | λ (`--kl-wdl`) | a | Hold, min model | Hold, linear model | Indicative KL(base ‖ cell) |
|---|---|---|---|---|---|
| S weak | **4.1** | 0.196 | 0.707 | 0.572 | ≈ 0.008 |
| S medium | **7.0** | 0.125 | 0.451 | 0.417 | ≈ 0.003 |
| S strong | **13.4** | 0.069 | 0.250 | 0.248 | ≈ 0.001 |
| Zq weak | **11.3** | 0.081 | 0.698 | 0.546 | ≈ 0.013 |
| Zq medium | **18.1** | 0.052 | 0.450 | 0.407 | ≈ 0.006 |
| Zq strong | **33.4** | 0.029 | 0.250 | 0.246 | ≈ 0.002 |

The KL column is **indicative only**: hold² × the unanchored KL. It is not a threshold. P0's
value head drifted KL 0.0072 from the base with NO value loss, so any cell's realised KL also
contains a policy-induced part. The thresholds that matter are gates (i) and (ii).

## Cells

All cells train on the pilot's frozen collection. They use the pilot's `COLLECT` (needed by
`--collect-from`, which refuses another seed) and its `$TRAIN` **verbatim, except
`--kl-wdl` and the value terms**:

```bash
TRAIN="--presentations 1 --mirror-prob 0.5 --batch-size 512 --lr 3e-6 --lr-floor 3e-6 --lr-schedule flat \
  --warmup-steps 0 --optimizer aurora --optimizer-hparams-source init-checkpoint \
  --optimizer-state fresh --init-optimizer-state fresh --train-target-temp 1.0 \
  --kl-policy 0 --sl-mix-frac 0 --holdout-frac 0.05 --probe-rows 2048 --no-eval"
```

This is the pilot's `TRAIN` with `--kl-wdl 0` removed; each cell passes its own.

| Cell | Value flags | Role |
|---|---|---|
| S-anchor λ ∈ {4.1, 7.0, 13.4} | S's flags + `--kl-wdl λ --kl-reference $SL` | screened, matchable |
| Zq-anchor λ ∈ {11.3, 18.1, 33.4} | Zq's flags + `--kl-wdl λ --kl-reference $SL` | screened, matchable |
| S-dose w ∈ {0.1, 0.03} | S's flags + `--w-wdl w --kl-wdl 0` | offline, descriptive |
| Truncated S | S's flags + `--kl-wdl 0 --snapshot-updates 353,635,988` | offline, descriptive (snapshots only) |
| P0-anchor | `--w-wdl 0 --kl-wdl 7.0 --kl-reference $SL` | offline, descriptive |
| Controls (no retraining) | the banked S, Zq and P0 (`rl-run1-pilot/arm_*`) | references |

**The dose arm tests the value's share of the TRUNK update, not the head's displacement.**
- Scale invariance predicts that `--w-wdl` < 1 leaves the value head's own movement nearly
  unchanged. So the dose arm cannot rule out the alternative "the anchor works only because
  the head moved less".
- It answers a different question: does shrinking the value's pull on the shared trunk alone
  avoid the damage? Its numbers are descriptive.

**The truncated-S control is the "moved less" control.**
- It is one extra unanchored S run that writes snapshots at 353, 635 and 988 of its 1,412
  updates. Under the same min model, those counts give displacements of 0.25, 0.45 and 0.70
  of the full run's (k = hold × 1412).
- So each snapshot is S's own direction, stopped at the anchored cells' predicted distance.
- It is scored in Stage A only. Its final model is an S replicate: its sha256 and its numbers
  are reported next to the banked S, and nothing else.

**The matched-displacement statistic r is report-only.**
- Definition: r = (paired clear-pair accuracy difference vs the base) / (top-decile
  `share_of_gap_closed`). That is ranking damage per unit of movement toward the search value.
  S's r is −0.0096 / 0.293 = **−0.033**.
- CI: a game bootstrap. Replicate i of the numerator and replicate i of the denominator use
  the banked bootstrap weights; the weight matrix's sha256 must match between runs. Every
  difference of r gets a 95% percentile CI over the 2,000 replicates, paired by replicate.
- **Scope.** r is computed only for S-target cells: the anchored S cells, the dose cells and
  the truncated-S snapshots, all with S as the reference.
  - **r claims are NOT COMPUTABLE for Zq-target cells.** Unanchored Zq's numerator CI
    includes 0 (−0.0007 [−0.0032, +0.0017]), so there is no ranking damage to have less of.
    Against S's r, any Zq cell would "pass" on noise.
  - r is not computable for the P0-anchor cell either, and it is not computed for inert
    cells, whose denominator is near 0.
- **Matched snapshot.** Each non-inert anchored S cell is matched to the non-inert truncated-S
  snapshot whose MEASURED `share_of_gap_closed` is nearest its own.
- **Readings:**
  - **"direction"** iff both r − r_S and r − r_snapshot have CIs entirely above 0;
  - **"distance"** iff only r − r_S does: less damage, but no less than S's own direction
    stopped at the same distance;
  - **"none"** otherwise: not better than S's r;
  - **"untestable (no usable snapshot)"** when every snapshot is VOID or inert.

**The P0-anchor cell is descriptive only.**
- It has no value loss, so its only value-specific pull is the anchor, against the
  policy-induced trunk drift.
- It does NOT match any S cell's anchor share of the trunk gradient: the anchor's gradient
  scales with the head's distance from the base, which differs between cells.
- Subtracting its effect from an anchored cell's would assume the two drifts add. That is not
  assumed here, and no subtraction is reported.
- It is offline-only. Matches on it would extend match 5's +2.2 and are not part of this
  screen.

## Deciding yardstick

### Stage A: an offline VETO, not a selector

**Instruments:** adapted copies of the banked instruments of
`rl-claude-20261119/pilot-value-diagnosis/scripts/`, banked at
`~/chess-artifacts/operations/rl-run1-value-anchor/stage_a/` (read-only). Each copy takes
the net list and the output directory as arguments. Every statistic is computed by the banked
lines unchanged.
- `diffs/` holds the unified diff of each copy against its original.
- `ORIGINALS.sha256` holds the originals' shas.
- `SHA256SUMS` covers everything. `run_stage_a.sh` checks it before running anything.

**The files run are these banked files, identified by sha256:**

| File | sha256 | Adapted from (sha256) |
|---|---|---|
| `collect_logits.py` | `956c238f88c7…` | `collect_logits.py` `676b45e7e6ee…` |
| `run_fits.sh` | `5017cea31e6e…` | `run_fits.sh` `3eeff6d1ed65…` |
| `analyze_holdout.py` | `3291449af64b…` | `analyze_holdout.py` `eb368ea89777…` |
| `sibling_order.py` | `17eeb159f691…` | `sibling_order.py` `4dd4ea8000ae…` |
| `supp_outcome_distance.py` | `2ad7f272d00a…` | `supp_outcome_distance.py` `67f72335402f…` |
| `run_stage_a.sh` (v3) | `afca672f5124…` | replaces `queue_after_fits.sh` `7ee033a63a12…`; v1 `002d7d4ca8fd…` and v2 `fafc514bfd31…` kept as `.v1` / `.v2` |
| `check_control.py` (v2) | `b72e711caba8…` | new: the step-0 exact-reproduction check; v1 `413b6e11cfd8…` kept as `.v1` |
| `train_check.py` | `5a2fa045d5e9…` | new: the per-cell field checks and gate (i) |
| `stage_a_decide.py` (v3) | `91d3ef26b34b…` | new: the rules below, applied mechanically; v1 `81cc5e42dd0f…` and v2 `1b2b2986af3a…` kept as `.v1` / `.v2` |
| `value_regret_recal.py` (report-only) | run UNMODIFIED from the bank | `6cddd7d5afd1…` |

What the adapted copies add:
- `analyze_holdout.py` also writes `holdout_boot.npz`: the bootstrap replicates of
  `KL_base_net_raw` and `share_of_gap_closed`.
- `sibling_order.py` also writes `sibling_boot.npz`: the replicates of the clear-pair paired
  difference.
- Both store the bootstrap weight matrix's sha256, for gate (ii) and for r.
- The v1 → v2 changes (r scope, snapshot matching, rerun handling, `CONTROL_FAIL`, the
  truncated-S final model scored) are in `diffs/*.v1_v2.diff`. `stage_a_decide.py` v2 was
  rerun on fabricated inputs in `fab_tests_v2/`: the reviewer's scenarios A and B, plus a
  scenario C that exercises the snapshot readings "direction" and "distance" and the Zq
  "NOT COMPUTABLE".
- The v2 → v3 changes are in `diffs/*.v2_v3.diff`:
  - `run_stage_a.sh` does the rerun rename itself;
  - `stage_a_decide.py` reads "untestable (no usable snapshot)" instead of "none (no usable
    snapshot)";
  - `stage_a_decide.py` no longer divides by a share of exactly 0. That was a v2 crash, found by
    the new scenario D.

  `fab_tests_v3/` reruns scenarios A–C, with C's comment corrected (S_k4.1 is VETOED, so its
  reading is "none"), plus a scenario D in which every snapshot is inert, giving "untestable".

**Instrument positive control.**
- *Already done on CPU* (`stage_a/cpu_proof/`): the adapted `analyze_holdout.py` and
  `supp_outcome_distance.py`, run on the banked logits and fits, reproduce the banked
  `holdout_analysis.json` and `supp_outcome_distance.json` exactly (parsed JSON equal). The
  replicate percentiles reproduce the banked CIs, for example S KL [0.01573, 0.01771].
- *At launch, Stage A step 0, run FIRST, before any training* (`run_stage_a.sh control`): the
  fits, `collect_logits.py` and `sibling_order.py` need the GPU, which is busy until ≈ 21:30.
  Step 0 needs only the banked nets, so a failure costs ≈ 0.4 GPU-h instead of the training.
  The whole adapted chain runs on the banked base / S / Zq / P0. It must reproduce all of these
  exactly:
  - the banked `holdout_logits.npz` arrays, bit-equal;
  - `recal_{S,Zq,P0}.json`;
  - `holdout_analysis.json`;
  - `sibling_order.json`;
  - `supp_outcome_distance.json`.

  Otherwise `check_control.py` writes `CONTROL_FAIL`, and `cells` refuses to run without
  `CONTROL_PASS`. **The screen's verdict is then "unread: instrument failed"**, and no cell is
  trained. A mismatch is reported with its size, and the instrument is never loosened to pass.
  A determinism fix may come only by amendment, with its own review.

**Statistics** (raw heads, no recalibration; `stage_a_decide.py`):
- **VETOED** iff the 95% CI upper bound of
  `sibling_order.json → paired_minus_base.all.clearplus_50plus.<cell>_raw` is below 0. That
  statistic is the paired clear-pair (≥ 50 cp) accuracy difference vs the base, with
  game-clustered CIs. S's was −0.0096 [−0.0129, −0.0064].
  - **The veto has NO power for Zq.** Unanchored Zq passes it: −0.0007. For Zq cells, Stage A
    decides only inert / not anchored.
- **INERT** ("did it learn anything?") iff
  `holdout_analysis.json → toward_search_top_decile.<cell>.share_of_gap_closed` (raw; 3,924
  rows with |search − base| ≥ 0.241) has a CI lower bound ≤ 0, or a point estimate below
  0.1 × the unanchored cell's.
  - The unanchored values are S 0.293 and Zq 0.346, so the thresholds are 0.029 and 0.035.
  - Dose cells, truncated-S snapshots and P0-anchor are compared with S's value, but
    descriptively only.
  - Inert cells are not matched: passing the veto by not moving at all is not a recipe.
- **NOT ANCHORED:** fails gate (ii) on `holdout_analysis.json → divergence.<cell>.KL_base_net_raw`.
- **No ranking.** The offline numbers may NOT rank survivors. There are three fit→Elo
  transfer failures on record:
  - v020: +3.1 cp audit gain, −9.6 Elo;
  - ft595: −41.0 Elo for better fit;
  - F-only: hold-out CE −0.043 nats for 0 cp.

  Stage A only removes cells.
- **Report-only:**
  - each cell's `divergence.<cell>.KL_share_removed`: the share of its KL to the base that its
    own (T, b) removes;
  - FEN-only 1-ply value regret (the banked `value_regret_recal.py`);
  - outcome distance (`supp_outcome_distance.py`);
  - the (T, b) fits;
  - r.

### Stage B: matches

- **Selection, fixed now by λ, not by offline score:** for each target that is not void, the
  **smallest-λ cell that is not VOID, VETOED, INERT or NOT ANCHORED**. That is at most two
  matches, one S-anchor and one Zq-anchor. Dose, truncated-S and P0-anchor cells are never
  matched. `stage_a_decide.py` writes the selection to `offline/cells/stage_a_decision.json`
  → `stage_b`.
- **Blind to recalibration.** Selection reads raw heads, and Stage B plays plain seats. Playing
  Zq's Stage B recalibrated was considered and not adopted.
- **Design.**
  - The cell vs the SL base, plain seats, seat-balanced.
  - 6,400 games: 3,200 per run, 1,600 openings.
  - Seeds: **2026112181** for the S-anchor match, **2026112182** for the Zq-anchor match.
    Neither appears in any pilot or follow-up match nor anywhere under `docs/`.
  - The follow-up launcher `~/chess-artifacts/operations/rl-run1-followup/run_followup_match.sh`,
    pinned by sha256 `7c8fd065865f3d5abcd8e9eb6f7fc1d48e882dfcfb88bbaa92d8696ce4a9510d`, with out
    root `rl-run1-value-anchor/`, the pilot's `$ARENA` flags, and the launch record's max-plies
    void rule.
- **Rules, as match 5:**
  - "damage" iff UB95 < 0;
  - "adoptable" iff LB95 > −10;
  - otherwise "unread".

  Match 5's tie-break is kept as written: below SE ≈ 2.55 both can hold, and then both facts
  are reported as "adoptable within the margin, with a measured loss". At the expected
  SE ≈ 3.37 it cannot trigger.
- **False nomination.** Each match nominates falsely with probability ≈ 2.5% when the true
  effect is at the −10 boundary. Across the two matches, the family false-nomination rate is
  ≈ 5%.
- **Pre-declared contrast, report-only and nearly uninformative:** the anchored cell's
  estimate minus its unanchored cell's banked result. Separate seeds, same pin; the SEs treat
  the matches as independent.

  | Target | Unanchored result | SE of the difference | 80%-power MDE |
  |---|---|---|---|
  | S | −20.0 (SE 8.5 at 800 games) | ≈ 9.1 | **25.6 Elo** |
  | Zq | −10.0 (SE 6.74) | ≈ 7.5 | **21.1 Elo** |

  **No anchored-vs-unanchored head-to-head is run in this record.** The owner's question is
  whether an anchored value recipe stops losing to the base. A sharper mechanism comparison
  can be bought later.

**What each outcome licenses:**

| Reading | Licenses |
|---|---|
| adoptable, S target | NOMINATES that value recipe (S plus λ) for a 2-generation run together with the policy, under its own record. Nothing is adopted here |
| adoptable, Zq target | NOMINATES the anchored Zq recipe only TOGETHER with a "Zq + (T, b)" alternative in the nomination record. It supports no claim that "the anchor fixes content" for Zq: match 7 showed Zq's loss is largely shape, and part of an anchored Zq cell's gain may be the anchor removing that shape error |
| damage | The anchor at that strength does not prevent the value damage |
| unread | Nothing |

**What counts as "the anchor is NOT the fix"** for a target:
- (a) every anchored cell of that target is vetoed, inert or not anchored, or the target is void;
- OR (b) its matched cell reads "damage".

An adoptable reading does not by itself show that the anchor's direction, rather than smaller
movement, is responsible. That claim needs r, as registered above.

## Confounds

- **Policy-induced drift.** P0 showed that the trunk moves the value even with no value loss
  (KL 0.0072). So an anchored cell also anchors against policy-induced drift. The P0-anchor
  cell describes that part alone; no subtraction is made.
- **One data-affecting change.** Each anchored cell differs from its banked unanchored cell
  only in `--kl-wdl` / `--kl-reference`: same collection, seed, batch plan (proved by
  `presented.npz`), pin and optimizer.
- **Match 7 exists.** Recalibrated Zq vs Zq read +21.4 [+15.1, +27.6] ("the loss is shape").
  Stage A already reports each cell's (T, b) fit and its `KL_share_removed`. Any recalibrated
  reading of the Zq cells is a follow-up under the match-8 record (PR #1476,
  `docs/experiments/2026-11-21-rl-run1-match8-zq-recal-vs-base.md`, not yet on main), not part
  of this design.
- **GPU nondeterminism.** The truncated-S run's final model may differ in the last bits from
  the banked S. Its snapshots are compared with the anchored cells, not with the banked S's
  weights.

## Cost

| Item | Cost |
|---|---|
| Training, 10 runs (9 cells + truncated S) | ≈ 7 min each (measured 398–440 s for S / Zq / P0) ≈ **1.2 GPU-h**. The anchor adds one no-grad reference forward per batch, and the snapshots add three hold-out measurements, so the realised wall time is reported |
| Offline Stage A | step 0 (first): 4 nets ≈ 0.4 GPU-h; cells: base + 9 cells + 3 snapshots + the truncated-S final = 14 nets. About 18 nets × ≈ 6 min ≈ **1.8 GPU-h**, capped at 0.24 of the card as banked |
| Stage B | ≤ 2 matches × 1.52 GPU-h ≈ **≤ 3.0 GPU-h** |
| **Total** | **≤ ≈ 6.0 GPU-h** |

## Commands

Outputs go to `~/chess-artifacts/operations/rl-run1-value-anchor/` (`V`). The pilot bank
(`P`) is read-only.

**Step 0 (instrument control, FIRST), training, take-effect check:**

```bash
V=$HOME/chess-artifacts/operations/rl-run1-value-anchor; P=$HOME/chess-artifacts/operations/rl-run1-pilot
PIN=$(cat $P/PIN.txt); PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-${PIN:0:9}
PY=$HOME/projects/chess/.venv/bin/python; SL=$P/sl_base.pt
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
SF=$HOME/local_stockfish_linux_latest/stockfish/stockfish-ubuntu-x86-64-bmi2
COLLECT="--openings-dir $P/openings --init-checkpoint $SL --device cuda --seed 2026112100 \
  --games 8000 --collect-procs 3 --concurrency 128 --sims 64 --full-prob 1.0 \
  --topk 16 --c-scale 0.1 --c-visit 50 --max-plies 450 --syzygy $TB --sf-path $SF"
TRAIN="--presentations 1 --mirror-prob 0.5 --batch-size 512 --lr 3e-6 --lr-floor 3e-6 --lr-schedule flat \
  --warmup-steps 0 --optimizer aurora --optimizer-hparams-source init-checkpoint \
  --optimizer-state fresh --init-optimizer-state fresh --train-target-temp 1.0 \
  --kl-policy 0 --sl-mix-frac 0 --holdout-frac 0.05 --probe-rows 2048 --no-eval"
S_V="--value-outcome-weight 0 --value-search-weight 1 --value-search-source stored_search_wdl"
Z_V="--value-outcome-weight 0.5 --value-search-weight 0.5 --value-search-source stored_search_wdl"
declare -A CELL=(
  [S_k4.1]="$S_V --kl-wdl 4.1 --kl-reference $SL"   [S_k7.0]="$S_V --kl-wdl 7.0 --kl-reference $SL"
  [S_k13.4]="$S_V --kl-wdl 13.4 --kl-reference $SL"
  [Zq_k11.3]="$Z_V --kl-wdl 11.3 --kl-reference $SL" [Zq_k18.1]="$Z_V --kl-wdl 18.1 --kl-reference $SL"
  [Zq_k33.4]="$Z_V --kl-wdl 33.4 --kl-reference $SL"
  [S_w0.1]="$S_V --w-wdl 0.1 --kl-wdl 0"            [S_w0.03]="$S_V --w-wdl 0.03 --kl-wdl 0"
  [P0_k7.0]="--w-wdl 0 --kl-wdl 7.0 --kl-reference $SL"
  [S_trunc]="$S_V --kl-wdl 0 --snapshot-updates 353,635,988")
(cd $V/stage_a && sha256sum -c --quiet SHA256SUMS) || exit 93
# step 0, before any training: banked nets through the adapted instruments must reproduce the bank exactly.
# A crash may be rerun once by running the same line again (the script renames the crashed dir itself and
# refuses a third run); a CONTROL_FAIL ends the screen as "unread: instrument failed".
$V/stage_a/run_stage_a.sh control || exit 1
cd $PINW && [ "$(git rev-parse HEAD)" = "$PIN" ] || exit 96
CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider \
  "tests/test_simple_rl_learn.py::test_kl_anchor_gradients_are_nonzero_and_reach_the_consumer" \
  tests/test_simple_rl_accum_steps.py || exit 1          # launch precondition: the code-level proof
train() { date -Is > $V/arm_$1.started
  PYTHONPATH=. $PY -u scripts/simple_rl_loop.py run --run-dir $V/arm_$1 --collect-from $P/collect \
    --generations 1 $COLLECT $TRAIN ${CELL[$1]} > $V/arm_$1.log 2>&1; echo "$? $(date -Is)" > $V/arm_$1.exit; }
CELLS="S_k4.1 S_k7.0 S_k13.4 Zq_k11.3 Zq_k18.1 Zq_k33.4 S_w0.1 S_w0.03 P0_k7.0 S_trunc"
for c in $CELLS; do train $c; done
for c in $CELLS; do                   # one whole rerun per crashed cell, BEFORE the single train_check run
  [ "$(cut -d' ' -f1 $V/arm_$c.exit)" = 0 ] && continue
  [ -z "$(compgen -G "$V/arm_$c.crash1*")" ] || continue      # rerun already spent: left crashed, train_check VOIDs it
  for f in "" .log .exit .started; do [ ! -e $V/arm_$c$f ] || mv $V/arm_$c$f $V/arm_$c.crash1$f; done; train $c
done
$PY $V/stage_a/train_check.py $V      # once: field checks + gate (i); writes VOID / TARGET_<T>_VOID markers
```

**Stage A cells** (after the training and `train_check.py`; step 0 already passed):

```bash
$V/stage_a/run_stage_a.sh cells || exit 1   # refuses without CONTROL_PASS; skips VOID cells; fits, logits, analysis,
                                            # outcome distance, sibling order, value regret (report-only), stage_a_decide.py
# a crash may be rerun once by running the same line again (self-rename; a third run is refused)
```

**Stage B**, for each target T whose `stage_b` entry names a cell `X`, with seed `SEED`
(2026112181 for S, 2026112182 for Zq):

```bash
F=$HOME/chess-artifacts/operations/rl-run1-followup
echo "7c8fd065865f3d5abcd8e9eb6f7fc1d48e882dfcfb88bbaa92d8696ce4a9510d  $F/run_followup_match.sh" | sha256sum -c --quiet || exit 93
[ "$($PY -c "import json;print(json.load(open('$V/offline/cells/stage_a_decision.json'))['stage_b']['$T'])")" = "$X" ] || exit 91
[ ! -e $V/arm_$X/VOID ] && [ ! -e $V/TARGET_${T}_VOID ] || exit 91
$F/run_followup_match.sh $V anchor_$X 1 A $SEED 3200 --candidate $V/arm_$X/gen_0001/train/model.pt --reference $SL || exit 1
$F/run_followup_match.sh $V anchor_$X 1 B $SEED 3200 --candidate $SL --reference $V/arm_$X/gen_0001/train/model.pt || exit 1
d=$V/arenas/anchor_$X; cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $V/arm_$X/gen_0001/train/model.pt \
  --a-games $d/anchor_${X}_1_A.games.jsonl --b-games $d/anchor_${X}_1_B.games.jsonl \
  --a-result $d/results.jsonl --b-result $d/results.jsonl \
  --a-label anchor_${X}_1_A --b-label anchor_${X}_1_B --out $d/pool.json > $d/pool.log 2>&1
```

**Recovery.** Nothing to recover: no live change. The pilot bank is read only; the arms
reference its collect, and `simple_rl/reference.py` documents that it never writes the
source.
