# PREREG (NOT LAUNCHED): the lc0 positive control

Status: **NOT LAUNCHED. Training is STOPPED and needs Josh's explicit go.**
Written 2026-08-15, before any converted shard has been trained on and before any
number from this arm is visible. Depends on PR #435 landing.

⚑⚑ **THIS FILE IS THE PRE-REGISTRATION, AND IT IS COMMITTED FOR THAT REASON.**
Until 2026-08-16 it lived as `scratchpad/lc0_positive_control/PREREG_DRAFT.md`, an
UNTRACKED file in one working tree — `git log --all` found nothing — while shipped
code hard-coded its numbers (`--sample 100000`, `--max-halfwidth-pp 0.392`) and cited
it as pre-committed. A pre-registration that can be edited after a number is seen,
with no record, is not a pre-registration; `memory/uncommitted_live_yaml_edits_lose_proven_wins.md`
is the same lesson one domain over (**NOT COMMITTED = NOT DEPLOYED**). Every later
change to this file must be an AMENDMENT with its own date and reason, appended
rather than edited in place, and the git history is what makes that checkable.

Amendments so far: the debug-corpus amendment (2026-08-15, below) and the guard-1
correction (2026-08-16, below). Both were written before any training step.

## Why this exists at all

Every verdict this project has produced was measured against the loop that is under
suspicion. We have never once fed our training stack data we did not generate. So
when the loop goes flat we cannot separate:

- **(H_target)** our TARGETS / data-generation loop are the defect, and the training
  stack is fine; from
- **(H_stack)** our TRAINING CODE or ARCHITECTURE is the defect, and no target would
  have saved it.

These have opposite remedies, and we have been unable to choose between them for
months. Clean external lc0 data is the discriminator.

## Hypothesis

Trained supervised on clean lc0 T91 rows, our production architecture (61.44M,
512×16×16) and trainer will show **held-out generalisation that is still improving
at the end of the budget**. If it does, H_stack is disfavoured and the defect is in
what we feed the loop. If it plateaus the same way production did, H_stack is live.

## ⚑ What this arm is NOT allowed to conclude

- **Fit is not the readout.** We already know the stack can FIT — absorption is 11.6×
  in-window. The banked failure is that policy *stopped generalising* ~iter 249. A
  training-loss curve here proves nothing and must not be quoted as a verdict.
- **This cannot judge Elo.** No arena, no selfplay, no PID. It is a statement about
  learning, not about strength.
- **A pass does not vindicate our targets.** It only says the stack is capable.

## Design

Supervised only. No selfplay, no MCTS, no PID, no curriculum. Production
architecture and optimizer (`aurora`, `matrix_optimizer_scope: mlp_out`) unchanged —
the point is to test THE STACK WE RUN, so any architecture change voids the arm.

**Train set**: converted T91 rows, tars `20260810-1417` … onward, EXCLUDING the
held-out window below.

**Held-out set**: the LAST 6 hourly tars by wall-clock, **time-disjoint** and never
trained on, frozen as an explicit row-id list before the first training step.
⚑ This is not optional caution: a held-out set drawn from the same window measures
exposure recency, not generalisation — banked as
`exposure_recency_dominates_heldout_ce`. Freeze it and record its sha256.

## Instrument resolution — COMPUTED FIRST, threshold derived from it

Held-out top-1 agreement with lc0's own visit-argmax. Computed 2026-08-15:

| comparison | n | halfwidth |
|---|---|---|
| unpaired, p≈0.35 | 50,000 | 0.418 pp |
| unpaired, p≈0.35 | 100,000 | 0.296 pp |
| **paired (same rows, two ckpts), discordance 0.10** | **100,000** | **0.196 pp** |
| paired, discordance 0.20 | 100,000 | 0.277 pp |

⇒ at n=100,000 paired we resolve ~0.2 pp, so a **2× material bar is 0.392 pp**.
Everything below is stated in halfwidths first and converted second, per
`compute_instrument_resolution_before_the_threshold`.

## PRIMARY yardstick (the one that decides)

⚑⚑ **A HELD-OUT PLATEAU ALONE IS NOT A VERDICT — IT IS TWO DIFFERENT OUTCOMES
WEARING ONE NUMBER.** A stack that plateaus because it has *converged on the data
available* looks identical, in held-out top-1, to a stack that plateaus because it
*cannot generalise*. Those have opposite remedies (more data / bigger net vs. fix the
trainer), so a single-series readout cannot decide this experiment. Caught while
drafting, before any number existed.

The discriminating signature is the one production actually showed at ~iter 249:
**fit kept improving while generalisation did not.** So the primary readout is the
JOINT reading of two slopes, both measured LAST vs MID-BUDGET on frozen row sets:

- `Δ_heldout` — paired top-1 on the frozen held-out rows (McNemar CI)
- `Δ_train` — paired top-1 on a frozen, equally sized sample of ROWS ALREADY TRAINED ON

| `Δ_heldout` | `Δ_train` | verdict |
|---|---|---|
| **≥ +2.0 pp**, CI lower > +0.392 pp | anything | **PASS** — stack learns and generalises. H_stack disfavoured. |
| within ±0.392 pp | **≥ +2.0 pp** | **⚑ H_stack LIVE** — the iter-249 signature reproduced on clean external data. Fit rises, generalisation does not. Most consequential outcome available. |
| within ±0.392 pp | within ±0.392 pp | **CONVERGED / CAPACITY-LIMITED — INCONCLUSIVE for H_stack.** Not a pass and not a failure. Pre-committed response: one 2× budget extension; if both slopes stay flat, the arm cannot answer the question and says so. |
| anything else | anything else | **AMBIGUOUS** — one 2× budget extension, then read once. No second extension; that is optional stopping. |

2.0 pp is ~10× the resolution, deliberately far above the noise floor: a stack
genuinely learning from 87M fresh positions should not be arguing in the third
decimal. Both slopes use the same n and the same estimator so the ±0.392 pp bar
applies to both.

## Guards that must pass before the primary is read

A verdict off a failed instrument is not a verdict, in either direction.

1. **NEGATIVE CONTROL — shuffled labels.** Same rig, policy targets permuted across
   rows. Held-out top-1 must collapse to chance. ⚑ **Chance is `E[1/n_legal]`, NOT
   `1/E[n_legal]`** — Jensen; this has bitten us three times and the two differ by
   ~2× in exactly the direction that puts a gate under its own floor. Compute
   `E[1/n_legal]` on the frozen held-out set and record it BEFORE the run.
   > ⚑⚑ **SUPERSEDED 2026-08-16 — see AMENDMENT 2. `E[1/n_legal]` is the wrong
   > quantity for this control by ~19×.** Kept here unedited because what the
   > pre-registration ORIGINALLY said is the thing a pre-registration is for.
2. **RANDOM-INIT FLOOR.** Same architecture, untrained, on the same rows. Must sit at
   the same chance level. If it does not, the evaluator is broken, not the net.
   > ⚑ **SUPERSEDED 2026-08-16 — see AMENDMENT 2.** A random-init net is a fixed
   > arbitrary function over a highly non-uniform argmax distribution and is NOT
   > expected to sit at the uniform-mover floor. The gate is a seeded BAND.
3. **⚑ `sf_wdl_frac: 0.0` MUST BE VERIFIED IN EFFECT, not configured.** lc0 shards
   carry no `sf_wdl`, and `train/losses.py` falls the SF component back to the raw
   one-hot game outcome when `has_sf_wdl=0` — so at production `sf_wdl_frac` roughly
   0.69 of the value weight would land silently on the deep outcome. Read the
   realized weight off the first training row and record it. A configured value is
   not an applied value.
4. **Held-out purity.** Assert zero row-id intersection between train and held-out,
   by id, not by tar name.

## ⚑⚑ AMENDMENT 1 (2026-08-15, written BEFORE any training step): THE CORPUS IS A *DEBUG* RUN

The premise as originally written was "clean external data **known to train a strong
net**". Provenance now establishes that is weaker than assumed, and the weakening is
in the direction that matters.

Read from lczero.org's training-runs page, not inferred:

| | run1 / **test80** | run2 / **test91** (what we have) |
|---|---|---|
| status | inactive | active |
| best net | 826344 | **913286** |
| upstream description | "Main run started April 4th 2022 (Partially pre-trained)" | **"Another debug run for training BT4 in RL mode"** |
| net shape | not shown | **15 blocks × 1024 filters** |

Two consequences, both of which must be in the readout:

1. **T80 was the flagship main line; T91 is explicitly a DEBUG run.** We could not take
   T80 — it is dead (newest non-empty tar 2025-09-24). So "lc0 data is known-good" is
   NOT established for this corpus at the strength the premise assumed.
2. **The teacher is far larger than the student.** 15×1024 (BT4-shaped, matching
   `docs/bt4.md`) generating targets for our 512×16 / 61.44M net. That is a
   capacity gap in the targets, not a like-for-like transfer.

⇒ **This asymmetrically restricts what a PLATEAU licenses, and not what a PASS
licenses.** A PASS still says the stack can learn and generalise — a debug corpus that
teaches our net is still a corpus that teaches our net. But a PLATEAU now has a second
live explanation ("T91 debug data is not good enough to learn from") competing with
"our training stack cannot generalise", and **this arm alone cannot separate them.**

**Pre-committed consequence:** if the readout lands in the `H_stack LIVE` cell
(held-out flat, train rising), that verdict is **PROVISIONAL, not final**, and the
named discriminator is a second corpus — no post-hoc substitute. Writing this down now
so a plateau cannot be promoted into "the training stack is broken" without it.

## ⚑⚑ AMENDMENT 2 (2026-08-16, written BEFORE any training step): GUARD 1's CRITERION WAS OFF BY 19×, AND GUARD 2 WAS NEVER A POINT

Filed as F2(b) of PR #438's second independent review, re-derived here. **No arm has
run and no number from this arm exists**, so this amendment cannot be threshold
shopping — it is an instrument correction, and per this file's own rule ("a verdict
off a failed instrument is not a verdict, in either direction") the ORIGINAL guards
could neither pass nor fail.

### Guard 1 — the shuffled-label control does not converge to `E[1/n_legal]`

The control scores each row's prediction against **another row's real target**. For a
uniformly random permutation `π` (fixed points included, each with probability `1/n`):

```
E[hit rate] = (1/n) Σ_i Σ_j 1[tgt_j = pred_i] / 1  =  Σ_m p_pred(m) · p_tgt(m)
```

— the **marginal collision rate** between the predictor's move-id distribution and the
target's, not the uniform-random-mover floor `E[1/n_legal]`. Measured on 100,000 real
converted T91 rows (`training-run2-test91-20260810-1417`):

| quantity | value |
|---|---|
| `E[1/n_legal]` — uniform-mover floor, what the rig printed | 0.063622 |
| `Σ_m p_tgt(m)²` — the shuffled floor when `p_pred ≈ p_tgt` (a trained net) | 0.003283 |
| ratio | **0.052× — 19× apart** |

distinct best-move ids 1803/1858, top move-id share 0.0223.

⚑ Getting the JENSEN direction right (`E[1/n]` vs `1/E[n]`, the correction already in
guard 1) did not make it the right QUANTITY. Both are floors for a *uniform mover*,
and the shuffled control does not produce one.

**Corrected guard 1.** The floor is a property of the scored run, not a constant, so it
is COMPUTED FROM THE SAME SCORE — `scripts/lc0_control_eval.py score --shuffle-targets`
prints `Σ_m p_pred(m)·p_tgt(m)` next to the observed rate and **exits 1** when the
observed rate sits more than `--negative-control-z` (default 5) standard errors ABOVE
it. That is the direction with a failure mode behind it: a shuffled control that still
agrees is a rig leaking row identity into the prediction. `E[1/n_legal]` stays in
`heldout.py chance`, relabelled as what it is — the uniform-mover reference — and is
not a gate.

### Guard 2 — a random-init net is not expected to sit at any point

Same measurement, other direction: an untrained net is a fixed arbitrary function over
a highly non-uniform argmax distribution, so it beats uniform guessing on frequent
moves. Four unseeded draws of the identical command on the identical 100,000 rows read
0.058770 / 0.062050 / 0.071630 / 0.073500 / 0.080590 — a 2.18 pp spread, 5.6× the
0.392 pp material bar, straddling 0.064577 in both directions. "It sits at chance" was
never a testable statement; the seed spread was the whole signal.

**Corrected guard 2.** `--seed` is applied before `build_model` and recorded in the
score metadata, and the gate is a BAND over several seeds against a trained
checkpoint. ⚑ **The band is not in this file, because it requires a trained
checkpoint and there is none.** It must be measured and appended here as AMENDMENT 3
BEFORE the primary readout, not alongside it.

### The constants this file is cited for

`--sample 100000` and `--max-halfwidth-pp 0.392` are hard-coded defaults in shipped
code that name this document as their source, so the derivation is now executable:
`lc0_control_eval.paired_halfwidth_pp(n, discordance)` reproduces every row of the
resolution table above, and `MATERIAL_BAR_PP = 2 × paired_halfwidth_pp(100_000, 0.10)`
= 0.392. `tests/test_lc0_control_drivers.py` pins both against the table.

## Confounds to state in the readout, not discover afterwards

- T91 is a **different lc0 run** from the test80 data behind our banked 2026-08-06
  comparison (test80 is dead — newest non-empty tar 2025-09-24). Measured T91 policy
  stats: entropy 1.5063 / max-prob 0.4956 / one-hot 0.64% / support 29, vs banked
  test80 1.399 / 0.562 / 1.3% / 30. **These are NOT the same population** and the
  banked numbers must not be used as this arm's reference. See the debug-run
  amendment above.
- **Corpus integrity is VERIFIED, and that is separate from corpus quality.** 130/130
  files byte-exact against the server's published index; all 130 archives open with
  every `.gz` member decompressing; every game blob an exact multiple of 8356 B with
  `version==6`, `input_format==1`, policy width 1858; 400 games / 48,564 positions
  replayed move-by-move against python-chess with 0 mismatches. ⚑ The server publishes
  **no checksums**, so our sha256s pin what we hold and cannot prove it matches what
  upstream intended. One file (`...20260813-1717.tar`, 11.6 MB vs a ~155 MB norm) is
  short **upstream** — it size-matches the index and parses cleanly to 614 games /
  53,083 positions — so it is a short hour, not a truncated download.
- Our value target and lc0's differ; `wdl_target` ← lc0 outcome, `search_wdl` ←
  lc0 `best_q`/`best_d`, `sf_wdl` deliberately ABSENT.
- ~3.7% of T91 games are chess960/DFRC and are dropped; ~1 row in 23,000 is dropped
  for the e.p.-blind repetition divergence.

## What each outcome licenses — pre-committed

- **PASS** ⇒ the training stack and architecture can learn. Effort routes to targets
  and the data-generation loop. Does NOT license "our targets are the defect" as
  proven; it removes the alternative.
- **H_stack LIVE** (held-out flat, train rising) ⇒ the training stack cannot
  generalise even on clean external data. Becomes the top priority and reframes
  every banked target finding. This would be the most consequential result the
  project has produced, and it is the outcome I expect least — which is exactly why
  the threshold is written down now.
- **CONVERGED / CAPACITY-LIMITED** (both flat) ⇒ the arm ANSWERS NOTHING about
  H_stack, and must be reported that way rather than as a pass. It does license one
  narrower claim: 87M lc0 positions were not enough to separate the hypotheses at
  this architecture, which is itself worth knowing before anyone proposes a bigger
  external-data run.
- **Guard failure** ⇒ no verdict in either direction. Fix the rig, re-run.

## Revert / cost

Offline arm. Touches no production weights, no live yaml, no replay window. The only
shared resource is the GPU, so it requires production training to be stopped —
which it already is, and which is Josh's call to change.
