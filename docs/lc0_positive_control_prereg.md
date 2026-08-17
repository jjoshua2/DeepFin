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

Amendments so far: the debug-corpus amendment (2026-08-15, below), the guard-1
correction (2026-08-16, below) and the TRAINER RE-POINT (2026-08-16, below). All
were written before any training step.

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

⚑ "Production" means the LIVE working tree's `configs/pbt2_small.yaml`, not `main`'s
committed copy, on BOTH axes. Enforced as launch guards 0/0b (architecture, against
`LIVE_ARCH_PIN`) and 0c (trainer, against `LIVE_TRAINER_PIN`). See AMENDMENT 3.

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
   > ⚑ EXTENDED 2026-08-16 by AMENDMENT 3: the same verification is now owed by
   > the CATEGORICAL head, whose rebuild drops an unavailable component's weight
   > onto the raw outcome by the identical rule. Measured, both bars, every step.
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
checkpoint and there is none.** It must be measured and appended here as a FURTHER
AMENDMENT (the next free number) BEFORE the primary readout, not alongside it.
> ⚑ This sentence reserved "AMENDMENT 3" for the band. The trainer re-point landed
> in that slot first, on 2026-08-16, and reserving an index in a document that only
> grows by appending was the mistake — an amendment is identified by what it
> contains, not by its number. The band is still owed and still unmeasured.

### The constants this file is cited for

`--sample 100000` and `--max-halfwidth-pp 0.392` are hard-coded defaults in shipped
code that name this document as their source, so the derivation is now executable:
`lc0_control_eval.paired_halfwidth_pp(n, discordance)` reproduces every row of the
resolution table above, and `MATERIAL_BAR_PP = 2 × paired_halfwidth_pp(100_000, 0.10)`
= 0.392. `tests/test_lc0_control_drivers.py` pins both against the table.

## ⚑⚑ AMENDMENT 3 (2026-08-16, written BEFORE any training step): "OUR EXACT TRAINER" WAS `main`'s TRAINER, AND THE VALUE OVERRIDE PROTECTED A NUMBER PRODUCTION DOES NOT HAVE

**No arm has run and no number from this arm exists**, so this cannot be threshold
shopping. It is the same correction AMENDMENT 2's architecture half received, on the
axis that did not get it: the arm's claim is "our EXACT net and trainer", and until
now only the NET was judged against the file production reads.

### The measurement

`trainer_kwargs_from_config()` on `configs/lc0_positive_control.yaml` against the
LIVE `configs/pbt2_small.yaml` differed in **13 kwargs**. Against `main`'s committed
copy — what every test in the tree checked — it differed in **2**, and `main` has
committed nothing to that file since the live branch diverged.

| kwarg | control (was) | LIVE |
|---|---|---|
| `warmup_steps` | 72 | **1000** |
| `w_categorical` | 0.3 | **1.0** |
| `rebuild_categorical_target` | False | **True** |
| `categorical_target_params` | blend 0.0 / search 0.0 | **0.69 / 0.31** |
| `w_sf_own_regret` | 0.7 | **0.0** |
| `w_sf_own` | 0.1 | **0.0** |
| `w_sf_eval` | 0.1 | **0.0** |
| `w_sf_move` | 0.02 | **0.05** |
| `sf_target_params.sf_policy_score_mode` | wdl | **cp** |
| `lr_T0` / `lr_T_mult` | 999999 / 1 | **5000 / 2** (both DEAD — see below) |
| `sf_wdl_frac` | 0.0 | 0.69 — the declared deviation |
| `search_wdl_frac` | 0.7 | 0.31 — the declared deviation |

### ⚑⚑ THE CONSEQUENCE THAT MATTERS: `game_frac` IS 0.00, NOT 0.30

The arm's value override was justified — in the config header, in the config test,
and in `value_blend_guard.PRODUCTION_GAME_FRAC` — as "hand the SF share to lc0's own
search value so `game_frac` stays at production's **0.30**". Live's blend is
`sf_wdl_frac 0.69 + search_wdl_frac 0.31 = **1.00**`, i.e. **`game_frac` 0.00**.
Production puts NO weight on the raw game outcome. The 0.30 was `1 - 0.50 - 0.20`
off `main`'s copy, and `search_wdl_frac: 0.70` was chosen to reproduce it — so the
override was *introducing* a 0.30 outcome share rather than preserving one, and the
guard's bar admitted exactly that.

**Re-derived:** `sf_wdl_frac: 0.0`, `search_wdl_frac: **1.0**`. Both value
components are then lc0's own root `best_q`/`best_d`, `game_frac` is 0.00 as live's
is, and `value_blend_guard`'s bar moves 0.30 → 0.00 with it. Rejected alternatives,
recorded so the choice is falsifiable: lc0's own 0.5 search / 0.5 outcome recipe
(moves `game_frac` to 0.50 — a second deviation on the one axis this arm holds
fixed, and substitutes lc0's recipe for ours); keeping 0.70/0.30 (now a deviation
rather than a preservation); and 0.31 search with the SF share simply dropped —
which is precisely what the LEAK produces, so it would make the chosen target
indistinguishable from the failure guard 3 exists to catch.

### The LR-schedule scare, and what it actually was

`lr_T0: 999999` / `lr_T_mult: 1` looked like the most disqualifying entry — an arm
running no cosine restart against a production that restarts every 5000 steps and
doubles. **Measured: both are DEAD.** They are read only by the
`CosineAnnealingWarmRestarts` branch, and both files run `lr_schedule: sqrt_release`,
which never constructs it — the live file deletes them with exactly that comment.
The control now deletes them too, landing the same realized defaults (5000 / 2) on
both sides. The REAL schedule difference was `warmup_steps` **72 vs 1000**, which is
now live's 1000. ⚑ A budget shorter than the warmup never reaches the base LR, so a
run with `--steps <= warmup_steps` is recorded `valid_control: false` with that
reason named.

### ⚑⚑ THE CATEGORICAL HEAD: ARMED, INERT, AND THE INERTNESS IS NOT IN THE CONFIG

Live runs `rebuild_categorical_target: true` with `blend_frac 0.69 /
search_blend_frac 0.31` and `w_categorical: 1.0`. The arm now carries all of it.
Measured on real converted rows (`data/lc0_rows/training-run2-test91-20260810-1417`):

* Converted lc0 rows carry **no `categorical_target` column** (19 array keys, and
  that is not one), so `rebuild_categorical_target_in_arrays` returns the batch
  UNCHANGED and `categorical_ce` is 0.0000 over 0 rows. **It does not degenerate.**
* But the mechanism IS armed. `targets.normalize_categorical_blend_fracs` DROPS a
  component whose row is missing and does **not** redistribute its weight, so if the
  corpus ever carried a `categorical_target`, `blend_frac`'s 0.69 would land on the
  raw game outcome — the WDL leak's exact shape, on the sibling value head, with no
  error and no metric named for it. Measured: the trained-toward value moves
  `0.6900·(outcome − E[sf])` = **0.345** on a row with `E[sf] = 0.5`.

⇒ the inertness is a property of `scripts/lc0_data_to_rows.py`'s OUTPUT, not of this
config, so it is not argued in a comment. `assert_categorical_rebuild_is_inert` reads
it off every training batch alongside the WDL blend, and the run refuses with no
checkpoint if it ever stops being true.

### What this amendment does NOT change

The hypothesis, the primary yardstick, the two slopes, the 2.0 pp bar, the 0.392 pp
resolution and every guard threshold are untouched. What changes is which training
recipe the slopes describe.

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
  lc0 `best_q`/`best_d`, `sf_wdl` deliberately ABSENT. ⚑ Since AMENDMENT 3 the WDL
  target is **100% `search_wdl`** (live's `game_frac` is 0.00 and the arm holds it
  there), so the game outcome enters the value target NOWHERE. Realized and
  measured, not configured: outcome 0.0000 / search 1.0000, mass 1.0000.
- ⚑ **Four heads production trains carry weights here that train NOTHING**, because
  the corpus has no target for them: `w_soft 1.0`, `w_future 0.01`,
  `w_categorical 1.0` and `w_sf_move 0.05` all sit at live's value over 0 rows.
  They are left at production's values rather than zeroed so the arm carries two
  documented deviations instead of six undocumented ones; the driver prints each
  head's realized ROW COUNT, and a head with 0 rows trained nothing. The
  consequence for the readout is that this arm's total loss is NOT comparable in
  magnitude to production's, and the policy head carries a larger share of the
  gradient than it does in the live loop.
- ~3.7% of T91 games are chess960/DFRC and are dropped; ~1 row in 23,000 is dropped
  for the e.p.-blind repetition divergence.

## ⚑⚑ AMENDMENT 4 (2026-08-17, written BEFORE any training step): THE REPLAY BUFFER WAS NOT PRODUCTION'S, AND THE DRIVER COULD NOT PRODUCE THE PRIMARY YARDSTICK

**No arm has run and no number from this arm exists**, so this cannot be threshold
shopping. Two independent defects, both found by PR #438's review, both closed
before any step.

### 1. A THIRD AXIS OF "OUR EXACT STACK", WITH NO INSTRUMENT ON IT

AMENDMENT 2 pinned the ARCHITECTURE to the live file and AMENDMENT 3 pinned the
TRAINER. Neither can see the REPLAY BUFFER: `DiskReplayBuffer` is constructed by
`tune/trainable_init.py` from `TrialConfig` fields that `trainer_kwargs_from_config`
does not read. `scripts/lc0_control_train.py` passed three buffer kwargs by hand and
took `chess_anti_engine/replay/disk_buffer.py`'s CONSTRUCTOR DEFAULTS for the rest,
its own comment claimed two deviations, the review found three, and the measurement
against the LIVE `configs/pbt2_small.yaml` found **seven**:

| kwarg | control (was) | LIVE |
|---|---|---|
| `shuffle_cap` | 20,000 | **100,000** |
| `shard_size` | 1,000 | **2,000** |
| `refresh_interval` | 5 | **4** |
| `refresh_shards` | 3 | **5** |
| `diff_focus_pol_scale` | 0.0 | **3.5** |
| `diff_focus_q_weight` | 0.0 | **6.0** |
| `input_planes` | None | **175** |

Both existing pins passed the whole time.

⚑⚑ **WHY THIS IS A CONFOUND AND NOT A TOLERANCE.** The arm's hypothesis is
*H_stack* — "the plateau is the training stack". A hot shuffle pool 5× smaller than
production's produces a plateau of the RIG, and in the held-out top-1 slope that is
**arithmetically indistinguishable** from the plateau H_stack is about. The
`H_stack LIVE` cell of the primary table would have been unattributable, and it is
the cell this document calls the most consequential result available.

**Closure.** `chess_anti_engine/eval/lc0_control_replay.py` is a third pin on the
same instrument as the other two (live file when `$CHESS_LIVE_PRODUCTION_CONFIG`
names one, `LIVE_REPLAY_PIN` otherwise), enforced as LAUNCH guard 0d. The two
remaining deviations are declared with reasons in `LC0_REPLAY_DEVIATIONS`, both
about the corpus being FIXED rather than streaming and neither touching hot-pool
size, refresh cadence or prioritisation: `shard_recency_exponent` 1.0 → 0.0 (uniform
shard draw) and `deterministic_refresh` False → True (the draw is a pure function of
the seed). `assert_buffer_kwargs_are_classified` reads `DiskReplayBuffer.__init__`'s
own signature and refuses unless every parameter is classified, so a knob added
later is a loud failure rather than a silent eighth deviation.

### 2. THE PRIMARY YARDSTICK REQUIRED TWO CHECKPOINTS AND THE DRIVER EMITTED ONE

This document's PRIMARY section reads the arm as the joint reading of `Δ_heldout`
and `Δ_train`, "both measured LAST vs MID-BUDGET". `scripts/lc0_control_train.py`
had one `train_steps` call and one `trainer.save`, so **the deciding statistic was
not producible by the rig at all** — a gate that cannot fire, one level up from the
gates AMENDMENT 2 and 3 fixed.

⚑⚑ **AND THE OBVIOUS FIX WOULD HAVE BEEN WRONG.** Two runs, or two `train_steps`
calls of N/2, are not ONE trajectory: with `lr_schedule: sqrt_release` and
`lr_release_cycle_steps: 0` — what this arm and production both run — `train_steps`
derives the release cycle from ITS OWN `steps` argument and restarts `local_step` at
0 on every call, so two half-calls run the release ramp TWICE. MID and LAST would
sit on different LR trajectories and part of the slope between them would be a
schedule artifact. `--mid-checkpoint-frac` (default 0.5) therefore writes
`checkpoint_mid.pt` from INSIDE the single call. A run that produces no mid
checkpoint records that in `summary.json`'s `validity_problems` and is not a valid
control.

### 3. `compare` COULD REPORT GUARD 1 AS THE PRIMARY SLOPE

`scripts/lc0_control_eval.py score` banks `shuffled_targets_seed` into every
artifact — the marker that says the run is **prereg guard 1**, the permuted-target
NEGATIVE CONTROL. `cmd_compare` loaded that metadata and used exactly one field of
it (`checkpoint`, for a print). Measured on PR #438's HEAD: a B artifact carrying
`shuffled_targets_seed: 0` compared against a real-target A at n=100,000 printed
`delta +0.0400 pp, CI [+0.0123, +0.0677]` and **exited 0**. The negative control
read as the learning slope and cleared the n floor, the halfwidth bar and the
zero-discordance refusal — every gate this file has.

`compare` now REFUSES, before the arithmetic so no number reaches the screen, when
the two artifacts' `shuffled_targets_seed` differ, when either is a negative
control, or when either FAILED its own negative-control gate. The first two are
waivable by `--allow-shuffled-contrast` for the one comparison that wants them; the
third is not, because a rig that manufactures agreement supports no verdict in
either direction.

### What this does NOT change

No threshold moves. The ±0.392 pp bar, the +2.0 pp effect size, the n=100,000
resolution point, the four guards and the outcome table are all as originally
written. This amendment changes only what the rig is able to measure.

## ⚑⚑ AMENDMENT 5 (2026-08-17, written BEFORE any training step): THREE GATES THIS RIG CARRIED BUT COULD NOT ENFORCE

**Still no arm has run and no number from this arm exists.** ⚑ **NO THRESHOLD MOVES
IN THIS AMENDMENT** — the ±0.392 pp bar, the +2.0 pp effect size, the n=100,000
resolution point, the four guards and the outcome table are exactly as originally
written. It changes only what the rig REFUSES.

Found by an independent review of PR #438's own fix commits. Amendment 4 closed the
blocking defect (a permuted-target negative control read as the primary slope at exit
0) and the review confirmed that by execution — refusal before the arithmetic, empty
stdout on a refused pair, a named test that kills the sophisticated mutant. The gap it
then found is one level up and it is the same gap three times: **the fixes were
guarded and the fields the fixes depend on were not.**

### 1. THE FIELD THE WHOLE GATE KEYS ON HAD NO WRITER→READER TEST
Every provenance test hand-banked its own meta dict, so the reader was tested in
complete isolation from the writer. Renaming the banked key on the WRITER side only
(`shuffled_target_seed`) left all 61 tests green while every real
`score --shuffle-targets` artifact read back `None` → labelled "real lc0 targets" →
zero provenance problems → **the negative control prints the slope at exit 0.** The
defect Amendment 4 closed, reopened by one token, with a green suite. Closed by
asserting the field on an artifact the real `cmd_score` wrote.

### 2. THE "NOT WAIVABLE" REFUSAL WAS WAIVABLE FROM THE SCORE SIDE
`compare` reads the negative-control bar off the artifact, which is right — re-judging
a banked z against whatever the module constant is at read time answers a question
nobody asked. But `--negative-control-z` had no ceiling and `score` banked whatever it
was handed, so `score --shuffle-targets --negative-control-z 1e9` writes an artifact
whose control sits **41.7σ above the floor** and is recorded as PASSING; `compare`
then prints the pre-fix slope at exit 0. Measured end-to-end by the reviewer.
**A bar banked by the run under test is a CLAIM, not a measurement**, so a run gated
more leniently than `NEGATIVE_CONTROL_Z` is now its own refusal. And the passing
readout now PRINTS the z and the bar, per this rig's own standard: a gate that
stopped running looks identical to a gate that ran and passed.

### 3. `valid_control: false` NEVER REACHED THE SCORE ARTIFACT
`lc0_control_train.py` disqualifies a run launched with `--allow-arch-drift` ("this is
NOT production's architecture"), `--allow-leak`, no purity receipt, no mid checkpoint,
or `--steps` under `warmup_steps` — and stamps `valid_control: false` into
summary.json. `cmd_score` banked twelve meta keys and that was not one of them, so a
checkpoint **the driver itself disqualified** scored clean and reported as the primary
slope at exit 0. Amendment 4 established the pattern (bank the provenance, refuse
before the arithmetic) and stopped at the field it had just created.

`score --summary <run>/summary.json` now banks it (defaulting to the checkpoint's own
directory), and `compare` refuses `valid_control: false` unwaivably. **An artifact with
NO validity record is also refused**, waivable by its own `--allow-unrecorded-validity`
— not by `--allow-shuffled-contrast`, because a waiver that clears more than its name
says is how a gate becomes a decoration.

⇒ **Operational consequence for this arm:** every `score` invocation must pass
`--summary`, and any `compare` whose output is quoted must show `neg-control:` on both
lines and must not have needed `--allow-unrecorded-validity`. A readout carrying that
banner is not a verdict.

## ⚑⚑ AMENDMENT 6 (2026-08-17, written BEFORE any training step): SIX SETTINGS THE RIG ACCEPTED AND DID NOT RECORD

**Still no arm has run and no number from this arm exists.** ⚑ **NO THRESHOLD MOVES IN
THIS AMENDMENT** — the ±0.392 pp bar, the +2.0 pp effect size, the n=100,000 resolution
point, the four guards and the outcome table are exactly as originally written. It
changes only what the rig RECORDS and REFUSES.

Found by a second independent review of PR #438. All six are one shape: **a value is
accepted, it deviates from the preregistered or production one, and the run still stamps
`valid_control: true`.** Five are cured by the machinery Amendments 4 and 5 built —
`validity_problems`, banked into the score artifact, refused before the arithmetic — and
the sixth is a real production-code arithmetic error.

1. **`--batch-size`** was assigned after every preflight, and `batch_size` is not a
   trainer kwarg, so guard 0c is structurally blind to it: a run at 32 against
   production's configured 512 changed the examples per step, i.e. the gradient-noise
   regime, and read VALID. ⇒ recorded in `validity_problems`.
2. **`--mid-checkpoint-frac`** clamps ANY positive value to an interior step, so `0.99`
   wrote a checkpoint, satisfied the "no mid-budget checkpoint" entry Amendment 4 added,
   and presented a LAST-vs-99%-budget contrast — the final 1% of training — as the
   preregistered LAST vs MID-BUDGET slope. ⇒ any fraction other than **0.5** is recorded.
3. **A `--shards` directory named twice** was staged twice under distinct
   `shard_NNNNNN.zarr` symlinks, so the buffer oversampled that hour — invisible to the
   coverage preflight (which SUMS over the list) and to the purity receipt (which
   compares SETS). ⇒ **REFUSED** at launch on resolved paths; de-duplicating silently
   would train on a corpus other than the one named.
4. **`summary.json` vouched for any checkpoint.** `valid_control` is a verdict about a
   RUN and the artifact named no FILE, so `score --summary <valid run>/summary.json
   --checkpoint <other run>/checkpoint.pt` banked `valid_control: true` for a checkpoint
   that summary had never seen — including a disqualified one — and `compare` could pair
   two trajectories' LAST checkpoints. ⇒ the run banks a content-derived `run_id` and a
   sha256 + role per checkpoint; `score` hashes the `--checkpoint` and REFUSES a summary
   that does not name it; `compare` refuses a `run_id` MISMATCH and a role pair that is
   not {mid, last} — **unwaivable, and judged on whichever artifacts carry the field** —
   and refuses an artifact with no identity at all (waivable by
   `--allow-unverified-trajectory`, which concedes only that the MISSING field's binding
   is unverified).
5. **The held-out POPULATION was ungated.** This document names it as "the LAST 6 hourly
   tars by wall-clock"; `freeze` gated the row COUNT at 100,000 and accepted any number
   of source directories, and a different number of hours carries different temporal
   correlation. ⇒ `freeze` REFUSES unless there are exactly six **distinct resolved
   paths** (a repeated directory is refused with no flag at all), or
   `--allow-source-selection` stamps the artifact; the frozen set banks resolved
   `source_paths` rather than basenames, so "the LAST 6 hourly tars by wall-clock" is
   auditable from the artifact; `score` RECOMPUTES the population from the frozen
   artifact (not from the stamp — a set predating the stamp would otherwise read as
   preregistered) and `compare` refuses it, waivable by `--allow-non-prereg-heldout`,
   with an ABSENT record refused separately under `--allow-unrecorded-heldout`.
6. **A production-code arithmetic error, on a path production does not visit.**
   `Trainer._warn_if_value_blend_leaks_to_outcome` multiplied the label shortfall by the
   RAW `sf_wdl_frac`/`search_wdl_frac` attributes while `compute_loss` renormalises them
   through `normalize_value_blend_fracs`. With `0.8`/`0.8` and 50% effective SF coverage
   the objective realizes a **0.25** leak and the warning reported **0.40** — 1.6×, enough
   to fire the 0.01 incident bar on a leak the trained objective never had. ⇒ the same
   helper, before the arithmetic.
   ⚑⚑ **BUT IT IS LATENT ON TODAY'S PRODUCTION CONFIG, AND AN EARLIER REVISION OF THIS
   LINE OVERSTATED THAT BY LABELLING IT "PRODUCTION CODE" WITHOUT QUALIFICATION.**
   `normalize_value_blend_fracs` only renormalises when `sf_wdl_frac + search_wdl_frac >
   1`, and the live `configs/pbt2_small.yaml` cannot reach it: `0.69 + 0.31` is **exactly
   1.0** in IEEE754 (measured, not assumed) and `sf_wdl_frac_floor == sf_wdl_frac ==
   0.69`, so `_dynamic_sf_wdl_weight` interpolates 0.69→0.69 and the PID ramp cannot lift
   the sum. **No live TB series carries a 1.6× error and nobody should go looking for
   one.** The states that reach it are a live-yaml edit or PB2 mutation of either frac
   (both are `TRAINER_WEIGHT_KEYS`, pushed onto the running trainer every iteration, and
   neither has a validator — CLAUDE.md category (c), so `search_wdl_frac: 0.8` lands
   silently) and this control arm's own oversubscribed configs, which is where the
   regression test lives. Found by the independent review of the fix wave; the remedy is
   the honest label, **not** a change to the blend, which is load-bearing.

Every waiver above is its OWN flag, and each refusal now carries the name of the one flag
that clears it as a FIELD (`ProvenanceProblem.waiver`) rather than as a substring of its
message — Amendment 5's `--allow-unrecorded-validity` scoped itself by matching prose,
which makes a waiver's reach a property of wording. ⚑ A waiver-tagged refusal is still
only as narrow as its CONTROL FLOW: the first version of finding 4's gate chained the
three identity checks as `if/elif/elif`, so the one waivable branch shadowed the two
unwaivable ones and the flag cleared exactly what its help text promised it could not.
Naming the waiver is necessary and not sufficient; the branches must be independent.

⚑ **`validity_problems` IS NOT A SOFT RECORD.** `valid_control = not
validity_problems` and `compare` refuses `valid_control: false` with **no waiver**, so
every entry — `--batch-size`, the mid fraction, `--allow-leak`, `--allow-arch-drift`, no
purity receipt, sub-warmup budgets — is terminal for the comparison. The choice these
entries make is LAUNCH vs ARTIFACT: the run finishes and writes its checkpoints (which is
what a plumbing smoke needs) and is then disqualified from supplying either side of the
primary readout. The driver's comments called this "recorded rather than refused", which
reads as "still comparable"; it is not, and the wording is corrected in the code.

⇒ **Operational consequence for this arm:** the deciding `compare` must be run on
artifacts whose `run_id` matches and whose roles are `mid` and `last`, from a `freeze`
that needed no `--allow-source-selection`, at the config's own `batch_size`, with the mid
checkpoint landing on the realized 0.5 of the budget (the knob is clamped to an interior
step, so `--steps 3 --mid-checkpoint-frac 0.5` lands at 33% and is recorded). Any of the
five new banners in the output means the readout is not the preregistered yardstick.

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
