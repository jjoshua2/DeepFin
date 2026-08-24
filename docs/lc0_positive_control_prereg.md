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

### Amendment 3 (2026-08-20): frozen held-out is **v2** — the purity gate fired on v1

Window 2's in-window `purity` measured **4,970 of 96,811 distinct v1 held-out inputs (5.1337%)
occurring in the train corpus** (record-level intersection 32 — the under-report the tool
warns about). The gate fail-fasted the window before any training compute; no model read
existed to select on, so rebuilding the split is identity-only, the same argument class as the
seed-1 re-freeze. Remedy = the gate's own `subtract`:

- **`data/lc0_control_heldout_frozen_v2.json` — 91,842 rows / 91,841 distinct x**, sha256
  `80ab2a7bc192c1356e249f3c0648b6a587ab4ef97f4250c0cc95bb057b2da2b6`; purity on v2 = PURE
  (0 exposed inputs, 0 record intersection over all 78,531,074 train rows).
- Resolution restated at n=91,842: the bar scales by sqrt(100000/91842) = 1.0435, so
  **±0.392 pp → ±0.409 pp** everywhere the verdict table reads it. The +2.0 pp material
  threshold is unchanged (still ~10× resolution).
- v1 (`data/lc0_control_heldout_frozen.json`, sha `b6f47be8…`) is retired and must not be
  scored against; the banked exposed ids are `data/lc0_control_exposed_20260820.json`.

Full narrative: experiment ledger, "window 2: the purity gate FIRED" (live branch).

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

### FROZEN STEP BUDGET — recorded 2026-08-19, before any `--steps` was passed

| | windows (×88) | `--steps` | sampled examples | corpus-size exposure |
|---|---|---|---|---|
| **initial** | 438 (MID at 219) | **38,544** | 19,734,528 | **0.25130×** |
| **single preregistered 2× extension** | 876 (MID at 438) | **77,088** | 39,469,056 | **0.50259×** |

Denominator is the **realized** training-corpus row count, **78,531,074**, summed
from every shard's `.zarray` shape across all 9,653 shards and cross-checked on
three independent arrays. The nominal `9653 × 8192 = 79,077,376` overstates by
0.69% (9,531 full shards + 122 partial) and must not be used.

**⚑ These are nominal corpus-size SAMPLE-EXPOSURE equivalents. They are NOT
epochs and NOT a unique-row fraction, and no claim about memorisation may be
built on them.** `DiskReplayBuffer` samples WITH REPLACEMENT on both halves —
uniform at `replay/disk_buffer.py:1666`, priority at `:1674`/`:1678`/`:1706`,
split `surprise_mix = 0.5` — drawing from a 100,000-row hot pool refreshed 5
shards every 4 batches, so rows repeat long before one corpus-equivalent
elapses. Independently, the `Δ_train` readout scores rows the net has already
trained on, and one exposure suffices to fit a row; no sub-epoch budget makes
`Δ_train` vs `Δ_heldout` divergence immune to memorisation. That is a property
of the verdict table, which is why the table reads the two slopes JOINTLY
rather than treating either alone as evidence.

**What justifies this budget:**
1. It is exactly 438 whole `--train-window-steps` windows, so MID lands on a
   window boundary (219) and MID and LAST are read at the SAME phase of the
   `sqrt_release` LR ramp. Reading them at different phases was a real defect
   in an earlier revision of this driver: `lr_release_min_scale: 0.1` means the
   end-of-window LR is 10× below base, so a LAST read at the trough against a
   MID read at full base compares two different training regimes.
2. ~38.5k optimizer steps is far past the 1,000-step `warmup_steps`, which the
   prereg elsewhere guards against under-running.
3. It stays below a single corpus-size exposure equivalent, which BOUNDS how
   much sampler reuse can have accumulated without claiming there is none.

**⚑ The budget is frozen. `--steps` takes 38,544 and nothing else.** The single
2× extension to 77,088 is available only for the CONVERGED/INCONCLUSIVE and
AMBIGUOUS rows of the table above. There is no second extension.

2.0 pp is ~10× the resolution, deliberately far above the noise floor: a stack
genuinely learning from ~20M sampled positions drawn from a 78.5M-row external
corpus should not be arguing in the third decimal. (An earlier draft said "87M
fresh positions". Both halves were wrong: the corpus is 78,531,074 realized
rows, not 87M or the nominal 79.08M, and nothing about the run is "fresh" —
see the frozen budget below.) Both slopes use the same n and the same estimator so the ±0.392 pp bar
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

## ⚑⚑ AMENDMENT 7 (2026-08-17, written BEFORE any training step): THE RIG COULD NOT PRODUCE THE NUMBER IT EXISTS TO PRODUCE

**Still no arm has run and no number from this arm exists.** ⚑ **NO THRESHOLD MOVES IN
THIS AMENDMENT** — the ±0.392 pp bar, the +2.0 pp effect size, the n=100,000 resolution
point, the four guards and the outcome table are exactly as originally written. ⚑ Launch
readiness is recorded separately and is NOT claimed here.

From a second independent review (of Amendment 6's own fix wave) plus five Codex threads.
Two items are Amendment 6's fixes over-correcting, which is the failure mode
`fixing_a_defect_class_reintroduces_it` names, and one destroyed data.

1. **The mid-budget guard's `!=` made every ODD budget permanently invalid.** Amendment 6
   moved the guard from the knob to the realized fraction, correctly, and compared with
   exact inequality — which `mid_step = int(0.5 × steps)` can only satisfy for EVEN
   budgets. `--steps 20001` realizes 0.499975, records a validity problem, and — because
   Amendment 6 deliberately made `valid_control: false` unwaivable — **a day of GPU would
   have been permanently unquotable over a 0.0025% discrepancy.** ⇒ the comparison is now
   tolerant, `MID_CHECKPOINT_FRAC_TOLERANCE = 0.01`, chosen so that (a) truncation is
   admitted for every budget ≥ 50 steps, (b) every deliberate misplacement still fires
   (`0.99` → 0.75, `--steps 3` → 0.333), and (c) the only regime where the bound is
   tighter than truncation is under 50 steps, which the `warmup_steps` entry already
   disqualifies — so **this entry can never be the sole reason a run is refused**, which
   is a test rather than a claim.
2. **A failed rerun into a populated `--out-dir` irreversibly deleted the previous
   completed run's `checkpoint_mid.pt`** — a day of GPU in production — while that run's
   `summary.json` went on banking the file's path, step and sha256 and its surviving
   `checkpoint.pt` still verified as `role: last`. The directory was left reading as a
   scorable success with half the deciding statistic gone, and the message "no checkpoint
   written (including the mid-budget one)" was FALSE about the directory. ⇒ a populated
   `--out-dir` is **REFUSED**; the escape (`--move-existing-aside`) **RENAMES** to
   `<out-dir>.superseded_<UTC>`. There is deliberately **no `--overwrite`** and no new
   delete path: a rename is reversible by hand, a delete is not.
3. **The deciding slope contained an LR anneal production never has.** With
   `lr_schedule: sqrt_release` and `lr_release_cycle_steps: 0`, `train_steps` derives the
   release cycle from ITS OWN `steps` argument, so the driver's single call held LR flat
   for `lr_release_start_frac` (0.80) of the **entire experiment** and annealed once: MID
   at 50% of budget sat at full base LR and LAST at `lr_release_min_scale` (0.1×) after a
   full anneal. Production calls `train_steps` once per ITERATION at ~88 steps, i.e. a
   sawtooth, and never places an anneal between two of its own checkpoints. An annealed
   endpoint scores better on held-out top-1 for reasons that are not the trainer learning
   — in the flattering direction for H_stack. The control yaml **documented** this;
   documenting a first-order confound in the primary yardstick is not controlling it. ⇒
   the driver now runs the budget in **windows of `--train-window-steps` (default 88,
   production's REALIZED steps/iteration — it is not a yaml key, because views-targeting
   derives it from ingest volume)**, so MID and LAST both land on a window boundary, i.e.
   at the same phase of the release cycle. A budget that is not an even number of windows
   records the deviation. ⚑ This is NOT the split Amendment 4 warned about: two calls of
   N/2 put MID mid-ramp and LAST at a bottom, while W equal windows put both at a bottom.
4. **The cluster key is banked; the clustered estimator is NOT.** Many plies share a
   `game_id`, so the per-row hit indicators are correlated within a game and both the
   reported CI and `MATERIAL_BAR_PP` (derived at n=100,000 **under independence**) are
   optimistic by the design effect. Estimating that effect needs the real corpus's
   plies-per-game distribution and is **deferred**; the KEY is banked NOW — in the frozen
   artifact and carried into every score npz — because it is free before the freeze and
   **unrecoverable after it**, and that asymmetry is the whole argument. `compare` prints
   the cluster count and states that its CI is row-level and optimistic. ⚑ The purity
   module rejects `game_id` as a CROSS-SPLIT instrument (it is `enumerate()`'s index over
   one conversion, so cross-split ids are disjoint by construction); as a WITHIN-set
   cluster key scoped by source directory it is valid, and the scoping matters — unscoped,
   two hours' game 0 would merge and UNDERSTATE the correlation.
5. **`--device` is now disqualifying and `--seed` is banked.** The comment deferring the
   device gate reasoned that "an entry here would disqualify every smoke"; measured, that
   is false — a clean 4-step CPU smoke already banks four entries — so the gate costs
   nothing and its absence left `realized_after_guard` banked-and-unread by every
   consumer. `--seed` seeds `torch.manual_seed` before `build_model` and the replay RNG
   and appeared in no artifact: `run_id` is content-derived, so it identifies a trajectory
   but cannot reproduce one, and the replay deviation `deterministic_refresh: true` is
   justified as "a pure function of the seed".
6. Two smaller ones: a legacy frozen artifact's BASENAME sources were being resolved
   against the **reader's** cwd (so a duplicate message named a path unrelated to the
   set), and `stage_shards` cleared only symlinks, so a non-symlink left in
   `staged_shards` would join the sampled corpus while `summary.json` named only
   `--shards`. The latter is now a refusal, not a deletion.

⇒ **Operational consequence for this arm, added to Amendment 6's list:** `--steps` must be
an even multiple of `--train-window-steps`, `--out-dir` must be fresh, and any quoted
readout must show the `game clusters` line — whose stated caveat (the CI is row-level) is
in force until the clustered estimator lands.

## ⚑⚑ AMENDMENT 8 (2026-08-17, written BEFORE any training step): AMENDMENT 7's OWN FIXES MADE THE ARM'S BUDGET UNLAUNCHABLE

**Still no arm has run and no number from this arm exists.** ⚑ **NO THRESHOLD MOVES IN
THIS AMENDMENT** — the ±0.392 pp bar, the +2.0 pp effect size, the n=100,000 resolution
point, the four guards and the outcome table are exactly as originally written. ⚑ Launch
readiness is recorded separately and is NOT claimed here.

From a THIRD independent review (of Amendment 7's fix wave) plus four Codex threads. The
headline item is the second consecutive instance of `fixing_a_defect_class_reintroduces_it`
on the SAME entry, which is why the fix this time is a change of KIND and not of arithmetic.

1. **Amendment 7 fixed item 1's arithmetic and left its OUTCOME — with a trigger set ~88×
   WIDER.** The mid-fraction tolerance admitted odd budgets, and then Amendment 7 item 3's
   cadence rule required `steps % (2 × window) == 0`: at the default window 88 that admits
   only multiples of **176**, i.e. it refused 175 of every 176 budgets **including this
   arm's own 20,000**, unwaivably, after the budget had been spent. The same failure as
   item 1, relocated one entry over. ⇒ three changes:
   - **the divisibility requirement is GONE.** The window count is a CEILING and the final
     window is SHORT, so no budget is refused and no step is discarded (Codex 3796113403:
     the floor plan trained `steps // window × window` steps while `summary["steps"]` kept
     reporting the request). A short final window still ends at the release-cycle bottom —
     `_scale_for_window_step` returns `min_scale` at `local_step == cycle_steps - 1` for ANY
     cycle length, measured, not argued.
   - **MID is SNAPPED to a window boundary** rather than being checked against a fraction
     band. This is the property the cadence fix was always for; see item 2.
   - **every disqualification is now evaluated BEFORE the first optimizer step and REFUSES
     the launch**, with `--allow-invalid-control` for plumbing smokes. Every entry in
     `validity_problems` is knowable from the arguments, the config and the plan, and twice
     a full budget ran only to be stamped `valid_control: false` — which `compare` refuses
     with no waiver. The class "a day of compute buys an artifact that cannot be quoted for
     a reason the command line already contained" is now closed rather than moved. The
     launch check and the artifact come from ONE function, so they cannot disagree.
   - ⚑ **What this trades:** a boundary can miss 0.5 by up to half a window, so the
     preregistered mid point is unreachable below `min_budget_for_mid_tolerance(88)` =
     **4400 steps** — above production's 1000-step warmup, so unlike Amendment 7's version
     this entry CAN now be a run's sole disqualification. Budgets in that band are refused
     **at launch** with the floor named in the message. The tolerance is NOT widened for
     short budgets: that would trade the prereg's mid point for the convenience of a run
     nobody needs (this arm runs 17,600–20,000).
2. **Both mid-point guards bounded PROXIES, and the property itself was unguarded.** At
   17,600 steps the ±0.01 fraction band is ±2 whole 88-step windows, so
   `--mid-checkpoint-frac 0.5028` put MID at LR scale **1.0** against LAST at **0.1** with
   both guards silent and `valid_control: true` — the exact anneal-in-the-slope confound
   Amendment 7 item 3 closed, reachable through a PASSING knob. ⇒ MID is snapped to a
   window boundary at the source, so MID and LAST are at the same phase of the release
   cycle by construction, and a second entry reports any future path that bypasses the snap.
3. **`cluster_keys_complete` was banked and read by NOTHING** (Amendment 7 item 4's own
   field). A key set with holes is worse than none: unkeyed rows collapse into one dropped
   empty cluster, so `distinct_clusters` UNDERSTATES the clustering and the `rows/cluster`
   figure overstates the cluster size — a number that looks like a stronger design-effect
   statement while being an artifact of the missing keys. ⇒ `compare` REFUSES a partial pair
   under its own `--allow-partial-clusters`, and a partial set never prints a ratio.
   Relatedly, `game_id` is an OPTIONAL shard field: the key builder read the column without
   its `has_game_id` mask, so an unset row took the int64 fill value and was banked as
   `<source>#0` — merged into game 0's cluster while `cluster_keys_complete` said True.
4. **The outcome bar was a constant in this tree, and the two keys it is made of are the
   two guard 0c must IGNORE.** `PRODUCTION_GAME_FRAC = 0.0` is correct for live's
   `0.69 / 0.31`, and both fracs are `LC0_TRAINER_DEVIATIONS` entries by necessity. A
   production move to, say, `0.50 / 0.20` leaves 0.30 of live's value target on the raw game
   outcome while this arm holds 0.00: guard 0c is silent by construction, and the arm would
   be stamped valid while no longer training production's objective. ⇒ the bar is DERIVED
   from the same reference signature the launch preflight judged against
   (`live_production_game_frac`), passed to the realized guard explicitly, banked in the
   artifact next to the pinned constant, and a divergence between the two is a launch
   refusal that names all three things to refresh together.
5. **The prereg's TWO deltas were indistinguishable in the artifact.** `score` classified
   every artifact by the HELD-OUT six-source rule and banked no statement of which
   population it scored, so a `Δ_train` sample spanning six directories banked an empty
   problem list and could be presented as `Δ_heldout`, while an honest train sample spanning
   more directories needed a MISLEADING held-out waiver to compare at all. ⇒ `score
   --population {heldout,train}` is banked, the six-source rule applies only to the held-out
   comparison, and `compare --population` must match what the artifacts record — a mismatch
   is **unwaivable**, an absent role has its own waiver.
6. **A refused comparison printed the slope anyway.** The resolution refusals (n below the
   prereg point, halfwidth over the bar, zero discordance) were evaluated AFTER the rates,
   delta, CI and p-value were on stdout — and a pasted readout that drops stderr and the
   exit code is how these numbers travel. ⇒ the resolution gate is built and checked before
   any result line, exactly as the provenance gate already was.
7. **Two Codex arithmetic items.** `--max-halfwidth-pp nan` was accepted, switched the
   n=100,000 floor off and compared FALSE against every finite halfwidth — both gates off,
   exit 0, slope printed; the override must now be finite and positive, and it waives the n
   floor only when it is at least as STRICT as the material bar (a looser bar is a
   relaxation, not a re-derivation). And the exact McNemar tail enumerated arbitrary-
   precision binomial coefficients, taking tens of seconds at 20,000 discordant pairs and
   minutes at 100,000 — i.e. slowest on the prereg's own readout; it is now summed in log
   space above 1,000 pairs, with the exact-integer path kept below that so the pinned value
   stays bit-identical.
8. **Sampling.** `_stratified_sample` gave each source an approximately EQUAL quota while
   its docstring said proportional, so on six hours of unequal size the small hours were
   overrepresented and the result depended on source ORDER. Now largest-remainder
   proportional, exact and order-independent.

⇒ **Operational consequence for this arm, REPLACING Amendment 7's line:** `--steps` no
longer needs to divide anything — any budget ≥ 4400 steps at the default window is accepted
and every requested step is trained — but the driver now **refuses to launch** unless the
run is a valid control, so `--out-dir` must be fresh, the purity receipt and the live config
must be supplied, and `--device`/`--batch-size` must match the config. A plumbing smoke
passes `--allow-invalid-control` and its artifact remains unquotable. Any quoted readout
must still show the `game clusters` line, and a `PARTIAL` there is a refusal rather than a
caveat.

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

## ⚑⚑ AMENDMENT 4 (2026-08-21, written AFTER training completed but BEFORE any score existed): the readout rig had two wiring defects and one missing input; the `Δ_train` sample is pinned here, with its dilution stated before the read

The 38,544-step run completed 08:14 with every launch/leak guard green. The first
scoring call then refused — `91842 of 91842 frozen rows were not found in the given
shards` — which is the n-shrink guard doing its job. No score, compare, or slope was
produced before this amendment; nothing here is conditioned on a number.

**Defects found (all in the staged runner script, none in the rig's gates):**

1. **The runner passed only the TRAIN corpus to `score --shards`.** The frozen
   held-out rows live in the six held-out hourly dirs (`data/lc0_rows_heldout/*/`,
   490 shards), disjoint from the 122 train dirs by construction. Fixed: held-out
   legs (seed band, negative control, MID/LAST held-out) scan the held-out dirs
   ONLY; train legs scan the train corpus ONLY. Scoping is also the cheap
   direction: `score` scans every shard it is given.
2. **The runner's `--population train` legs pointed at the HELD-OUT frozen file.**
   They would have scored the held-out rows twice under a train label — the exact
   mislabeling `--population` was added to prevent, one level up. Fixed: the train
   legs read the train-side frozen artifact below.
3. **The `Δ_train` artifact did not exist.** The prereg requires "a frozen, equally
   sized sample of ROWS ALREADY TRAINED ON" and no train-side freeze was ever run.

**Pinned `Δ_train` construction (run from the pinned worktree):**

```
python3 scripts/lc0_control_heldout.py freeze \
    --shards <the 122 train dirs> \
    --out data/lc0_control_train_frozen_v1.json \
    --sample 91842 --seed 2 --allow-source-selection
```

`--sample 91842` matches the held-out n exactly ("equally sized"); `--seed 2` is
pinned here before the draw (held-out v1 used 1; 0 is refused by the tool);
`--allow-source-selection` is the deliberate train-side stamp — the six-source rule
is a held-out property, and `score --population train` / `compare --population
train` are the consumers that read the stamp in that role.

**⚑⚑ DILUTION, stated before the read, thresholds UNCHANGED.** The budget realized
0.25130× corpus-size sample exposure WITH replacement, so the expected fraction of a
uniform train-corpus sample that was drawn at least once during training is
≈ 1 − e^(−0.2513) ≈ **22.2%** (an upper bound: priority sampling concentrates draws,
lowering the unique fraction). The verdict table's `Δ_train ≥ +2.0 pp` cell was
written for "rows already trained on"; on this uniform sample the same true
per-trained-row effect appears diluted ≈ 4.5×. Pre-committed consequences:

- The table's thresholds are NOT rescaled. A post-hoc division of `Δ_train` by the
  22.2% estimate may be used as interpretation COMMENTARY only, never as a verdict
  input.
- If `Δ_heldout` reads flat and `Δ_train` lands in (0.409, 2.0) pp — a plausible
  diluted H_stack signature — the verdict row is "anything else" → **AMBIGUOUS** →
  the single preregistered 2× extension, which also raises exposure to 0.503×
  (expected trained fraction ≈ 39.5%), sharpening this same instrument without
  changing it.
- Enumerating the actually-drawn rows by replaying the sampler was considered and
  rejected: the priority half's draw depends on data values, so a faithful replay
  is a second full pass of the training loader, and a cheaper index-only replay
  would enumerate a DIFFERENT population than the one trained on.

**Also corrected, same session, before execution:** the two non-readout baseline
commands (deep-SF value baseline, sigma probe) pointed at the salvage export root;
the trainer file lives at `seeds/slot_000/trainer.pt` inside it (identity confirmed
from the export manifest: `picked_row_training_iteration: 595`,
`checkpoint_000594`). Path fix only; no parameter changes.
