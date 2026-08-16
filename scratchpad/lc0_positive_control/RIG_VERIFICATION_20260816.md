# lc0 positive control — RIG VERIFICATION (2026-08-16)

**Status: PLUMBING ONLY. The arm was NOT run and no conclusion about the RL loop is
drawn or licensed here.** Everything below is a property of the apparatus, measured
at smoke scale. The prereg is `PREREG_DRAFT.md` in this directory; it is unchanged.

Companion to that prereg, which requires four guards to pass before its primary
readout may be read. Three of them are apparatus questions and are answered here.

---

## 1. ⚑⚑ The `sf_wdl` fallback — VERIFIED IN EFFECT, not configured

### The defect, read off `train/losses.py` directly

`compute_loss` builds the value blend as
`game_frac * outcome + sf_wdl_frac * sf_component + search_wdl_frac * search_component`,
and

```python
if sf_wdl_probs is not None:
    sf_component = sf_effective_b * sf_wdl_probs + (1.0 - sf_effective_b) * blend_fallback_target
else:
    sf_component = blend_fallback_target        # blend_fallback_target = game_oh
```

with `sf_effective = has_sf_wdl * keep`. lc0 shards carry neither `sf_wdl` nor
`has_sf_wdl`, and `_get_mask` defaults an absent mask to **0.0**, so BOTH branches
resolve the SF component to the raw one-hot game outcome for every row. No error, no
warning, no metric named for it.

### Measured, through the production loss, on real converted lc0 rows

The value target `compute_loss` actually built was intercepted at its
`soft_cross_entropy` call and decomposed back into component shares by least squares
(`tests/test_value_blend_guard.py`).

| config | intended `game_frac` | realized outcome share | search share | target sums to |
|---|---|---|---|---|
| production blend (`sf 0.50 / search 0.20`) on lc0 rows | 0.30 | **0.80** | 0.20 | 1.000 |
| this arm (`sf 0.00 / search 0.70`) on lc0 rows | 0.30 | **0.30** | 0.70 | 1.000 |
| production blend on SF-LABELLED rows | 0.30 | 0.30 | 0.20 | 1.000 |

⇒ at production settings **0.50 of the value target — 62.5% of all outcome-borne
mass — silently relocates onto the deep game outcome**, and the third row shows this
is a property of the DATA, not of the weights, so nothing about production changes.

⚑ The `target sums to` column is load-bearing. An earlier version of this
measurement computed the outcome share as `1 - search_share`, which assumes the
components still sum to 1 — and a mutation that deleted the fallback branch entirely
(`sf_component = zeros`) PASSED that test, because losing 0.50 of the mass and moving
0.50 of the mass read identically. Reported here because the same shortcut is
available to anyone re-deriving this.

⚑⚑ **CORRECTION (independent review, 2026-08-16).** This section previously credited
the `sums to 1.0` assertion with killing that mutant. It did not: `_decompose`'s
caller asserted `shares["outcome"] == 0.80` first, so the mass line was never
reached. The reviewer's diagnosis — "the least-squares solve is the killer" — is
right about *which* assertion fired. But the further claim that the mass line is
therefore *redundant with* `residual < 1e-5` is wrong, and the distinction matters:
dropping the SF component leaves the target at `0.30·one_hot + 0.20·search`, which is
exactly in the span of the remaining basis, so the residual is ~0 while the mass is
0.50. Residual cannot see lost mass; the shares cannot distinguish lost from moved.
The assertion was SHADOWED, not dead. Fixed by moving the mass check to the TOP of
`_decompose`, before any share is returned, so each assertion owns a defect the other
is blind to.

### Realized weights off an actual training step

Read by wrapping `trainer.compute_loss` and recording the kwargs it was CALLED with
(not `trainer.sf_wdl_frac`, which is an attribute a live reload can change between
the read and the step). 8 steps, batch 512, `configs/lc0_positive_control.yaml`:

```
sf_wdl_frac (realized)                 0.000000
search_wdl_frac (realized)             0.700000
game_frac (intended outcome share)     0.300000
sf_labelled_frac (rows with sf_wdl)    0.000000
leaked_to_outcome                      0.000000
outcome_borne_frac (game_frac + leak)  0.300000
```

Same wrapper on a config restored to the production blend (`--allow-leak`, real lc0
shards):

```
sf_wdl_frac (realized)                 0.500000
search_wdl_frac (realized)             0.200000
sf_labelled_frac (rows with sf_wdl)    0.000000
leaked_to_outcome                      0.500000
outcome_borne_frac (game_frac + leak)  0.800000
```

### ⚑ Is `sf_wdl_frac` PID-driven? — VERIFIED AGAINST THE CONTROLLER

`tune/trainable_config_ops.py:1406` recomputes the weight EVERY iteration from
`trainable_metrics._dynamic_sf_wdl_weight(sf_wdl_start=tc.sf_wdl_frac, ...)` and
assigns `trainer.sf_wdl_frac` whenever the result is not `None`. That function
returns `None` iff `sf_wdl_start <= 0`.

Confirmed by calling the real function, not by reading its docstring: swept over
regret `{0.0, 0.0075, 0.05, 0.10, 0.35, 0.70, 1.0, 5.0}` at `sf_wdl_start=0.0` it
returns `None` at every point, and the same sweep at `sf_wdl_start=0.50` returns a
weight — so the `None` result is not the function being inert. Mutating
`if sf_wdl_start <= 0` to `if sf_wdl_start < -1` fails the test.

⇒ **`sf_wdl_frac: 0.0` disables the ramp at its source; the override cannot expire
mid-run.** `sf_wdl_frac_floor` is never read while the start is 0 and is zeroed
anyway. (This arm does not run the tune trainable at all, so the ramp is doubly
unreachable — the assertion does not rely on that.)

### The guard, and where it lives

`chess_anti_engine/train/value_blend_guard.py`. Three checks at three levels, in
`scripts/lc0_control_train.py`:

0. **launch, architecture** — `assert_control_matches_live_architecture` (added
   2026-08-16, review F1). See §6.1.
1. **launch, config** — `assert_pid_cannot_reassert_sf_wdl`.
2. **launch, corpus** — the converter's own `run_config_problems` is IMPORTED and
   reused, not restated, driven by MEASURED SF-label coverage
   (`shard_dir_sf_wdl_coverage`) rather than `any()`. See §6.4.
3. **realized, per-step** — the wrapped `compute_loss` kwargs plus the batch's own
   `sf_wdl_rows`, judged by `assert_no_silent_outcome_fallback` at `max_leak=0.0`
   AND `max_outcome_borne=0.30`. Judged on the WORST step, not the first or the
   mean, so a leak that begins partway through a long run cannot be diluted under
   the bar. ⚑ `--allow-leak` no longer skips it — see §6.2.

The blend arithmetic is NOT duplicated: `losses.normalize_value_blend_fracs` was
extracted from `compute_loss` and is called by both, so the guard cannot drift from
the criterion it checks.

**No RAISE wired into `Trainer`/`tune`, deliberately — but a LOG-ONLY readout now
is.** A hard raise inside `train_steps` would be a new fatal path on the production
trial (CLAUDE.md category (b): the iteration loop has a `finally:` and zero
`except`, so it kills the trial), for a condition production cannot reach — 712/713
live shards carry `has_sf_wdl`. The reviewer agreed with declining the raise and
recommended the middle path, which is now implemented: `TrainMetrics` publishes
`sf_wdl_labelled_frac` (`sf_wdl_rows / batch_rows`, row-weighted), so
`leaked_to_outcome = sf_wdl_frac × (1 − sf_wdl_labelled_frac)` is computable from
TB, and `Trainer._warn_if_value_blend_leaks_to_outcome` logs a warning above a 0.01
leak. No raise, no new fatal path — and the 2026-05 realized-`sf_wdl_frac`-0.45
episode would have been a TB series instead of a reconstruction.

### Mutation battery — the tests are not vacuous

| mutant | expected | result |
|---|---|---|
| baseline | pass | ALL PASS |
| `losses.py`: SF fallback → `zeros_like` | fail | FAILED |
| `losses.py`: blend renormalisation dropped | fail | FAILED |
| `value_blend_guard.py`: `leaked_to_outcome` → `0.0` | fail | FAILED |
| `trainable_metrics.py`: ramp not disabled at start 0 | fail | FAILED |

---

## 2. Held-out split — frozen, time-disjoint, pure BY ID

**Held-out window**: the last 6 hourly tars by wall-clock (from the filename stamp,
which is the data's hour; local mtimes are download times and sort differently):
`20260815-1917 · 2017 · 2117 · 2217 · 2317 · 20260816-0017`.

| | |
|---|---|
| pool rows (400 games/tar, all 6 converted `VERDICT: PASS`) | 279,180 |
| pool duplicate row ids | **3** (1.1e-5) |
| frozen sample (stratified over all 6 hours, seed 0) | **100,000** |
| frozen unique ids | 100,000 |
| row id version | `lc0_control_row_id_v1_x_policy_blake2b128` |
| **sha256 of the frozen artifact** | `e7ceb225f4c7caf10b7872e7b45b34346a0698096d2fca210de57a190fd46ba1` |

**Purity, by id and not by tar name**: 0 intersecting ids against a time-disjoint
train conversion (`20260810-1417`, 48,360 rows). The check FIRES: injecting one
held-out hour into the train list yields **16,668 intersecting ids and exit 1**.

⚑ Why the id is a content digest of `(x, policy_target)` and not `(game_id,
ply_index)`: `game_id` is `enumerate()`'s index within ONE conversion invocation, so
converting train and held-out separately restarts it at 0 and the id sets are
disjoint BY CONSTRUCTION — the assertion would pass on a 100%-contaminated split.

The 3 pool duplicates were inspected: each is a pair of DIFFERENT games reaching an
identical position with an identical lc0 policy target but a different game outcome
(`wdl_target` 1 vs 2). So the id is a *position+label* identity rather than a record
identity, and the purity check's false-positive rate is bounded by 1.1e-5.

---

## 3. Chance level — `E[1/n_legal]`, on the frozen rows

Computed on exactly the 100,000 frozen held-out rows, not on the pool:

| | |
|---|---|
| mean `n_legal` | 27.6351 |
| **`E[1/n_legal]` — CHANCE TOP-1** | **0.064577** |
| `1/E[n_legal]` — WRONG (Jensen) | 0.036186 |
| ratio | **1.7846×** |

⇒ quoting `1/E[n]` would put the negative control's floor **44% below its own floor**.

**Random-init floor (prereg guard 2)**, same 100,000 rows, untrained net:
**0.062050**. That sits at the chance level (0.25 pp below it), so the evaluator is
not manufacturing agreement.

---

## 4. Ingest path — shards → replay buffer → training step → checkpoint

`scripts/lc0_control_train.py`, 8 steps × batch 512 = 4,096 rows, CUDA, no
`torch.compile`, 8.69 s of training wall time. 63,084,128 trainable params (this
branch's pinned production number; `tests/test_lc0_control_config.py` asserts the
control's ModelConfig is field-identical to `configs/pbt2_small.yaml`'s rather than
pinning the constant).

Buffer is the production `DiskReplayBuffer`, opened `read_only=True`.

| head | loss | rows that trained it |
|---|---|---|
| `policy_loss` | 3.094168 | 4,096 |
| `wdl_loss` (= `blended_wdl_loss`) | 1.082501 | 4,096 |
| `wdl_onehot_loss` (diagnostic) | 1.084320 | — |
| `moves_left_loss` | 0.030380 | present |
| `soft_policy_loss` | 0.000000 | **0 — no target in lc0 shards** |
| `future_policy_loss` | 0.000000 | **0** |
| `sf_move_loss` | 0.000000 | **0** |
| `sf_eval_loss` | 0.000000 | **0** |
| `categorical_loss` | 0.000000 | **0** |
| `volatility_loss` / `sf_volatility_loss` | 0.000000 | **0** |
| `loss` (total) | 4.177277 | |

Phase split (`policy_loss_phase_*`): open 3.477 (n=1560) · mid 3.231 (n=1017) ·
end 2.611 (n=1519).

⚑ The zero-loss heads keep their production weights. They are INERT, not applied —
`compute_loss` returns a zero loss for a head whose target is absent. The row counts
above are the proof; the weights are left alone so the arm carries exactly one
documented deviation instead of a dozen undocumented ones.

**Throughput, for sizing the real run:** conversion runs at ~230 rows/s
single-threaded (48,360 rows in 210 s), so the full 87M-position corpus is
**~105 h on one core** and wants parallelising by tar. Row-id hashing is ~31k rows/s
(~0.8 h for the full corpus) and is not the bottleneck.

---

## 5. Yardstick — paired top-1 vs lc0's visit-argmax (McNemar)

`scripts/lc0_control_eval.py`. Exercised end to end on the frozen 100,000:

```
A  random init                    top-1 0.062050
B  smoke checkpoint (6 steps)     top-1 0.096710
paired rows          100000
discordant  b(A only)=4434  c(B only)=7900  discordance=0.1233
delta (B - A)        +3.4660 pp
95% CI               [+3.2494, +3.6826] pp   (halfwidth 0.2166 pp)
exact McNemar p      1.4412e-216
```

⚑ **This is a pipeline check, not a learning result, and must not be read as one.**
Two arbitrary checkpoints differ; the point is that the paired estimator resolves the
difference and reports a CI.

**The prereg's resolution claim is CONFIRMED empirically.** It predicted 0.196 pp at
discordance 0.10 and 0.277 pp at 0.20; realized discordance 0.1233 gave 0.2166 pp,
on the interpolation. The **2× material bar of 0.392 pp** therefore stands as written.

The prereg's second slope (`Δ_train`, frozen already-trained rows) uses the same two
commands with different `--frozen`/`--shards`; exercised, works. ⚑ Its frozen set
must be built at the SAME n as the held-out one — the prereg's bar applies to both
and n=5,000 gives a 1.07 pp halfwidth, which would silently move the bar. **That is
now a gate, not prose**: `compare` refuses (exit 1) any pairing whose 95% halfwidth
exceeds `--max-halfwidth-pp` (default 0.392), and `freeze` refuses to write an
artifact with fewer rows than `--sample`. See §6.5.

---

## 6. Review follow-up (2026-08-16) — what the independent review changed

An independent reviewer returned **MERGE-WITH-CHANGES** with 5 findings (3× P1) and
confirmed all 6 open Codex threads by execution. Every load-bearing number in §§1–5
reproduced. This section records what changed, and the evidence that each change
takes effect. **Author of the fixes is neither the author of the PR nor the
reviewer.**

### 6.1 F1 (P1) — ⚑⚑ THE ARM WOULD HAVE TRAINED AN ARCHITECTURE PRODUCTION DOES NOT RUN

Measured, three ways:

```
this branch  configs/pbt2_small.yaml         63,084,128 trainable (unique storage)
this branch  configs/lc0_positive_control     63,084,128
LIVE tree    configs/pbt2_small.yaml         61,444,448   (sum(numel) 77,173,088)
```

The live file carries the bt4heads bundle promoted 2026-08-15 — `aux_policy_head_dim:
128`, `categorical_head_coupled: true`, `policy_embedding_mode: linear`. **Two of the
three touch the policy head, which is this arm's ONLY yardstick.**
`aux_policy_head_dim` is not in this branch's `utils/config_yaml.py` schema, so this
branch cannot build the live net at all (CLAUDE.md category (a): fatal at launch).

**Decision: option (b) — the arm refuses to launch, loudly, rather than being
re-pointed.** Option (a), basing the arm on the live branch's model code, would mean
either shipping the bt4heads promotion inside a rig PR (a production-path change
smuggled into an instrumentation change, and it needs its own ledger entry and
review) or rebasing this PR onto a branch that does not exist on `origin`. Neither
is a change to make on this PR's behalf. The premise "test THE STACK WE RUN" is a
LAUNCH precondition, not a merge precondition, so the right shape is a guard that
blocks the launch and names what is missing.

New module `chess_anti_engine/eval/lc0_control_arch.py`:

* `LIVE_ARCH_PIN` — the live `model:` section, both param counts, the branch and
  commit they were read from, and the date.
* `live_production_config_path()` — reads `$CHESS_LIVE_PRODUCTION_CONFIG`, and
  **never defaults to an in-tree path**. The repo is public, so the live path is
  supplied by the operator, not committed.
* `assert_control_matches_live_architecture()` — judges against the LIVE FILE when
  the env var names one, against the recorded pin otherwise, and says which in the
  provenance string it returns. A drift raises `ControlArchitectureDrift`.
* `unique_storage_param_count()` — counts by unique `untyped_storage().data_ptr()`,
  never `sum(v.numel())` over the state_dict. `lc0_control_train.py` now uses it.
* `scripts/lc0_control_arch_pin.py` — `--emit` to regenerate the pin from a live
  config, `--check` as a standalone gate.

Wired as **launch guard 0** in `lc0_control_train.py`. Today it FIRES:

```
$ PYTHONPATH=. python3 scripts/lc0_control_arch_pin.py --check
REFUSING: the lc0 control's architecture is NOT production's ... Drift:
  aux_policy_head_dim: control='<absent>' live=128
  categorical_head_coupled: control='<absent>' live=True
  policy_embedding_mode: control='<absent>' live='linear'
  trainable params (unique storage): control=63084128 live=61444448
```

⚑ `tests/test_lc0_control_arch.py` is written so that the situation IMPROVING breaks
it: `test_the_pin_records_an_architecture_this_tree_cannot_build` fails the moment
`aux_policy_head_dim` enters this branch's schema, and
`test_the_pin_names_the_bt4heads_keys_the_in_tree_config_lacks` fails the moment the
in-tree production config gains them. Both failures mean "regenerate the pin and
re-point the control", and neither can be silenced by copying a number across.

⚑ `test_a_control_matching_the_stale_pin_still_fails_against_the_live_file` is the
one that proves the instrument is not another in-tree copy: a control matching the
committed pin EXACTLY still raises when `$CHESS_LIVE_PRODUCTION_CONFIG` disagrees.

### 6.2 F2 (P1) — the headline per-step guard could not fail; now it can, and does

`--allow-leak` used to skip the realized assert as well as the launch guards, and
launch guard 1 refuses every `sf_wdl_frac > 0` config unconditionally, so no input
to the script reached a firing state. `--allow-leak` now downgrades **only the two
launch guards**. Constructed and observed:

```
$ ... --config <production-blend copy> --shards <lc0 rows> --allow-leak
⚑ --allow-leak: IGNORING launch guard — the PID sf_wdl ramp is not disabled ...
  sf_wdl_frac (realized)  0.500000   leaked_to_outcome  0.500000
  outcome_borne_frac      0.800000
REALIZED VALUE-BLEND GUARD FAILED — no checkpoint written.
EXIT != 0, and out-dir/checkpoint.pt does not exist.
```

Covered by `test_the_realized_guard_fails_the_run_and_writes_no_checkpoint`, which
also asserts the absence of the checkpoint. Mutant M1 (delete the guard call) is
killed by it.

### 6.3 F6 / Codex #2 — the second door: an all-outcome value target with ZERO leak

`leaked_to_outcome = sf_wdl_frac × (1 − sf_labelled_frac)` is 0 whenever
`sf_wdl_frac` is 0 — including `sf_wdl_frac: 0.0` / `search_wdl_frac: 0.0`, which
trains **100% of the value target on the raw one-hot outcome**. Two changes:

* `assert_outcome_is_not_the_whole_target` gates on `outcome_borne_frac` against a
  bar of **0.30 — production's own `game_frac`**, not a number invented here.
  `assert_no_silent_outcome_fallback` now checks both bars.
* `run_config_problems` no longer returns `[]` on the first line when any SF label
  exists. The collapse check is label-INDEPENDENT and always runs.

⚑ Correction to Codex #2 as filed: the mixed-corpus path could NOT permit a nonzero
`sf_wdl_frac` (launch guard 1 refuses that corpus-independently). The reachable
defect is the `sf=0 / search=0` door above, which the reviewer demonstrated passing
all three old guards.

### 6.4 Codex #2, second half — `any()` is not label coverage

`shard_dir_sf_wdl_coverage()` replaces the boolean, reads `has_sf_wdl` through the
LAZY shard loader (the 175-plane inputs are never decoded), and the preflight prints
per-directory coverage. A **partially** labelled corpus is now refused outright:
neither regime's reasoning applies to an average of two corpora.

### 6.5 F5 / Codex #5 — "0 intersecting ids" was not "0 exposure". Measured: 450

Re-derived on exactly the shipped smoke split, with the new gate:

```
frozen held-out rows 100000
frozen distinct x    96847
train rows scanned   48360 (unique ids 48360, unique inputs 47060)
intersecting ids     0     <-- what the shipped gate reported: PURE, exit 0
EXPOSED inputs       450   (0.4647% of held-out x)   <-- THE GATE, exit 1
```

`input_ids` (x-only) is now the gate; `row_ids` (`x, policy_target`) stays as the
record identity used to address a row. Both are in the artifact and both are
printed. `pool_duplicate_inputs` is reported next to `pool_duplicate_ids`: **10,130
vs 3** in the pool, **3,153 vs 0** inside the frozen 100,000 — the same 1,000× gap
the reviewer found.

⚑ **The shipped frozen artifact is superseded.** Re-frozen with the same sources and
seed; the `row_ids` list is **byte-identical to the shipped one** (verified), so the
draw did not move — only the artifact schema did.

```
row id version   lc0_control_row_id_v2_x_policy_plus_x_only_blake2b128
sha256           95829c261f6352eae1cd7da6417afb17b009a87849765668bb2799cf27dfb562
```

`load_frozen` refuses a v1 artifact and refuses any artifact missing the exposure
ids, so the weaker gate cannot be reached by re-using an old file.

### 6.6 Codex #1 and the Δ_train n — a bar derived at n=100,000, enforced nowhere

* `freeze` refuses (exit 1, **and writes no file**) when the pool cannot supply
  `--sample` rows.
* `compare` refuses (exit 1) when the 95% halfwidth exceeds `--max-halfwidth-pp`
  (default **0.392**, the prereg's material bar). `--allow-underpowered` downgrades
  it to a banner. The reviewer's own demonstration — 5,000 paired rows, 1.0721 pp
  halfwidth — now exits 1.
* `cmd_compare` gains its first regression test: hand-built cells `b=10, c=30,
  n=10,000` pin delta, halfwidth and the exact p against
  `scipy.stats.binomtest(10, 40, 0.5).pvalue = 0.0022214337732293643`.

### 6.7 Codex #6 — purity passed on an empty train directory

`purity_against_train` raises `EmptyTrainCorpus` when it scans zero rows, and the
driver exits 1. A check that scanned nothing reported PURE and exit 0.

### 6.8 Codex #3 — the scorer tolerated a partial checkpoint load

`lc0_control_eval._load_trainer` no longer calls `Trainer.load` (which uses
`load_state_dict_tolerant(..., require_complete=False)` and only trips below 50% of
keys). It loads the state dict directly with `require_complete=True`. Given F1 this
was not hypothetical: scoring a live-production checkpoint with this config would
have silently blended trained and fresh tensors under the checkpoint's name.

### 6.9 F4 / Codex #4 — the random-init floor was unseeded, and there was no band

`--seed` is applied **before** `build_model` and recorded in the score `.npz` meta.
`--shuffle-targets` builds prereg guard 1, the shuffled-label negative control, by
permuting the per-row TARGET across rows — not by permuting the hit vector, which
would preserve the hit rate exactly and give a control that cannot fail.

⚑ NOT DONE: the band itself. A band needs several seeds scored on the frozen rows
against a trained checkpoint, and there is no trained checkpoint. The rig can now
produce the band; the band is not in this document.

### 6.10 F3 (P1) — zero test coverage on the drivers

`tests/test_lc0_control_drivers.py` (new) drives all three `main()` entry points.
Mutation battery, 14 mutants, each applied in a `cp -r` copy (never `git checkout`),
each verified absent from the mutated file (`count(target) == 0`), each run twice —
the lc0-control suite MINUS the killing test, and the killing test alone. Results in
the PR comment.

The reviewer's own double mutant (`worst -> readouts[0]` AND deletion of the guard
call) is now M2 and M1, and each is killed by a test that fails alone while the rest
of the suite passes.


---

## Reproduction

```bash
CONV=scripts/lc0_data_to_rows.py
for t in 20260815-1917 20260815-2017 20260815-2117 \
         20260815-2217 20260815-2317 20260816-0017; do
  PYTHONPATH=. python3 $CONV convert \
    --data data/lc0_training/training-run2-test91-$t.tar \
    --limit-games 400 --out <work>/heldout/$t
done
PYTHONPATH=. python3 scripts/lc0_control_heldout.py freeze \
  --shards <work>/heldout/* --out <work>/frozen.json --sample 100000 --seed 0
PYTHONPATH=. python3 scripts/lc0_control_heldout.py purity \
  --frozen <work>/frozen.json --train-shards <work>/train/*
PYTHONPATH=. python3 scripts/lc0_control_heldout.py chance \
  --frozen <work>/frozen.json --shards <work>/heldout/*
PYTHONPATH=. python3 scripts/lc0_control_train.py \
  --config configs/lc0_positive_control.yaml --shards <work>/train/* \
  --out-dir <work>/run --steps 8 --no-compile
PYTHONPATH=. python3 scripts/lc0_control_eval.py score \
  --frozen <work>/frozen.json --shards <work>/heldout/* --out <work>/a.npz
```

⚑ **At launch the frozen set must be REBUILT over the full held-out conversion and
its new sha256 recorded in the ledger.** The 100,000 above is drawn from a
400-games-per-tar smoke conversion (279,180 of the window's ~2.8M rows), so it is a
verified apparatus, not the artifact the arm will read.

---

## What is NOT verified

* **No arm was run.** Everything here is 8 steps or fewer on 400 games per tar.
* **The full-corpus conversion has not been done** (~105 core-hours) and the 130-tar
  set has not been converted end to end. Only 7 of 130 tars were touched.
* ~~**The negative control (prereg guard 1, shuffled labels) has no rig.**~~ Built
  2026-08-16: `lc0_control_eval.py score --shuffle-targets`. It has NOT been RUN on
  a trained checkpoint — there is no trained checkpoint.
* **`torch.compile` was OFF** in every run here (`use_compile: true` in the config).
  It changes throughput, not the objective, but the compiled path is unexercised.
* **`Trainer`/`tune` do not RAISE.** The value-blend guard protects this arm's entry
  point only; the trainer carries a log-only readout — see the decision note in §1.
* **⚑⚑ THE ARM CANNOT BE LAUNCHED YET.** The param count here is this branch's
  63,084,128, not the live tree's 61,444,448 (post-bt4heads). The claim that the
  config test "follows a rebase automatically" was WRONG: `aux_policy_head_dim` is
  not in this branch's `utils/config_yaml.py` schema, so this branch's code cannot
  build the live network at all, and the in-tree config diff is structurally unable
  to see it. `preflight_architecture` now REFUSES the launch. See §6.1.
* **Nothing in this document was produced on production's architecture.** Every
  smoke run since 2026-08-16 carries `--allow-arch-drift` and stamps
  `valid_control: false` into its `summary.json`.
* **`search_wdl_frac: 0.70` is a judgement call**, not a measured optimum. It holds
  `game_frac` at production's 0.30; lc0's own recipe would be 0.50.
