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

1. **launch, config** — `assert_pid_cannot_reassert_sf_wdl`.
2. **launch, corpus** — the converter's own `run_config_problems` / `shard_dir_has_sf_wdl`
   are IMPORTED and reused, not restated, so the "do these shards carry an SF label"
   question is answered by reading the shards.
3. **realized, per-step** — the wrapped `compute_loss` kwargs plus the batch's own
   `sf_wdl_rows`, judged by `assert_no_silent_outcome_fallback` at `max_leak=0.0`.
   Judged on the WORST step, not the first or the mean, so a leak that begins
   partway through a long run cannot be diluted under the bar.

The blend arithmetic is NOT duplicated: `losses.normalize_value_blend_fracs` was
extracted from `compute_loss` and is called by both, so the guard cannot drift from
the criterion it checks.

**Not wired into `Trainer`/`tune`, deliberately.** A hard raise inside `train_steps`
would be a new fatal path on the production trial (CLAUDE.md category (b): the
iteration loop has a `finally:` and zero `except`, so it kills the trial), for a
condition production cannot reach — 712/713 live shards carry `has_sf_wdl`. The
guard is on the entry point where an lc0-shard launch can actually happen, and is an
importable function if production ever wants it. **Decision, flagged for review.**

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
and n=5,000 gives a 1.07 pp halfwidth, which would silently move the bar.

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
* **The negative control (prereg guard 1, shuffled labels) has no rig.** The chance
  level it is judged against is computed; the shuffle arm itself is not built.
* **`torch.compile` was OFF** in every run here (`use_compile: true` in the config).
  It changes throughput, not the objective, but the compiled path is unexercised.
* **`Trainer`/`tune` are unguarded.** The value-blend guard protects this arm's entry
  point only — see the decision note in §1.
* **The param count is this branch's 63,084,128**, not the live tree's 61,444,448
  (post-bt4heads). The config test asserts equality WITH PRODUCTION rather than a
  constant, so it follows a rebase automatically, but the arm should be re-measured
  after one.
* **`search_wdl_frac: 0.70` is a judgement call**, not a measured optimum. It holds
  `game_frac` at production's 0.30; lc0's own recipe would be 0.50.
