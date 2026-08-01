# Which training targets can be rebuilt from a stored shard row

A target change normally only affects **newly generated** data, so it cannot be
read until the ~1.5M-row replay window turns over — measured 2026-07-27 at ~99
iterations, **~18 h** (50 % at ~9 h). That latency is the rate limiter on the
whole target-experiment loop.

Some targets escape it. Shards (schema v2+) store the raw Stockfish MultiPV
rows the live pipeline built its SF targets from, so those targets can be
recomputed from scratch at sample time under new parameters —
`train.rebuild_sf_targets` (default OFF), implemented in
`chess_anti_engine/train/target_builder.py`. When it is on, an
`SfTargetParams` change applies to **the whole SF-labelled window on the next
iteration** — see "Coverage over labelled rows is total" below.

Most targets do not escape it. This file says exactly which, why, and what
extra storage would move a row from the second table to the first. Every claim
below was verified against 10 live shards (18 307 rows) from the iter-168
window on 2026-07-27; the "verified" column gives rows checked and the worst
disagreement observed, which for the rebuildable ones is fp16 storage noise.

**⚠ That sample is now known to have been drawn inside a Stockfish-desync
episode** (next section). The per-target *identities* below are unaffected — a
row whose SF label is detached from its position is still bit-equal to its own
neighbouring row, which is all those checks assert — but every *rate* derived
from that sample (the coverage figures, and the 12.6 % / 22.6 % / 95.6 %
population shares quoted further down) is measured on contaminated data and
should be treated as indicative only.

## Coverage over labelled rows is TOTAL — and the shortfall column is a desync alarm

**⚠ CORRECTED 2026-08-01. This section previously claimed a structural 5.38 %
coverage gap. There is no such gap: the floor is exactly zero.** The old figure
came from a 10-shard sample (18 307 rows, 956 labelled rows without
`sf_multipv_raw`) drawn on 2026-07-27 — which was itself a Stockfish-desync
episode day. Of the 260 possible 10-consecutive-shard windows written that day,
**192 read exactly 0.000000** and exactly one lands near 5.4 %; that one
reproduces the old numbers to within rounding (17 844 labelled / 16 892 with
raw / 952 missing / **0.0534**). The sample was the anomaly, not the norm.

The selfplay writer stamps `sf_multipv_raw` and `sf_label_meta` on a labelled
row **together** (`selfplay/stockfish_turn.py::_stamp_sparse_sf_labels`), so
every SF-labelled row carries raw and the rebuild reaches **100 % of the
SF-labelled window**. An `SfTargetParams` change is a clean swap, not a mixture
of two target regimes.

Both causes this document previously offered are refuted:

* **Pre-raw-schema rows.** Real, but long dead. The schema shipped 2026-06-10
  (`49436d3a0`) and a 1.5M-row window turns over in ~18 h, so no such row could
  survive to a July measurement. Confirmed: across every retained shard pool
  from 2026-07-02 onward, **zero shards lack the `has_sf_multipv_raw` array.**
* **The "no-scoreable-candidates fallback" being a benign structural cost.** It
  is `if not rows: return None` in `_collect_sparse_pv_rows`, and it fires
  exactly when **not one** of Stockfish's 40 MultiPV moves is legal at the
  position queried. That is not benign — it is the fingerprint of a desynced
  UCI engine answering a *different* position, so the whole SF label block on
  that row is detached from its row. Same measurement as the live
  `_SF_NO_LEGAL_PV_WARN_RATE` counter and the offline
  `eval/value_optimism.py::sf_multipv_missing_rate` gate.

Measured 2026-08-01 over **6 535 unique shards / 11 052 418 labelled rows**
(all retained salvage pools + the live window + the quarantine set, deduplicated
by file identity and bucketed by the day each shard was *written*):

* **90.5 % of individual shards read exactly 0.000000**, and **10 of 23 days read
  exactly 0.000000 across 5 463 097 labelled rows** — including 07-06..07-09
  (1 621 603 rows), 07-22, 07-24, 07-25 and 07-26.
* The non-zero mass sits in contiguous **episodes** (332 shards above 10 %, up to
  0.67), not in a chronic background. Four days carry a tiny residue (4 to 258
  missing rows each, confined to 1-9 shards) that looks like episode tapers.
* The desync is fixed by PR #297 and the contaminated live shards were
  quarantined 2026-08-01 (122 shards), so new data reads zero.

**⚑ Read `sf_rebuild_policy_frac` below `sf_rebuild_wdl_frac` as CONTAMINATION,
not as cost.** The two columns share a denominator (all batch rows) and a
healthy labelled row always sets both presence flags, so on clean data they are
**equal** and their difference is the poisoned-label share of the batch:
0.000000 on clean days, **0.192** over the quarantined shards. That makes the
pair a per-iteration desync detector the training loop already computes — but
only while `rebuild_sf_targets` is ON, since with the flag off both read 0.0 by
construction and the difference is uninformative. Do not read a gap as "the
rebuild could not reach those rows".

Five `progress.csv` columns report what actually happened, per iteration:

| column | meaning |
|---|---|
| `sf_rebuild_policy_frac` | rows whose `sf_policy_target` was rebuilt / rows in the batch |
| `sf_rebuild_wdl_frac` | rows whose `sf_wdl` was rebuilt / rows in the batch |
| `sf_rebuild_masked_frac` | **rows** that lost ≥1 cross-ply target / rows in the batch |
| `sf_rebuild_masked_p0_frac` | rows whose `has_sf_p0` was set **before** the mask / rows in the batch |
| `sf_rebuild_masked_volatility_frac` | same, for `has_sf_volatility` |

`sf_rebuild_masked_frac` counts ROWS, not flags. Counting flags made it read
**2.0** on a window where every masked row carried both `has_sf_p0` and
`has_sf_volatility` — a column named `_frac` that exceeds 1.0 is not a coverage
number, and it hid the real per-row rate behind the number of flags per row.
The two `masked_*` columns decompose it per flag, and they are **PRE-mask**
presence fractions, which makes them the outage detector while a rebuild
runs: the mask zeroes the flags indistinguishably from "never recorded", so
`has_sf_p0_frac` — documented in `trainable_report.py` as the sf_p0 outage
detector — is pinned at exactly 0.0 for the duration of any rebuild
experiment and can detect nothing there. `sf_rebuild_masked_p0_frac → 0`
with the flag on is the reading that means "the selfplay workers stopped
recording sf_p0 rows".

All five read **0.0** with the flag off, so a non-zero value *is* the proof
the flip reached the batch pipeline. The transition log line proves only the
config push — it fires from the setter, before any batch is built. Do not use
`has_sf_p0_frac → 0` as the proof either: on a window with no p0 rows it
already reads 0 and proves nothing — and with the flag ON it reads 0 by
construction (the mask), which proves even less.

**Denominator caveat, all five columns:** the accumulator receives counts only
from batches that actually went through the rebuild, so each `_frac` is a
fraction of the REBUILT batches, not of every batch the iteration trained on.
With the flag steady for a whole iteration the two denominators coincide; on
the single iteration where a live flip lands mid-way, the row mixes pre- and
post-flip batches and the fracs cover only the post-flip subset — a one-row
transient to read around, not a coverage regression.

On the row produced by `eval_full_pass` — the frozen ruler, and the only eval
production runs — all three stay 0.0 **by construction** (it pins the rebuild
off; next section), so a non-zero value there is itself the alarm that the ruler
moved. That alarm is readable as the `progress.csv` columns
**`test_sf_rebuild_policy_frac` / `_wdl_frac` / `_masked_frac`**, not only as a
TensorBoard scalar: TB event files rotate per Ray session, and an alarm whose
only reading lives in a rotating sink is not an alarm. The SAMPLED
`Trainer.eval` is explicitly *not* a ruler and does rebuild, mirroring the
training distribution; it has no production caller.

**Each measurement owns its own counter.** `drain()` read-and-RESETS, so a
single trainer-wide accumulator would not merely blur the train and eval rows —
it would move counts between them. The async holdout eval calls
`_compute_metrics` on the *same* `Trainer` from its own thread while the next
iteration is training (`distributed_async_test_eval: true` in the production
config), so its drain would publish the TRAINING path's counts on the `eval`
row and leave the `train` row short by an unknowable amount: the proof-of-effect
becomes an undercount and the ruler alarm above fires every iteration on
nothing. `_compute_metrics` therefore builds a fresh accumulator per eval and
threads it down through `coverage=`. Note that reasoning about which paths
*accumulate* does not catch this — the full pass accumulates nothing and still
drains. Pinned by `test_an_eval_does_not_drain_the_training_coverage_counters`.

## The frozen holdout ruler is NOT rebuilt

`_full_pass_host_batch` — the deterministic holdout pass that produces
`test_loss` — pins `rebuild_sf_targets=False`, the same way and for the same
reason it already pins `mirror_prob=0.0`: *a ruler must not acquire a
dependency on a training-side config knob.*

Without the pin, flipping the flag would have:

* scored the model against **rebuilt** `sf_policy_target` / `sf_wdl` instead of
  the stored ones, and
* dropped `w_sf_own` (0.1) and `w_sf_volatility` (0.05) out of `total`, because
  the rebuild masks the two cross-ply targets it cannot move — measured at
  **12.6 % of live rows** losing the first and **22.6 %** losing the second,
  with `sf_own_ce` going to exactly 0.0 purely from the flip.

Both change what `test_loss` MEANS with no `holdout_generation` bump, and the
second moves it **down** — a definitional fall that reads as improvement and
that `_update_best_model` would promote across. That is the G16 / PR #277 shape
recorded in `docs/rl_loop_audit.md`: *freezing the SET does not freeze the
MEASUREMENT.*

The alternative — let the rebuild through and bump `holdout_generation` — was
rejected on the merits, not on cost. A ruler that re-parameterises itself with
the training target **cannot measure that target's effect**: the bump would
fire exactly when the experiment starts, so every arm would sit on a fresh
instrument and the holdout could never read the experiment at all. The bump
makes the corruption visible; the pin keeps the measurement usable.

### What the pin does NOT cover

* **The SF-derived legs of `test_loss` are contaminated for the duration of a
  rebuild experiment.** `sf_move_ce` scores the model against capture-time
  targets while it trains against rebuilt ones, so that column moves for a
  definitional reason (a softer target has a higher CE floor) in the *opposite*
  direction — it reads as degradation. The `wdl` leg is affected too whenever
  the `sf_wdl_cp_*` knobs move, since the WDL target blends `sf_wdl` at
  `sf_wdl_frac`. The `policy`, `policy_own`, `soft` and `value` legs are
  untouched.
* **So the holdout is not a valid yardstick FOR a rebuild experiment.** Use an
  external one per `docs/eval_protocol.md` — `arena_standard` matched_sims,
  `value_regret`, or `audit_targets`. The pin exists so the holdout keeps
  measuring *drift in everything else* across the experiment, not so it can
  judge the experiment.
* It does not touch the pre-existing hole that editing a **loss weight** moves
  `test_loss` definitionally with no generation bump either. Out of scope here.
* The pin lives on `_full_pass_host_batch`. Both production holdout call sites
  reach it (`tune/trainable_phases.py:286` passes `full_pass=True` to the async
  eval, `:290` calls `eval_full_pass` directly), but a future caller routing the
  holdout through the SAMPLED path would rebuild again. That is exactly the
  ruler-identity shape PR #282 exists to catch, and it is not re-solved here.
* It is independent of PR #282 (`holdout_generation` tracks the ruler, not the
  set) and does not conflict with it: #282 derives the ruler id from the
  full-pass batch producers, and this pin makes those producers *invariant* to
  `rebuild_sf_targets`, which is the property #282 wants to be able to assert.
  Note #282 would **not** have caught this on its own — the flag changes a
  runtime value, not the descriptor or the source digest. The pin is the same
  move #282 already makes for `steps` and `mirror_prob`, under the same stated
  rule: *a knob that cannot move the number must not be able to move its
  identity.* Adding the flag to the ruler id instead would have broken that
  rule in both directions — the id would move on a knob that (once pinned)
  cannot reach the measurement, and reaching the unpinned runtime value would
  have required plumbing config into a descriptor #282 deliberately built out
  of arguments and source only.
* **The pin is GUARDED by #282, which is the observation that makes (b) safe
  against future drift** — and it beats the argument above, because it can be
  executed. The pin lives *inside* `_full_pass_host_batch`, and that function
  is one of the frames #282's `measured_by` hashes. So an edit that un-pins the
  ruler moves the ruler id and fires the handover. Measured on the actual
  merged tree (2026-07-28, #282 in `main`): pin intact
  `v1:full_pass:2efe658b4e778870`, pin removed
  `v1:full_pass:277d0421e7ff0ea5`. The argument from #282's descriptor
  explains why (a) was unavailable; this is why (b) stays true.
* **SETTLED: the deploy produces ZERO `holdout_generation` bumps.** #282's
  `measured_by` hashes four frames this PR edits — `_prepare_host_arrays`,
  `_full_pass_host_batch`, `_compute_metrics` and `_iter_full_pass_batches` —
  so the merged ruler id differs from #282's alone even though the number does
  not. But **both landed in `main` before the single pending restart**, and
  today's live checkpoints carry no `holdout_ruler` at all: an absent id reads
  as *no evidence* and is adopted without bumping. Generation stays **1**,
  which is what #282's pre-registered PASS condition expects.

  The other case — #282 deploying at one restart and this PR at a later one —
  would have produced ONE bump (1 → 2) and handed `best/` over. It does not
  arise here. Either way it is a FALSE positive, #282's declared direction of
  error, and must not be read as the ruler moving. **If a bump IS observed at
  the deploying restart, that is a KILL signal for #282's condition 3, not a
  shrug.**
* **DONE — the merge of `main` (carrying #282) into this branch.** It conflicts
  in exactly one hunk, `_compute_metrics`'s `_build_metrics(...)` call:
  `eval_ruler=ruler` (#282) vs `**eval_coverage.drain()` (this PR). **Both
  lines are kept.** GitHub reported each PR individually mergeable because
  neither was on `main` yet, so the conflict only surfaced at the second merge;
  `trainable_report.py` auto-merged clean even with the new `test_` keys.
* **BOTH golden ruler pins moved — not just the full-pass one.** This PR edits
  frames in *both* `measured_by` lists (`_prepare_host_arrays` and
  `_compute_metrics` are shared; `_sample_batch_host` and
  `_iter_prefetched_batches` are the sampled branch's), so updating only the
  full-pass constant would still leave `main` red on
  `test_the_production_ruler_id_is_pinned`. Recomputed against the **actual
  merged tree**, not predicted:

  | constant | #282 alone | deployed (merged) |
  |---|---|---|
  | `PRODUCTION_FULL_PASS_RULER` | `v1:full_pass:c8fb48a79e804bb4` | **`v1:full_pass:2efe658b4e778870`** |
  | `PRODUCTION_SAMPLED_RULER` | `v1:sampled:e3cc3241626a581f` | **`v1:sampled:d6f7cabecd8e6f67`** |

  (Historical: the #283 review follow-up later moved both again — to
  `v1:full_pass:bed3d8e3799e997d` / `v1:sampled:610f05cf817b4783` — by making
  the rebuild gate fail-closed in `_prepare_host_arrays` /
  `_sample_batch_host`. Same declared-false-positive shape; see the ledger's
  2026-07-30 addendum.)

  Both are interpreter-independent, re-verified on the merged tree: the seven
  covered frames digest identically on CPython **3.10.12, 3.11.14 and
  3.12.12**. (#282's earlier `ast.unparse` digest was interpreter-dependent —
  do not reuse any hex quoted before that change.) The ledger's quoted hex is
  updated to match in the same commit.

## Rebuildable from the row itself

| target | source on the same row | param-dependent? | verified |
|---|---|---|---|
| `sf_policy_target` | `sf_multipv_raw` + `sf_legal_mask` | yes — `sf_policy_temp`, `sf_policy_label_smooth`, `sf_wdl_use_cp_logistic`, `sf_wdl_cp_slope`, `sf_wdl_cp_draw_width` | 16 822 / 16 822 rows, max TV **2.2e-4** at the live params |
| `sf_wdl` | `sf_label_meta` | yes — the three `sf_wdl_cp_*` knobs | 17 778 / 17 778 rows, max abs **2.4e-4** |
| `categorical_target` | `wdl_target` + `sf_wdl` (+ `search_wdl`) | yes — `categorical_blend_frac`, `categorical_search_blend_frac`, `hlgauss_sigma` | already shipped as the separate `train.rebuild_categorical_target` flag |
| `priority_sf_search_gap` | `sf_wdl` + `search_wdl` | yes, via `sf_wdl` | 17 778 / 17 778 rows, max abs **9.2e-4** — **but see the note below: rebuilding it in a batch is useless** |

`priority_sf_search_gap` is arithmetically rebuildable and practically not: it
is a **sampling** priority, consumed once at ingest to weight the row in the
buffer index. Recomputing it inside a sampled batch changes nothing about which
rows were sampled. Moving a sampling knob still requires re-ingest, i.e. it is
still an 18-hour question (or an offline `retarget_retrain.py` run over a copied
shard dir).

## Stored, but nothing to rebuild

These are SF-derived yet carry **no** `SfTargetParams` dependence, so they never
go stale under a rebuild and must be left alone.

| target | why it is parameter-free |
|---|---|
| `sf_move_index` | SF's `bestmove` from the UCI reply, not a function of our scoring. A naive "argmax of `sf_multipv_raw`" rebuild reproduces it on only **95.6 %** of rows (16 822 checked) — so a rebuild would *corrupt* 4.4 % of them. Don't. |
| `sf_p0_regret`, `sf_played_regret`, `sf_played_rank`, `sf_played_move_index`, `future_sf_regret_*` | normalized centipawn regret / rank orderings (`finalize._sf_move_score`, cap 1000 cp). No temperature, no cp→WDL mapping. |
| `sf_legal_mask`, `legal_mask`, `future_legal_mask`, `moves_left` | positional or game-length facts. |

## NOT rebuildable — the source is on a different row

Both of these **are** functions of what the rebuild moves, but the source row is
not in a randomly sampled batch. `rebuild_sf_targets_in_arrays` therefore
**masks** them (clears the presence flag) rather than leaving them pointing at
capture-time values while their source moves underneath.

| target | is exactly | verified | consequence |
|---|---|---|---|
| `sf_p0_policy_target` | ply **t-1**'s `sf_policy_target` | 2 276 / 2 276 rows, max TV **0.0** (bit-exact) | masked ⇒ the `w_sf_own` own-move teacher (0.1) is inert while the rebuild is on. `w_sf_own_regret` (0.7) is NOT masked. |
| `sf_volatility_target` | `abs(sf_wdl[t+6] - sf_wdl[t])` | 2 374 / 2 374 rows, max abs **4.9e-4** | masked ⇒ `w_sf_volatility` (0.05) is inert while the rebuild is on. |

Masking is **unconditional** — not "only when the params actually moved". A
control run (flag on, capture-identical params) and a treatment run must mask
the *same* rows, or the paired comparison is confounded by the own-move teacher
switching on and off between arms.

Three rejected alternatives, for the record:

* **Leave them stale.** This is the defect the masking exists to prevent: the
  loop's only external move-teacher would train on capture-time targets while
  its sibling `sf_policy_target` moved, and nothing would report it.
* **Re-temper the stored `sf_p0_policy_target`** (`p' ∝ p^(T0/T')`). Measured on
  800 live rows against a true rebuild: sharpening 0.012 → 0.006 is fine (mean
  TV 0.0013), but **softening is not** — 0.012 → 0.05 gives mean TV **0.066**,
  p90 0.25, max 0.84, against a true rebuild's 7e-5. The stored target is fp16
  and at temp 0.012 keeps only ~18 non-zero entries; softening has to resurrect
  a tail that fp16 already flushed to zero. Softening is the direction any
  `sf_policy_temp` experiment goes, so this is not a substitute.
* **Also mask the SF-labelled rows without `sf_multipv_raw`.** ⚠ **This bullet
  used to argue against masking a "~5.4 % structural" slice. That population
  does not exist on healthy data — it is exactly the desynced rows** (see the
  coverage section), so the trade-off it weighed was fictional. What is left is
  a genuinely different and much easier call: on a clean window there is
  nothing to mask (the slice is empty, so masking costs zero coverage), and on
  a contaminated window the right response is to quarantine the shards
  (`scripts/quarantine_desync_shards.py`), not to paper over them at sample
  time. Masking is therefore still not implemented, but for the opposite
  reason: not "the cure is worse than the disease", but "the disease is a data
  incident with its own detector and its own remedy". An experiment that
  nonetheless wants belt-and-braces can mask `has_sf_move` where
  `has_sf_multipv_raw` is 0 and say so in its ledger entry — on a clean window
  that is a no-op, which is the point.

## NOT rebuildable — the source was never stored

| target | built from | what the worker would have to store |
|---|---|---|
| `policy_target` | MCTS visit counts at the root, discarded after `finalize` | sparse root visits: `(K=64) × (move_idx int16, visits int32)` ≈ **384 B/row** |
| `policy_soft_target` | the same visit counts under `soft_policy_temp` | same 384 B/row — one field unlocks both |
| `future_policy_target` | ply **t+2**'s MCTS distribution | the visits above, *and* a cross-ply copy |
| `search_wdl` | the net's own MCTS root value average | nothing: it *is* the raw datum, there is no parameter to re-point |
| `volatility_target` | `abs(search/net wdl[t+6] - [t])` | cross-ply *and* search-derived |
| `wdl_target` | the game result (+ Syzygy / adjudication rescoring) | the raw datum |
| `priority`, `priority_policy_kl`, `priority_q_delta` | the generation-time model's surprise (KL of search vs prior, Q delta) | these are properties of *the model that played the game*; no storage makes them re-derivable for a different model. Re-weighting requires re-ingest. |

### Storage that would close the two masked rows

* `sf_p0_multipv_raw` — `(48, 5) int16` = **480 B/row uncompressed**. On the
  measured shards `sf_multipv_raw` compresses to 153 B/row at ~92 % coverage;
  `sf_p0` coverage is ~22 %, so expect ≈ **+35-40 B/row, about +2.5 %** on a
  1 588 B/row shard. `finalize._build_replay_samples` already computes
  `prepare_multipv(prev_idx)` for `sf_p0_regret`, so it is a one-line reuse of a
  value that is already in hand.
* `sf_label_meta_t6` — `(6,) int32` = **24 B/row**, unlocks
  `sf_volatility_target`.

Both only help data generated *after* they ship, so they do not shorten the
first readout — they shorten every readout after it.

## Before flipping the flag live

Flipping `train.rebuild_sf_targets` to `true` is a **training-affecting
change**: it re-points the SF targets for the SF-**labelled** rows in
the replay window in one iteration and takes the `w_sf_own` /
`w_sf_volatility` legs out of training. Per the experiment protocol it needs a
`docs/experiment_ledger.md` entry with ONE deciding yardstick as an exact
command and a pre-committed kill threshold **before** launch, and the yardstick
has to be an external one (see "What the pin does NOT cover"). Proof it took
effect: `sf_rebuild_policy_frac` > 0 on the first new row of `progress.csv`.

**Expect `sf_rebuild_policy_frac` ≈ `sf_rebuild_wdl_frac` ≈ the SF-labelled
fraction of the batch (0.996-0.997 on the current live window), and expect the
two to be EQUAL.** ⚠ This
paragraph previously told operators to expect **0.92** and to read that as
definitional rather than as a regression. That was wrong, and it is exactly the
reading that would have made a live desync invisible: the 0.919 came from
16,822 / 18,307 on the contaminated 07-27 sample, where the missing 5.4 % were
poisoned rows. The denominator being all batch rows *is* deliberate — it makes
the column a fraction of the training batch, which is what "how much of what I
trained on was re-pointed" asks — but on healthy data every SF-labelled row
rebuilds, so `policy_frac` and `wdl_frac` land on the same number.

**A gap between them is the alarm.** `wdl_frac − policy_frac` is the share of
batch rows whose Stockfish label is detached from its position; it reads
0.000000 on clean data and 0.192 over the shards quarantined 2026-08-01. Treat
any sustained non-zero value as a desync incident and check the worker logs for
the `sf label health` line.

**The deploying restart will ROTATE `progress.csv`.** The rebuild reports
**five** `sf_rebuild_*` keys, each with a `test_` twin — ten columns in all.
Three arrived with #283 (`sf_rebuild_policy_frac` / `_wdl_frac` /
`_masked_frac`) and two with the #288 review follow-up (`_masked_p0_frac` /
`_masked_volatility_frac`), so BOTH deploys rotate: #283's and then #288's.
Each rotation changes the report schema, so
`_rotate_progress_csv_if_schema_changed` (`tune/harness.py`, PR #262) moves the
existing file aside and starts a new one. That is correct behaviour and
`scripts/ratchet_slope.py` and `scripts/train_watchdog.py` already follow
rotations — but it is visible at the restart and must not be read as data loss.

## What this does and does not buy

**Does**: any experiment whose only lever is `SfTargetParams` —
`sf_policy_temp`, `sf_policy_label_smooth`, `sf_wdl_use_cp_logistic`,
`sf_wdl_cp_slope`, `sf_wdl_cp_draw_width` — plus the categorical-blend family
under its own flag. Those move from "wait ~18 h for window turnover" to
"applies to **the whole SF-labelled window** at the next iteration", per
"Coverage over labelled rows is TOTAL" above. Rows with no SF label are
untouched because they have no SF target to re-point, so a verdict may be
written as a clean swap over the labelled population — but confirm it per
iteration from `sf_rebuild_policy_frac == sf_rebuild_wdl_frac`, rather than
assuming it.

Note the interaction, and its price: the SF param keys are also **worker reco
keys** — all five of them (`WorkerSession._RECO_RESTART_KEYS`,
`worker.py:2897,2918`) — so one yaml edit retunes the capture params for new
data *and* re-points the rebuild for the old data; the window stays homogeneous
instead of spending 18 h as a mixture of two target regimes. But being a restart
reco key means that edit forces a full worker session rebuild, which in
`worker.py`'s own words "abandons the 256 in-flight games and collapses
curriculum throughput for ~2 iters". That cost is pre-existing and not
introduced by the rebuild; it is written down here because the interaction reads
as free and is not. `rebuild_sf_targets` itself is trainer-side only and
deliberately not a reco key, so flipping the flag alone costs no games.

**Does not**: anything search-derived. `policy_target`, `policy_soft_target`,
`search_wdl`, `volatility_target` and every `priority_*` come from the search
that was actually run, and no re-parametrisation of stored bytes recovers a
search that did not happen. **C17 (duplicate leaves / virtual loss) is squarely
in this category** — it changes which nodes get visited, so its readout still
costs a full window turnover. Sampling/priority knobs are also excluded: they
act at ingest, not at sample time.
