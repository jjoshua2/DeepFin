# SF-anchored value bootstrap

Research direction, September 8, 2026. The initial modest-dose proposal below
led to matched teacher diagnostics. The [updated next contrast](2026-09-08-bootstrap-next-value-and-policy-contrasts.md)
is **B100V50: equal SF and BT4 value supervision with B100 policy targets fixed**.
V10 remains optional; no value-mixed training or match has launched. The later
dose choice does not change the historical proposal or establish a best dose.

## What the existing evidence supports

The user's assessment that game outcome `z` added little is consistent with the
historical bootstrap screens, within their tested scope. The September 2
[Z-residual record](../experiment_ledger.md#2026-09-02-0620z--z-residual-screen-verdict-prereg-2026-09-01--operator-amendment-h1h4-no-λ--002-is-supported-on-the-deep-sf-ruler-arm-c-qzsegment-stays-parked-the-sf-blind-mate-hypothesis-is-untestable-on-an-sf-ruler) reports 6,263 rows from 1,447 games:
Q-only MAE against deep SF was 0.0117, versus 0.5795 for Z alone. A 1% Z blend
worsened held-out MAE by 0.00137 [0.00104, 0.00172], despite a small MSE gain.
Earlier Q/Z training comparisons also lost on their internally matched,
zero-history protocol; that history limitation prevents treating them as a modern
matched-history verdict. These results do not establish that outcomes are useless
or that SF misses no value information: an SF reference cannot reveal blind spots
shared by shallow and deep SF.

The current original-corpus control stores SF's cp-derived parametric WDL
distribution, from the sample's side to move, in `search_wdl`; `wdl_target` stores
the game outcome. The offline configuration's
`search_wdl_frac: 1` therefore means **SF value supervision**, despite the column
name; its `sf_wdl_frac: 0` does not remove that anchor. The BT4 policy treatments
preserve these value arrays. The search uses the trained `wdl` head, so an auxiliary
`sf_eval` change alone would not test the proposed mechanism.

Existing BT4 and recent Ceres policy banks do **not** retain matched teacher WDL
triples. The Ceres adapter's finite-value check is not a saved value bank or a
value-quality result. Ceres's exact training-data lineage also remains unresolved;
a different network does not establish independent errors or freedom from LC0 bias.
See [Ceres readiness](2026-09-08-ceres-teacher-readiness.md) for the qualified scope.

## One focused contrast

Let `S` be the unchanged stored SF WDL and `N` one qualified neural teacher's WDL,
both normalized in win/draw/loss order from the sample's side to move. Compare:

| Control | Challenger |
| --- | --- |
| Main value target `S` | Main value target `(1 − α) S + α N` |

A modest `α = 0.10` is an illustrative starting point, not a selected optimum or
launched weight. Choose one teacher for the first training comparison after checking
semantics, coverage and cost. Do not simultaneously vary teacher, dose, policy
recipe and training horizon. Keep the same policy target bytes, rows, order, masks,
initialization, training budget and other loss settings. Shared-trunk learning may
still change the learned policy; the controlled quantity is policy supervision.

The question is whether complementary value supervision improves playing strength,
not whether it reproduces SF more closely. SF agreement and held-out calibration
are useful diagnostics, but an SF-agreement threshold must not veto the very
complementary signal being tested. Use matched paired games for the deciding
comparison, with a protected higher-search readout when that is the registered
question. A same-seed development win remains subject to fresh confirmation.

## Useful preparation before training

Bank WDL on the already selected 128 full-history training positions first. Retain
original input/row identities, raw three-output values, head names, model/runtime
pins and batch metadata. BT4 probability output must not be softmaxed again; the
Ceres primary value logits require their declared conversion. Neither Ceres
`value2` nor an action-value head is interchangeable by name. Policy temperature
0.5 is not a justified WDL temperature. Qualify orientation, normalization and
batch consistency before treating either triple as a label.

The current BT4 labeler requests policy only; it is not discarding returned WDL.
A future qualified group could request `[policy, named_WDL]` in one call and retain
the three values, avoiding another full inference later. Incremental cost remains
unmeasured and old coverage remains policy-only; no active labeler change is implied.

A value rewrite needs explicit provenance and evidence that policy supervision and
all intended non-value fields survive unchanged. Current policy-only qualification
schemas do not automatically admit a changed value target. Trace the actual main
WDL consumer and any auxiliary-target coupling; the current original corpus lacks
categorical targets, so its categorical rebuild is inert. Reuse a completed control
only if the exact trainer, objective, policy and canonical schedule match. A newer
runtime/objective requires a fresh matched control.

Record the chosen teacher/head, mixture weight, qualified corpus/runtime, actual
labeling cost, training/match budget and stopping/readout rule before launch.
These are remaining experiment details within the ongoing research scope, not
additional per-test permission gates. No value labels, rewrite, training or match
were produced for this record.


## First matched value readout

The [registered CPU collection](https://github.com/jjoshua2/DeepFin/pull/579#issuecomment-5589203654)
completed all 128 rows for both teachers. Raw BT4 winner probabilities and Ceres
primary/secondary logits are retained. Independent review recomputed every
reported statistic from native arrays and checked the row identities, inherited
selection weights and SF/outcome bindings. Neither teacher was run again for the
review. The [evidence manifest](evidence/value-bootstrap/manifest.json) binds the
[per-row outputs](evidence/value-bootstrap/matched-value-rows.json),
[full descriptive readout](evidence/value-bootstrap/matched-value-readout.json),
[matched input arrays](evidence/value-bootstrap/matched-inputs.npz) and both
[collection](evidence/value-bootstrap/collector-source.txt) and
[analysis](evidence/value-bootstrap/readout-source.txt) sources.

Use centered value `q = W − L`; expected score is `(1 + q) / 2`. These are weighted
descriptions of the existing training sample, not accuracy against a truth label.

| Target/head | Mean absolute q | Mean draw probability | Mean WDL entropy (nats) |
| --- | ---: | ---: | ---: |
| Stored SF | 0.87286 | 0.05854 | 0.24434 |
| BT4 winner | 0.93224 | 0.04914 | 0.08850 |
| Ceres primary, T1 | 0.93066 | 0.05180 | 0.09336 |
| Ceres secondary, T1 | 0.94795 | 0.04388 | 0.05384 |

BT4 and Ceres primary differ by only **0.00950 mean absolute q** and **0.00792
mean WDL total variation**. Their q residuals relative to SF have weighted
correlation **0.978**. Both are more decisive than the stored SF target. That
supports starting with one operationally convenient BT4 value candidate, rather
than paying for two similar teacher arms immediately. It does not establish
which teacher is better or whether either corrects SF bias. Only six sampled
positions have `|q_SF| ≤ 0.1`, while 111 have `|q_SF| ≥ 0.8`; the sample is weak
evidence about subtle equal-position or fortress disagreements. Ceres's native
head/temperature blend remains a different package, recoverable from the saved
raw logits without repeating inference.

A value-mixture win could partly reflect changed confidence. Keep that possible
mechanism explicit, and consider a suitable SF-only confidence control if the
first value candidate is promising. Do not reinterpret this diagnostic as
calibration, tune temperatures against its outcome labels, or discard the
complementary-teacher hypothesis because of SF disagreement.

## Lower-cost value backfill and future capture

An additional [input check](evidence/value-bootstrap/derived-input-equivalence.json)
found bitwise-identical BT4 feeds for all 128 stored float16 tensors versus the
original float32 inputs. The existing converter already does the necessary work:
consumed history/castling/color planes are binary; rounding the rule50 plane
recovers every clipped integer counter from 0 through 100; remaining consumed
metadata is overwritten and extra features are discarded. The full 175-plane
input still differs after float16 storage, so this does **not** recover the
original full-tensor input key or establish history authenticity.

This provides a route to label existing derived rows directly, without replaying
millions of raw histories. A producer must retain source-qualified shard/row
identity and enforce the encoding/domain contract while consuming the actual
corpus. Full-corpus coverage and throughput have not been measured. A bounded
reusable prefix can establish cost before allocating the complete value pass.

[PR #580](https://github.com/jjoshua2/DeepFin/pull/580) separately adds optional
WDL retention to future raw BT4 labeling. It preserves old policy-only groups and
reports their missing value coverage. It has passed focused tests, static checks
and independent review, but the running labeler has not adopted it at this update.
No value-mixed training has launched.
