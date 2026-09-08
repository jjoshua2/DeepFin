# SF-anchored value bootstrap

Research direction, September 8, 2026. Consider a modest BT4 or Ceres WDL
contribution while retaining the Stockfish value anchor and holding policy
supervision fixed. This is a conditional value experiment, not a new teacher grid,
a trained result or a replacement for the current policy and horizon comparisons.

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
