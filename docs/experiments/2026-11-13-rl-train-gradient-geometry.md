# E3: objective-gradient geometry on fixed TRAIN roots

Current status: technically complete; primary geometry effect unresolved.
The premeasurement record is preserved at commit `6a2f790e5`; the scientific
files remain unchanged. This is a separate diagnostic after
the [completed outcome-bundle study](2026-11-13-rl-resolved-outcome-bundle.md).
That study remains closed with the donor retained and no registered playing gain.

## Preregistration

The question is whether the tested 20% verified-outcome bundle changes alignment
between policy-family and value-family gradients at the same parameters and
inputs. This measures an objective interaction on ingested training examples.
It does not measure applied Aurora/AdamW updates, explain a playing result, or
select an optimal recipe.

Use one identity-selected retained row from each of the 32 qualified TRAIN
games, including all 15 natural and 17 strict-tablebase games. Minimize
SHA256 of `E3-gradient-root-v1:20261113:` plus physical row identity within each
game; all retained rows are eligible. Fix numeric source-game order and the
first row as the repeat-control sentinel. Root identities were frozen from the
accepted metadata before new arrays, model predictions or losses were accessed.
There is no outcome filter, replacement, augmentation or HOLDOUT access.

At each of the frozen donor, FULLA40 and FULLB40 states, evaluate both original
target maps using the same eager, gradients-enabled evaluation forward. The
donor is the sole primary state; endpoints are descriptive. Preserve the actual
loss weights, masks, denominators, fallbacks and coupled-head graph. Measure
policy/soft/future family P, main WDL W, categorical C and remaining common
auxiliaries U over all 423 deduplicated physical shared-trunk parameters,
57,533,899 elements. Count tied Smolgen weights once and preserve their aliases.
The actual architecture has no phase-output adapter.

The primary per-game contrast is
`cos(gP, gWB + gCB) - cos(gP, gWA + gCA)` at the donor, explicitly using sums of
the separately measured component gradients. Report its equal-game mean and
10,000-resample paired game-bootstrap percentile 95% interval, seed 2026111501.
An undefined primary cosine makes the entire primary result inconclusive; do
not discard a root. An interval excluding zero supports a geometry difference
on this retained TRAIN sample. Its sign is not an improvement criterion.
Natural/tablebase, endpoint, layer and optimizer-route summaries are descriptive.

The proposed bound is one 600-second GPU owner, 570-second inner deadline,
cores 20 and 22, two Torch threads and one resident model at a time. At most 99
forwards and 600 gradient calls include the fixed measurements and controls.
There are no optimizer steps, scheduler advances, new teacher queries, checkpoint
writes, playing games or automatic extensions. A fresh nonstepping optimizer
object may supply the qualified parameter map; its state must remain empty.

Reconstruct actual source-ordered scalar totals on every row. Record the
separately backpropagated family-sum versus source-total gradient residual on
the sentinel under each state and target map. BF16 intermediate rounding can
make these gradients differ; this residual is a diagnostic, not a requirement
for bit identity. A repeated sentinel requires exact input, target, forward and
None-gradient coverage, component relative-L2 gradient drift at most 1e-5 and
primary cosine/contrast drift at most 1e-5. Norms at or below 1e-12 make a primary
cosine undefined. These are prospectively chosen measurement-resolution gates,
not proven backend bounds; three sentinels do not bound all numerical variation.
Preserve failures and partials without retuning tolerances or rerunning.

The implementation, immutable bindings and final protocol still require
independent acceptance and a fresh resource admission. No scientific or
implementation preparation alone authorizes execution. Bulk artifacts remain
under `~/chess-artifacts/operations/rl-week-one-20261112-v1/`, outside checkouts.
The week-one and long-term strength objectives remain active.

## Frozen protocol and engineering qualification

The [scientific protocol](evidence/rl-e3-gradient/scientific.json),
[fixed roots](evidence/rl-e3-gradient/root-roster.json) and
[command](evidence/rl-e3-gradient/command.json) are frozen before measurement.
The [source review](evidence/rl-e3-gradient/source-review.json) accepted the
corrected helper. The superseded source findings and exact corrective diff are
retained; neither source version was run against research models during review.

[Synthetic CPU controls](evidence/rl-e3-gradient/cpu-readout.json) passed in
4.182 seconds. They exercised actual loss-term capture, fractional and absent
masks, fallback exposure, differentiable components, connected-zero versus None
gradients, and analytic geometry including undefined cosines. CUDA remained
uninitialized; no research arrays, checkpoint or engine was loaded. These checks
do not qualify real model gradients, BF16 repeat stability or performance.

The [artifact manifest](evidence/rl-e3-gradient/artifact-manifest.json) maps exact
compact copies to their external sources. Final one-way binding review and fresh
resource admission remain separate. No GPU measurement or strength result is
reported in this premeasurement record.

## Completed measurement

The fixed GPU stage completed in 529.468 seconds against its 600-second owner
cap, using 525.52 recorded CPU seconds and peak allocated CUDA memory of
709,286,400 bytes. All 96 state/root measurements and three repeat sentinels
completed: 99 forwards and 600 component/control gradient calls. There were
zero optimizer steps, checkpoint writes, new teacher annotations or HOLDOUT
array reads. All owned processes closed. The wall time includes CPU hashing,
gradient transfers and statistics; it is not exclusive GPU kernel time.

The primary donor contrast is +0.0091304, with the registered paired game
bootstrap 95% interval [-0.0816912, +0.0928776]. It includes zero: the average
alignment effect of the tested outcome bundle remains unresolved on these
32 retained TRAIN roots. This is neither equivalence nor evidence that the
bundle improves or harms playing strength. Individual contrasts span both
signs; no root or outcome class was excluded.

| Fixed state | Mean P versus A value cosine | Mean P versus B value cosine | Mean paired difference |
| --- | ---: | ---: | ---: |
| Donor, primary | 0.034661 | 0.043791 | 0.009130 |
| FULLA40, descriptive | -0.042290 | -0.023201 | 0.019089 |
| FULLB40, descriptive | -0.031325 | -0.031389 | -0.000064 |

The endpoint rows are descriptive, with no checkpoint selection or additional
inferential gate. The same 32 physical games underlie all three rows; these
are not 96 independent samples. Game-bootstrap uncertainty is conditional on
this retained TRAIN cohort and fixed root sample. Shared opening-family
correlation is not separately corrected.

At the donor, median whole-trunk component-gradient norms were 27.874 for P,
3.964 for search-only WDL, 8.326 for search-only categorical, 10.191 for
outcome-bundle WDL, 8.353 for outcome-bundle categorical and 0.0121 for remaining
common auxiliaries. These are separately measured, weighted B1 evaluation
components. They are not training B256 norms or applied optimizer updates, and
ratios of medians are not medians of per-root ratios. Larger norms do not by
themselves imply a useful direction or justify changing a weight.

All three sentinel repeats had zero measured component relative-L2 drift and
zero primary cosine/contrast drift. This establishes the controls at the three
sentinels, not universal backend determinism. The maximum observed relative
residual between separately backpropagated family sums and the actual combined
source-total gradient was 0.0078073. This predeclared BF16 diagnostic was not a
pass threshold; the primary quantity remains the specified sum of separately
measured WDL and categorical gradients. It must not be relabeled as the actual
Aurora/AdamW update direction.

A separate 1.149-second finite CPU readout reproduced all 96 reported primary
cosines from saved Gram matrices, checked root ordering, same-input A/B host
signatures, source loss arguments, repeat flags and layer/optimizer-route
aggregates, and exactly reproduced the 32-game 10,000-resample NumPy bootstrap.
It reused the saved observations without another model forward or gradient.
The [raw compact readout](evidence/rl-e3-gradient/actual-readout.json),
[metadata verification](evidence/rl-e3-gradient/metadata-verification.json) and
[portable Gram statistics](evidence/rl-e3-gradient/gram-sufficient-statistics.json)
preserve these observations and their external source identities.

This study validates the specified measurement and leaves its primary contrast
unresolved. It does not explain the earlier playing nonpass, establish that
clipping caused it, or select a learning-rate, dose, teacher or loss-weight
change. The donor remains retained. The completed E2TB decision and playing
panels remain closed; no Elo or external-engine result follows from E3. A next
learning comparison requires its own hypothesis, controls and fixed readout.
The week-one and long-term strength objectives remain active.

The [independent actual review](evidence/rl-e3-gradient/actual-independent-review.json)
and [root acceptance](evidence/rl-e3-gradient/root-actual-acceptance.json) accepted
the complete technical delivery and the unresolved primary classification.
At that acceptance snapshot, all 87 recorded owner stages and 337 recorded
process births were closed. Week-one GPU-intended owner time was 9,202.945
seconds against the 172,800-second ceiling; E3 added 529.468 seconds and did not
change the closed E2TB study's charged 2,814.029 seconds. Publication CPU
validation is additional work recorded separately. The
[results manifest](evidence/rl-e3-gradient/results-artifact-manifest.json)
preserves exact supporting copies while leaving all premeasurement scientific
files unchanged.


## Separate teacher and selection follow-up

A [source/metadata audit](evidence/rl-e3-gradient/teacher-selection-audit.md.txt),
with [independent review](evidence/rl-e3-gradient/teacher-selection-audit-review.json),
resolved two preparation questions without new model inference. In the actual
E2TB routes, upstream retention used own-network/search disagreement and policy
KL; the Stockfish-gap field was attached afterward. Production replay gap weight
was zero, and the fixed uniform learner bypassed its priority sampler. There is
no evidenced Stockfish-numeric selection through those traced routes. This does
not prove that disabling its process would preserve scheduling, integrity checks
or every RNG event, and uniform retained-row sampling does not restore dropped
positions.

The linked conversion receipt documents teacher hash `1d3c…` as the original
`BT4-it332-vanilla-winner.onnx` artifact and `809d…` as its FP16-internal,
FP32-I/O conversion. These are documented variants of the same named artifact,
rather than evidence of two independently trained teachers. The static conversion
receipt establishes neither numerical equivalence nor the exact export/head
mapping, training pedigree or complete original 100M-row target lineage. This
additional lineage evidence does not turn current P1 Stockfish reply/value labels
into the donor's original P0 teacher distributions or justify an anchor weight.
It informs a separately specified future teacher study; it is not an E3 result
or an explanation of the playing nonpass.
