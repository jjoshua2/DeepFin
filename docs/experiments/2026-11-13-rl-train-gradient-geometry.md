# E3: objective-gradient geometry on fixed TRAIN roots

Status: preparation; no execution or result. This is a separate diagnostic after
the [completed outcome-bundle study](2026-11-13-rl-resolved-outcome-bundle.md).
That study remains closed with the donor retained and no registered playing gain.

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
