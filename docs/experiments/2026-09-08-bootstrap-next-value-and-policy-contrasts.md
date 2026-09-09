# Next bootstrap contrasts: value first, selective SF policy use

September 8, 2026 local time; the dose analysis completed September 9 UTC.
**B100V50 is the selected first substantive value comparison; it has not trained
or played.** The completed [SoftSF10/B100 result](2026-09-08-soft-sf-qualified-training-sample.md#completed-softsf10-versus-b100-result)
favors B100 at both budgets. This motivates separating useful SF information from
forcing the whole policy to imitate SF rankings. The priorities below are research
choices, not a launched queue or a claim that their targets improve play.

## Ongoing storage work supports the whole roadmap

Continue SSD-to-external transfers and **verified local reclamation** alongside
research so the approximately 100M-position bootstrap has room. The
[capacity record](2026-09-08-bootstrap-storage-capacity.md#continuing-storage-objective--september-8-update)
tracks working data, labels, rewrite/transfer temporaries and space actually
recovered. The 150 GiB reserve is an operating floor, not full-scale capacity
proof. G20T1's verified archive now supports completed reclamation of its 2,309
local shards, with both summaries retained; G20T05's copy operator has started.
Preserve live sources and selected controls while continuing through eligible
cold policy recipes. Assess only eligible payload components of already-verified
archived pools, preserving checkpoint and holdout exclusions.

## First: B100V50 with policy supervision fixed

Keep B100's 100% sharpened BT4 policy target bytes and compare its SF-value control
with **50% normalized stored SF WDL + 50% normalized native BT4 WDL**. Both triples
are W/D/L from the sample's side to move; BT4 probabilities are not softmaxed again.
Change only the main value target, preserving rows, order, masks, initialization,
training budget and other loss settings. Shared-trunk learning may still alter the
learned policy; it is policy supervision that stays fixed.

The existing 128-row weighted training sample gives a reason for a substantial
contrast, not an optimal-dose estimate. After simulated float16 target storage:

| BT4 value share | Mean WDL total variation from SF | Mean absolute change in q=W−L | Weighted fraction with absolute q change ≥ .01 |
| --- | ---: | ---: | ---: |
| 10% | .00645 | .00679 | 18.0% |
| 30% | .01936 | .02037 | 43.8% |
| 50% | .03227 | .03395 | 52.4% |

All 128 stored triples change even at 10%; V10 is not a no-op. V50 offers a clearer
intervention while retaining half the SF anchor. V10 remains an optional follow-up;
there is no mandatory three-dose training ladder. The sample has 111 strongly
decisive SF rows and only six near-equal rows. Draw-mass changes can be substantial
without large q shifts. These weighted, correlated training observations establish
neither value accuracy nor corrected bias, calibration, or population precision.

The frozen V10 preparation is preserved. [PR592](https://github.com/jjoshua2/DeepFin/pull/592)
merged reusable alpha arithmetic and strict B100V50 admission, retaining exact
historical V10 defaults and producer compatibility. The new V50 rewrite plan is
[frozen and independently reviewed](2026-09-08-bootstrap-storage-capacity.md#account-for-the-next-value-rewrite),
with no rewrite or training launch. Complete native-WDL
coverage, exact policy/non-value preservation and the actual corpus/schedule pins
remain required evidence. The full label pass is ongoing; partial coverage is not
a qualified full-corpus recipe. The planned comparison uses unchanged B100 only
if the original trainer, objective and canonical one-epoch schedule still match.
A new runtime or objective needs a matched control. Freeze the actual training and
two-budget match manifests and costs before execution; no new run is claimed here.

The [prospective comparison registration](../../scratchpad/bt4_joint20/publication_20260908_storage_objective_v1/v50/TRAINING_COMPARISON.md)
retains fresh seed-zero initialization and the original B100 trainer, 18,910,484-row
canonical schedule, 36,935 updates and 420 windows. It proposes the established
ordered 100-simulation SPRT and protected 400-simulation fixed 128-pair probe
against B100. The **27,000-second combined GPU-stage ceiling** covers training
and both arenas; CPU qualification, schedule checks, arena preparation and the
value rewrite have separate allocations. It is not a 7.5-hour end-to-end budget.
Actual corpus qualification and executable training/match manifests remain pending.

## V50 training and arena draft — September 8 evening update

The [complete draft bundle](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/v50-training-arena-draft.tar.gz)
now contains the original comparison registration, training and matched-arena
proposals, exact prospective command allocation, finalizer and validation evidence.
The [per-member manifest](evidence/bt4-bootstrap/v50-training-storage-progress-manifest.json)
pins the original bytes. **These are unlaunched drafts, not qualified executable
training or match manifests.** The alpha coordinator is pinned to reviewed
`5afc1e2f0923704b2bb8f1aa7ffe45cc15141757`; the original trainer and historical
invalid-control limitations remain unchanged.

Draft envelopes are rejected by the launcher. Final binding needs the genuine
completed V50 derive/value-rewrite summaries and dataset qualification, then one
completed prospective schedule. It never manufactures a BT4 policy-mix receipt.
The proposed schedule uses the existing completed SoftSF runner with only its
profile binding changed: CPUs 0,1, two verifier workers, 1,800 seconds including
cleanup, a 150 GiB reserve and 32 MiB sampled output cap. The original training
command still uses 16 planning and 16 loading workers. The later single CPU arena
preparation checks the actual completed checkpoint and full-history opening panel;
its 300-second stage allocation is separate from the GPU-stage ceiling above.

Parent review found that the first finalizer called launch freshness validation
at the completed-training transition. Successful training creates the state/run
directories, so that call would always refuse arena preparation. The corrected
[finalizer snapshot](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/v50/finalize.py.data)
compares the immutable draft and saved training manifest, preserves completion,
input and checkpoint identity checks, and invokes the existing completed-training
metadata admission. It does not claim to verify checkpoint contents before the
later CPU preparation. Seven synthetic cases passed: existing completed outputs
are accepted; altered profile, input pins, checkpoint path, summary hash, budget
or saved manifest are refused. The [independent parent review](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/v50/completed_transition.parent_review.json)
closed the finding. Original author evidence and the prior finalizer remain in
the bundle, with the [correction receipt](../../scratchpad/bt4_joint20/publication_20260908_v50_storage_v1/v50/completed_transition.author_validation.json)
identifying the superseded hash. No operational corpus, schedule, model, training
or arena was processed for this preparation.

## Next policy question: attenuate large SF deficits

Preserve BT4's relative ordering and probability ratios among moves SF considers
acceptable, while reducing mass on large raw-SF deficits. This asks SF to flag
suspected tactical mistakes without replacing the ranking of every plausible move.
A diagnostic uses B100 temperature-.5 probabilities B and raw d9 score deficit Δ:

`weight(m) = max(0.1, exp(−max(0, Δ(m) − gap) / 100))`

Normalize `B(m) × weight(m)`. Moves inside the gap all receive weight one, so their
ratios and ranking survive normalization. The .1 bound is a relative multiplier,
not a minimum output probability. The existing 4,096-row bank gives:

| Gap | Weighted original mass exposed to attenuation | Mean target TV | Top move changes |
| --- | ---: | ---: | ---: |
| 100 cp | 13.43% | .05177 | 3.86% |
| 200 cp | 6.32% | .03102 | 2.10% |

This measures intervention size, not tactical correctness or a best gap. It leaves
all mate-containing rows unchanged: **27.56% of weighted rows**. That flag means
any legal alternative has a mate-encoded score, so it also excludes positions
with a losing tactical alternative. A future categorical treatment must distinguish
winning-mate groups, nonmate choices with losing alternatives, and all forced
losses, without treating differences across the mate band as ordinary centipawns.
That treatment is proposed, not measured here. A subsequent
[parent descriptive category audit](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/tactical_attenuation/MATE_FOLLOWUP.md#saved-bank-category-count)
found 604 rows with only losing-mate alternatives (14.75% weighted), 398 with a
winning mate (9.72%), and 127 with every move a forced loss (3.10%). These counts
were not independently re-audited here and do not establish BT4's mass on those
moves or the truth of the mate claims. The d9 sample cannot establish
search-depth stability. The [first candidate selected for implementation qualification](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/tactical_attenuation/FIRST_CANDIDATE.md)
uses gap 100 cp, decay scale 100 cp and relative floor .1, with that categorical
mate treatment and unchanged SF value. This chooses a substantive contrast, not
an optimum. It needs full legal-move raw d9 scores; the truncated rank sidecar is
insufficient. Exact source joins, stored-target semantics and non-policy preservation
remain to be qualified. No policy training has launched.

A later joint correction could use SF move deficits, weighted by BT4 policy mass,
and the root SF/BT4 value disagreement to decide when correction is warranted.
Small curves could be fitted against deeper SF on a source-game-separated split,
with thresholds and held-out evaluation frozen before fitting. Root BT4 WDL is
one value for the position, not a value for each move; no successor BT4 values are
banked by this diagnostic. A deeper-SF reference shares SF's biases, so improved
agreement would qualify a diagnostic fit, not establish playing strength. Keep
this joint policy/value mechanism separate from the fixed-policy V50 contrast,
and require the eventual paired arena to decide whether it helps.

## Near-tie reranking remains optional

Reverse near-tie reranking is a lower-priority alternative. On the saved 4,096-row
bank, restricting to BT4's top three moves with probability at least .8 times its
top probability gives multiple choices on 13.58% of weighted rows; SF would pick
a different move on 6.15%. A rank-only permutation could redistribute the same
probability values within that set, preserving its mass and entropy before storage.
It would change BT4's ranking among acceptable moves, unlike attenuation, so keep
the questions separate. These are decision counts, not measured rewritten targets
or better play; no threshold is selected as optimal.

## Evidence and scope

[Value/reverse-policy aggregates](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/value_policy_analysis/readout.json),
[analysis plan](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/value_policy_analysis/analysis_plan.json)
and lossless per-row outputs are bound in the
[publication manifest](evidence/bt4-bootstrap/softsf-results-value-priorities-manifest.json).
[Attenuation aggregates](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/tactical_attenuation/readout.compact.json)
and their [independent arithmetic review](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/tactical_attenuation/independent_review.json)
are also included; the original 8,192 row-gap records remain externally pinned.
All diagnostics reuse saved training banks; no engine search, teacher inference,
corpus rewrite or training was performed for these new aggregates. The result
review preceded this publication; its reviewer authored this prose, and a separate
parent review covers publication accuracy. Playing strength remains the deciding
measurement for any subsequently qualified recipe.
