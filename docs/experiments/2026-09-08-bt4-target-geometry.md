# BT4 target geometry: stored ties and a raw-cp control

The fixed training sample shows that C's selected set can be much wider than three
moves, and that softmax over raw effective cp changes more than overall sharpness.
It softens the sampled nonmate groups while restoring score distinctions erased
by the WDL mapping in some saturated/mate positions. This establishes a distinct
control that can be constructed on the bank; it selects no temperature or training
run and says nothing about playing strength.

## Scope and existing population evidence

The [preregistration](../../scratchpad/bt4_joint20/target_geometry_v1/preregistration.md)
reuses exactly 128 seen-training rows from 43 source-qualified games, originally
sampled without replacement with seed20260906 from the first 8,192-row SF shard.
This is **one shard, not a representative sample of the 18,910,484-row corpus**.
Its mean SF/C entropies, 0.7472/0.6603 nats, differ from the full-corpus
0.5690/0.5155. No additional positions, inference, corpus scan or gameplay were run.

The already published [C corpus summary](../../scratchpad/bt4_joint20/publication_originals/data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_sfclose_C20T05/bt4_policy_mix_summary.json)
reports 8,174,861 rows with multiple selected moves (43.229%), 4,951,025 with a set
wider than the stored SF maxima (26.181%), and 3,434,000 with multiple stored SF
maxima (18.159%). Those counts already answer how often C can redistribute mass;
they do not give selected-size, raw-score-spread or saturation histograms.

## What the C filter actually selects

C redistributes the SF mass in the union of **all stored SF maxima** and the first
three d9 ranks within 20 effective cp of rank one, using sharpened BT4 (T=0.5).
The three-rank cap applies only to the added rank window. The analysis uses the
existing rank and candidate-set functions with retained original MultiPV ordering.

| Sample property | Rows / 128 |
| --- | ---: |
| C set size 1 / 2 / 3 / greater than 3 | 65 / 25 / 21 / 17 |
| Multiple stored SF maxima | 29 |
| C set wider than stored maxima | 36 |
| Stored maxima contain different raw effective cp | 21 |
| Entire legal q vector exactly flat | 10 |

The largest selected set has 60 moves. In all 21 rows with unequal raw scores among
stored maxima, those maxima already have identical mapped q; float16 equality
alone does not explain their lost distinction. Nine of the ten globally flat-q
rows contain a mate-band observation. A flat row can also contain only one legal
move, so flatness alone does not prove harmful saturation.

The source summary's `temp_recovery_skipped_saturated=503` is not a population
saturation count: the zero-floor recovery path returns before incrementing it for
exactly flat vectors. Its support range 1–84 counts positive probabilities after
float32→float16 storage, while `policy_support_lost_to_float16` counts move entries,
not positions. This analysis uses its separately registered q-spread definition.

## Raw-effective-cp distributions

For each legal move, the control is `softmax(effective_cp / temperature_cp)`.
Scores retain the generator's mover perspective and mate-distance encoding;
there is no new clipping or mate remapping. “Stored” below means float64 softmax,
then the existing float32→float16 path, then legal normalization for measurement.
Support means strictly positive entries, including numerically tiny probabilities.

| Target | Entropy, ideal / stored | Mean support, ideal / stored | Stored mean top-one mass | Stored mean mass outside C |
| --- | ---: | ---: | ---: | ---: |
| Actual SF | — / 0.7472 | — / 8.63 | 0.7307 | 0.0831 |
| Actual C | — / 0.6603 | — / 8.66 | 0.7718 | 0.0831 |
| Raw cp T=10 | 0.543007 / 0.543010 | 21.79 / 9.88 | 0.7849 | 0.0417 |
| Raw cp T=20 | 0.877252 / 0.877249 | 21.79 / 15.51 | 0.6817 | 0.1386 |
| Raw cp T=40 | 1.386119 / 1.386118 | 22.53 / 19.48 | 0.5403 | 0.2860 |
| Raw cp T=80 | 1.928881 / 1.928890 | 22.54 / 21.62 | 0.3907 | 0.4399 |

The largest pre-normalization stored mass error across these candidates is 0.0004121.
Storage removes tiny tails with little entropy change. Banked BT4 raw/T=0.5 have
mean entropy 2.1871/1.3286 and mean support 27.58 each; these are their existing
teacher distributions, not newly float16-materialized candidates. Complete top-one
concentration, tail, support and per-row results are in the
[compressed readout](../../scratchpad/bt4_joint20/target_geometry_v1/readout.json.gz).

The registered SF evaluation strata use rank-one d9 effective cp: below −200,
inclusive [−200,200], above 200. Any legal move in the existing mate band
(`abs(effective_cp)>32000`) sends that row to a separate group. These are
policy-observation strata, not changes to stored value labels.

| Stratum (rows) | SF entropy / support | C entropy / support | cp10 stored | cp20 stored | cp40 stored | cp80 stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Below −200 (31) | 0.2945 / 4.35 | 0.3918 / 4.35 | 0.5295 / 10.23 | 0.9316 / 15.61 | 1.5348 / 19.81 | 2.1416 / 19.97 |
| Within ±200 (8) | 0.2472 / 1.88 | 0.2180 / 2.25 | 0.7855 / 13.00 | 1.2796 / 16.25 | 1.7503 / 25.38 | 2.1668 / 29.38 |
| Above 200 (50) | 0.5078 / 7.12 | 0.4683 / 7.14 | 0.7093 / 13.42 | 1.1525 / 21.18 | 1.7698 / 26.50 | 2.4196 / 30.68 |
| Mate present (39) | 1.5167 / 15.33 | 1.2105 / 15.33 | 0.2908 / 4.44 | 0.3986 / 8.00 | 0.7013 / 9.03 | 1.0819 / 9.72 |

Each candidate cell is mean entropy / mean support. Even T=80 is sharper on average
than C in the mate-present group, while T=10 is softer in all three nonmate groups.
Raw-cp targets preserve encoded mate distances that q may collapse; their effect
cannot be summarized as uniformly softening SF. These small correlated groups,
especially the eight near-even rows, do not estimate population effects.

## Interpretation and evidence

T=10 is closest to C's sample mean entropy and T=20 to SF's among the four tested
values. This is a descriptive comparison, **not a final temperature choice**.
A raw-cp control would test score-to-policy mapping as well as tail width. A later
training comparison must keep value/history arrays fixed and use matched runtimes;
[current main supports uninterrupted multiple epochs](../toolchains.md#uninterrupted-offline-game-epochs),
but its objective normalization differs from the frozen H20 runtime.

The full 18.91M-row raw-score join remains unqualified. The old corpus lacks the
new optional row-provenance arrays; the retained raw observations and existing
shuffle/join checks provide a route, but the top-eight rank sidecar and rounded SF
probabilities cannot reconstruct all-legal raw scores. No full control corpus was
created here.

The [evidence manifest](evidence/bt4-bootstrap/target-geometry-manifest.json) binds
original and published hashes, including the losslessly compressed readout,
[analysis script](../../scratchpad/bt4_joint20/target_geometry_v1/analyze.py),
[prelaunch semantic review](../../scratchpad/bt4_joint20/target_geometry_v1/prelaunch_review.json),
[execution receipt](../../scratchpad/bt4_joint20/target_geometry_v1/execution.json),
and [completed independent review](../../scratchpad/bt4_joint20/target_geometry_v1/independent_review.json)
(PASS).
The once-only process exited zero in 0.723 seconds (0.269 seconds in the analysis
body), with GPU hidden, two CPUs and low priority. Existing NPZ/raw training banks
remain external and are identified by hash; historical scripts retain original
host paths and are evidence, not portable launch instructions.

## Bank reuse check for a larger Soft-SF sample

A subsequent inspection found that the existing 4,000-position audit bank retains
112,172 raw d9 lines, including 3,072 explicit mate lines, but lacks the original
training source/shard/row/history join and actual stored C targets. Its normalized
FEN join cannot establish that it is a training-only sample. It was therefore not
used to choose a Soft-SF temperature.

The already-reviewed 128-row, 43-game sample above remains the available
source-qualified comparison. Its C entropy is 0.660261 nats; the stored and
renormalized raw-cp targets at 10, 20, 40 and 80 cp have mean entropies 0.543010,
0.877249, 1.386118 and 1.928890 respectively. These values were reused without
repeating the target calculation. Ten cp is the exploratory nearest value, but
one selected shard does not establish representative corpus statistics or strength.

The [bank-readiness evidence](evidence/bt4-bootstrap/soft-sf-bank-readiness-manifest.json)
preserves the inspection plan, original bank pins, source inspection and result.
The inspection body took 3.149 seconds with GPU hidden and two low-priority CPU
cores. A representative source-qualified training sample and the full raw-score
join remain needed before a corpus-wide entropy-matched control is ready.
