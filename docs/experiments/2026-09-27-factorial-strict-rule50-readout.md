# Strict six-man factorial target readout — September 27

Status: completed historical A–D evaluation readout. This record publishes the
replacement rule50-aware arenas for the four seed121 checkpoints; it does not
register new training or games. The frozen [analysis protocol](evidence/factorial-strict-rule50-20260927/protocol.md)
was pinned before three of the four strict edge results were read. The
[frozen readout](evidence/factorial-strict-rule50-20260927/four-edge-readout.json)
and [independent small-receipt recomputation](evidence/factorial-strict-rule50-20260927/independent-recompute.json)
agree on all four scores and the shared-opening bootstrap.

## Question and fixed comparison

The factorial varied two target recipes while keeping the 58,090,688-row exact
epoch, seed121 initialization, 113,459 updates and model family matched. The
policy choice was BT4-only (A, C) versus an equal BT4/Ceres blend (B, D), with
each policy teacher sharpened separately to T=0.5 before mixing. The main value
choice was equal SF/BT4 (A, B) versus equal thirds SF/BT4/Ceres (C, D), using
the registered Ceres calibration. The question here is how those four saved
checkpoints play against each other under the corrected evaluation protocol.
The prior [factorial readout](2026-09-22-factorial-readout-next24h.md) and
[Syzygy audit](2026-09-23-bootstrap-syzygy-correctness-audit.md) retain the
training and correction history.

The original four arenas had `syzygy_max_pieces=0` and empty Syzygy paths.
Their games lack move histories, so they cannot be repaired by retrospective
tablebase adjudication. The replacement arenas use strict six-man, rule50-aware
root/leaf search and adjudication, with required WDL and DTZ access and saved
PGNs. Each edge passed its frozen receipt for 256 games on 128 shared two-color
openings, 400 simulations per side, training search shape and prior temperature
1.0. An incomplete or unresolved edge would have blocked the four-edge readout.
These strict matches compare the same trained target arms under the new arena
protocol; the old arenas are excluded from every strict estimate.

| Candidate − reference | Target change, other target held | Old no-tablebase Elo | Strict Elo | Strict paired 95% interval |
| --- | --- | ---: | ---: | ---: |
| B−A | BT4-only → equal BT4/Ceres policy; SF/BT4 value | +6.79 | **+6.79** | [−24.72, +38.40] |
| D−C | Same policy change; equal-third value | −16.30 | **+6.79** | [−23.56, +37.24] |
| C−A | SF/BT4 → equal-third SF/BT4/Ceres value; BT4-only policy | +21.74 | **+25.83** | [−6.28, +58.40] |
| D−B | Same value change; mixed policy | +17.66 | **+25.83** | [−3.73, +55.78] |

The strict intervals are the preregistered paired-opening mean ±1.96 standard
errors, transformed from score to Elo. The earlier values describe the actual
tablebase-disabled matches only. Their change of sign or magnitude is not an
estimate of a causal tablebase effect: the replacement matches are new games.
All four strict intervals include zero, and each allows a gain above 30 Elo.

## Shared-opening analysis and decision

All four strict banks use the same opening FEN at each pair ID. The protocol
therefore averages the two policy edges and two value-recipe edges in **score
space**, resampling the same pair IDs across all edges in 20,000 seeded
nonparametric bootstrap draws. Intervals below are unadjusted 95% percentile
intervals for opening sampling. The Elo columns transform each score estimate
and endpoint against 50% for description; they are not coefficients of an
additive factorial Elo model.

| Contrast | Score effect, percentage points | Paired bootstrap 95% interval, points | Descriptive Elo and interval |
| --- | ---: | ---: | ---: |
| Average policy change, (B−A + D−C)/2 | +0.977 | [−2.344, +4.199] | +6.79 [−16.30, +29.25] |
| Average value-recipe change, (C−A + D−B)/2 | +3.711 | [+0.586, +6.738] | +25.83 [+4.07, +47.11] |
| Policy-edge heterogeneity, D−C minus B−A | 0.000 | [−6.055, +5.859] | — |
| Value-edge heterogeneity, D−B minus C−A | 0.000 | [−6.445, +6.445] | — |

The policy average narrowly lies inside a descriptive ±30-Elo band at this
opening-sampling confidence level, but **neither individual policy edge** does.
The value-recipe average favors equal thirds on this conditional seed/panel; its
interval still allows a gain above 30 Elo, while both individual value edges
allow zero. The zero heterogeneity point estimates do not establish equal
effects on the two backgrounds; both intervals are broad. These two
heterogeneity differences are distinct conditional comparisons, not one
identified factorial interaction.

This readout does not select a universally best target. It has one training
seed and one checkpoint per arm. Its intervals cover opening-pair sampling for
those checkpoints, not training-seed variation, the 500M source mixture, or
deployment strength. Four edges and several contrasts were examined with no
familywise multiplicity adjustment. The original A–E training summaries retain
`valid_control=false` against the historical sampler/control, so this is a
matched within-factorial checkpoint result rather than a validated comparison
with the earlier control. The value intervention adds Ceres while reducing
**both** SF and BT4 weights. It cannot isolate an intrinsic Ceres-versus-SF
value effect or show that SF supervision is harmful.

## Separate SF-free E−D context

The earlier strict E−D arena used D's mixed policy and compared D's equal-third
SF/BT4/Ceres value recipe with E's half BT4/half Ceres, with no SF component.
Its separate 256-game, 128-pair result is **E−D +13.58 Elo** with paired 95%
interval [−14.35, +41.68]. The nonnegative point estimate met that
experiment's precommitted provisional-E cost/quality screen, while the interval
still permits a disadvantage and a gain above 30 Elo. E−D was not part of the
four-edge joint bootstrap, and removing SF also reallocates weight to BT4 and
Ceres. It does not establish SF-free value supremacy or that selective SF
corrections lack value. The second seed remained held in the published
[E−D readout](2026-09-23-bootstrap-syzygy-correctness-audit.md#september-23-reboot-recovery-and-completed-ed-readout).

The later Selected-E versus E test asks a different one-teacher cost/quality
question with 576 precommitted pairs under the strict protocol. Its separate
[completed readout](https://github.com/jjoshua2/DeepFin/pull/911) reports
−3.016 Elo [−17.737, +11.694] and passes its own −20 Elo lower-bound gate
for measuring annotation cost. That result is not part of the four-edge joint
bootstrap and does not establish an optimal target recipe.

## Evidence and verification boundary

The [compact evidence index](evidence/factorial-strict-rule50-20260927/README.md)
pins the protocol, frozen result, independent recomputation, validation receipt,
and E−D context. The independent audit hashed each small plan, PASS receipt and
game log, reconstructed candidate score from result and color for every game,
required both halves and exact opening-FEN alignment, and reproduced paired
means, standard errors and the registered shared-pair bootstrap. The original
strict arena verifiers supply the game-level PGN/rule50 and tablebase assurances;
this publication audit did not replay PGNs, hash tablebase files or checkpoints,
read the corpus, or inspect current training. Source operation IDs and content
hashes identify the retained external banks without embedding local host paths.
