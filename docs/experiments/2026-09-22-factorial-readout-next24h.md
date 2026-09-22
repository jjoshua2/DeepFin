# Completed 58M factorial and next-day queue — September 22

## Decision and evidence

The value intervention has the more consistent positive signal. Keep BT4+Ceres
value as a serious candidate and test SF removal directly. The mixed policy is
not yet preferred: its two measured effects have opposite signs. These are
working decisions under uncertainty, not claims that either teacher is useless.
All four arms completed 58,090,688 rows and 113,459 updates from identical
initial weights, with one exact epoch and zero same-game repeats within batches.

| Change | Other setting | Contrast | Elo | Paired 95% interval |
| --- | --- | --- | ---: | --- |
| BT4-only to equal BT4/Ceres policy | SF50/BT4-50 value | B-A | +6.79 | [-24.95, +38.63] |
| BT4-only to equal BT4/Ceres policy | Equal thirds SF/BT4/Ceres value | D-C | -16.30 | [-46.88, +14.04] |
| SF50/BT4-50 to equal thirds SF/BT4/Ceres value | BT4-only policy | C-A | +21.74 | [-12.49, +56.40] |
| SF50/BT4-50 to equal thirds SF/BT4/Ceres value | Equal BT4/Ceres policy | D-B | +17.66 | [-13.93, +49.54] |

Each match is 256 games / 128 opening pairs at 400 simulations with prior
temperature 1.0. All four registered matches completed without truncation.
Both policy teachers are sharpened separately to T=0.5 before equal mixing.
Ceres value retains its registered calibration. The value intervention replaces
one-third of the original SF/BT4 mixture with Ceres, so it lowers both existing
teachers. It cannot isolate SF harm or Ceres benefit by itself.

A synchronized bootstrap across the 128 shared opening pairs (50,000 draws)
gives an average policy effect of -4.76 Elo [-27.22, +17.69] and an average value
effect of +19.70 Elo [-3.40, +43.66]. These summarize observed contrasts;
they are not independent training replications or a fitted universal rating.
The two ways of estimating an interaction need not agree exactly because these
are separate noisy head-to-head matches. Do not force Elo transitivity to infer
an unplayed matchup, or count four matches as four independent training seeds.

The [independent banked-data analysis](evidence/factorial-next24h-20260922/four-edge-readout.json)
verifies completion, checkpoint lineage, game-bank hashes and shared opening
identities. Its resampled single-edge intervals differ slightly from the original
registered intervals shown in the table. The uncertainty covers opening-pair
sampling for these trained checkpoints, not population training-seed variance.

## Optimization readout and idle diagnosis

The complete BT4 pipeline screen passed exact input/output parity on 8,236 rows.
Median labeling plus deep verification was 19.332s original serial, 13.461s
projected serial, and 12.468s projected with bounded prefetch. Projection gives
1.436x throughput; prefetch adds 1.080x over projected serial (7.38% less time),
passing its registered 5% threshold. Together the observed time reduction is
35.5%. Fixed ABCCBA order, two observations per variant and one warm shard limit
extrapolation; this is not a 500M fleet throughput claim.

The packed GPU comparison failed before its first optimizer update: the observer
requested `ply`, while the real sampler emits `ply_index`. Its CPU qualification
remains valid, but the failed GPU run establishes no throughput result. Preserve
that run and retry only with a reviewed schema correction and fresh output root.
The queue then had no remaining work, and its supervisor exited. The recorded
`current_gpu` field still named the failed item; host process and GPU inspection,
not that stale field, established idle status.

## Preregistration: SF-free value and a second matched training seed

E keeps D's equal BT4/Ceres policy and replaces its equal-third main value target
with half native BT4 and half calibrated Ceres, with no SF component. The full
E preparation qualifies all 7,108 shards, 35 cohorts and 58,090,688 rows; aggregate
qualification SHA256 is
`4325d6bb319d278d5c844b0e16a82a3211dab1224d9c6443d0bbc25b82af1f03`.
Do not reconstruct native BT4 by subtracting rounded stored mixtures.

Train E at seed121 against the completed D121 control, then independently train
D122 and E122 from the same seed122 initialization and compare them. This fixes
both policy and exposure within each seed, tests SF removal, and supplies a
second training seed without selecting it based on the first outcome. Training
uses the same frozen model/config/runtime and exact one-epoch sampler as the
original factorial. Require complete coverage, finite losses, matched initial
weights, matching planned/realized schedules and valid artifact bindings before
admitting either arena. The second seed changes both initialization and training
order together, identically for D122 and E122.

Each E-D arena uses the fixed 256-game, 128-pair, 400-simulation protocol; retain
all game records and paired intervals. Report both seed contrasts and their
prespecified equal-weight score-margin average, with a synchronized opening-pair
bootstrap where identities align. Two training seeds do not justify a precise
estimate of population seed variance. Positive average with both directions
positive favors E provisionally; negative average with both directions negative
favors D; mixed directions leave the decision unresolved. No extra games based
on closeness to zero, no automatic deployment, and no changing targets mid-run.

Plan three complete training jobs at approximately 13 hours each plus two arenas
and the bounded storage retry: roughly 40 hours of useful backlog, exceeding the
requested next 24 hours. Runtime caps are recovery bounds, not expected durations.
Respect the existing September 26 queue deadline and each job's resource/STOP
checks. A failed admission must remain failed with evidence; never mark a partial
epoch complete to keep the GPU busy. Queue registration and launch verification
will be appended after independent review.
