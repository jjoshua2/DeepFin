# V50 training horizon: four completed bootstrap arenas

## Question and completed scope

Does continuing the existing 35,314,577-position V50 bootstrap beyond one epoch
improve playing strength? Four bounded matches were registered before launch:
epochs 2, 3 and 4 against epoch 1 at 400 simulations, plus epoch 4 against epoch 1
at 100 simulations. All four completed their fixed 256 games (128 color-swapped
opening pairs), without an adaptive game-count extension. This is bootstrap
training, not an RL-loop comparison.

## Results

Positive scores favor the later checkpoint. Intervals are the arena's nominal
95% opening-pair intervals transformed to Elo; they are conditional on these models
and development openings, without correction for selecting among several horizons.

| Checkpoint | Simulations | Score | Elo versus epoch 1 | Nominal 95% interval |
| --- | ---: | ---: | ---: | --- |
| Epoch 2 | 400 | 68.55% | +135.39 | +101.96 to +171.40 |
| Epoch 3 | 400 | 73.83% | +180.15 | +144.67 to +219.49 |
| Epoch 4 | 400 | 75.98% | +200.02 | +164.11 to +240.26 |
| Epoch 4 | 100 | 76.95% | +209.45 | +170.34 to +253.98 |

Continuing training gives a substantial improvement over the one-epoch checkpoint.
Point estimates improve through epoch 4, but the latest increment is uncertain.
Using the same 128 opening pairs to compare scores against the common anchor:

| Difference in anchor score | Percentage points | Nominal 95% interval |
| --- | ---: | --- |
| Epoch 3 minus epoch 2, 400 simulations | +5.27 | +0.01 to +10.53 |
| Epoch 4 minus epoch 3, 400 simulations | +2.15 | −3.15 to +7.45 |
| Epoch 4 minus epoch 2, 400 simulations | +7.42 | +1.19 to +13.65 |
| Epoch 4 at 400 minus 100 simulations | −0.98 | −6.74 to +4.79 |

These secondary intervals use the mean of per-opening paired score differences
plus/minus 1.96 sample standard errors. They are not direct later-epoch head-to-head
Elo estimates. The results do not establish a plateau or a search-scaling interaction.
In particular, the last row does not say that either engine becomes weaker with
more search: both opponents receive the changed search budget.

## Lineage and settings

The three later checkpoints belong to one continuous optimizer trajectory, with
full model, optimizer and scheduler restoration. They are not independent training
replications. The original checkpoint has 68,974 updates; the later boundaries have
137,948, 206,922 and 275,896. The continuation used epoch sampling seeds 102–104,
with continuous augmentation state within that continuation. All use the same
21-root corpus and V50 recipe. No teacher-mixture change was made in this contrast.

| Actual training epoch | Checkpoint SHA256 |
| --- | --- |
| 1 | `6d36f93d040c8babed040159419279af62a444f6466f42b720e12bb72d67ab02` |
| 2 | `0d6bb2ef7c33adc62b60bf53d7911f825985e4f5ee0665cdfe75955eb82ec9b5` |
| 3 | `235534778882e069ca14213fbeb30d646ebb702b945098c05be95d951a2c3607` |
| 4 | `e579bd08f90009924941bb5c773d01af836f6500f95a0250eca2918d7b58679c` |

Arena settings were shared: opening seed 20260917, prior temperature 1.0,
training search shape with noise enabled, rolling concurrency 128, compile enabled,
evaluation batch 4096, 16 book plies and a 300-ply cap. Arena seed is an
opening/search seed, not an additional trained model. The four matches consumed
about 61.5 minutes in total. This measures that search configuration, not an
unqualified deployment-strength estimate.

## Decision and remaining question

Prioritize continued training on the admitted 50,548,069-position corpus, starting
from the completed 35M epoch-4 checkpoint. The selected next horizon is four total
passes over the expanded corpus: seed 105 first, then seeds 106–108, retaining
intermediate checkpoints. This is a prospective choice, not a completed result.
Use bounded 400-simulation readouts against the preserved 35M epoch-4 checkpoint.

That comparison tests the combined benefit of more training and expanded data.
A same-donor, same-additional-update control on the original 35M corpus remains
necessary to isolate the unique-data contribution; it must not be described as
already run. No new policy/value recipe has been selected by these horizon matches.

## Registered expanded-corpus readouts

Four future jobs are queued behind the running first expanded pass: each expanded-corpus epoch 1–4 versus the same
35M epoch-4 anchor, fixed 256 games at 400 simulations. Use common opening seed
20260918, the same search settings above and no automatic game-count extension.
The first readout follows the first expanded pass; the remaining three follow the
three-pass continuation. The latter preserve the intermediate boundaries and are
not independent trained seeds. Compare scores and uncertainty before deciding
whether more training or a direct finalist comparison is useful; a positive estimate
need not exclude zero to earn further compute.

Candidates are bound only after successful training terminal and completion
receipts, matching the registered training-plan hash, completed-summary hash and
actual checkpoint hash. Failed training cannot supply an earlier accidental
checkpoint. Every arena must finish all 128 pairs and pass actual-bank validation.
The [adopted queue evidence](evidence/expanded50m-readouts-20260918.json) records
commands and identities; it is not evidence that any future game has completed.
Each job allows 2,500 seconds plus 30 seconds of outer termination cleanup, with
32 GiB available RAM and 150 GiB disk floors and an exclusive GPU lease. Independent
review corrected exited-leader cleanup before adoption.

## Evidence and verification

[Compact bank evidence](evidence/v50-horizon-20260918.json) records all four
complete-bank hashes, scores and secondary paired differences. Bulk games and
launch receipts remain under
`scratchpad/bt4_joint20/takeover_20260916/v50_epoch_curve_arena_v1/`;
each arm has `arena.games.jsonl`. Independent readout checked all 1,024 game
records, complete pairs, common pair identities/openings and checkpoint lineage.
No games were regenerated for this analysis.
