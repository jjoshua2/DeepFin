# SF-free BT4+Ceres target preparation

## Question and registered comparison

Does Stockfish value supervision help the current bootstrap when the BT4:Ceres
ratio stays fixed? Arm E retains factorial D policy exactly and replaces its
equal-thirds SF/BT4/Ceres WDL with half native BT4 and half calibrated Ceres WDL.
The working hypothesis remains that BT4 and Ceres can help both heads. The
historical online-RL SF guidance does not establish SF necessity in this bootstrap.

The [continuation readout in PR #809](https://github.com/jjoshua2/DeepFin/pull/809)
records the completed contrasts and preregisters E versus D: the same 58,090,688
rows, 7,108 shards, 35 cohorts, seed 121, batch 512, initialization, optimizer,
losses and one exact epoch of 113,459 updates. D and its two matches finish first.
One fixed 256-game, 128-pair, 400-simulation arena uses prior 1.0 and seed
2026092101. Report the paired estimate and uncertainty. A nonnegative estimate
supports provisional E with its collection-cost benefit; a negative estimate
favors D for this screen. Neither establishes universal teacher usefulness.
Do not extend the sample to chase significance. Caps: CPU preparation 16 hours,
training 20 hours plus 2 hours paused, arena 1 hour; honor existing resource floors,
exclusive GPU lease and STOP markers. Full qualification and immutable training
plans remain required before launch.

## Implementation

`scripts/bootstrap_sffree_targets.py` consumes hash-pinned cohort manifests and
original native BT4 WDL banks. It never subtracts rounded V50 values to recover
BT4. Policy uses the same D construction; value is 0.5 normalized native BT4 plus
0.5 Ceres, with Ceres equal to 0.6 softmax(primary / 0.55) plus
0.4 softmax(secondary / 1.5). Policy and WDL are stored as float16 schema-2 overlays.

The builder verifies sealed base membership, original teacher model/output and
attribute pins, every cached teacher array digest, game/ply identity, and exact
LC0 feed bytes for each row. Both policy and main-value masks must be fully active.
It rejects mutated teachers, input/output overlap, partial teachers and reused
output directories. It checks STOP and a 150 GiB free-space reserve while building.
Original historical producer metadata is pinned and checked through the existing
cached-bank validator; this does not independently reproduce all historical
inference invocations. A completed cohort alone is not full training admission.

## Evidence on September 21

The [coverage receipt](evidence/2026-09-21-sffree-targets/coverage.json) establishes
metadata coverage of all 58,090,688 rows and all 7,108 shards, with native BT4 and
both Ceres value heads. It is explicitly a metadata-only audit. All 35 prepared
manifest identities are [banked](evidence/2026-09-21-sffree-targets/manifests.json).
Full preparation must still verify actual array and row-feed bytes.

The [real-data smoke](evidence/2026-09-21-sffree-targets/real-smoke.json) checks
one 8,192-row shard from each of three source families: original20m, common G10,
and adapted saved-raw labels. All 24,576 rows pass schema-2 storage qualification,
retain D policy and every non-value array exactly, and have changed WDL targets.
Target generation takes 4.45, 6.15 and 4.37 seconds per shard; output sizes are
1.58, 1.57 and 1.62 MB. These small, cached observations are not full-corpus
throughput estimates. Each smoke uses an isolated copied base and new seal.
The initial attempt built and compared cohort00 successfully, then used the
schema-1 qualification API in the harness and failed. That attempt remains at
`/home/josh/chess-artifacts/operations/factorial58-sffree-smoke-20260921`;
the corrected schema-2 harness uses the separate `...smoke-v2-20260921` namespace.
No teacher inference or outcome selection was rerun.

Fifteen focused tests pass, including real target-overlay construction, wrong
feed bytes despite matching IDs, changed teacher content/provenance and both
missing supervision masks. Focused Ruff, basedpyright and vulture pass. The
whole-repository gate reports 23 type errors in seven unchanged baseline test
files and no introduced diagnostics; baseline reproduction is being banked with
the companion loader optimization. Independent implementation review approved
the builder after adding the explicit policy-mask guard.

## Current boundary

Full CPU preparation and E training have not launched at this record's creation.
Bulk artifacts live under `/home/josh/chess-artifacts/operations/`; the manifests
point to original teacher/base banks. Publishing this builder does not alter the
active frozen factorial runtime or queue. The implementation PR is stacked on
`research/factorial58-teacher-mix`, whose schema-2 overlay support is required.
