# Direct prior-temperature checks for S0 and E0

## Question and fixed tests

Does removing the inherited search-prior softening help the existing SF-only
network S0 and SF+BT4 exact-tie network E0 individually? Comparing E0 against S0
under two temperatures cannot answer this: both networks might improve or worsen.
Run each checkpoint against itself, changing only its search-prior temperature.

| Cell | Candidate | Reference | Simulations per side |
| --- | --- | --- | --- |
| S0 | S0, prior 1.0 | same S0, prior 1.5 | 100 |
| E0 | E0, prior 1.0 | same E0, prior 1.5 | 100 |

Use the completed exact-epoch checkpoints and runtime pinned by the
[untempered-search amendment](2026-09-06-bt4-untempered-search.md). No retraining,
teacher-temperature change, model mixing or production configuration edit.
The existing G20T1 training and G20T05 materialization finish before this runs.

Each cell has 1,000 games / 500 color-swapped opening pairs. Retain the original
book and seed 42, 16 opening plies, maximum 300 plies, matched simulations,
training search shape, move-selection temperature 0.1, compile on, no rolling,
128 concurrent games and evaluation batch 4096. Pass explicit candidate
`policy_temp=1.0` and reference `policy_temp=1.5`. Check the realized headers:
checkpoint paths must be identical within each cell and every other realized
search setting must match. Checkpoint bytes and runtime are bound by the manifest.

Prior temperature acts on search probabilities. This test does not change the
network's raw move ranking, and its result cannot establish the best target
temperature for retraining or whether a BT4-only student would be useful.

## Readout and next action

Complete both fixed-size cells; do not stop early on strength. For each network,
report the candidate's paired score and pentanomial Elo interval. A nominal 95%
interval wholly above zero supports removing softening at 100 simulations;
wholly below zero supports keeping softening there; overlap means inconclusive,
not equivalence. These are two exploratory comparisons without a joint-error
guarantee. A loss at 1.0 would show that 1.5 works better in this tested search,
not establish that the network is intrinsically too sharp or that 1.5 is optimal.

Read both banks before launching the amended global grid. If neither network
shows a clear loss at 1.0, continue the registered common-1.0 grid, retaining any
uncertainty. If either shows a clear loss, hold that grid and preregister a
400-simulation check before choosing its shared temperature. No automatic
asymmetric per-network tuning or claim that this result transfers to the new
global mixtures. The two same-net effects are separate results, not a direct
comparison of E0 and S0 strength. Training-recipe confirmation remains necessary.

## State and resources

Use `scratchpad/bt4_joint20/global_run03/calibration` and freeze this document as
`global_run03/calibration_preregistration.md`. This is an additive amendment;
preserve the previous global, mixture-admission and SF-close snapshots unchanged.
The old prior-1.5 E0-versus-S0 banks cannot substitute for either new cell.

Run after the reviewed safe boundary and verified S0 training adoption:

```bash
/usr/bin/python3 scripts/bt4_joint_experiment.py --phase baseline --execute
/usr/bin/python3 scripts/bt4_joint_experiment.py --phase calibration --execute
```

Use the same shared GPU lease and 150 GiB free-disk reserve. Both cells count
toward the existing cumulative 30 GPU-hour global cap across run02 and run03;
this adds no budget. Allow one GPU hour for admission per 100-simulation cell,
but charge actual elapsed time including failed attempts. Preserve raw banks,
closed charges and launch identities. The driver owns interruption and cleanup;
no new training or concurrent arena is launched by this diagnostic.

## Status

Prospective. No calibration games have been launched or read. The replacement
handoff ends after calibration so its readout can inform the next stage.
