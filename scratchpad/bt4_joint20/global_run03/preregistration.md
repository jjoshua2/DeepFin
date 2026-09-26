# BT4 comparisons with an untempered search prior

## Prospective correction

The user identified the inherited search-prior temperature 1.5 as compensation
for an older, overly sharp network. It should not silently define the evaluation
of newly trained target mixtures. Use explicit prior temperature **1.0 on both
sides** for the main comparisons, keeping move-selection temperature 0.1 and all
other registered search settings unchanged. One is a neutral no-transform prior,
not an asserted optimum for every model.

This amends the evaluation in [global mixing and search scaling](2026-09-05-bt4-global-search-scaling.md)
and the [SF-close follow-up](2026-09-06-bt4-sf-close-followup.md). It leaves training
recipes, rows, seed, optimizer, values and runtime unchanged. At this decision,
S0 and the E0 search curve were complete; G20T1 was training and G20T05 was
materializing. No global-mixture playing result had been read. The old exact-tie
curve at prior 1.5 remains evidence for that setting, never a prior-1.0 baseline.

Prior softening may shift the target-sharpness recipe that wins. That is a
plausible interaction, not proof that all earlier sharpening gains were artifacts.
In particular, deep-SF-agreement audits have their own concentration preference;
changing search temperature does not retrospectively change those measurements.

## Fixed comparison

Keep the original S0, G20T1 and G20T05 recipes and all pinned training inputs.
G20T1 is 80% normalized stored SF plus 20% full legal raw BT4; G20T05 uses the
same mixture with BT4 teacher temperature 0.5. Teacher temperature and the
search-prior transform are separate operations. Preserve the ongoing G20T1
exact epoch and published G20T05 materialization; do not restart useful training
merely to change future evaluation flags.

Evaluate E0, G20T1 and G20T05 against S0 at 25, 100 and 400 simulations per side,
1,000 games / 500 color-swapped opening pairs per cell, on the original book and
seed 42. Explicitly pass:

```text
--cand-gumbel policy_temp=1.0 --ref-gumbel policy_temp=1.0
```

All original fixed protocol settings otherwise remain: training search shape,
matched_sims, opening plies 16, maximum plies 300, move temperature 0.1, compile on,
no rolling, 128 concurrent games and evaluation batch 4096. Verify the realized
candidate and reference headers both contain `gumbel.policy_temp == 1.0`.
Do not pool or reuse old arena banks merely because checkpoint paths match.
Re-evaluating E0 at 1.0 provides the matched incumbent anchor for the global
mixtures and makes the old/new temperature interaction interpretable.

The 100-simulation paired 95% Elo interval is primary: lower bound above zero
is promising, upper bound below zero is loss, otherwise inconclusive. Report
400 simulations separately as deployment-relevant. Run all three budgets even
if shallow results lose. The 10,000-replicate aligned-opening-pair bootstrap
(seed 20260903) compares score400 minus score25 and G20T1 minus G20T05 against
common S0. Common-opponent score differences are not direct head-to-head Elo.
No optional stopping, winner promotion or inference-optimal-target claim.

After this global screen, run the single SF-close C20T05 recipe directly against
E0 at the same explicit prior 1.0, preserving its separate 12 GPU-hour budget and
all three search budgets. E0 uses teacher temperature 1, so this still compares
widening-plus-sharpening as a package. Historical E0 runtime remains incompletely
stamped. Independently confirm the selected recipe with another training seed
and fresh openings before recommending it for the 100M restart. A small later
inference-temperature screen remains optional and separate from this fixed test.

## State, adoption and cumulative accounting

Use new global state `scratchpad/bt4_joint20/global_run03` and SF-close state
`scratchpad/bt4_joint20/sf_close_run02`. Freeze this record as the former's
`preregistration.md` and the updated SF-close record as the latter's snapshot.
Keep the original source/corpus/run paths: targets and training were not changed.
The existing global audit receipts remain in `global_run02` and their descriptive
admission remains bound to `global_run01/preregistration.md`. That target-admission
record is distinct from this new evaluation record and must not be rewritten.

Allowlist adoption of completed S0 and G20T1 training only. Require their original
completion receipts to equal freshly validated checkpoint/summary identities;
preserve original runtime/manifest/receipt hashes in adoption records. Never adopt
old arenas. The G20T05 training run must still be absent before its new launch.
Retain the frozen Python 3.10.12 / Torch 2.11.0+cu128 / NumPy 1.26.2 runtime and
native extension identities, with source-normalized schedule verification.

The global cap remains 30 GPU hours cumulatively across run02 and run03, including
old prior-1.5 arenas and any interrupted stage time. Read old charges directly;
never copy them into the new root. Do not add run01 separately: run02/previous
already contains the reconciled 1,441.670 seconds from the rebooted S0 attempt.
Budget exhaustion preserves partial work and requires a prospective decision,
not a scientific loss verdict. Keep the shared GPU lease, generation jobs and
150 GiB free-disk reserve.

## Safe transition and commands

The old driver has no pause-after-training switch. A separate reviewed watcher
waits for both the G20T1 completion receipt and G20T05 published-mix completion.
Only then may it request old coordinator stop; never signal the trainer early.
The coordinator's own cleanup handles an obsolete arena if one started in the
boundary race. Preserve that bank as superseded and charge its time. If the CPU
materializer unexpectedly finishes late, obsolete arena time need not be tiny.
Require old coordinator exit, free driver lock and closed charges before the new
runner launches. No force-kill escalation or in-place code edit. The previously
queued SF-close handoff was cancelled via its own STOP marker.

From the reviewed new checkout, after that boundary completes:

```bash
/usr/bin/python3 scripts/bt4_joint_experiment.py --phase baseline --execute
/usr/bin/python3 scripts/bt4_joint_experiment.py --phase treatments
/usr/bin/python3 scripts/bt4_joint_experiment.py --phase treatments --execute
```

The baseline phase adopts the verified completed S0 without retraining; the
treatments phase adopts G20T1 and continues with fresh evaluation banks.

Use the reader with `--prior-temperature 1.0` and explicit checkpoint/cell paths;
its default is also 1.0. Old bank analysis must explicitly request 1.5 and remain
in a separate report. The complete launch manifest records exact commands,
code/input hashes, training-adoption origins and effective runtime. Later readouts
append to this document without changing its executed snapshot.

## Status

Prospective amendment. Replacement runner and boundary watcher require review
before launch. Current training/materialization remain active and unchanged.
