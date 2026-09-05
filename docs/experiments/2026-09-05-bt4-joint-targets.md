# BT4 near-tie policy targets for the NNUE restart

**Superseded before A-F training:** The user reopened this design on September5.
The coordinator was intentionally stopped; preserve its artifacts. The active
plan is [global BT4 mixing and search scaling](2026-09-05-bt4-global-search-scaling.md).
The historical plan and diagnostic below remain evidence, not queued work.

## Objective and authorization

Continue the September 4 six-arm experiment using its completed E0 control and
banked teachers. On September 5 the user authorized autonomous training, evaluation
and evidence-supported follow-ups, keeping available GPU compute productive until
there is a defensible recipe. A target-quality proxy alone does not decide strength.
The production restart remains a separate adoption decision.

This carries forward the preregistration and later corrections in the wise-cloud
checkout's frozen `docs/experiment_ledger.md`, September 4 17:58Z entry and
15:17/15:30/15:46/15:50 EDT amendments. That historical branch contains newer
experiment entries than main's ledger. New readouts live here.

## Banked evidence and frozen inputs

Paths below are relative to the experiment checkout root.

- Source: `data/nnue_derived/armB/qtemp_0.0005_hist_20m`, 18,910,484 rows in
  2,309 shards, derived from the first 20M raw rows; source summary SHA256
  `391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef`.
- BT4 bank: `data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m`.
  One network evaluation per position, zero search nodes; already complete.
- Rank bank: `data/lc0/sf_d9_rank_sidecars/armB_qtemp0005_hist20m_top8`, complete.
- Admitting receipts: `scratchpad/bt4_joint20/audit_<cell>_v2.json`.
  All six passed; no repeat audit or teacher inference is needed.
- E0: `runs/armB/qtemp_0.0005_hist_20m_bt4_toptie_a100_epoch_v3/checkpoint.pt`,
  SHA256 `c7d0bb38f952150db004b29699e0509437bcec7928d79cd44f69125ffa5fa817`.
  Summary reports all 18,910,484 rows and 36,935 batches realized, no same-game
  repeats, matching planned/realized hashes, and complete=true.
- E0 plan hash: `185fa5b2bd676246de2329f6b281b61e586cda2dffcb60f4ca2c500f97187041`.
  Mixed source directories change source identities; compare each arm's own
  planned/realized hashes, not raw hashes across different corpus paths.

The older exact-tie treatment won its recorded 1,000-game/100-simulation arena
against plain SF by +74.4 Elo [56.4,92.8]. That checkpoint used replacement
sampling. E0 uses the registered exact-epoch sampler and is the reference for
all new target comparisons. Its `valid_control=false` flag concerns reproduction
of an older historical control; it is not evidence of an incomplete E0 epoch.
The cancelled S1/N1 extension pipeline stays cancelled.

## Six fixed treatments

Eligible actions are the union of all stored SF maxima and the first K d9 ranks
within the stated cp gap of rank one. Preserve total probability mass inside the
set and every probability outside it. Redistribute the selected mass according
to `(1-alpha)*stored_SF + alpha*conditional_BT4_T`. Values/features/priorities
remain unchanged. This experiment does not test a global BT4 mixture.

| Arm | K | Gap cp | Alpha | BT4 temperature |
| --- | ---: | ---: | ---: | ---: |
| A | 2 | 10 | 1 | 0.5 |
| B | 2 | 10 | 1 | 2 |
| C | 3 | 20 | 1 | 0.5 |
| D | 3 | 20 | 1 | 2 |
| E | 2 | 10 | 0.5 | 2 |
| F | 3 | 20 | 0.5 | 2 |

Run A through F; do not pick cells from their audit ranking. Each trains from
scratch with seed0, batch512, `lc0_positive_control.yaml`, `game_epoch`, steps0,
planner/loader workers16/16 and training window88. Require exactly one complete
pass with no missing/duplicate ordinal or same-game repetition. Preserve the
E0 trainer/config/scheduler pins and use its wise-cloud checkout with explicit
`/usr/bin/python3`, currently Python3.10.12/Torch2.11.0+cu128. E0's log establishes
Python3.10 but does not independently stamp its historical Torch version. Do not
introduce main's Python3.13/Torch2.14 environment into this comparison.

## Decision and follow-ups

Each arm plays E0 at 100 simulations, training search shape, seed42, no rolling,
1,000 games in 500 color-swapped opening pairs. Bank every game and preserve
checkpoint, search and opening-book identities. The existing arena's pair-unit
95% Elo interval defines SUCCESS (lower bound >0), KILL (upper bound <0), or
INCONCLUSIVE (otherwise, prefer E0). Do not stop early on a favorable running score.

Among successful arms, compare common-opening pair outcomes with a paired bootstrap
(10,000 replicates, seed20260903). A positive lower bound on score advantage is
required to displace the current leader; unresolved ties prefer B,A,E,D,C,F.
These common-opponent score differences are not direct head-to-head Elo.
Use fresh validation before calling a selected winner broadly optimal.

Retain the September4 corrected follow-up: only if C wins, consider R=K5/20/alpha1/T0.5
and W=K3/14/alpha1/T0.5 against C, changing rank and window separately. Stop expansion
when the defined gain is not established; do not resurrect the superseded coupled
K5/window proposal or unbounded wider-rank ladder. After selecting the 100-simulation
winner, record its low/high-simulation persistence comparison before new games.
If another arm exposes a distinct useful question, document its hypothesis, simpler
control and bounded budget prospectively under the user's continuing authorization.

## Compute, scheduling and recovery

Six existing-size epochs plus arenas are estimated at about 21h20m from E0's
2h49m50s training and a prior 43m32s arena, excluding materialization/host variance.
Budget up to 30 GPU hours for the initial six and the short control diagnostic;
a failed stage is operational failure, not a negative scientific result. Preserve
completed work and correct the cause before choosing a fresh run or safe resume.

Use `scratchpad/gpu0_experiment.lock` for one GPU stage at a time. Materialize at
nice19, at most one corpus ahead of the current GPU arm. Keep the active 12+4 CPU
G10 generators unchanged. Sources are copied normally (about12GiB each), never
hardlinked and then modified. Preserve at least150GiB free disk; no automatic
artifact deletion. Logs/state go to `scratchpad/bt4_joint20/af_run01` and new outputs
use distinct `bt4_joint_<arm>_<parameters>` names. Never overwrite an existing run
or append new games without the arena's validated resume path.

## September 5 prospective sampler diagnostic

While A materializes, compare completed E0 to the older replacement-sampled
exact-tie checkpoint using 400 requested games/100 simulations, training shape,
seed42, no rolling, maximum900 seconds. This checks the combined sampler/training-vintage change on the already-winning
exact-tie target; the older replacement run used36,960 steps versus36,935 for E0,
so it is not a perfectly isolated sampler-effect estimate. Report completed opening pairs,
truncation and pair-unit interval; never pool with A-F or use it to select a target.
An unresolved diagnostic consumes no additional automatic budget. A clear large
regression requires checking protocol/lineage before interpreting later A-F results.
Artifacts: `scratchpad/bt4_joint20/af_run01/E0_sampler_diagnostic.*`.

## Execution and readouts

2026-09-05: E0 and both teacher banks recovered; six audit admissions confirmed.
Current G10 banks have about16.57M published positions and continue independently.
No A-F train or arena existed when this continuation was recorded.

### GPU backlog filling during CPU preparation

Arm A's dense target materialization is CPU-bound and progresses much more slowly
than its initial copy. Resume the existing G10 BT4 raw labeler during GPU gaps:
retain the same sources, ONNX model, output root, batch1024, threads16,24GiB
allocator ceiling and16-shard lease batches. It has its own writer lock, preserves
completed sidecars and releases the shared GPU lease after child teardown. Its
15-second inter-batch yield lets the A-F driver's10-second lock polling take
priority when an arm is ready. The September4 pause was for training priority;
archive those markers before resuming under the user's September5 instruction to keep
useful GPU work running. No generator, source manifest or label semantics change.

This labeling is part of the planned100M teacher bank and is recorded separately
from the30-GPU-hour A-F training/arena budget. Stop it at its next boundary if it
does not yield to a ready trainer. Retain150GiB free disk and existing refusal
behavior. No cancelled S1/N1 training is resumed.

### E0 sampler/training-vintage diagnostic readout

The900-second play budget completed192 opening pairs/384games, stopping before
the fourth chunk (928seconds reported, excluding earlier startup). Pentanomial
counts were WW37, WD/DW19, DD/WL74, LD/DL26, LL36; candidate E0 score0.49349,
Elo-4.5 with95%CI[-37.3,+28.2]. Verdict: INCONCLUSIVE. This does not establish
a sampler gain or a regression and does not change A-F's scientific reference.
The16 requested games beyond the time budget are not automatically rerun.
The raw games and appended result are preserved as
`af_run01/E0_sampler_diagnostic.{games,results}.jsonl`.

After the diagnostic released its GPU lease, the resumed G10 labeler acquired it
and emitted new raw BT4 sidecars. A's independent CPU materialization continues.
