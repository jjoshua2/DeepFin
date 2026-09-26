# Sharpened BT4 among close SF moves

## Question and scope

Following the [untempered global screen](2026-09-06-bt4-untempered-search.md), test one
sharpened BT4 treatment among SF-close moves directly against the existing
exact-tie model E0. The user explicitly restored this question after clarifying
that global mixing does not replace selection by SF closeness. This is a new
one-arm follow-up, not a revival of the [superseded six-arm plan](2026-09-05-bt4-joint-targets.md).
Finish the running global screen first. A separate BT4 probability-floor arm is
lower priority because sharpening already suppresses its tail smoothly.

## Fixed recipe and rationale

Use the previously audited C recipe: `sf-cp-window`, SF d9 rank cap 3,
20 effective-cp window, alpha 1.0, BT4 temperature 0.5. The eligible set is the
union of every original stored SF maximum and d9 top-three moves within 20 cp
of d9 rank one. Preserve the set's existing SF probability mass and redistribute
it according to conditional sharpened BT4. Leave probabilities outside the set
unchanged. Temperature 0.5 squares relative BT4 probabilities before normalization.
Alpha 1 applies only within the selected set; it is not a global pure-BT4 target.

On the existing 4,000-position audit this widens support beyond exact ties on
48.425% of positions, versus 31.65% for top-two/10 cp. The stored rank-gap scan
places this moderate region near a descriptive plateau. Choose this coarse,
pre-existing recipe for substantive expansion, not the scan's finest proxy optimum.
The audit uses FEN-only teacher inputs; the full training bank preserves history.
These observations justify a candidate, not a playing-strength prediction.

The old C audit passes current treatment invariants and its original SF proxy
gate, so reuse it rather than repeating teacher inference or choosing another
recipe through a new proxy scan. Playing results decide the follow-up.

## Inputs and runtime

Paths below are relative to the experiment checkout root.

- Source `data/nnue_derived/armB/qtemp_0.0005_hist_20m`: 18,910,484 rows,
  2,309 shards; summary SHA256
  `391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef`.
- BT4 `data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m`: summary SHA256
  `68b32a41e89c03737aa28c89310d9ac744f6b1e5afcbfba198d2a0155bd646b3`.
- SF ranks `data/lc0/sf_d9_rank_sidecars/armB_qtemp0005_hist20m_top8`: summary
  SHA256 `165c43e5180146a37e308b9d2dc204190f9cbc5e8a971e0daf5ec8acfa7542fa`.
- Audit `scratchpad/bt4_joint20/audit_C_k3_g20_a100_t050_v2.json`: SHA256
  `7e54c847c02cf3e2e2ad37d34ab5913317f8f93d66ffc7dac7c91acd1604ce7e`.
- Rank-gap scan `scratchpad/bt4_joint20/rank_gap_marginal_scan.json`: SHA256
  `12e692625986d3d4cc500851906f3206c776d5fcb9955ac5af2831e6d2f90e68`.
- E0 `runs/armB/qtemp_0.0005_hist_20m_bt4_toptie_a100_epoch_v3/checkpoint.pt`:
  SHA256 `c7d0bb38f952150db004b29699e0509437bcec7928d79cd44f69125ffa5fa817`.

Use frozen wise-cloud trainer/config/scheduler from the global screen, Python
3.10.12 / Torch 2.11.0+cu128 / NumPy 1.26.2, with the banked native-build hashes.
Train from scratch with seed 0, batch 512, game_epoch, steps 0, planner/load
workers 16/16, and windows of 88 steps: exactly 36,935 batches and all source rows.
No global-model initialization, optimizer/runtime migration or changed value labels.
Validate each run's realized schedule plus source-normalized cross-arm schedule;
physical path hashes alone differ between copied corpora. Preserve non-policy
arrays through the mixer's copy-only path and independently check a published
shard sample. Do not claim that a sample proves full-corpus byte equality.

## Deciding comparison

Direct candidate versus E0, 1,000 games / 500 color-swapped opening pairs at each
of 25, 100 and 400 simulations per side. Use the same book as the global screen,
seed 42, opening plies 16, maximum plies 300, temperature 0.1, training search
shape, matched_sims, compile on, no rolling, 128 concurrent games and evaluation
batch 4096. Explicitly pass `--cand-gumbel policy_temp=1.0` and
`--ref-gumbel policy_temp=1.0`; verify both realized priors. This supersedes the
original inherited 1.5 setting before SF-close training or games. Run all three
cells regardless of the shallow result.

The 100-simulation paired 95% Elo interval is the primary strength measure:
lower bound above zero is promising, upper bound below zero is a loss, otherwise
inconclusive. Report 400 simulations separately as deployment-relevant; a gain
there is not erased by a 100-simulation loss. No optional stopping or promotion
from a running score. Check complete banks, matching protocol and actual
checkpoint/build identities before interpretation.

Use aligned opening-pair bootstrap, 10,000 replicates / seed 20260903, for
score400 minus score25 against E0. A positive lower bound means its relative
advantage over E0 increases, not that either model improves absolutely with search.
These direct-E0 Elo numbers cannot be subtracted from globals' direct-S0 Elo to
claim head-to-head strength. If warranted, compare the leading global and close
recipes directly before confirming the selected recipe.

This is an exploratory seed-zero recipe screen with nominal intervals. E0 used
BT4 temperature 1, so the intervention combines widening and sharpening; it does
not isolate widening alone. E0's historical NumPy/runtime was not independently
stamped. A favorable result requires independent training-seed and fresh-opening
confirmation before a recipe is recommended for the 100M restart. An extra
exact-tie/T0.5 attribution control is optional if it would change the decision.

## Compute, scheduling and recovery

Separate cap: 12 GPU hours for this one training epoch and three arenas, including
failed GPU-stage time. Expected cost is about 6 hours based on the fresh S0/E0
stages; host variance and queue time can change the wall-clock horizon. Stage
admission estimates remain conservative, and budget exhaustion preserves partial
work without a scientific loss verdict.

Use the shared `scratchpad/gpu0_experiment.lock`, one GPU stage at a time. Preserve
both G10 generators and their disk monitor. Materialize at nice 19, one CPU corpus
at a time; do not compete with the in-flight sharpened global materializer. Maintain
at least 150 GiB free disk and never delete old partial corpora automatically.

The implementation and immutable launch manifest must be reviewed before launch.
Freeze the new script and record before execution. Do not edit or move any live
runtime checkout. Preserve partial training on interruption; resume arenas only
through their validated fingerprint path. A separate state directory and distinct
output names prevent adoption of old A-F outputs by accident.

## Execution status

Prospective record. No SF-close training has launched. The global screen continues
unchanged. From the reviewed, frozen main-based follow-up checkout, plan and run:

```bash
/usr/bin/python3 scripts/bt4_joint_experiment.py --phase sf-close
/usr/bin/python3 scripts/bt4_joint_experiment.py --phase sf-close --execute
```

The execute path requires global-screen completion. It uses state
`scratchpad/bt4_joint20/sf_close_run02/sf-close`, corpus
`data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_sfclose_C20T05`, and run
`runs/armB/qtemp_0.0005_hist_20m_bt4_sfclose_C20T05_epoch_v1`. The independently
pinned preregistration is `scratchpad/bt4_joint20/sf_close_run02/preregistration.md`;
this snapshot and the existing legacy audit have different roles. The mixer uses
the existing audit's original gate mode; it must not bind that old receipt to a
new experiment-record hash.

From that same reviewed checkout, read the resulting banks with the frozen
training runtime (set `experiment_root` to the deployed experiment checkout):

```bash
experiment_root="$HOME/projects/chess"
PYTHONPATH="$experiment_root/.dev/worktree/wise-cloud" /usr/bin/python3 scripts/bt4_joint_readout.py --profile sf-close --prior-temperature 1.0 \
  --reference "$experiment_root/runs/armB/qtemp_0.0005_hist_20m_bt4_toptie_a100_epoch_v3/checkpoint.pt" \
  --cell "C20T05:25=$experiment_root/scratchpad/bt4_joint20/sf_close_run02/sf-close/C20T05.s25.arena.games.jsonl" \
  --cell "C20T05:100=$experiment_root/scratchpad/bt4_joint20/sf_close_run02/sf-close/C20T05.s100.arena.games.jsonl" \
  --cell "C20T05:400=$experiment_root/scratchpad/bt4_joint20/sf_close_run02/sf-close/C20T05.s400.arena.games.jsonl"
```

Bank the reader's JSON output, all game logs, completion receipts and
source-normalized schedule verification. Missing cells remain unread.
Immutable script/run identities and review evidence are recorded
before launch; later readouts append to this main-based document without editing
the executed snapshot.
