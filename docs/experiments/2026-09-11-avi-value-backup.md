# AVI-style one-ply value backup

Status: implementation and bounded diagnostic tooling only. No full-corpus AVI sidecar,
training run, search change, or playing-strength result is claimed.

## Motivation

[The Surprising Effectiveness of Approximate Value Iteration in Self-Play](https://arxiv.org/abs/2609.09094)
reports that approximate value iteration (AVI) can learn stronger value functions than
AlphaZero-style self-play in several smaller games. DeepFin is not an outcome-only
AlphaZero baseline: its bootstrap already uses strong searched Stockfish values. The
relevant question is therefore narrower and harder:

> Does imposing a one-ply Bellman-like relationship between a parent and all legal
> successor positions add useful value information on top of DeepFin's strong SF
> anchor?

This record does not propose replacing Gumbel search. The main intended downstream
comparison is an AVI-supervised evaluator used by the existing Gumbel search.

## Implemented mechanism

`chess_anti_engine/mcts/one_ply.py` provides a shared one-ply value primitive.
For every parent position it:

1. enumerates every legal move, without a policy top-k filter;
2. copies the board with `stack=True`, preserving the move stack consumed by the
   history encoder;
3. resolves terminal successors from the exact python-chess result under the declared
   claim-draw convention;
4. batches every non-terminal child through one frozen evaluator;
5. converts child W/D/L logits to probabilities, flips W/L into the parent's
   side-to-move perspective, and maximizes `q = P(win) - P(loss)`;
6. carries the selected child's full W/D/L distribution back to the parent.

Ties retain python-chess legal-move iteration order. A twofold repetition is *not*
turned into an exact draw by this primitive. That distinction is deliberate: native
DeepFin search separately uses an LC0-style twofold-as-draw pruning convention, which
is a search heuristic rather than a proven terminal result.

Dynamic-relation checkpoints are supported by computing the same relation matrices the
model expects. The helper also exposes a normalized WDL mixture primitive for later
root-distillation and backed-up target construction.

## Bounded diagnostic

`scripts/avi_value_diagnostic.py` accepts a frozen checkpoint and DeepFin's replayable
seed format (`FEN` or `<start_fen> | <moves>`). It banks, for each selected root:

- frozen root W/D/L;
- one-ply backed-up W/D/L;
- selected move and centered value;
- legal, terminal and network-evaluated child counts;
- whether the selected successor was exact terminal;
- root-versus-backup target differences.

The compact JSON receipt pins checkpoint and seed-file SHA256 identities and records
model input/history/relation settings, collection counts, elapsed time and the NPZ hash.
The tool refuses overwrite and defaults to at most 4,096 distinct usable positions.
This is a mechanism/cost instrument, not an accuracy or strength verdict.

Example (illustrative, not a launch instruction):

```bash
PYTHONPATH=. python scripts/avi_value_diagnostic.py \
  --checkpoint <frozen-checkpoint.pt> \
  --positions <preselected-history-bearing-seeds.txt> \
  --out <new-output>.npz \
  --device cuda --batch-size 4096 --max-positions 4096
```

## Intended controlled training comparison

The first training comparison should keep policy supervision, rows, initialization,
row order, update budget, trainer/runtime and non-value targets fixed. Let `S` be the
existing SF value target, `R` the frozen network's direct root WDL and `B` its one-ply
backup. A proposed first dose is `alpha = 0.25`:

| Arm | Main value target | Purpose |
| --- | --- | --- |
| A | `S` | unchanged value supervision |
| D | `0.75 S + 0.25 R` | root self-distillation control |
| B | `0.75 S + 0.25 B` | one-ply successor-backup treatment |

B versus A asks whether the package helps. B versus D asks whether the successor backup
adds anything beyond mixing in a frozen neural teacher. The 25% dose is a starting
experiment, not a selected optimum.

The existing `wdl` head is the value consumed by MCTS. Changing only `sf_eval` or an
auxiliary value head would not test this mechanism.

## Gate before training

Do **not** launch a full-corpus training arm from this PR alone. Derived bootstrap shards
retain the model input tensor and row identities but are not, by themselves, a portable
proof that every row's original move stack can be reconstructed. A full-corpus AVI
sidecar therefore still needs a qualified common-history binding from each training row
to a replayable board. It must preserve the exact row set and fail rather than silently
falling back to FEN-only encoding.

Before training, require a producer/qualifier that establishes:

- complete row coverage on the chosen common cohort;
- exact source/history identity and frozen checkpoint identity;
- unchanged policy and every non-value target;
- a stamped value-source identity distinguishing root-distillation from one-ply backup;
- successful consumption by the actual main `wdl` loss;
- a matched control under the same current training implementation.

This separation is intentional. The diagnostic can establish semantics, effect size and
collection cost without claiming a training-ready corpus it has not constructed.

## Proposed first readout

On a preselected 4,096-position history-bearing mechanism bank, record at minimum:

- mean and quantiles of root-versus-backup centered-value change;
- mean WDL total variation;
- fraction of roots whose best one-ply move is terminal;
- legal children and network evaluations per root;
- positions/children per second and total collection cost;
- slices for roughly equal, decisive, tactical and endgame positions when the bank
  carries those strata.

A later training candidate should be judged primarily by matched Gumbel play, not by
one-ply greedy strength alone. Teacher agreement and value loss are diagnostics, not a
playing-strength substitute. If the one-ply treatment wins, a repeated frozen-snapshot
AVI round can be registered separately; online continuously refreshed targets are a
different experiment.

## Scope and recovery

This PR changes no production config, live distribution, search defaults or training
admission. The diagnostic writes new files only and refuses overwrite. It can be stopped
without affecting existing generation, Ceres collection, G10 value work or training.
