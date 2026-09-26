# AVI-style one-ply value backup

Status: one-ply evaluation, a bounded diagnostic, a provenance-gated successor
collector and a value-only B100 rewriter are implemented. No real AVI sidecar has
been collected, no full training corpus has been qualified, and no AVI training or
playing-strength result is claimed.

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

## Implemented one-ply mechanism

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
model expects. The helper also exposes a normalized WDL mixture primitive for the
root-distillation and backed-up targets below.

## Bounded mechanism diagnostic

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

## Provenance-gated corpus collector

`scripts/avi_successor_sidecar.py` is the corpus-scale observation producer. It
intentionally does **not** trust a derived shard's stored float16 input tensor as a
complete description of legal-child history. Every selected derived row must carry the
existing `row_provenance.npz` binding back to its original raw schema-3 SF row.

For every row admitted to collection the producer:

- verifies source/config/shard/physical-row and worker/game/ply provenance;
- replays `history_root_fen` plus `history_uci` from the raw row;
- verifies the replayed FEN and history-aware `input_key`;
- rechecks the derived row's `stored_input_key` against its actual stored `x`;
- re-encodes the replayed root and requires float16 identity with the stored source
  tensor before evaluating any child;
- requires the frozen checkpoint's history and extra-feature encoding to match the
  source derive summary;
- banks both direct-root WDL and the all-legal one-ply backup from the same frozen
  evaluator.

The output is a new immutable sidecar directory with one compressed sidecar per selected
source shard and `avi_successor_sidecar_summary.json`. The summary binds the teacher
checkpoint, source summary, source storage identities, raw-history storage identities,
producer identities and WDL payload hashes. Collection refuses overwrite, honors STOP
markers and a disk reserve, and defaults to **one shard** rather than making a
full-corpus run easy to launch accidentally.

A partial selected-shard bank can support cost/mechanism work. The downstream rewriter
requires `full_source_coverage: true`; a partial bank cannot silently become a training
corpus.

## Value-only B100 rewriter

`scripts/avi_value_rewrite.py` consumes a complete AVI sidecar, the corresponding
original SF-derived corpus, and the B100 policy corpus. It first checks that B100 is the
expected global BT4 policy recipe and that its non-policy bytes still match the original
SF corpus. It then copies B100 and changes only `search_wdl`.

The two implemented treatment modes share the same frozen collection:

| Mode | Main value target at neural weight `alpha` | Purpose |
| --- | --- | --- |
| `root` | `(1-alpha) S + alpha R` | ordinary frozen-network root self-distillation control |
| `backup` | `(1-alpha) S + alpha B` | all-legal one-ply successor-backup treatment |

Here `S` is the unchanged stored SF WDL, `R` is the frozen network's direct root WDL,
and `B` is its selected one-ply backup. `alpha=0.25` is the proposed first dose, not a
selected optimum. The producer normalizes probability mass before blending, stores the
result in the existing float16 WDL channel, verifies stored mass, and proves that every
non-value file remains unchanged.

The rewrite receipt and each shard's `value_target_postprocess` stamp include the exact
AVI summary SHA256 and per-shard sidecar hash in addition to checkpoint, mode and dose.
A later training admission must pin that full rewrite identity; the generic
`derive_value_scheme` / `derive_value_source` pair alone is **not** being declared a
training-admission contract by this record.

## Intended controlled training comparison

The first training comparison should keep policy supervision, rows, initialization,
row order, update budget, trainer/runtime and non-value targets fixed. At the proposed
`alpha = 0.25`:

| Arm | Main value target | Purpose |
| --- | --- | --- |
| A | `S` | unchanged value supervision |
| D | `0.75 S + 0.25 R` | root self-distillation control |
| B | `0.75 S + 0.25 B` | one-ply successor-backup treatment |

B versus A asks whether the package helps. B versus D asks whether the successor backup
adds anything beyond mixing in a frozen neural teacher. Policy supervision and existing
Gumbel search remain unchanged.

The existing `wdl` head is the value consumed by MCTS. Changing only `sf_eval` or an
auxiliary value head would not test this mechanism.

## Gates before training

Do **not** launch a full-corpus training arm from this PR alone. The implementation makes
previously implicit blockers executable, but no real corpus has passed them yet.
Before training require, at minimum:

- a completed sidecar on the intended full common cohort, with no FEN-only fallback;
- measured collection cost and a successful bounded real-checkpoint pilot;
- a completed `root` and/or `backup` rewrite with exact source and sidecar identities;
- an explicit training admission that pins the rewrite receipt, including the exact AVI
  summary hash rather than inferring identity from mode/dose alone;
- proof that the changed `search_wdl` reaches the actual main `wdl` loss while policy
  and every other intended target remain fixed;
- the same current trainer/runtime, initialization, row schedule and update budget for
  the matched control, or a fresh control when historical compatibility fails;
- a registered match budget/readout and recovery plan before spending training compute.

These gates preserve the distinction between implementation capability, corpus
qualification, training evidence and playing strength.

## Proposed first readout

On a preselected 4,096-position history-bearing mechanism bank, record at minimum:

- mean and quantiles of root-versus-backup centered-value change;
- mean WDL total variation;
- fraction of roots whose best one-ply move is terminal;
- legal children and network evaluations per root;
- positions/children per second and total collection cost;
- slices for roughly equal, decisive, tactical and endgame positions when the bank
  carries those strata.

If that mechanism/cost screen is useful, the one-shard provenance collector is the next
bounded integration test before allocating a complete bank. A later training candidate
should be judged primarily by matched Gumbel play, not by one-ply greedy strength alone.
Teacher agreement and value loss are diagnostics, not a playing-strength substitute.

If the one-ply treatment wins, a repeated frozen-snapshot AVI round can be registered
separately. Online continuously refreshed targets are a different experiment.

## Scope and recovery

This PR changes no production config, live distribution, search defaults or training
admission. All new collection/rewrite tools write new namespaces and refuse overwrite.
They can be stopped without altering existing generation, Ceres collection, G10 value
work, B100 data or training. No live job is adopted or restarted by merging this code.
