# Bounded multi-root search (PR5a)

This is an explicit **CPU-F32, fixed-batch, offline cohort runner**, not a replacement
for the UCI application. It owns 1–16 independent Bend search trees, visits roots
in rotating order, gathers at most one leaf per root per sweep, executes the existing
native batch backend and returns each result to its original tree/ticket.

```sh
bash native/bend_engine/multi_root/build.sh /tmp/new-cohort \
  /path/to/checkpoint.pt2 /path/to/libtorch/share/cmake /path/to/verified/bend
DEEPFIN_BEND_MODEL_PACKAGE=/path/to/checkpoint.pt2 \
  /tmp/new-cohort/build/deepfin-bend-multi-root --threads 1 -- \
  4 2 0 0 'startpos' 'startpos moves e2e4 e7e5'
```

Arguments after `--`: maximum completed simulations per root (1–256), maximum
search depth (1–32), per-root real-neural-row budget (0 disables, otherwise 1–256),
and diagnostics (0/1), then one quoted position per root. Use `startpos` or
`fen <six FEN fields>`, optionally followed by `moves <complete legal history>`.
All positions/history are validated before loading the model. Roots use fresh
4,096-node arenas and immutable input order IDs; the runner exits after the cohort.
Model load/binding must match the existing trusted CPU package contracts. One process
binds one package; incompatible models, encodings or dtypes cannot share a batch.

Each gather pass stops at the physical batch size or the end of the available
root sweep. Partial batches dispatch immediately; there is no wait-to-fill heuristic.
Visited roots rotate behind unvisited roots. Leaf histories, legal entries and
inputs are reconstructed independently; transpositions do not merge paths. Completed
and rule-draw simulations cost no neural row. All real output logits are checked
for finiteness before any neural row is backed up. A failed or malformed model
call terminates with error rather than emitting a successful cohort summary.

`cohort_root` records per-root completed simulations, accepted/executed neural rows,
rule-draw replies, used nodes, stop code and budgets. An unmet neural budget remains
explicitly false: a terminal root, depth/simulation limit or exhausted arena is not
an equal-neural-budget comparison. `searched_move=false,bestmove="0000"` means no
expanded continuation exists, including adjudicated roots that still have legal
moves. This is an offline result, **not a UCI bestmove protocol** or an implicit
legal fallback. Diagnostic mode prints all tree nodes and selected-leaf paths/replies.

`cohort_work` uses `deepfin.multi-root-work.v1`, distinct from PR1's UCI schema but
with the same real/physical/accepted-work definitions. One backend call is not one
simulation; physical padding is never accepted. Forward calls and batch histograms
are global, while accepted rows belong to roots. Do not sum aggregate and per-root
rows. Failure exits have no successful summary. There are no cancelled or in-flight
rows at a successful final report because this runner is synchronous and bounded.

The millisecond-resolution cohort clock begins after position validation/model
loading, before shared policy-map/buffer initialization; it includes the first
inference (no excluded warmup) and optional per-leaf diagnostics, but excludes final
node/report formatting. Gathering and normalization/backup are composite CPU phases.
Queue wait and H2D/GPU/D2H timings remain null. These numbers are not an external
command-to-decision benchmark, fixed-wall comparison or steady-state EPS study.

## Explicit qualification

`qualify.sh GENERATED_RUNNER.c ORACLE_BINARY NEW_OUTPUT` compiles test-only callbacks
at five batches and both input widths, plus two UBSan configurations. It compares
all final tree bits and decisions with batch one. `test_backend.c` is never linked
into the LibTorch product. `verify.py` validates actual selected paths/tickets,
CBoard input bits, every real logit, legal priors/WDL, all final tree fields and
real/physical accounting against independent existing references. For real models,
supply both `--package` and `--checkpoint`; otherwise it expects the test callback.

```sh
python -m native.bend_engine.multi_root.verify \
  --binary /tmp/new-cohort/build/deepfin-bend-multi-root --oracle /path/to/oracle \
  --batch 4 --channels 175 --package /path/to/checkpoint.pt2 \
  --checkpoint /path/to/checkpoint.pt --report /tmp/cohort-model.json
```

No production defaults change. Async multi-root polling/cancellation, live root
arrival/removal, persistent self-play integration, bounded wall-time admission,
measured bucket selection, trained-model CUDA qualification and same-tree concurrency
remain later work. Existing PR4 UCI responsiveness is not weakened or inherited by
this separate synchronous executable. No speedup or Elo gain is implied by fewer
forward calls. The experiment record specifies actual qualification and its limits.

## Qualification evidence

[The experiment record](../../../docs/experiments/2026-09-22-native-multi-root.md)
contains the completed deterministic/UBSan matrix, actual CPU selected-leaf model
checks, exact source/package identities and limits. Deterministic final trees are
bit-identical across batch sizes; real model outputs are qualified numerically,
not claimed bit-identical. The primary real batch-four cohort executes 34 accepted
rows in 9 forwards with 2 padding rows, compared with 34 singleton forwards.
This is a work-count observation, not a measured speedup.
