# Native NNUE / DAG / FastQ production-Gumbel readout

`python scripts/nnue_gumbel_readout.py` is the end-to-end measurement bridge
between the native CPU verifier work and the production Gumbel search.

It deliberately does **not** implement another game loop or another Gumbel
search. It imports `scripts/gen_random_selfplay_shards.py` and calls that file's
`play_game()`, so all three cells use the same C `MCTSTree`, candidate selection,
per-root simulation budget, target construction, pending-leaf `CBoard` binding,
and terminal handling.

## The three cells

| `--arm` | search | canonical position DAG | purpose |
|---|---|---|---|
| `nnue-qsearch` | existing qsearch | no | baseline |
| `nnue-qsearch-dag` | **same** qsearch | yes | isolate reuse only |
| `nnue-fastq` | FastQ-4+ | yes | reuse + new tactical algorithm |

`nnue-qsearch-dag` is the control that makes attribution possible. PR #472
already proves it is bit-identical to `nnue-qsearch`; a difference between those
two cells here therefore measures the evaluate-once substrate under real Gumbel
leaf traffic. A further difference between `nnue-qsearch-dag` and `nnue-fastq`
is search-policy work.

## DAG lifetime

A DAG-backed worker owns one arm context for its lifetime, but the graph is
reset **before each game**:

```text
worker
  game 0: reset -> search every ply with one persistent graph
  game 1: reset -> search every ply with one persistent graph
  ...
```

The reset retains allocations. This gives two useful properties at once:

1. nodes discovered by an earlier ply in the same game remain reusable on later
   plies, which is the cross-ply benefit the DAG exists to measure;
2. unrelated games cannot accumulate semantic nodes forever.

The report therefore names graph sizes `*_per_game`. `memory_peak_per_worker_bytes`
is different: because reset retains capacity it is the worker's resident DAG
allocation high-water mark, not the sum of positions seen over the run.

No DAG-backed provider is installed in `MCTSTree`. `nnue-qsearch-dag` and
`nnue-fastq` still declare `requires_gil`, and the tree still refuses them. The
gen-0 integration obtains the tree's pending `CBoard` leaves and calls
`arm_handle_eval` externally with the GIL held; this readout reuses that exact
path.

## Knob ownership is strict

The command accepts both qsearch-family and FastQ-family flags so the three cells
can be driven by one tool, but it refuses a knob the selected provider does not
consume.

For `nnue-qsearch` / `nnue-qsearch-dag`:

- `--nnue-resolver-max-depth`
- `--nnue-qsearch-max-ply`
- `--nnue-qsearch-check-plies`
- `--dag-node-cap` (**DAG cell only**)

For `nnue-fastq`:

- `--fastq-max-qply`
- `--fastq-node-cap`
- `--fastq-delta-margin`
- `--fastq-recapture-exempt`

Unspecified consumed knobs are read from the compiled `_nnue_ext` constants,
not copied as Python numeric defaults. Configuration is applied before
`arm_open()` and then read back from the **context that will actually run**.
The tool raises if requested and realized snapshots differ.

The stats surface is likewise provider-owned: FastQ is read only through
`fastq_stats()`, qsearch through `arm_stats()`, and DAG resource/canonical-table
state through `arm_dag_stats()`. The extension intentionally raises on the wrong
surface, so the readout never turns an all-zero wrong-counter block into a
plausible measurement.

## Suggested first matrix

Use identical seeds and search settings for all three commands:

```bash
PYTHONPATH=. python scripts/nnue_gumbel_readout.py \
  --arm nnue-qsearch --nnue-pack big.pack \
  --games 64 --workers 8 --sims 32 \
  --json data/readout_qsearch.json

PYTHONPATH=. python scripts/nnue_gumbel_readout.py \
  --arm nnue-qsearch-dag --nnue-pack big.pack \
  --games 64 --workers 8 --sims 32 \
  --json data/readout_qsearch_dag.json

PYTHONPATH=. python scripts/nnue_gumbel_readout.py \
  --arm nnue-fastq --nnue-pack big.pack \
  --games 64 --workers 8 --sims 32 \
  --json data/readout_fastq.json \
  --bank-leaf-observations data/readout_fastq_leaves.jsonl
```

`--all-root-moves` defaults on for this harness and `--topk` therefore defaults
to the engine's exported maximum legal-move count. This matches the native-arm
quality/readout cells rather than silently ranking a Gumbel-random subset of a
uniform-prior root.

When banking is enabled each worker writes a separate `*.wNN.jsonl` file, so
there is no shared-file lock or interleaved JSON. Rows retain FEN, raw internal
value, game/ply cluster, pack hash, cp mapping, and the realized arm knobs. The
raw value makes a later scale correction or deep-SF reanalysis possible without
replaying the games.

## Read the output in this order

1. **Throughput**: `plies_per_s` / `games_per_h`.
2. **Actual arm work**: `arm_io.nnue_evals_per_top_level_call` and the raw
   provider counters.
3. **FastQ tripwire**: `fastq.budget_trip_rate`. A nontrivial rate is a finding,
   not an invitation to silently raise the cap.
4. **Reuse**: FastQ/qsearch-DAG within- and cross-call counters plus
   `dag_per_game.canonical_hit_rate`.
5. **Memory**: `dag_per_game.nodes_peak_per_game` and
   `memory_peak_per_worker_bytes`.
6. **Search shape/coverage**: root-budget and termination data in each worker's
   detail record, plus the banked leaf population for the standardized deep-SF
   quality readout.

The harness does not claim that similarity to qsearch is strength. The deciding
quality comparison remains the standardized deep-Stockfish target-quality
readout described by the AZ-purity framework. The point of this script is to
produce the **production-shaped leaf population and raw observations** needed to
run that decision honestly.

## Non-goals

This PR does not:

- change FastQ search semantics;
- change `MCTSTree` or migrate Gumbel onto `CaePositionDag`;
- make the DAG concurrent;
- add quiet-check or mate-verifier logic;
- tune FastQ's defaults;
- implement the lazy-eval / PEXT speed-plan PRs.

Those decisions should consume this readout rather than precede it.
