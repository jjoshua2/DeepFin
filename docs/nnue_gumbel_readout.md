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

## ⚑⚑ The oracle, and what voids the decomposition

The control cell's bit-identity is a **free oracle**, and this harness spends it
rather than throwing it away.

Everything a game depends on is a pure function of the seed:

* `gumbel_c.py` draws exactly `legal_idx.size` uniforms per ply;
* `sample_action_with_temperature` draws **zero** at `temperature <= 0`, and
  both production and this tool run `DEFAULT_TEMPERATURE = 0.0`;
* `sample_starting_board` short-circuits when there is no opening book, which
  is this tool's only mode.

So at one seed `nnue-qsearch` and `nnue-qsearch-dag` **must** play byte-identical
games. Each cell publishes

* `games_detail[i].digest` — the game trajectory/termination digest;
* `games_detail[i].search_digest` — the exact improved-policy/legal-mask plus
  returned-root-value sequence for that game;
* `games_digest` and `searches_digest` — ordered whole-cell digests of those two
  independent views.

and when both cells are in one report the top-level `oracle` block compares them.

> **If `oracle.digests_agree` is `false`, the decomposition is VOID.** It
> now requires BOTH game-trajectory equality and improved-policy/search-output
> equality. Equal played moves alone are insufficient: sequential halving can
> absorb changed leaf values while still choosing the same move.
> The report says so in `inadmissible_reasons`, `admissible` goes `false`, and
> the process exits **2**. The same holds across two separate invocations: the
> digests are comparable whenever `provenance.seed` and the search block match.

The old trajectory-only oracle was measured and shown to absorb several
perturbed leaves before a move changed. That experiment is retained below as the
reason the current schema also digests the improved-policy/search output: the
stronger oracle catches a changed target even when the played move is unchanged.
Perturbing evaluated leaf values in the DAG arm only (+500 internal units on one
leaf of the first *N* leaf batches, `--games 2 --sims 8 --max-plies 12 --seed 7`):

| perturbed leaf values | `digests_agree` | exit |
|---|---|---|
| 0 | true | 0 |
| 1 | true | 0 |
| 5 | true | 0 |
| **6** | **false** | **2** |

⚑ Read the top of that table as a real property, not a weakness: sequential
halving **absorbs** a handful of changed leaves without changing a move, so the
digest is a test of the search's *decisions*, which is the thing the wall-clock
comparison depends on. It is not a bit-level checksum of the evaluator; the
value-level parity proof for that is #472's, and `tests/test_qsearch_dag_parity.py`
is where it lives.

## DAG lifetime, and the cadence is a knob

A DAG-backed worker owns one arm context for its lifetime. `--dag-reset` chooses
when the graph is cleared:

| `--dag-reset` | meaning |
|---|---|
| `game` (default) | reset before every game — the preregistered policy |
| `never` | one graph for the worker's whole game list |
| `every-N-games` | reset every N games |

`fastq_design.md` §4.4 is explicit that the persistence policy is to be **chosen
from measurement, not assumed** — and this harness *is* that measurement. A tool
that could only produce the one preselected point could not inform the choice it
exists to inform. The realized cadence is in `provenance.dag_reset` and in each
cell's `dag_reset`.

The reset retains allocations. This gives two useful properties at once:

1. nodes discovered by an earlier ply in the same cadence window remain
   reusable, which is the cross-ply benefit the DAG exists to measure;
2. graphs cannot accumulate semantic nodes forever.

The per-game work fields are DELTAS of the cumulative `arm_dag_stats()`
snapshot. This matters for `never`/`every-N-games`: summing absolute snapshots
would count game 1 again in game 2, game 3, and so on. Resource peaks remain
absolute snapshots. `memory_peak_per_worker_bytes` is the worker's resident DAG
allocation high-water mark, not the sum of positions seen over the run.

No DAG-backed provider is installed in `MCTSTree`. `nnue-qsearch-dag` and
`nnue-fastq` still declare `requires_gil`, and the tree still refuses them. The
gen-0 integration obtains the tree's pending `CBoard` leaves and calls
`arm_handle_eval` externally with the GIL held; this readout reuses that exact
path.

## Knob ownership is strict

The command accepts both qsearch-family and FastQ-family flags so the three
cells can be driven by one tool. In a multi-arm matrix, a supplied knob is legal
when at least one selected cell consumes it, and each `ResolvedArmConfig` copies
only its own fields. In a single-arm invocation, foreign knobs are still refused.

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

⚑ **`--dag-node-cap > 0` is REFUSED on `nnue-qsearch-dag`.** `set_arm_config`'s
own docstring says why: above 0, a node that trips the cap stands pat and
increments `dag_budget_trips`, "so an arm with a binding cap no longer matches
the oracle". A capped DAG cell is not a control. `--allow-binding-dag-node-cap`
exists for deliberately measuring the capped arm, and the report is inadmissible
anyway if `provider_stats.dag_budget_trips > 0` — the cap *actually binding* is
the condition, not merely being set.

Unspecified consumed knobs are read from the compiled `_nnue_ext` constants,
not copied as Python numeric defaults. Configuration is applied before
`arm_open()` and then read back from the **context that will actually run**;
the tool raises if requested and realized snapshots differ, and "requested" is
the caller's own dict rather than the setter's echo (a setter that clamped and
echoed the request would otherwise compare a clamped value with itself). Both
the requested and the realized values are published, each under its own name:
`arm_config` and `arm_config_realized`.

The stats surface is likewise provider-owned: FastQ is read only through
`fastq_stats()`, qsearch through `arm_stats()`, and DAG resource/canonical-table
state through `arm_dag_stats()`. The extension intentionally raises on the wrong
surface, so the readout never turns an all-zero wrong-counter block into a
plausible measurement.

## Counter identities that must hold

`fastq_design.md` §7 names two counter identities, and every term of both is in
the report. They are now checked rather than merely publishable:

| field | identity | source |
|---|---|---|
| `identities.evaluate_once_identity_ok` | `nnue_evals + nodes_created_in_check == nodes_created` | `fastq_stats()` |
| `identities.dag_state_identity_ok` | `state_inits + state_makes == node_count`, per snapshot | `arm_dag_stats()` |

A violation makes the cell inadmissible and the process exit 2. The artifact is
still written first: a gate that raised before the JSON existed would destroy
the evidence for the finding it had just made.

## The matrix — one command, comparable cells

⚑ **The three cells must be run with identical settings, and that now includes
banking.** The previous version of this page enabled `--bank-leaf-observations`
on the FastQ cell only. Banking is a `board.fen()` + `json.dumps` + write **per
evaluated position, inside the timed window** — so it was charged to exactly the
cell whose speedup was the headline. The flag is now matrix-wide: it applies to
every cell or to none, `provenance.banking` records which, and each worker's
file is named for its `(arm, repeat, worker)` so three cells aimed at one path
cannot merge.

```bash
PYTHONPATH=. python scripts/nnue_gumbel_readout.py \
  --arm nnue-qsearch --arm nnue-qsearch-dag --arm nnue-fastq \
  --nnue-pack "$NNUE_PACK" \
  --games 64 --workers 8 --sims 32 --repeats 3 \
  --json data/readout_matrix.json
```

`--repeats N` runs the whole cell set N times **interleaved** — `(repeat, then
cell)`, never all of arm A before any of arm B. A fixed-order single pass
measures the machine's first hour against its third; interleaving spreads
thermal drift, page-cache warming and an arriving neighbour job across every
cell instead of loading them onto whichever ran last. `order` records the
sequence that actually ran.

To bank raw end-to-end trace observations, add the flag to the **same**
command so every cell pays it. **These files are NOT a paired deep-SF evaluator
quality sample**: each arm drives its own Gumbel search, so changing FastQ can
change which later leaves exist. A paired quality experiment needs a frozen
driver population with shadow arms that do not feed values back into MCTSTree.
This PR deliberately does not claim that attribution:

```bash
PYTHONPATH=. python scripts/nnue_gumbel_readout.py \
  --arm nnue-qsearch --arm nnue-qsearch-dag --arm nnue-fastq \
  --nnue-pack "$NNUE_PACK" \
  --games 64 --workers 8 --sims 32 \
  --json data/readout_matrix_banked.json \
  --bank-leaf-observations data/readout_leaves.jsonl
```

`--all-root-moves` defaults on for this harness and `--topk` therefore defaults
to the engine's exported maximum legal-move count. This matches the native-arm
quality/readout cells rather than silently ranking a Gumbel-random subset of a
uniform-prior root. `--topk` is validated in the parent **before** any worker
spawns.

Bank files are opened `"x"`: a rerun that would append into the previous run's
rows fails instead. Rows carry FEN plus `halfmove_clock`, `hash_stack_len`,
`hist_len`, and `fen_reconstructs_full_search_state`. FEN does NOT carry the
CBoard repetition hash stack: a row with that flag false must be excluded by any
FEN-only history-sensitive scorer rather than silently reconstructed as a fresh
position. Rows also carry the raw internal value, an `is_mate` flag,
the game/ply cluster, the pack hash, the cp mapping, the realized arm knobs, the
`run_id` / `seed` / `repeat` / `worker_id` that identify the run, **and the three
mate-band constants** (`RESOLVER_MATE_BASE`, `RESOLVER_MATE_PLY_STEP`,
`RESOLVER_MAX_PLIES`). Those last three are what make a later cp reconstruction
self-contained: above the band floor the raw value is a mate distance in plies,
and a reader who had to supply the build vintage's constants from outside the
artifact would silently run the mate rows through the centipawn slope.

## Provenance: proving the cells are one experiment

`provenance` carries everything that would make two cells incomparable if it
differed — and the report checks the ones that can:

`run_id` · `started_utc` · `pack_path` · `pack_file_sha256` ·
`pack_source_sha256` · `kernel` (avx2 vs scalar — a **multi-fold** wall factor) ·
`seed` · `games_per_cell` · `workers` · `sims_floor` · `topk` · `max_plies` ·
`all_root_moves` · the cp triple · `nice_requested` **and** `nice_realized` ·
`banking` · `dag_reset` · `repeats` · `arms` · `python` · the Git HEAD,
tracked-diff SHA-256 and dirty flag at **both** matrix endpoints (an unavailable
Git snapshot is itself inadmissible) · the native module pathname SHA-256
snapshots · and, authoritatively, the GNU build-ids read from the already mapped
`_lc0_ext` / `_nnue_ext` / `_mcts_tree` ELF images.

Mixed kernels, mixed niceness, a pack whose file hash is not the one the parent
hashed, and disagreeing per-worker arm configuration each make the report
inadmissible or are recorded in `provider_stats_conflicts`. ⚑ A worker
configuration disagreement is **recorded, not raised on**: raising at the
aggregation step throws away a finished multi-hour run, and the generator
already settled that question the same way (`NnueArmStats.context_conflicts`).

## Read the output in this order

1. **Is it admissible?** `admissible` and `inadmissible_reasons`, then
   `oracle.digests_agree`. Everything below is void if these are not clean.
2. **Throughput**: `search_plies_per_s` for the arm comparison — it divides by
   the widest worker's *search* window. `plies_per_s` is the end-to-end figure
   and includes pool startup and each worker's `ext.load()` + `arm_open()` mmap;
   `setup_wall_s` is how much that was. The pack file hash is computed once in
   the parent, before any clock starts, and is not in either window.
3. **Actual arm work**: `arm_io.nnue_evals_per_top_level_call` and the raw
   provider counters. Read `provider_stats_classification` beside them: the
   `store_endpoint_sizes` group is summed across workers as a **resource
   endpoint total**, not a count of positions the run saw.
4. **FastQ tripwire**: `fastq.budget_trip_rate`. A nontrivial rate is a finding,
   not an invitation to silently raise the cap. ⚑ `null` means `calls == 0`,
   i.e. nothing ran — it is not the healthy zero.
5. **Reuse**: FastQ/qsearch-DAG within- and cross-call counters plus
   `dag_per_game.canonical_hit_rate`.
6. **Memory**: `dag_per_game.nodes_peak_per_game` and
   `memory_peak_per_worker_bytes`.
7. **Search shape/coverage**: root-budget and termination data in each worker's
   detail record, plus the banked **end-to-end trace population** for diagnosing
   how each arm changes the leaves production Gumbel actually visits.

The harness does not claim that similarity to qsearch is strength, and these
end-to-end banks are **not** the input population for a paired evaluator-quality
verdict: each arm helped choose its own later leaves. The deciding standardized
deep-Stockfish evaluator-quality comparison must be a separate frozen-driver /
shadow-arm experiment in which all candidate evaluators score the same positions
without feeding values back into MCTSTree. This readout measures production
throughput, reuse, search shape, and endogenous trace distributions; it does not
silently turn those endogenous populations into paired quality evidence.

## Exit codes

| code | meaning |
|---|---|
| 0 | every gate passed |
| 2 | the report was written and is **inadmissible** — read `inadmissible_reasons` |

`assert_admissible(report)` is the library-side equivalent for a caller that
wants the exception instead.

## Non-goals

This PR does not:

- change FastQ search semantics;
- change `MCTSTree` or migrate Gumbel onto `CaePositionDag`;
- make the DAG concurrent;
- add quiet-check or mate-verifier logic;
- tune FastQ's defaults;
- implement the lazy-eval / PEXT speed-plan PRs.

Those decisions should consume this readout rather than precede it.
