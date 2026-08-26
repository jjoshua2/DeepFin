# Reusable structural-position DAG

This layer is the storage substrate for the native CPU tactical search and a possible later Gumbel migration. It is deliberately **not** another MCTS implementation.

## Core invariant

A structural chess position exists at most once in a `CaePositionDag`:

```text
parent A ─┐
          ├──> position X ──> children
parent B ─┘
```

The node has no parent pointer. Incoming edges are ordinary `(action, child_id)` graph edges, so two move orders that transpose really do converge on the same node id. This is stronger than the current `MCTSTree` transposition helper, which recognizes a donor but creates a separate recipient MCTS node with copied expansion data.

The canonical table uses open addressing and exact structural equality after the 64-bit hash probe. A hash collision therefore costs probes but cannot merge positions or overwrite the canonical node.

## What belongs in node identity

A DAG node is the **current structural position** that determines legal moves:

- piece placement / colors;
- side to move;
- castling rights;
- en-passant only when a capture is actually available.

This is the same semantic boundary as `cboard_transposition_key()`.

⚑ **Castling is compared masked to its four defined bits** (`CAE_DAG_CASTLING_MASK`). `CBoard.from_raw()` stores the caller's low byte unmasked, while `cboard_compute_hash()` indexes `ZOBRIST_CASTLING[castling & 0xF]` and movegen only ever tests WK/WQ/BK/BQ — so two boards differing above bit 3 have the same hash *and* the same legal moves. Comparing the raw byte split that one position into two nodes sharing a hash. The mask is canonicalization to what the rest of the engine already consumes, not input validation: rejecting such a board would make the DAG refuse a position the evaluator and movegen accept.

The following are intentionally **not** node identity:

- halfmove clock;
- repetition stack/history;
- DeepFin encoded-history planes;
- search visits/Q/priors/virtual loss;
- path-specific terminal/solved status.

Those are path/search-context properties. Two histories can reach the same structural position while differing on a fifty-move claim, repetition, or DeepFin neural input. A future Gumbel consumer may share the structural node and legal edges, but must keep history-sensitive evaluation state in its own overlay. This separation is what allows real chess transpositions to merge without laundering path-dependent draw semantics into a position value.

### Repetition means the structural store is not an acyclicity proof

Chess can return to the same structural position by reversible moves. Because repetition history is intentionally outside canonical node identity, a search can encounter an edge whose structural child is an ancestor node. The storage permits that representation; static NNUE reuse is still sound.

This is tested, not just asserted: `1.Nf3 Nf6 2.Ng1 Ng8` closes a four-edge cycle whose child id **is** the root id, and `dag_children()` reports that back-edge like any other edge.

The **path-aware search overlay** must decide repetition/fifty-move terminal semantics before recursively following such an edge. In the intended tactical search, a repetition back-edge is adjudicated on that path rather than recursively expanded forever. A future Gumbel migration has the same obligation. So `CaePositionDag` means canonical transposition-node storage; consumers must not infer that structural interning alone proves the reachable graph is mathematically acyclic.

## NNUE first consumer

`_nnue_ext` layers `CaeNnueState` and a static value over each canonical node.

For a new root:

```text
refresh NNUE state once
-> evaluate once when not in check
-> publish canonical node + payload
```

For a new child:

```text
parent CaeNnueState
-> incremental make from PR #469
-> evaluate once when not in check
-> publish node
```

For a transposition hit:

```text
probe canonical node
-> add another incoming edge
-> reuse existing CaeNnueState/value
-> NO incremental make
-> NO NNUE evaluation
```

An in-check node is still represented and still owns an accumulator state, but its static value is absent (`None` on the Python probe surface). NNUE is undefined in check; the future tactical solver must resolve evasions rather than consume a sentinel.

The accumulator array is explicitly 32-byte aligned because the AVX2 kernels use aligned loads. The NNUE DAG retains the same `CaeNnueWeights` mapping created by `_nnue_ext.load()`; it does not compile or load a second evaluator.

## Action integrity

`dag_intern_child(parent, action, child)` does not trust that the action and child correspond. The DAG reconstructs the canonical parent structure, first requires `action` to occur in that parent's generated legal-action set, then pushes it and requires the resulting structural position to equal the supplied child before mutating the graph or evaluating NNUE.

The explicit legal-membership check matters because `cboard_push_index()` is defensive around malformed actions; push-and-compare alone could otherwise accept an illegal no-op if a caller supplied the unchanged board as the alleged child.

This is a deliberate guard against the repository's recurring failure mode: a parameter can be accepted while the production path silently uses something else.

## Rerooting and lifetime

Rerooting changes only `root_id`; descendants and transposed nodes stay allocated and reusable. `dag_reset()` clears graph semantics and counters but keeps allocations and the retained weight mapping, so a caller can reuse the storage without repeated malloc/mmap churn.

`dag_open()`'s `initial_nodes` is bounded by `CAE_DAG_MAX_INIT_NODES` (`INT32_MAX/4`) and an out-of-range value raises **`ValueError`**. The bound is enforced twice on purpose: at the Python surface, where it can name the argument, and inside `cae_position_dag_init()` itself, because that function derives both the edge capacity and the hash size as `initial_nodes * 2` in int32 — an unchecked capacity wraps negative and reaches `malloc` as an enormous `size_t`. The graph layer is meant to be reused by the tactical search next, so the check belongs next to the arithmetic rather than in whichever caller happens to remember it.

No garbage collector is added in this PR. The first tactical-search consumer is expected to operate on bounded graphs; memory usage is explicit in `dag_stats()`. Once real workloads show the required lifetime, reclamation can be designed around reachability/generations instead of guessed in advance.

## Threading: single-threaded, and enforced rather than promised

The construction API is single-threaded, and `dag_intern_root()` / `dag_intern_child()` **hold the GIL across `state_init` / `state_make` / `evaluate`** so that constraint enforces itself: with no release window inside those functions, another Python thread cannot interleave between the canonical probe and the publish.

This is a deliberate reversal. The first version released the GIL around each NNUE call, which cost nothing to write and made the documented constraint unobservable:

- 6 threads interning the same 20 children produced **87 nodes for 21 distinct structural positions**, and 66 `RuntimeError`s from the duplicates' failed links;
- worse, `cae_nnue_state_make()` reads `&h->states[parent_id]` inside that window while another thread's publish can run `cae_nnue_dag_grow_payload()`, which `free()`s that array — a use-after-free reading an accumulator already handed back to `malloc`.

With the GIL held, the same probe reads 21 nodes for 21 positions, no duplicate ids and no errors, and a 2-ply 6-thread probe (2400 requests, growth to 512 payload slots mid-run) reads 421 for 421 with every value equal to a full refresh. The calls are µs-scale, so nothing measurable is given up.

A later Gumbel migration with concurrent walkers must **not** simply reinstate `Py_BEGIN_ALLOW_THREADS` here. It needs real synchronization first: single-owner checks or a lock spanning probe → publish → link, plus payload storage a concurrent grow cannot free under a reader (stable chunks or RCU-style retirement).

## Observable invariants

`dag_stats()` exposes:

- nodes / edges / root;
- hash probes, hits, inserts, collision steps;
- edge reuses and canonical node reuses;
- NNUE root refreshes (`state_inits`);
- incremental child makes (`state_makes`);
- actual static evaluations (`nnue_evals`);
- allocated DAG and NNUE payload bytes.

The headline invariant is the exact identity:

```text
state_inits + state_makes == node_count
```

Every canonical node was published by exactly one accounted NNUE state construction, and no node exists without one. `tests/test_nnue_position_dag.py` asserts it at *every* stats read.

**It is the headline because it is falsifiable, and it has fired.** A 6-thread probe against a build that released the GIL around `cae_nnue_state_make()` read `21 == 87`: threads that both missed the canonical probe both published the same structural position, and each duplicate's `link()` then failed, so its work was never accounted.

**It now holds on every path, allocation failure included.** A new child's edge is *reserved* — `cae_position_dag_reserve_edge()`, which grows the edge arrays if needed — **before** the node is published to the canonical table, so the link that follows cannot allocate and cannot fail. Publishing first was the earlier shape, and it was wrong in a way the identity itself exposed: an edge-growth failure after publication left a node in the table that no edge reached, a retry found it and reported *reuse*, its NNUE work was never accounted, and the identity was broken permanently rather than transiently. A construction now either completes or leaves the DAG untouched, and an out-of-memory edge reservation raises `MemoryError` rather than reporting an allocation failure as a `RuntimeError`.

⚑ **`nnue_evals <= node_count` is not an invariant worth reading.** It holds by construction on every path — a node is published at most once per evaluation — and duplicating nodes only widens the margin, so it is precisely blind to the failure it appears to watch. Measured against a deliberate double-publish mutant: the identity fires (`5 == 9` false) while the old relationship stays green and merely loosens, from `5 <= 5` to `5 <= 9`. It was the documented headline before this review.

Counters that do not mean what their names suggest:

| counter | what it actually counts |
| --- | --- |
| `hits` | canonical-table probe hits — **the transposition signal**. A new parent reaching an already-interned position increments it; so does a re-request of a position already interned (including a plain `dag_lookup()`), so read it against `probes`/`inserts`. |
| `node_reuses` | **not** the transposition signal: it additionally counts a repeated identical `(parent, action)` request, which never probes the table at all. |
| `edge_reuses` | only that caller redundancy — an exact duplicate `(parent, action, child)` edge request. Never a transposition. |
| `collision_steps` | linear-probe **displacement**: occupied slots stepped over, whatever their key. Not a count of 64-bit key collisions, which are far rarer than this number. |
| `probes` | `find_position()` calls, `dag_lookup()` reads included. |

A true transposition raises `hits` and `node_reuses` and leaves `state_makes` and `nnue_evals` unchanged for that request.

The Python test also pins the complete stats key set and `memory_bytes == dag_memory_bytes + nnue_payload_bytes`, so a `Py_BuildValue` format drift cannot silently shift or omit trailing metrics.

## First search consumer: `nnue-qsearch-dag`

The DAG's first search-side consumer is a **retrofit**, not a new search. `nnue-qsearch-dag` runs the existing quiescence — the same move policy, ordering, ply/depth budgets and fail-soft arithmetic, out of the *same* `cae_qsearch_node()` function — and changes only where a node's stand-pat number comes from. `nnue-qsearch` is left untouched and becomes the **oracle**.

That is the point of doing this before the tactical search proper. A new search has nothing to be checked against; a retrofit has an exact one, so the DAG substrate can be proved correct on its own before anything relies on it:

| property | how it is measured |
| --- | --- |
| bit-identity | every row of a ~470-position corpus (openings, middlegames, tactics, in-check, endgames) returns the oracle's exact value, and the twelve search-shape counters match as a block |
| corpus span | the corpus is asserted to CONTAIN the classes the claim leans on — promotions, legal en passant, and in-check rows whose evasions are all quiet — rather than left to the RNG that happens to supply a few |
| evaluate-once | `nnue_evals` is strictly below the oracle's, and equals `dag_nodes_interned` |
| reuse accounting | `nnue_evals + dag_hits_within_call + dag_hits_cross_call == qnodes` |
| persistence | a second call over a shared subtree costs strictly fewer evaluations than a cold store; re-running an identical call costs **zero** |

`tests/test_qsearch_dag_parity.py` owns all of it, and runs the six substrate-parity tests against **both** nets from one body: a parametrised `eval_pack` fixture yields the synthetic pack (always) and the real net (when `CAE_NNUE_TEST_PACK` is set, skipped otherwise, since CI has no 111 MB net).

⚑ **The real-net arm is not decoration — it kills a mutant the synthetic arm does not.** Under the "cache a fail-high search value in the node payload" mutant, `test_the_dag_holds_the_static_evaluation_and_never_a_searched_one[real]` FAILS while `[synthetic]` passes: on the PSQT-only pack that fixture's quiescence value and static value happen not to separate the poisoned node. A synthetic-only gate would have reported that mutant caught by three tests when it is really caught by four, and the missing one is the test named after the property.

### The numbers, over the 467-row corpus

Both nets, both at `resolver_depth=12 qply=3 check_plies=1` (what the test module runs):

| | synthetic PSQT pack | real net `nn-f68ec79f0fe3` |
| --- | --- | --- |
| quiescence nodes, both arms | 62,010 (equal, as required) | 59,081 (equal) |
| oracle NNUE evaluations | 62,010 (one per node) | 59,081 |
| DAG NNUE evaluations | **53,863** — **13.1%** fewer | **51,203** — **13.3%** fewer |
| probe hits | 6,061 within + 2,086 cross | 5,714 within + 2,164 cross |
| cross-call share of hits | 25.6% | 27.5% |
| `dag_memory_bytes` | 362 MB / 53,863 nodes ≈ 6.6 KB/node | ≈ same (the state size is the net's, not the pack's) |

⚑ **The saving grows with quiescence depth, so the test module's figure is a floor twice over.** At the C defaults (`32/4/1`) the same corpus and the same real net read oracle 201,671 → DAG 160,026, a **20.6%** saving with 8,293 cross-call hits — deeper search revisits more, which is the direction that matters for the tactical consumer this substrate exists for. The module runs the shallower config to stay cheap in CI, not because it is the interesting one.

⚑ **Both nets agree bitwise on all 467 rows**, and the corpus is checked to be able to tell them apart first: 430 distinct non-mate values and 467/467 non-zero on the real net. Agreement over a corpus that scored 0 everywhere would be worth nothing, which is how a symmetric fixture once made a whole module vacuous.

⚑ **The saving is 13.1%, not a multiple, and the memory figure is the reason to care.** A `CaeNnueState` is dominated by a 2×1024 `int16` accumulator, so the store costs ~6.6 KB per canonical position — 362 MB after one 467-position corpus. §4.4 of `docs/fastq_design.md` deferred the eviction/reset cadence decision until there were measurements; these are they, and they say the cadence question is real rather than theoretical. On this corpus a `dag_reset()` per position would forfeit only the 25.6% of hits that are cross-call, which is the trade the cadence decision is actually about.

The 13.1% is also a *floor* for what a workload with more overlap would see: these rows come from 24 independent scripted games plus crafted FENs, so most of them share nothing. The shared-subtree measurement is the other end of the range — re-running an identical call costs **zero** evaluations, and a capture child of an already-searched parent costs 133 against a cold store's 160.

### The counters, and the watermark that splits the hits

`arm_stats()` carries `nnue_evals` for **every** arm, counted inside the code that calls the evaluator. It is not a restatement of `qnodes`: for the incremental and refresh substrates the two are equal by construction (each quiescence node evaluates its own stand-pat), and for the DAG substrate they diverge by exactly the reuse achieved. That is what makes "evaluate once per canonical position" an observation rather than a description of an intention.

The DAG-only counters split a probe hit into **within-call** and **cross-call** using a node-id watermark: ids are dense, monotonic and never recycled, so the `node_count` captured at the start of each top-level call partitions every later hit exactly — below the mark is a node an earlier call created. Cross-call hit rate is the number the eviction/reset cadence decision waits on, so it has to be a measurement and not an estimate. `dag_enabled` distinguishes "this arm has no store" from "this arm has one that did nothing"; a bare `0` cannot.

### What the search may never write back

The DAG stores a `CaeNnueState` and the **static** NNUE value — window-independent, history-free facts. No alpha–beta result is ever written into a node. A backed-up value depends on the `(alpha, beta)` it was searched under and, with cross-call persistence, on a path this graph deliberately does not model; caching one would make a position answer differently depending on which window reached it first.

This is asserted at the node rather than argued: `arm_dag_value()` exposes the stored number, and the test requires it to equal `evaluate()` — first *proving* on that fixture that the quiescence value and the static value differ, so the assertion cannot pass vacuously. A separate two-window fixture searches the same positions under a narrow window (inside a parent's quiescence, where they fail high or low) and then under the full window at top level, through one persistent store, and requires both answers to be the window-correct ones.

Two path-sensitive verdicts are consequently never interned at all: in-check positions (handed to the resolver, and NNUE is undefined there) and drawn positions (decided from the halfmove clock and repetition history, which are not node identity).

### The node budget

`set_arm_config(..., dag_node_cap)` caps the expanding quiescence nodes one top-level DAG evaluation may spend. It **ships off** (`0`), and off is what every parity assertion runs under — a binding cap makes the arm return a value that is deliberately not the oracle's, so it cannot be a default. On a trip the node stands pat, which is window-independent and matches what the ply-budget cutoff already returns, and `dag_budget_trips` counts it.

The cap is consulted by this arm only. `CaeArmCtx.dag_node_cap` is set from the configuration for a DAG-backed context and to a hard `0` for every other, in one place, and `arm_stats()` reports *that field* — so a caller who sets the knob and reads `0` back off `nnue-qsearch` is being told the truth rather than shown the global it just wrote. Check resolution is not budgeted: forced evasions are mandatory shared correctness work, and charging them here would make the knob mean "how much correctness may this arm skip".

### Threading, again

The store's probe → evaluate → publish → link path is not atomic and a concurrent publish can `free()` the accumulator array another thread is reading — the same failure measured above. So the provider's vtable sets **`requires_gil`** (`CaeValueProvider`, `_value_provider.h`), which declares `eval()` non-reentrant, and two consumers act on it:

- `_nnue_ext` does not release the GIL around a batch evaluated through such an arm;
- **`MCTSTree` refuses to install one at all**, in `resolve_provider_export()` — after the capsule is resolved and before anything is installed, which is the only point *both* install routes pass through.

⚑ **An earlier version of this section claimed the second half and did not implement it.** The predicate was a list of vtable pointers inside the publishing module that only that module's own batch loops consulted; the tree never saw it. The tree's exclusion actually rested on the DAG arm being *unreachable* — absent from `CAE_VALUE_PROVIDER_MODULES`, no capsule exported — and the name table's own comment invites callers to install a provider by passing its capsule directly. Exporting `qsearch_dag_arm_capsule` symmetrically with the other two arms would therefore have handed tree threads the non-atomic path with the GIL released while the guard named in these docs never ran: a value accepted and then silently ignored, inside the fix for that defect. The flag now lives in the vtable, every consumer reads the same field, and `tests/test_check_resolver.py` installs a fake provider through the real capsule ABI to prove the refusal fires for a capsule the name table has never heard of.

Not publishing the capsule remains true and is ergonomics; the vtable flag is the enforcement. Installing this arm in the tree must wait for real synchronization, not for a name to be added to a table.

## Intended next layers

The next PR can implement the tactical expansion/termination policy on this graph: checks, captures/promotions, SEE, selective deepening, and backup. That search should keep path-specific repetition/fifty-move state outside the structural node and terminate repetition back-edges before recursive expansion.

A later Gumbel migration can reuse the same canonical nodes/edges and place sequential-halving statistics in a search-local edge overlay instead of storing `N/W/prior/parent` on the position node. This PR intentionally does not change current Gumbel behavior.
