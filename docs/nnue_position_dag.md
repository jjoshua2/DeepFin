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

**It is the headline because it is falsifiable, and it has fired.** A 6-thread probe against a build that released the GIL around `cae_nnue_state_make()` read `21 == 87`: threads that both missed the canonical probe both published the same structural position, and each duplicate's `link()` then failed, so its work was never accounted. The identity is therefore also the alarm for a published-but-unlinked node — which is why the counters are incremented *after* a successful link and must not be moved earlier to make the error path "tidy". Such a node is not leaked or unreachable: it is in the canonical table, so a retry finds it and the request becomes an ordinary transposition that only has to add the edge. (When the link fails because the edge array could not grow, the API now raises `MemoryError` rather than reporting an allocation failure as a `RuntimeError`.)

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

## Intended next layers

The next PR can implement the tactical expansion/termination policy on this graph: checks, captures/promotions, SEE, selective deepening, and backup. That search should keep path-specific repetition/fifty-move state outside the structural node and terminate repetition back-edges before recursive expansion.

A later Gumbel migration can reuse the same canonical nodes/edges and place sequential-halving statistics in a search-local edge overlay instead of storing `N/W/prior/parent` on the position node. This PR intentionally does not change current Gumbel behavior.
