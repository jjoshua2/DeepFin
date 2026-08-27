# Incremental NNUE state in tactical search

## Goal

The native NNUE evaluator currently full-refreshes the feature transformer at every
qsearch node. That is the wrong cost model for a search tree: a child differs from
its parent by one legal move, while a full refresh re-adds every active HalfKA and
FullThreats weight row.

`nnue-qsearch` now carries an NNUE state alongside the `CBoard` copies qsearch already
creates. `nnue-qsearch-refresh` preserves the previous implementation as an exact
oracle and benchmark baseline.

## Why there is make but no unmake

The current search is already **make-on-copy**:

```text
parent CBoard
    memcpy -> child CBoard
    push(move) -> child position
```

The NNUE state follows the same ownership rule:

```text
parent NNUE state
    copy -> child NNUE state
    apply(parent -> pushed child) feature delta
```

There is deliberately no NNUE `unmake()` in this PR. An unmake log is valuable when a
single mutable board is used by a depth-first search and siblings reuse the same state.
Here every sibling already owns a board copy. Reversible accumulator bookkeeping would
therefore add complexity and another sibling-corruption failure mode without avoiding a
copy that the search already requires conceptually.

If profiling later shows the ~5 KiB NNUE-state copy is the bottleneck, the next change
should convert **both** board and accumulator to one mutable DFS stack. That is the point
at which make/unmake becomes a coherent optimization rather than an isolated API.

## Update rule

### HalfKAv2_hm

When a perspective's king square is unchanged, compare the parent's and child's
`piece_on[64]` arrays. For every changed square:

1. subtract the old HalfKA feature row, if any;
2. add the new HalfKA feature row, if any;
3. apply the same subtraction/addition to the PSQT accumulator.

This naturally covers captures, en passant, promotions and castling-rook movement.

If that perspective's king moved, its orientation and king bucket changed, so that
perspective falls back to the existing exact full refresh.

### FullThreats

A move can alter sliding attacks far from its source/destination, so predicting a local
threat delta is fragile. Instead each NNUE state caches the sorted active threat feature
indices for both perspectives. A child regenerates the cheap relation/index list and
merges the old and new sorted sets:

- equal index: no accumulator work;
- old-only index: subtract its weight/PSQT row;
- new-only index: add its weight/PSQT row.

Only changed threat rows touch the 1024-wide accumulator.

The cache holds 128 active relations, matching the normal Stockfish active-threat
ceiling. This is an optimization boundary, not a validity assumption: if either state
exceeds it, the affected child uses the existing full refresh and remains correct.

## Check-resolution boundary

The shared check resolver currently transports only `CBoard`, not NNUE state. If qsearch
plays a checking move, the resolver searches the evasions and the eventual resolved leaf
initializes a fresh NNUE state. Non-check qsearch edges stay incremental.

That is intentionally conservative for this PR. Threading evaluator state through the
resolver would widen a correctness-critical shared interface before we know how much of
the remaining wall time is actually in check excursions.

## Gates

`tests/test_nnue_incremental.py` requires production `nnue-qsearch` and the retained
`nnue-qsearch-refresh` oracle to have:

- exactly equal returned values;
- exactly equal resolver/qsearch work counters;
- a dense non-zero synthetic PSQT net so feature-update mistakes are observable;
- optional exact parity on the real NNUE pack when `CAE_NNUE_TEST_PACK` is available.

`scripts/nnue_incremental_bench.py` runs the same pair on deterministic natural positions,
refuses to report a speed ratio if values or search-work counters disagree, and then
prints the incremental/full-refresh wall-clock ratio.
