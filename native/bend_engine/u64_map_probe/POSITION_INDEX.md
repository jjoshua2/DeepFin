# Collision-safe canonical-field position index

`PositionIndex` uses the existing U64Map as a hash-to-chain-head index and a
separate fixed array of complete identity records. A numeric hash match only
selects candidates: equality checks all eight bitboards and all three metadata
fields. Distinct canonical positions sharing the same 64-bit hash receive
different IDs instead of being merged. The numeric-only IdRegistry is unchanged.

## Input and lifetime contract

`Identity` contains six piece bitboards (pawn, knight, bishop, rook, queen, king),
white and black occupancy, side to move, four-bit castling rights, and canonical
en-passant square. The test adapter mirrors `chess_workloads.structural` and
`_position_dag.h`: pseudo-legal EP availability selects the square, with **64**
representing absent EP in this unsigned Bend record. Pinned EP is deliberately
retained by that existing conservative identity rule.

The caller must supply already-canonical fields and the same deterministic hash
for the same identity. The index does not parse FEN, validate board legality,
normalize castling/EP fields, or repair inconsistent hashes. A different hash
for identical fields can create a different binding. Raw constructors/internal
helpers are not safe deserialization or mutation interfaces.

`new(bits)` accepts 1..16, allocates 2^bits hash buckets and 2^(bits-1) identity
records, and returns None for invalid sizes before deriving record allocation.
`intern(index, hash, identity)` returns the updated owner and Assigned{id},
Known{id}, or Full. New IDs are consecutive record indices starting at zero;
there are no deletion, recycling, reset or reserve/commit operations. `get`
returns Maybe<U32> and `length` returns the number of distinct records.

The capacity counts **identities**, not distinct hash values. A colliding position
consumes its own record even though it shares an existing hash-table entry. Known
positions remain accessible after the record array fills. Failed admission changes
neither existing records nor their IDs. Hash heads store record indices; record
links use Maybe<U32>, so record zero is not an empty-link sentinel.

Each new record is prepended to its same-hash chain. The owning index prevents
concurrent mutation through its public API; each traversal has a size-derived
bound. Lookup is not worst-case constant time: a deliberately colliding group
requires full-record comparisons. Storage includes both arrays, so earlier
numeric-map memory/timing results do not describe this implementation.

This is an append-only **structural-index candidate**, not a production DAG or a
neural evaluation cache. It stores no edges, search visits, neural outputs or
external node pointers. IDs are local to this index lifetime, not cross-generation
handles. History, halfmove clock and model version remain outside structural
identity and must be retained by a future consumer. Allocation failure during an
insertion is not a recoverable external transaction; the separate IdRegistry's
reservation API cannot be applied to this owner's storage.

## Qualification

```sh
python -m pytest tests/test_bend_map_position_index.py
python -m native.bend_engine.u64_map_probe.position_index \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-position-index --chess
```

The independent reference is a dictionary indexed by (hash, complete fields),
not a copy of the chain or probing algorithm. Base fixtures vary each of eleven
fields while holding the hash fixed; fill same-hash groups to record capacity;
read back the oldest and newest members; mix separate hash groups; and cover
constructor bounds. Field-sensitivity vectors need not be legal chess positions.
Maximum allocation is a small record smoke, not full population of this new
identity-record layout. The older full-capacity numeric-map test is not relabeled
as full-capacity PositionIndex coverage.

With --chess, the existing fresh compiled CBoard producer supplies its unchanged
legal-walk corpus. Every observation is replayed in bounded 32-record chunks with
both its original hash and an intentionally constant hash. Full identity must
still give stable distinct IDs within each chunk. There is no claim of identity
across fresh chunk owners. Existing history/halfmove/EP/castling boundary pairs
are also replayed with a forced equal hash. Corpus generation is not modified,
and these are not recorded production access schedules.

Every operation's ID, disposition and size is compared exactly, followed by reverse
readback. Generic, portable-U64, explicit BMI2/POPCNT and UBSan repeat the same
fixtures. Three wrong implementations must compile and run normally before
failing the oracle: hash-only identity, dropping the collision chain, and rejecting
known keys at capacity. Crashes and compiler errors do not count as detecting them.

Local qualification passed all 26 base cases / 636 operations / 685 output rows
in all four modes and all three executed mutations with the pinned Bend compiler
and Clang 17. All 39 new Python tests passed with global conftest disabled. The
container lacks python-chess/the checkout's native extension, so fresh CBoard
replay and full static/locked-environment results are qualified separately by
hosted CI and reported on PR #876. No older result substitutes for those checks.
No speed/memory/playing-strength claim, compiler change, live setting, merge or
deployment. Self-review only; no independent review or formal proof.
