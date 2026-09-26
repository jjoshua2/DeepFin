# Append-only numeric key-to-ID registry

`IdRegistry` is a small consumer of the existing `InsertOnce` API. It owns the
map AND the next-ID state, so callers do not have to update a separate counter
correctly after every lookup, failed insertion, or counter overflow.

## API and lifetime

`new(bits, first)` creates an empty registry. `bits` has the original U64Map
contract (1..16, half of 2^bits buckets may hold entries). `first` is the first
U32 ID to issue, not a restored counter or the number of existing entries.
Use zero for ordinary dense IDs, or a different start to reserve a prefix.

`intern(registry, key)` returns the updated owner and:

| Outcome | Effect |
| --- | --- |
| `Assigned{id}` | Adds a new key with the next ID, then advances the counter |
| `Known{id}` | Returns the existing ID; no counter or dictionary change |
| `TableFull{}` | Missing key cannot fit; no ID consumed and no eviction |
| `IdsExhausted{}` | No unused U32 ID remains; no state change |

The maximum U32 ID (4,294,967,295) is a usable last ID. Once assigned, `next_id`
becomes None; no arithmetic increment of that value occurs. An existing key
continues to return Known after either limit. When both limits apply, a missing
key returns IdsExhausted. `get` is read-only lookup and returns Maybe<U32>;
`inspect` reports entry count and the optional next ID without consuming it.
Zero IDs, maximum IDs and zero/full-width keys are ordinary values.

There is no erase, reset, overwrite or ID-recycling operation. Bindings and issued
IDs are stable only for this registry lifetime when using these public operations.
Two fresh registries can issue the same numeric ID: consumers must scope handles
to their owning registry/generation. Reconstructing exposed storage, extracting
its table and mutating it, or calling low-level helpers with invalid states is
outside the API contract. Public constructors are not a deserialization boundary.
This is a single-owner module, not concurrent atomic interning.

ID allocation is distinct from node allocation. Assigned does not prove a node
was successfully created, that two position hashes denote the same full board,
or that a history/model-sensitive evaluation can be reused. A consumer needing
external allocation failure recovery can use the reservation protocol below;
there is deliberately no rollback API that could recycle an already-issued ID.
No existing engine, graph, tree or neural cache is wired to this registry.

## Reserve, prepare external data, then commit or abort

Use `reserve(registry, key)` instead of immediate `intern` when caller preparation
can fail. It returns either `Settled{registry, outcome}` (Known, TableFull or
IdsExhausted, with no mutation) or `Prepared{reservation}` for an admissible new
key. Prepared has NOT inserted the key or advanced the counter.

The reservation owns the map and prelocated vacant slot. The original registry
is unavailable until the reservation is consumed, so its slot cannot become stale
through another public registry operation. There is one outstanding reservation
per owner, not a concurrent or multi-writer transaction protocol.

1. `candidate(reservation)` returns the reservation plus its proposed U32 ID.
2. Prepare the caller's payload privately. Do not expose that ID as a committed
   handle or make the payload reachable through it yet.
3. On explicit preparation failure, `abort(reservation)` restores the registry
   without publishing a binding or advancing its counter. The proposed ID can
   be reused by a later attempt, including for a different key.
4. On success, `commit(reservation)` returns the registry and `Assigned{id}` for
   a valid reservation. It admits into the owned vacant slot without rehashing
   or searching again, then advances the counter. Publish the prepared payload
   only after that outcome, according to the consumer's ownership contract.

A proposed ID is not an issued ID: abort does not retract or recycle any committed
handle. U32_MAX can be reserved and aborted repeatedly; only its commit exhausts
the counter. Hits, full maps and exhausted IDs never produce a reservation or
require caller allocation. The existing immediate `intern` API is unchanged.

The registry does not invoke an allocator, catch exceptions, clean up the caller's
payload, or make external objects and map storage crash-atomic. The caller must
explicitly abort on recoverable errors and release its private staged resources.
Process termination, native allocation failure during commit, forged reservations,
raw-table mutation, and cross-thread publication are not rollback guarantees.
Full-position/hash-collision and history/model checks are still separate contracts.

## Correctness checks

```sh
python -m pytest tests/test_bend_map_registry.py tests/test_bend_map_reservations.py
python -m native.bend_engine.u64_map_probe.registry \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-id-registry --chess
```

The independent reference uses a Python dictionary and an unbounded integer
counter. Every operation checks returned IDs, outcome, size and optional next ID.
It does not reproduce the open-addressing algorithm or mask overflow to U32.
Thirty-five boundary/configuration/mixed cases cover all admitted table exponents,
revisits at capacity, failure preservation, the last one/two/four IDs, simultaneous
limits, wrapped high-half collisions and maximum-address smoke. The maximum table
is not exhaustively filled. Four existing CBoard corpora add chronological interning
and final readback; IDs are assigned by Bend, not preassigned by the fixture.
Generated legal walks are not production search access traces.

The reservation extension adds eleven separate boundary/mixed cases and, with
`--chess`, an abort/get/commit/get replay over each unchanged CBoard corpus.
Existing keys must settle without reserving; unsuccessful attempts leave no ghost
binding or ID gap. Ordinary intern and staged commit are exercised together.
The trace's `a` command models an explicit caller failure, not an actual malloc
failure or external allocator integration. Old fixture definitions stay unchanged.

Four original broken implementations must compile and exit normally, then disagree
with the unchanged oracle: advance on Known, advance on TableFull, wrap IDs to zero,
and reject a known key after exhaustion. Three new mutations consume an aborted
ID, publish on abort, or commit the wrong key. A crash, timeout or build error is
not accepted as semantic mutation detection. Generic, portable-U64, explicit
BMI2/POPCNT and UBSan builds repeat fixtures, not independent games. Existing
source-only numeric-map CI includes this extended entry point without a new
workflow or any speed gate. All 22 new Python tests are selected by its map glob.

Local checks before the initial registry publication: 35 new Python cases passed
with --noconftest. The exact pinned Bend compiler generated the driver, and 35
cases / 2,506 operations / 2,573 rows matched in generic, portable and BMI2/POPCNT
builds under Clang 17. The enclosing 200-second local command timed out before
completing UBSan and mutations; this was not an end-to-end native qualification.
The initial full hosted run and later reservation-extension results are recorded
separately on PR #876; historical tests do not substitute for current-source CI.

This is correctness work, not a performance experiment. No speed, memory, playing
strength, formal proof, or independent review is claimed. Earlier benchmarks and
sources retain their original identities. Self-review only; no merge, deployment,
compiler upgrade, live configuration change or third-party source import.
