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
external allocation failure recovery must design that transaction before adoption;
there is deliberately no rollback API that could recycle an already-exposed ID.
No existing engine, graph, tree or neural cache is wired to this registry.

## Correctness checks

```sh
python -m pytest tests/test_bend_map_registry.py
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

Four broken implementations must compile and exit normally, then disagree with
the unchanged oracle: advance on Known, advance on TableFull, wrap IDs to zero,
and reject a known key after exhaustion. A crash, timeout or build error is not
accepted as a passing negative control. The four native build modes reuse fixtures,
not independent games. Existing map/insert-once/replay tests remain unchanged.

Local checks before publication: 35 new Python cases passed with --noconftest.
The exact pinned Bend compiler generated the new driver, and 35 cases / 2,506
operations / 2,573 rows matched in generic, portable and BMI2/POPCNT builds under
Clang 17. The enclosing 200-second local command timed out before completing UBSan
and mutations; this is not an end-to-end native qualification. The full locked
hosted workflow separately checks all modes, mutations and fresh CBoard replay;
its result is recorded in the PR rather than assumed from partial local coverage.

This is correctness work, not a performance experiment. No speed, memory, playing
strength, formal proof, or independent review is claimed. Earlier benchmarks and
sources retain their original identities. Self-review only; no merge, deployment,
compiler upgrade, live configuration change or third-party source import.
