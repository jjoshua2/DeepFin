# Single-search lookup or insert

`InsertOnce.get_or_insert(table, key, candidate)` is an opt-in operation on a
valid **U64Map** owner. It calls `U64Map.locate` once and reuses the vacant slot
for admission, rather than calling get and then searching again with put.
It returns the updated table owner plus one of:

| Result | Meaning | Dictionary effect |
| --- | --- | --- |
| `Inserted{value}` | New key accepted; value is the candidate | Adds exactly one entry |
| `Existing{value}` | Key already present; value is its stored value | None, even at the entry limit |
| `Saturated{}` | Missing key cannot be admitted | None; no eviction |

The distinction is useful for stable key-to-ID bindings. A caller proposing ID 7
for a key already bound to ID 3 receives Existing{3}; the binding stays 3. A
caller allocating consecutive IDs advances its counter only for Inserted, not
for Existing or Saturated. This module is not an ID allocator: counter overflow,
global ID uniqueness, external node lifetime and identity checks remain caller
responsibilities. Ordinary U64Map.put intentionally still overwrites, and remove
still erases; insert-once does not freeze an owner against those other operations.

Use it only with the hash table's placement invariant, never a populated ScanMap
owner despite their shared type. Zero and maximum U32 values and full-width U64
keys remain ordinary data. Exposed helper functions and storage constructors are
not untrusted deserialization or stale-slot APIs. This is a single-owner operation,
not a thread-safe atomic insertion or a worst-case constant-time guarantee.

No engine/cache consumer is changed. Numeric key equality still does not establish
full-position, history or model identity. The earlier collision and neural-cache
limitations apply unchanged. The source avoids a second lookup; no wall-time
speedup, reduced allocation or playing-strength claim follows from that fact.
Historical benchmark sources, observations and results are not relabeled.

## Regression checks

```sh
python -m pytest tests/test_bend_map_insert_once.py
python -m native.bend_engine.u64_map_probe.insert_once \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-insert-once --chess
```

The dictionary oracle uses membership/setdefault, independently of open addressing.
The native protocol mixes insert-once with ordinary get/put/remove, covering full
admission, hits at capacity, stored zero/MAX values, deletion/reinsertion, wrapped
high-half collisions, deterministic churn and maximum-address smoke. The existing
CBoard corpus is replayed with a DIFFERENT proposed ID on each revisit, then all
final bindings are read back. The first value must survive; replaying the same
proposal on every visit would miss an accidental overwrite.

Three broken implementations must compile and exit normally, then fail the exact
trace: overwriting an existing value, returning the proposed instead of stored
value, and rejecting an existing key. A crash or compiler error is not accepted
as mutation detection. Generic, portable-U64, explicit BMI2/POPCNT and UBSan builds
repeat the same fixtures. A fresh output directory preserves failed evidence.
Persistent numeric-map CI includes these tests alongside the unchanged map checks,
without any new benchmark run or speed threshold.

Local checks: 31 new Python tests passed without repository-wide conftest; native
16-case/2,478-operation traces passed in all four modes using the pinned compiler
and Clang 17. Two enclosing local tool invocations timed out before finishing the
mutation section; remaining commands were completed and every retained trace and
all three mutations checked separately. Neither interrupted harness invocation is
claimed as an end-to-end pass. Hosted CI separately qualifies the locked environment,
the CBoard replay and the complete entry point. Current results are recorded on
PR #876. Self-review only; no independent review, formal proof or deployment.
