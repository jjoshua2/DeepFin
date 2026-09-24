# Exact numeric-key map candidate

Independent follow-up to the bend-collections discussion and PRs #864/#874.
This module targets main directly: it does not need or change the scheduler,
search arenas, compiler, model, crypto modules or any live configuration.
No third-party collection source is copied or vendored.

## Contract

`U64Map.bend` is a single-owner, bounded `U64 -> U32` map. `new(bits)` accepts
1..16: physical buckets are `2^bits`, and maximum entries are `2^(bits-1)`.
For example, `new(10)` allocates 1,024 buckets for at most 512 distinct keys.
Invalid sizes return None before allocating. Zero and U64_MAX are valid keys;
zero and U32_MAX are valid values. Occupancy is explicit, not a key sentinel.

`get` and `remove` return the updated owner and Maybe<U32>; missing and zero
remain distinct. `put` returns Added, Replaced(previous), or Full. Replacing
an existing key still works at the entry limit, while a rejected new insertion
leaves contents and size unchanged. Removal repairs the probe chain using
backward shifts, including a cluster wrapping across the final bucket. All
probe/repair loops have a bucket-count bound. No tombstones accumulate.

Use the public new/get/put/remove/length operations and preserve the constructor's
invariants. The exposed Table/Slot constructors and internal helpers are not a
validated serialization API. Concurrent access and arbitrary owning payloads
are not supported. Values can represent node IDs, but no node arena is wired in.
The concrete Slot type deliberately avoids a new quantity-generic owning-array
layout dependency. No compiler workaround, generated-C patch or unsafe cast is
introduced by this module.

This is NOT a transposition/evaluation cache: it has no eviction, replacement
policy, model identity or history-aware cache key. Bucket collisions are resolved
by full 64-bit equality. A collision between two caller-supplied position hashes
is a separate issue that a numeric map cannot resolve on its own.

## Qualification

```sh
bash native/bend_engine/install_ci_bend.sh
python -m native.bend_engine.u64_map_probe.run_probe \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-u64-map
python -m pytest tests/test_bend_u64_map.py
```

The runner checks the existing compiler fingerprint, generates native C and
compares every returned value, disposition and size against Python dict semantics.
There are 32 fixture cases, 7,042 operations and 7,103 exact output rows per build
mode. Generic, portable-U64, native-target and UBSan builds repeat the same cases,
not independent datasets. Fixtures cover limits/replacement, deletion/misses,
reuse, deterministic churn, high-bit keys, same-low-half colliding keys and
wrapped mixed-home clusters. All admitted constructor sizes are exercised; the
maximum size gets a high-address smoke, not a full 32,768-entry stress test.

Three deliberately broken maps must compile, exit normally and then disagree
with that same oracle: low-half-only equality, missing backshift, and ignored
entry limit. A compiler failure, sanitizer crash or timeout is not accepted as
successful mutation detection. A fresh output directory preserves failed runs;
summary, input/expected hashes and actual stdout/stderr are retained. The text
protocol is a test transport, not production input handling or a timed hot path.

Local final qualification passed all four modes and three negative controls
with the pinned Bend sources, Bun 1.4.2 and Clang 17. The 34 Python cases passed
locally without global conftest; that is not the locked hosted test environment.
Two earlier outer-container command timeouts interrupted otherwise partial
checks; neither was counted as a complete qualification. Hosted results are
reported separately in the PR and source-only CI artifacts.

## Adoption boundary

No speed, memory, playing-strength, formal-proof or independent-review claim is
made. This first increment qualifies behavior, not the hash quality or the fastest
layout. Next compare actual numeric lookup/update/delete workloads and equal
memory budgets before using this instead of another map or cache. The string-keyed
upstream benchmark is not a numeric-map baseline. Review is self-review only.
