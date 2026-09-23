# Owning FIFO and traversal screen

This is the first bounded follow-up to reviewing
[Giulio2002/bend-collections](https://github.com/Giulio2002/bend-collections).
It independently implements the standard two-list FIFO algorithm and the
finished-state traversal pattern. No upstream source, crypto, dependency, hub
import or compiler upgrade is included. Licensing and exact-version qualification
remain prerequisites for importing upstream modules.

## What is implemented

`Queue.bend` supports `Queue<a, T>` for `T: Kind(a)`: push/pop transfer the value
and return the owning queue. There is deliberately no copying `peek` API. The
invariant is `front ++ reverse(rear)`, with cached size equal to both list lengths.
Operations from `new` preserve FIFO order. Push is O(1); pop is amortized O(1)
along a single consumed state history and O(n) on a reversal. This is not a
concurrent queue, worst-case constant-time deque, bounded scheduler or ring buffer.
There is no automatic capacity limit; a future scheduler must enforce admission.

The existing `Search.select` now returns when a step reaches its fixed point,
rather than repeatedly reading that same leaf until its fuel expires. Backup
returns when inactive. Fuel-zero behavior, selection scoring, tie breaks, backup
signs, pending identities, capacity and all public function signatures remain.
The old `select_step`/`backup_step` helpers are unchanged. No live process changes.

## Qualification

Use the **unchanged** pinned U64 compiler. From the repository root:

```sh
bash native/bend_engine/install_ci_bend.sh
COMPILER=build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae
python -m native.bend_engine.collections_probe.run_probe \
  --compiler-root "$COMPILER" --cc clang-18 --benchmark \
  --report artifacts/bend-collections.json
python -m native.bend_engine.collections_probe.session_check \
  --compiler-root "$COMPILER" --cc clang-18 \
  --report artifacts/bend-collections-sessions.json
python -m pytest tests/test_bend_collections.py tests/test_bend_search_sessions.py tests/test_bend_root_advance.py
```

The owning test queues actual `Array<U64>` values, covering high bits, empty
pops, fill/drain, mixed operations and reuse. Every emitted value and final size
is checked against Python's independent deque in generic/native/UBSan modes.
An executable, type-preserving LIFO mutation must fail the same oracle; compiler
failures are not counted as semantic rejection. Separate probes test actual
Search selection and backup at depths 0..7, zero/short/long fuel, terminal node
statuses and inactive backup with an invalid sentinel (which must not be read).
The existing root/session oracle separately exercises actual legal chess trees,
reply rejection, capacity, cancellation and root advancement.

## Timing limits

`benchmark.bend` compares list-append rotation against the new FIFO at 16, 256
and 4096 **U32 IDs**, with 20,000 rotations per sample and alternating order.
The first pair at each size is warmup. Every arm's checksum and final length
must match Python's deque. Raw milliseconds are retained; samples shorter than
20 ms are explicitly flagged and must not be used for a precise speedup ratio.
These are native CPU loop measurements, not allocation counts, root-ownership
costs, real scheduler latency, GPU utilization, useful neural EPS or playing
strength. Construction is outside the timed loop; runtime lifetime/drop costs
can still affect this screen. Do not infer a scheduler win from ID-only timings.
No claim is made that this equals the upstream library or its benchmark.

## Adoption gates

The queue is opt-in and **not wired into** the current bounded multi-root runner.
Before adoption, benchmark owning roots or root IDs plus their actual store,
retain root/ticket/history/row routing, enforce admission bounds and check
cancellation/fairness against the existing scheduler oracle. Compare a bounded
ring too. Keep direct arrays for fixed buffers. Arena growth and a U64-keyed
map/cache require separate capacity, sentinel, collision and identity contracts.
The String-keyed upstream map is not adopted or used to stringify chess keys.

These are native tests and self-review, not formal proof or independent review.


The current-compiler integration wrapper compiles the existing session driver and CBoard support with the same flags and calls the unchanged independent session/root oracles. The legacy bitboard/session toolchain manifest remains untouched; this screen explicitly verifies the CI installer's aaeb9bc9 84-file source fingerprint.

The session driver has existing foreign transport I/O. Its build explicitly requires and records the exact pinned compiler's 18-definition foreign-dependency notice; it is not labeled a pure proof. Unexpected diagnostics fail. Collection/traversal compilation still requires no stderr.
