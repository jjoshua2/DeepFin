# Bounded owning ring and calibrated collection comparison

## Preregistration

Continuation of PR #864 at `185106dfbb271abe41a81c5256d87ecb7493d257`.
Compiler: unchanged `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, verified
84-file fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

Hypothesis: moving owning roots through a two-list FIFO avoids list-append
traversal, while a bounded array ring may avoid reversal latency. Neither
hypothesis establishes scheduler throughput, neural EPS or playing strength.
The existing ID-only benchmark remains historical evidence, not a ratio claim.

Implement a bounded ring using ownership-preserving Array.swap, never Array.get
on noncopyable elements. Full push must return its incoming value and preserve
the queue. Empty pop must preserve the queue. Accept logical capacities 0..4096,
including nonpowers of two; allocate a power-of-two backing array of distinct
empty slots. Capacity zero rejects every push without indexing. Above 4096 is
rejected before allocation. API invariants apply to values constructed by new;
the language does not hide public representation constructors. This is a
single-owner collection, not a concurrent or lock-free queue. The compiler's
Array implementation determines actual runtime costs; no O(1) native-access
claim follows from the source representation.

Controls: the existing FIFO, list-append rotation, and the new ring all move
actual Search.Tree values created by Search.start (4096 nodes each), plus
separately owned U64 history arrays. Every turn updates the root node visit
count and completed counter. Every sample validates the checksum, complete
root order, counters, epoch, request, capacity, pending sentinel, stop, depth
limit and observed U64 history value against independent Python expectations.
These are synthetic payloads; no legal search or evaluator runs in this loop.

Correctness gates: exact deterministic bounded-ring trace against Python deque
at capacities 0/1/2/3/7/16/17/4096, constructor rejection at 4097/U32_MAX, high-bit
payloads, overflow, empty pops, wraparound, mixed operations and reuse. Run in
generic/native/UBSan modes. An executable wrong-head-advance mutation must
compile, exit normally and be rejected by the same oracle. Owning-root sample
contracts cover three arms, three sizes and three work budgets in three modes.
Any disagreement blocks publication, not a relaxed tolerance.

Timing: 1/16/64 owning roots, one native worker. Double a shared work count from
32768 to at most 20 million until all three arms take at least 100 ms. Retain
calibration observations but exclude them from comparison. Measure all six
permutations of arm order at that common work count. A size supports a ratio
only when every one of its 18 measured samples is at least 50 ms. Retain every
millisecond observation, work count, output hash and source identity. Report
medians descriptively; no confidence interval, tail-latency claim or universal
winner from this screen. Construction and final drain/checks occur outside the
Bend timer; mutation, collection dispatch and allocator behavior occur inside.

Budget: isolated hosted CPU checks, two Torch threads, one compiler job, no GPU,
training, model export, live checkout, config, scheduler or compiler changes.
Recovery: source-only PR; no merge or deployment. Ring and owning benchmark
remain opt-in under collections_probe. Review is self-review, not independent
review or a formal ownership/correctness proof.

## Reproduction

```sh
bash native/bend_engine/install_ci_bend.sh
COMPILER=build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae
python -m native.bend_engine.collections_probe.owning_benchmark \
  --compiler-root "$COMPILER" --cc clang-18 --benchmark \
  --report artifacts/bend-owning-collections.json
python -m pytest tests/test_bend_owning_collections.py
```

The Bend executable reads DEEPFIN_COLLECTION_BENCH as three decimal fields:
arm (0=list, 1=FIFO, 2=ring), root count (1..128), and turns (0..20 million).
The Python harness sets this only in its child process environment.

## Readout

Pending native execution. The 32 independent Python parser/oracle/timing tests
passed locally; that result does not qualify the Bend implementation. Compiled
results and any failed attempts will be added after execution.
