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

The completed qualification below supersedes the intermediate failures.
All raw timing samples and compact diagnostic evidence are retained.

### Initial static failure

Run 35934827960 stopped at Ruff B023: push/pop closures were nested inside the
capacity loop. The helpers are now scoped to a per-capacity function, with no
suppression and unchanged expected trace. No native check ran in that attempt.
Self-review also added rejection of duplicate timing identities, unequal work
counts and impossible durations, plus Search/Chess source fingerprints.

Run 35935123234 passed focused static checks, all 61 collection Python tests,
and the existing native FIFO/traversal/mutation checks. The first ring compile
then rejected destructuring r in the second Boolean branch after its use in
the first. Dedicated push_open/pop_open helpers now destructure their own
parameters. Neither the compiler nor the behavioral oracle changed.

Run 35935298901 passed 37,416 bounded-ring trace rows in each of generic,
native-target and UBSan builds with identical hashes, and rejected the compiled
wrong-head mutation. The owning-root benchmark then failed name resolution:
its length helpers followed their caller. Helpers and the recursive dump law
are now declared before use. No runtime expectation changed; timings had not
started, and these partial checks are not a completed qualification.


Run 35935524101 again passed ring and prior native contracts, then rejected
the benchmark's mutual-recursion forward law as an unfilled live claim. The
drain now uses an IO(Bag) single-item helper and direct structural recursion;
no forward law, unsafe escape or compiler modification is used. Timing and
owning-root execution remained unrun in that attempt.


### Owning-payload failure and representation amendment

Run 35935785219 passed the ring's Array<U64> trace/mutation checks and all 18
list/FIFO generic owning cases, then failed the very first ring-owning case:
arm 2, one root, zero turns. The executable exited 1 before any output with
`bend: memory fault (machine stack overflow?)`. This was a real failure, not a
performance result; no owning ring was adopted.

The isolated diagnosis run 35936064206 compiled those exact sources normally
and with Clang 18 ASan/UBSan. Both list and FIFO controls passed; the ring
failed in both builds. UBSan identified an invalid shift in generated `blk_at`
with exponent 4294967293. Inspection of the retained generated C found the
recursive generic `Ring.slots` builder allocating a two-word Maybe slot while
specialized `Array.swap(Maybe<Work>, ...)` addressed it as a sixteen-word slot.
The unsigned block-class subtraction underflowed (1 - 4). This explains the
misleading memory-fault message; increasing stack size is not a fix.

Before any calibrated timings, the ring's internal carrier was changed to an
empty/singleton `List<&1, T>`. Its recursive carrier is boxed rather than
flattened with T, so allocation and access use a consistent slot shape under
this pinned backend. The tradeoff includes singleton allocation, so the ring
is not an allocation-free queue. The same public result types, capacity
rules, trace expectations, mutation, and owning payload checks were retained.
The compiler and generated C were not modified. This is a source-level
representation workaround for the exercised case, not a general compiler fix.

Diagnosis provenance: source commit `5e2c2a5eca3ce2b5e183bd98b93cccd1646eaf71`,
run `35936064206`, artifact `10783161833`, ZIP SHA-256
`b8a46427727721d2604bcde4565063799be7fefabb2f954bfd016f3abdbf2ce5`.
Generated C SHA-256:
`87dd6cb2d1184984576acf2a01cd11a4a622cf612f6afdb2fbc8fda23c6ced39`.


### Completed owning qualification

Run [35936460996](https://github.com/jjoshua2/DeepFin/actions/runs/35936460996), source `c7b00c19a69bdc2242e9f1931695b2bbb73f8a7a`.
Whole-repository Ruff/Basedpyright/Vulture and explicit collection static checks passed.
All 61 collection Python cases passed without skips (36 new, 25 existing).
Ring: 37,416 identical exact output rows per generic/native/UBSan mode; the compiled wrong-head mutation was rejected.
Owning payloads: 81 native contract executions (27 per mode), checking order, counters, identity, root state and observed history.
Existing FIFO/traversal/LIFO controls passed unchanged. Modes repeat fixtures; they are not disjoint datasets.

| Roots | Turns per measured sample | List median ms | FIFO median ms | Ring median ms | Reliable samples |
|---|---|---|---|---|---|
| 1 | 4194304 | 186.0 | 282.0 | 225.5 | True |
| 16 | 4194304 | 675.5 | 185.0 | 235.0 | True |
| 64 | 2097152 | 1149.5 | 145.5 | 169.0 | True |

Six measured permutations per size; calibration excluded. Raw samples, output hashes and exact source hashes are retained in [evidence](evidence/bend-owning-collections/summary.json).
These are collection-plus-root-update timings, not useful neural EPS, search throughput, Elo, allocation counts or tail-latency measurements. Descriptive medians from one host are not an independent review or a broad statistical performance claim.
On this host, FIFO/list median ratios favor FIFO by 3.65x at 16 roots and 7.90x at 64 roots. With one root, list-append was fastest. The boxed ring did not beat FIFO at 16 or 64 roots by median; overlapping/noisy individual samples make the smaller FIFO/ring differences less conclusive. Prioritize a FIFO scheduler experiment rather than adopting the ring as a performance upgrade.
The boxed ring avoids the demonstrated unboxed layout failure for the tested workloads; no general compiler correction is claimed. Neither ring nor FIFO is newly wired into production scheduling. No merge, deployment, GPU/model execution, compiler update or live configuration change occurred. Self-review only.
