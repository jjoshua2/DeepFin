# Bend perft: scalar selections and conditional list construction

## Scope and discovery

Follow-up to [the perft baseline](2026-09-18-bend-perft-baseline.md) / PR #775.
Base core: `6b4533e6736fb75ecb43a4cc2dd4ab883fc54ce2`.
Compiler stays `57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`.
No production, compiler, chess-rule or perft-depth changes.

Exploratory Clang 17 `-pg` profiling found heavy runtime destructor/reference
traffic. Inspection of generated C identified two avoidable sources:
`Bool.pick(U64, ...)` boxes its two-word alternatives, and eager selection
between `acc` and `Con{m, acc}` constructs/shares/discards list branches even
when only one branch is needed. Source monomorphism avoids the former; an
explicit conditional list constructor avoids the latter. The U64 helper still
evaluates its scalar operands eagerly. Chess move order/filtering is unchanged.

Exploratory local timings selected this candidate (not a preregistered discovery):
scalar selection alone took 0.4301 / 0.3508 seconds vs 0.9646 / 0.7924 baseline
for startpos d5 / Kiwipete d4. The follow-up list-branch screen took 0.2592 /
0.1935 seconds vs 0.4322 / 0.3362 for scalar-only on that same local run.
Those two rounds are not combined into a cross-run timing verdict.

## Hosted confirmation plan

Record before running the hosted confirmation: compile previous core and full
candidate with the same adapters, source-pinned compiler, Clang flags and CPU.
Use the existing benchmark's rotating order with three samples per engine,
full warmups, and exact node checks. Startpos d5 = 4,865,609, Kiwipete d4 =
4,085,603. Keep only if all correctness checks pass and candidate median time
is at least 10% lower on both fixtures; otherwise retain only the diagnostic
and revise/revert the source changes. No noisy timing gate is added to CI.

The correctness gate remains all four native build modes, 30 fixed positions,
33 perft/divide cases, child states, seeded-play samples and invalid requests;
python-chess supplies the second fixed-position oracle in hosted validation.
Cheap parser/instrumentation tests do not run the compiler or perft.

Budget: one hosted single-CPU confirmation, bounded by existing subprocess and
job timeouts; no GPU, training, live checkout, or routine depth increase.
No changes to attack tables, special moves, king-safety policy or compiler.
Recovery: discard this isolated optimization, not reset earlier branches.

## Allocation diagnostic

`profile_allocations.py` instruments only disposable generated host C. Anchors
must occur exactly once; compiler fingerprints still apply. It resets/enables
four counters after warmup and stops before output/final table destruction.
Runs use one thread. Output node counts are checked on warmup and traversal.

These are dynamic calls in an INSTRUMENTED build: `heap_alloc` includes runtime
freelist requests (not OS malloc), `rfc_wrap` creates reference wrappers,
`rfc_bump` increments references, and `term_drop` is an entry count, not an object
or byte count. Instrumentation can change optimization. No timing from these
builds is used for performance claims; no peak-memory reduction is inferred.

Local startpos d5 previous -> candidate: heap requests 92,997,716 -> 25,304,456;
reference wrappers 23,423,583 -> 0; reference bumps 11,359 -> 0; destructor
entries 33,887,668 -> 5,062,890. Kiwipete d4: 73,504,419 -> 19,239,248;
18,415,716 -> 0; 42,403 -> 0; 29,905,124 -> 4,183,465 respectively.
All four node totals matched. This supports the mechanism, not a complete time
attribution or proof that all remaining allocations are necessary.

## Reproduce

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
git show 6b4533e6736fb75ecb43a4cc2dd4ab883fc54ce2:native/bend_engine/legal_probe/Chess.bend > /tmp/Chess.before.bend
python -m native.bend_engine.legal_probe.profile_allocations --baseline-chess /tmp/Chess.before.bend --report artifacts/bend-allocations.json
python -m native.bend_engine.legal_probe.benchmark --baseline-chess /tmp/Chess.before.bend --report artifacts/bend-scalar-timing.json
```

## Readout

Hosted confirmation pending. Local full four-mode correctness and 55 cheap
parser/instrumentation tests passed. Self-review only; no independent reviewer,
GPU qualification, universal formal proof or production adoption claimed.
