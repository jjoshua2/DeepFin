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

Recorded before running the hosted confirmation: compile previous core and full
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

Local Clang 17 and hosted Clang 18 produced the same counts:

| Position | Runtime call | Previous | Candidate |
| --- | --- | ---: | ---: |
| Startpos d5 | heap_alloc | 92,997,716 | 25,304,456 |
| Startpos d5 | rfc_wrap | 23,423,583 | 0 |
| Startpos d5 | rfc_bump | 11,359 | 0 |
| Startpos d5 | term_drop | 33,887,668 | 5,062,890 |
| Kiwipete d4 | heap_alloc | 73,504,419 | 19,239,248 |
| Kiwipete d4 | rfc_wrap | 18,415,716 | 0 |
| Kiwipete d4 | rfc_bump | 42,403 | 0 |
| Kiwipete d4 | term_drop | 29,905,124 | 4,183,465 |

All four node totals matched. This supports the mechanism, not a complete time
attribution or proof that all remaining allocations are necessary.

## Reproduce

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
git show 6b4533e6736fb75ecb43a4cc2dd4ab883fc54ce2:native/bend_engine/legal_probe/Chess.bend > /tmp/Chess.before.bend
python -m native.bend_engine.legal_probe.profile_allocations --baseline-chess /tmp/Chess.before.bend --report artifacts/bend-allocations.json
python -m native.bend_engine.legal_probe.benchmark --baseline-chess /tmp/Chess.before.bend --report artifacts/bend-scalar-timing.json
```

## Hosted readout: retain the optimization

[Confirmation run 35395073985](https://github.com/jjoshua2/DeepFin/actions/runs/35395073985)
passed. Published executable sources: commit
`7601ac1eb18077185e1f7b5e2e137b8ec16ed491` (subsequent readout changes are docs only).
The source-hash check verified all six feature files against the local candidate.
An earlier attempt stopped before tests on one extra README blank line; the
whitespace was corrected without changing expected hashes or executable code.

Hardware: AMD EPYC 7763 64-Core Processor, pinned CPU 0, one thread.
Compiler: Ubuntu Clang 18.1.3, `-O3 -march=native`.
Same adapters/table setup and balanced rotating order for previous Bend,
candidate Bend and actual CBoard PEXT. Three measured traversals each, after
full warmup in each process. CLOCK_MONOTONIC boundaries exclude initialization,
printing and final table destruction. Every warmup and measured total matched.
This is the uninstrumented build, separate from the allocation counters.

| Position | Exact nodes | Previous Bend | Candidate Bend | CBoard PEXT | Old/new speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Startpos d5 | 4,865,609 | 1.030985214 s | 0.291212262 s | 0.080086298 s | 3.540x |
| Kiwipete d4 | 4,085,603 | 0.894671215 s | 0.230244919 s | 0.089373104 s | 3.886x |

The >=10% confirmation threshold passes on both fixtures; time falls by about
71.8% / 74.3%. Candidate throughput is 16.71 / 17.74 million leaf nodes/s.
CBoard remains 3.636x / 2.576x faster: no claim of beating C, generalizing to all
CPUs/positions, or improving playing strength. This is a small controlled
performance screen, not a broad statistical performance study.

### Raw measured nanoseconds (trial order within each engine)

| Position | Engine | Trial 0 | Trial 1 | Trial 2 |
| --- | --- | ---: | ---: | ---: |
| Startpos d5 | candidate Bend | 291212262 | 290801785 | 292654290 |
| Startpos d5 | previous Bend | 1061404857 | 1030985214 | 1024100617 |
| Startpos d5 | CBoard | 80086298 | 80492781 | 79934654 |
| Kiwipete d4 | candidate Bend | 231448746 | 226178108 | 230244919 |
| Kiwipete d4 | previous Bend | 894671215 | 878289131 | 897005610 |
| Kiwipete d4 | CBoard | 89230628 | 89455489 | 89373104 |

Within each position the execution orders were candidate/previous/CBoard,
previous/CBoard/candidate, then CBoard/candidate/previous.

### Validation and provenance

- Generic C, forced-portable helpers, native-target C and UBSan: PASS.
- Per mode: 30 positions with exact move sets/child states, 33 perft/divide
  cases, 128 checked random plies, 16 sampled random checks, 7 invalid inputs.
- Independent python-chess fixed-position checks: 30 PASS.
- Focused Ruff/Basedpyright: PASS. All 55 cheap parser/instrumentation tests: PASS.
- No default perft-depth, production-path, compiler-fork or recurring benchmark
  change. The temporary confirmation workflow is absent from the feature diff.
- Self-review only; no independent reviewer, GPU qualification, universal formal
  proof or production adoption claimed. Full repository CI is reported separately.

Artifact `bend-scalar-confirmation`, ID `10567572354`, on the confirmation run,
contains correctness, allocation, timing and published-commit reports. ZIP SHA256:
`eb9f7c15db17327e139bedc407734315b04308da20af9431a17b052e54ff8cef`.
The artifact is retained for 30 days; raw timing/counter evidence is banked above.

Report SHA256:
- timing: `8170f8fefdf90ce7bd3c6a459afd8ba5c6f340eb619f26a78ff391e995a41e2b`
- allocations: `d7bdecaafc60c8f35f73c5bf9af10abe5893af31e3cdeecf7fc25260273b4747`
- correctness: `22ee09515b686403491628a8db8c93b4c5067100aa0e0e076310f5a3ee2022b7`

Core SHA256:
- previous: `55e4cb0e466deb9cbda820ae0ce8c368ecee9b03b007cb7a0cec6891bc9f516a`
- candidate: `3c60db6c73a215e09043e63dff41d93643c77edb7f66237e47fa851362b181f9`

Full source identities and compiler commands are included in the timing report.
