# Bend perft: native baseline and conservative king-safety filter

## Scope and controls

This follows the legal/perft prototype in PR #774, stacked on #773. No
production move generation, inference, configuration or training is changed.
The U64 compiler remains pinned to `57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`.
Old core: `7e33ebe5c7aeada732fad63034181c0138ed4ca4` / `legal_probe/Chess.bend`.

An exploratory local run found the unoptimized Bend prototype much slower than
CBoard. The candidate was selected after those exploratory measurements; this
is not a preregistration of the discovery. It skips full make/check filtering
only for non-exposing ordinary moves outside check, using a conservative
first-blocker mask. King moves, EP and all check evasions keep the full filter.
The safety argument is not a machine-checked universal chess-correctness theorem.

## Confirmation plan (recorded before the GitHub benchmark)

Question: does this limited filter optimization improve the prototype while
preserving exact chess behavior, and how far is it from the native C baseline?
Controls: compile old Bend core, new Bend core and actual CBoard traversal with
the same Clang, flags, CPU, input and last-ply bulk-counting convention. The old
and new Bend variants share the compiler, input loader and timer adapter. Only
the core source differs. No cross-runner NPS comparison is used as the verdict.

Primary measurements: exact startpos d5 = 4,865,609 and Kiwipete d4 = 4,085,603;
three raw timing samples per variant, median and spread, full warmup each process.
Keep the candidate only if all correctness gates pass, median traversal improves
by at least 10% on both positions, and no other existing validation regresses.
Otherwise retain the benchmark and revise/revert the optimization. Small samples
are a development screen, not a statistical or cross-hardware guarantee.

Budget: one single-core local screen and one hosted Actions confirmation, each
bounded by the runner/command timeouts. No GPU, training or expensive permanent
CI expansion. Existing native correctness depths and all original perft tests
are unchanged. Benchmarks are explicit/manual; ordinary pytest gets parser tests.

Timer: CLOCK_MONOTONIC after input/table construction, full warmup and validation;
stop before printing or releasing the final table owner. Bend's timer-boundary
IO dispatch is included. Same available CPU affinity on Linux, with each engine
occupying each execution-order slot once per complete rotation cycle. The C
baseline publishes its realized PEXT/magic backend. The JSON records raw samples,
source hashes, compiler flags, input and backend.

Recovery: this is an isolated branch, not a live deployment. Revert only the
candidate legal-filter change if its confirmation fails. Do not reset a live
checkout or alter the compiler pin to make a result pass.

## Validation

Local Clang 17 and hosted Clang 18.1.3: all four build modes (generic,
forced-portable, native, UBSan) passed 30 position fixtures, 33 perft/divide cases,
128 trace plies and 16 sampled random legal-set/divide comparisons per mode.
All seven invalid-input cases were rejected. Six new pin/blocker fixtures
exercise the optimization's boundaries. Hosted python-chess independently
matched all 30 fixed-position legal sets and child states.

45 isolated parser tests passed; focused ruff and basedpyright passed. These
checks do not import the full training environment or execute benchmarks in
ordinary pytest.

The first hosted attempt passed correctness but stopped at a timing-adapter
assertion before producing a benchmark report: the four-argument foreign effect
also stores an IO continuation, so its constructor arity is five, not four.
The guard was corrected after inspecting the emitted constructor; it was not
removed. The final run additionally balanced execution order by complete cycles.
The chess optimization did not change between those attempts.

## Hosted readout

Final confirmation: [Actions run 35393024617](https://github.com/jjoshua2/DeepFin/actions/runs/35393024617),
artifact `bend-perft-confirmation` (ID 10567320086).
Tested source commit: `92313002f3ab1c6fc8ebe0d5206bda4cb35c9f9a`.
The final feature commit retains identical executable sources, with only this
readout and temporary-workflow cleanup differing from the tested tree.
Full benchmark JSON SHA-256:
`bea42e7cbb324ad177c3b383a249675b89802b6d7ebf4aa9f9bbb7b22a4ed767`.

AMD EPYC 9V74 hosted runner, CPU affinity 0, one thread, Clang 18.1.3,
`-std=c11 -D_POSIX_C_SOURCE=200809L -O3 -march=native` for all measured hot loops.
CBoard reported PEXT. This is the actual CBoard legal-generator/copy/push path,
not Python recursion; it is not a measurement of the deployed GCC/LTO build.
Both traversals bulk-count legal moves at depth one. Counts matched on every
warmup and measured run.

| Position | Nodes | Old Bend median | New Bend median | CBoard median | Old/new speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Startpos d5 | 4,865,609 | 1.872319 s | 1.071798 s | 0.078277 s | 1.747x |
| Kiwipete d4 | 4,085,603 | 1.545862 s | 0.926017 s | 0.088158 s | 1.669x |

Raw nanosecond samples, in each engine's trial order, retained here beyond the
artifact retention window:

| Position / engine | Trial 0 | Trial 1 | Trial 2 |
| --- | ---: | ---: | ---: |
| Start / old Bend | 1879534404 | 1872318999 | 1857014192 |
| Start / new Bend | 1071798159 | 1071797447 | 1117646819 |
| Start / CBoard | 78377175 | 78277344 | 78012664 |
| Kiwipete / old Bend | 1545861652 | 1712880598 | 1536895344 |
| Kiwipete / new Bend | 929718493 | 922990330 | 926017245 |
| Kiwipete / CBoard | 88258331 | 88157920 | 88050633 |

Decision: retain this limited optimization. Both positions exceed the 10%
confirmation threshold, with exact counts and all targeted correctness checks
passing. New Bend remains approximately 13.69x and 10.50x slower than CBoard.
The useful result is a reproducible baseline and a measured prototype improvement,
not evidence to replace production C or a language-wide speed conclusion.

## Remaining limitations

Self-review only, not an independent review. No universal proof, GPU/multicore
comparison or chess-strength result. The prototype retains functional board/list
allocation; profiling generated code and data representation is the next question,
not an established bottleneck diagnosis.

The inherited PR #774 run `35386735510` already failed its ordinary CPU suite in
`test_the_lease_watchdog_never_touches_a_foreign_marker` (missing `lease.log`),
before this change. Its lint and PEXT jobs passed. The older moving-release Bend
installer also has its separately documented metadata failure. These unrelated
paths are unchanged; the new feature is not claimed repository-wide green solely
because the targeted confirmation passed.
