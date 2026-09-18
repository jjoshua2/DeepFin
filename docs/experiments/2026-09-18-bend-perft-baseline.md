# Bend perft: native baseline and conservative king-safety filter

## Scope and controls

This follows the legal/perft prototype in PR #774, stacked on #773. No
production move generation, inference, configuration or training is changed.
The U64 compiler remains pinned to `57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`.
Old core: `7e33ebe5c7aeada732fad63034181c0138ed4ca4` / `legal_probe/Chess.bend`.

An exploratory local run found the unoptimized Bend prototype much slower than
CBoard. The candidate was selected after those exploratory measurements; this
is not a preregistration of the discovery. It skips full make/check filtering
only for provably non-exposing ordinary moves outside check, using a conservative
first-blocker mask. King moves, EP and all check evasions keep the full filter.
This argument is not a machine-checked universal chess-correctness theorem.

## Confirmation plan (before the GitHub benchmark)

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
IO dispatch is included. Same available CPU affinity on Linux, rotating/reversed
execution order. The C baseline publishes its realized PEXT/magic backend.
The JSON records raw samples, source hashes, compiler flags, input and backend.

Recovery: this is an isolated branch, not a live deployment. Revert only the
candidate legal-filter change if its confirmation fails. Do not reset a live
checkout or alter the compiler pin to make a result pass.

## Validation at implementation

Local Clang 17: all four build modes (generic, forced-portable, native, UBSan)
passed 30 position fixtures, 33 perft/divide cases, 128 trace plies and 16 sampled
random legal-set/divide comparisons per mode. All seven invalid-input cases were
rejected. Six new pin/blocker fixtures exercise the optimization's boundaries.
The local environment lacks python-chess; the hosted gate runs that second oracle.
45 isolated parser tests pass, without importing the full training environment.

## Readout

Hosted confirmation and compact numerical evidence will be recorded after the
actual run. No result for that run is asserted by this initial record.

Limits: no universal proof, no GPU/multicore comparison, no chess-strength result.
A scalar U64 microbenchmark cannot substitute for this whole-traversal benchmark.
The prototype retains functional board/list allocation and is not a replacement
for optimized CBoard merely because it has faster bitboard primitives.
