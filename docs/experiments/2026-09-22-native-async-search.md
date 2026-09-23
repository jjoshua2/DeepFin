# Native asynchronous selected-leaf search

## Preregistration — September 22, 2026

PR4b, based on #836 at a63409180bc337591a97d8877c5dbc86483381c6.
No merge, deployment, live hardware use or private checkpoint upload. Preserve
synchronous CPU F32 as the default; async singleton mode requires an explicit
DEEPFIN_BEND_ASYNC=1. No CUDA or production Gumbel qualification is implied.

Hypothesis: Bend can own search selection, encoding, rule handling, request
identity, stop/deadline decisions and physical retirement while a single native
worker executes only copied CPU input tensors. A slow model callback must not
block readiness/stop, publish cancelled output or corrupt a replacement search.
The meaningful comparison is behavior and identity, not throughput or Elo.

Controls: unchanged Search core and existing material/perft, accounting and
selected-leaf neural oracles; same saved untrained 5,043,005-parameter CPU-F32
batch-one package used by PR1. Compare synchronous and asynchronous selected-leaf
outputs using the existing numerical tolerances. No new model export.

Acceptance: generated real application builds on the unchanged compiler; original
material gates pass; an external test-only blocked evaluator demonstrates stop,
readiness, command-flood deadlines, shutdown drain and replacement-search safety.
Each go has one process-monotonic epoch and at most one bestmove. Cancellation is
charged at admission and reported separately from confirmed physical completion.
Decision reports can have unconfirmed work; later neural_retired records cannot
be confused with a new search's neural_work report. A replacement go can be queued
behind retirement, but its own wall clock includes that wait. No timer or output
backpressure hard-real-time guarantee is made.

Budget: bounded hosted CPU validation, one full Bend generation (reuse generated C
for subsequent unchanged-source tests), one compiler job and two Torch threads,
no training, CUDA execution, model export or game arena. Preserve failures and
source/build identities. Do not suppress original checks, change compiler pins or
relax numeric tolerances. Self-review is not independent review.

## Readout

Pending hosted full-application qualification. The first local full type-check
exceeded the 4 GiB authoring container limit; small runtime checks passed, which
is not evidence that the full application compiles or runs correctly.
