# Bounded Bend multi-root batching (PR5a)

## Preregistration — September 22, 2026 (America/New_York)

Scope: the first PR5 slice, based on #840 at
`31f376752d0baea75a6fbd65bf59f8bd1d92202c`. A separate, opt-in headless runner
owns one to sixteen independent trees and one fixed CPU-F32 model package.
It gathers at most one selected leaf from each root per sweep, rotates visited
roots behind unvisited roots, and resumes each tree only with its own validated
row. Normal UCI, live jobs, search selection, encoders and the native backend are
unchanged. No merge or deployment. This is not the complete adaptive scheduler.

Hypothesis: cross-root batching can preserve serial per-tree search semantics,
full history inputs and exact compute accounting without creating same-tree
concurrency or introducing a host chess coordinator. This slice has no throughput
or playing-strength hypothesis. Single-cohort synchronous calls cannot establish
responsiveness or cross-game steady-state speed.

Controls: a row-independent deterministic test callback at fixed batches 1, 2, 4,
8 and 16, with 146 and 175 input planes; actual saved untrained CPU model at batches
one and four. Compare all final tree fields and root results under deterministic
batch/serial execution. Verify actual leaf paths and ticket identities, complete
CBoard inputs, real/padded rows, legal probabilities/WDL and final trees against
the existing independent search reference. Real model outputs must match eager
singleton inference with inherited 2e-6 absolute / 2e-5 relative tolerance; legal
probabilities use 2e-7 / 3e-6. Do not relax gates after observing failures.

Test terminal and automatic draws, promotions and en passant, same-board/different
histories, partial batches, repeated roots, per-root neural limits, invalid inputs,
backend failure, and a nonfinite final row. No successful summary or neural backup
may follow a first-batch failure/nonfinite row. A deliberate swapped-row backend
is a verifier negative control, not something arbitrary logits can self-identify.

Budget: one fresh Bend runner C generation on a hosted CPU runner, bounded test
builds, two Torch threads, one compiler job, one Bend runtime thread. Reuse the
saved untrained 5,043,005-parameter checkpoint and singleton package; export only
one CPU batch-four package if the previous transient package is unavailable.
No training, live GPU, matches, repeated full UCI builds or production checkpoints.
Normal and UBSan deterministic controls are separate from LibTorch qualification.

Record source/compiler/model identities and compact reports. Raw traces/model
packages are not committed or uploaded in new artifacts. Failure prevents clean
feature publication; preserve diagnostics and fix the failing layer without
changing compiler, numerical thresholds or unrelated engine code. Self-review is
not independent review or formal proof.

## Readout

Pending hosted qualification. Local type checking is not native execution evidence.
