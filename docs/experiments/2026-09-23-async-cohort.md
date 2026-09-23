# Asynchronous bounded cohorts and per-root cancellation

## Preregistration — September 23, 2026

PR5b, on #847 at dbe12bd862bbe04573f0e4d3aeb26c910406f74c. No merge,
deployment, live training or GPU use. CPU-F32 only, existing fixed-batch model.
Synchronous cohort execution remains the default and its output contract is preserved.

Hypothesis: a single batch execution worker lets the Bend cohort owner service
control input while a forward runs, without allowing a cancelled row into a tree
or cancelling unaffected roots that share the same forward. No speed/Elo claim.

Scope: opt-in async bounded cohorts; readiness, per-root cancellation, stop-all,
and graceful quit. This is not dynamic root admission/removal, persistent self-play,
a measured batch-selector, or GPU qualification. A worker callback must finish all
physical accesses before returning. The native worker knows batches, not roots;
only Bend associates tickets and rows with trees and cancels individual roots.

Controls: unchanged synchronous cohort runner, deterministic row-independent
callback, actual selected-leaf CPU model, explicit blocked callback before release.
Require same no-cancel trees/results and counts as sync; exact admission/physical/
accepted/cancelled reconciliation; no output on pending, duplicate or wrong-token
poll; no slot reuse before physical completion; shutdown join; input snapshots;
full, partial and repeated batches at supported sizes and both widths; malformed
commands leave state unchanged. Cancellation must not refund computation. Backend
failure/nonfinite rows still fail even when cancelled; no successful summary.

Budget: one bounded hosted CPU qualification, two Torch threads and one compiler
job; fresh cohort C generation, deterministic normal/UBSan checks, and the existing
untrained CPU fixture. Reuse singleton export; at most one batch-four export from
the identical weights. No training, games, live hardware or new model architecture.
No tolerance relaxation: inherit 2e-6 absolute/2e-5 relative logit and existing
policy/WDL tolerances. CPU time summaries are descriptive, not warmed benchmarks.

Recovery: preserve failed evidence, fix the affected layer, retain compiler and
oracle contracts. Publish only a checked source diff. Independent review is not
available; self-review must be labeled. Component checks are not substituted for
full Bend application/worker/model composition checks.

## Readout

Pending qualification.
