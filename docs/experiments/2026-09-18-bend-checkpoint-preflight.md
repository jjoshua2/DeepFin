# Bend real-checkpoint readiness, artifact retention and reuse

## Decision and scope

Follow #781 (`d7517e5823d5b01239fae71472907e174ec591fb`) without another language,
search, perft or synthetic-model milestone. No connected CUDA host or trained
checkpoint is available in this environment. The next real experiment should not
export before discovering a broken toolchain, erase its evidence on failure, or
require recompiling a successful model package just to retry a later stage.

Inspection found all three in the initial runner: source verification occurred
inside Bend compilation after export; its unconditional TemporaryDirectory removed
exports on both success/failure; native-start failures closed the stderr temporary
file before the caller could read the underlying loader diagnostic.

## Change

Add an explicit no-forward preflight, optional NEW retained work directory and
exact-checkpoint package reuse. Preserve the default disposable behavior. Reuse
validates package hash/version, checkpoint identity/state selection, resolved model
config, encoding, bucket and target, then reruns all numerical/search checks.
Progress records distinguish preflight readiness from actual qualification. They
retain prior completed groups and bounded native stderr, record per-stage elapsed
time, and restore the calling process's compilation-cache environment. No model,
Bend, CBoard, math tolerance, test depth or trained/GPU qualification is changed.

Preflight strictly loads and checks weights on CPU, verifies the pinned compiler
and required tools, and checks the requested target. CUDA queries may initialize
a context. A free-memory snapshot is neither a reservation nor a prediction of
peak compilation/inference demand; the caller still needs a safe compute window.

## Confirmation plan (before hosted execution)

Keep the current saved-transformer fixture purely as regression evidence, not a
new model milestone. Export from one immutable fixture checkpoint in a retained
directory, run the existing control/batched/cancellation checks, then reuse the
same package from the same checkpoint in a NEW directory. Both must pass unchanged
CPU logit tolerances and complete search comparisons. Assert identical per-root
structural outcomes; native call counts may vary with batch arrival timing.
Verify the package hash is unchanged and the second report has no export stage.

Exercise the no-forward preflight, all old cheap contracts plus new identity,
workspace, path-alias, early-failure, atomic-report and stderr-cleanup cases. A
changed checkpoint/package/batch/state selection must fail before native execution.
Retain one deliberate post-export failure and verify package survival. Record
preflight as not_run, never as neural/CUDA evidence. Keep the original native
qualification, single-row/batching/session regressions when confirming the modified
shared startup wrapper. No added recurring exports/reuse runs: only cheap tests
join ordinary pytest; one bounded, temporary hosted confirmation handles the
new execution path. The permanent path-scoped workflow's depth/cost defaults stay.

Budget: one CPU hosted confirmation, 15-minute job cap, one compilation worker,
two inference threads. No GPU purchase/allocation, training, live process access,
checkpoint download, production merge or deployment. Recovery is discarding this
isolated branch, never resetting earlier work. Self-review only.

## Readout

Pending hosted confirmation. Local preliminary preflight passed with strict
weights and `qualification: not_run`; native execution and reuse are checked
separately. A preflight or retained .pt2 file cannot establish model correctness,
portability, trained provenance or end-to-end speed.
