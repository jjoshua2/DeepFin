# Native asynchronous request-lifecycle foundation

## Scope and implementation — September 22, 2026

Continuation of the neural-search scaling plan after #832. The reference head is
`3e308ec5c491cd9f427e923d63c2f80ac18598b7`. The initial delivery was a **PR4a patch candidate** because GitHub write actions
were unavailable. The continuation restores publication and adds a real CPU-model
qualification target. It is still not completed PR4 or asynchronous UCI. No merge,
deployment or live-training-hardware use is authorized by this record.

The implementation is a single bounded native worker with a Bend-owned interface.
It copies the input at admission, preserves immutable request identity, separates
logical cancellation from physical callback completion, consumes terminal replies
once, rejects token recycling and poisons failed slots. It has no tree or scheduler
logic. The callback contract requires synchronous physical completion; no GPU
asynchrony is inferred from returning a host function.

The implementation files are new; the experiment index gains a link. Normal UCI, search, compiler, CUDA backend,
model bridge, production settings and PR1 accounting remain unchanged. The test
callback is not LibTorch or a neural model. `DEEPFIN_BEND_ASYNC=1` is implemented
only in this isolated interface/probe; it does not enable an async engine.

## Local validation and limits

The qualification script and reports accompany the patch. Checks use the existing
verified compiler revision
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` (Bend 2.0.21 + U64), unchanged 84-file
source fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The local C/C++ toolchain is Clang 17.0.0, Bun 1.4.2 and Python 3.13.5. There is
no claim of the repository's locked whole-suite or hosted native qualification.

The bounded worker test covers 433 assertions and 128 repeated admissions in each
mode. It includes a held callback while polling/cancellation succeeds, all identity
components, eight competing submitter threads with one successful admission,
malformed buffer dimensions, return-code and exception failures, cancellation that
cannot hide failure, output tail preservation, stable owned storage, duplicate
completion rejection and exhausted-token admission. The generated Bend/C test
exercises submit/cancel/drain/reuse and caller-input mutation after admission.

Both the normal and instrumented local runs passed. Each run also passed the
generated Bend/C cancellation/reuse probe, five option controls and the missing-
native-link control. Reports `normal.json` and `sanitized.json` retain the outcomes.

Normal execution and the instrumented lane are reported separately. C++ worker
and adapter use ASan + UBSan; the generated Bend C uses UBSan only. A full generated
C ASan attempt failed in the Clang backend with an unsupported dynamic stack
realignment/musttail combination. This was a compiler failure, not a successful
sanitizer run. The compiler and generated source were not altered to bypass it.
ThreadSanitizer and independent review were not performed.

The script validates five disabled/malformed flag cases and a missing-native-link
case. Its exact stdout expectations retain the two distinct execution tokens and
an unknown result on duplicate polling. Scoped Ruff 0.16.8, Basedpyright 1.40.1 (zero errors/warnings), Vulture
2.16, Python byte compilation and shell syntax checks passed. Basedpyright used
its explicit bundled typeshed path after the direct Node entrypoint initially
failed to locate standard-library stubs. No source type checks were suppressed.
Source manifests and tool versions identify the delivered bytes. These observations are functional checks, not throughput measurements.

## Excluded integration and next acceptance criteria

A full standalone integration draft was attempted, but bounded local compiler
qualification did not complete in the constrained container. That unqualified
wiring is excluded. No `standalone/main.bend`, `NativeEvaluation.bend`,
`NeuralWork.bend` or `CMakeLists.txt` change is in this patch. The scope was reduced
rather than claiming UCI responsiveness or shipping unverified engine changes.

Next, integrate Bend-owned awaiting/draining states with exact ticket matching and
PR1 pending/accepted/cancelled accounting. Require stop/readiness/deadline tests with
a deterministic blocked evaluator, exactly one `bestmove`, rejection of stale
replies into new epochs, correct physical drain at shutdown, and unchanged real
CPU-model outputs. GPU execution, trained-model fidelity, CUDA memory/latency and
production Gumbel search remain separate acceptance gates.

This record documents completed local work and limitations; it is not a
retroactive preregistration or a claim of formal proof. Commands and detailed
ownership constraints are in `native/bend_engine/async_probe/README.md`.

## CPU-model continuation preregistration — September 22, 2026

Hypothesis: the same CPU LibTorch/AOTI model can execute through the native worker
without changing input snapshots or numerical results, publishing cancelled work,
or releasing callback storage early. This is separate from a Bend UCI/search
integration or any GPU claim. Reference remains #832 at `3e308ec5`.

Controls: same saved untrained 5,043,005-parameter, 175-plane CPU-F32 batch-one
checkpoint/package from PR1. Package SHA256
`9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
No new model export. Direct and worker execution run in distinct native processes,
each with trace off/on. All four execute six identical deterministic synthetic
tensors; the third worker request is cancelled, the fifth repeats the first.
Real traces must retain the cancelled physical forward while caller publication
is absent. Every other result is published once. Existing singleton bridge and
normal UCI are unchanged; the separate probe links the actual adapter and bridge.

Deciding gate: bit-identical direct/worker traces and quiet/traced publications,
exact independently generated input bytes, untouched output tail and cancelled
output, stable bridge pointers and one input-tensor allocation. All six outputs
must match independent eager singleton evaluation with predeclared inherited
CPU tolerances of 2e-6 absolute and 2e-5 relative. Invalid identities and duplicate
polls cannot consume a request. Pending-work shutdown checks require callback completion before shutdown returns,
for both cancelled and non-cancelled requests. Failure tests deliberately write partial private
output before returning/throwing; none may reach the caller. No timing/Elo gate.

Budget: one bounded CPU-only hosted job, one compiler job and two Torch threads;
one small Bend probe generation, normal and instrumented worker checks, one
C++ model-probe build and 24 native forwards total. No full-engine generation,
training, game arena or live GPU. The cheaper parser tests run in ordinary pytest;
model/native execution is opt-in only. Reports/identities are retained, not raw
neural traces or model weights. Publish a clean feature branch only after gates
pass. Keep temporary publishing workflows out of the feature diff.

Recovery: keep failed evidence; fix the failing layer without changing compiler,
model identity, inherited tolerances or original engine tests. No test success may
be presented as UCI responsiveness or trained/GPU qualification. Self-review only
unless another actual reviewer is separately obtained.

Continuation local checks: 38 new parser/trace/audit tests pass; normal and
instrumented lifecycle probes pass. An initial combined compile command exceeded
the tool's execution window; the remaining instrumented compile and tests were
completed separately. Local lint launcher cannot import its Node wrapper; locked
hosted type/lint checks are required rather than treating that failure as success.

Hosted model readout: pending this explicit qualification, not inferred from the
prior callback tests.
