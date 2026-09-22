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

## Initial patch local validation and limits

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

## Hosted continuation readout — September 22, 2026

**[Run 35798303801](https://github.com/jjoshua2/DeepFin/actions/runs/35798303801)
passed every qualification stage before publishing the clean feature branch.**
Job `106982621420`. Executable-source feature commit:
`b7c2216828df78fa56e0ec83a9df96fe5f3e63b2`. A following documentation-only
commit records this readout; it does not change any qualified executable source.

The exact base is #832 at `3e308ec5c491cd9f427e923d63c2f80ac18598b7`.
The 17-path source manifest in the downloaded artifact matches the local authored
files. There are sixteen new implementation/test/documentation files plus one
experiment-index entry. No existing engine, search, accounting, exporter, model
bridge, CUDA, compiler or production configuration is changed. A separate
`async_probe/CMakeLists.txt` builds only the opt-in CPU-model verification target.

### Actual results

| Gate | Observed result |
| --- | --- |
| Whole-repository Ruff, Basedpyright, Vulture plus explicit verifier lint | Pass; zero type errors or warnings |
| New parser tests plus existing accounting/benchmark/broker tests | 127 passed; zero failures, errors or skips; includes 38 new cases |
| Existing Bun singleton-binding/compiler contracts | 33 passed |
| Native worker, each normal/instrumented mode | 440 assertions; 128 reuse round trips; all pass |
| Generated Bend/C foreign interface, each mode | Cancellation/drain/reuse/input-snapshot and duplicate-poll checks pass |
| Disabled/malformed-option and missing-native controls | Five option rejections and missing-link rejection pass per mode |
| Actual CPU-model adapter | Four native processes, six forwards each; 24 forwards total |
| Worker publication in each of two worker processes | Five completed results, one cancelled result withheld |
| CPU eager singleton comparison | Maximum logit error 5.960464477539062e-7; inherited 2e-6 absolute / 2e-5 relative tolerances retained |

The worker suite's increase from the initial patch's 433 to 440 assertions includes
shutdown with pending cancelled and non-cancelled callbacks. Both must finish before
shutdown returns. Backend failures deliberately write private partial output first;
no bytes may escape to the caller, and a cancelled failure still poisons the slot.

Actual model execution used the same **untrained 5,043,005-parameter 175-plane
CPU-F32 batch-one checkpoint/package**, without export. The direct and worker
processes execute the actual existing LibTorch/AOTI bridge, not the deterministic
test callback. All six physical forwards are retained in each trace, including the
worker's cancelled request. Published worker outputs omit only that request.
Direct/worker trace bytes are identical; quiet/traced output bytes are identical;
input-snapshot and repeated-input checks pass. Each process has stable bridge
buffer addresses and exactly one bridge input-tensor allocation. These allocation
observations do not cover LibTorch/AOTI internal allocations.

The real-model driver is a C++ qualification executable using the actual worker
adapter. The generated Bend foreign interface is qualified separately with test
callbacks. Their composition inside the full Bend UCI/search application is **not**
qualified by these two component results. Inputs here are synthetic tensors, not
Bend-encoded search leaves; published results are not accepted search evaluations.

Environment: locked Python 3.13.15, Torch 2.14.0+cpu, uv 0.12.10, Bun 1.4.2,
Clang/Clang++ 18.1.3 on hosted Linux x86-64, two Torch threads and one compiler
job. The existing compiler revision and 84-file fingerprint above are unchanged.
Normal and instrumented modes are both executed. The instrumented C++ worker and
adapter use ASan+UBSan, while generated Bend C uses UBSan only; neither LibTorch
nor compiled model internals are sanitizer-qualified. ThreadSanitizer, full-engine
rebuild, full ordinary pytest suite, perft and chroot isolation were not run here.
The ordinary PR checks run separately after publication.

### Integrity, transport and review

The initial payload transfer had a base64 mismatch. A transfer-only preflight
(run 35798069945) preserved and downloaded those bytes. The correction was checked
against the locally authored compressed and uncompressed hashes **before** source
application or execution. Qualification then applied the exact authored patch; no
source, compiler, numerical threshold or oracle was changed to bypass a failure.
The preflight is transport evidence, not an additional successful test run.

Self-review, not independent review or formal proof. The prior local type-checker
launcher failure remains a local tooling limitation; the locked hosted
whole-repository and explicit verifier checks passed without suppressions.

### Retained evidence and remaining integration

Artifact **10725151032**, `deepfin-pr4a-async-qualification`, retains the exact
source patch/manifest, JUnit and lint output, compiler/build identities, lifecycle
and real-model reports, dependency list and executable digest for 30 days. It
contains no model package, binary or raw neural trace.

- Artifact ZIP SHA256: `001967f7f7a162271cd32a65dd930579980a0d81cfc4daa514c545f687628b6a`.
- Qualified source patch SHA256: `86fe915786316a809ecd097de106d91a72b1698982d9b4c329dacafc6df92497`.
- Model-probe executable SHA256: `b57d25c7179e97557d0ae3b238bf0cfb17cd54c001299c3d76f7ce7634985535`.
- Checkpoint SHA256: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Model-package SHA256: `9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.

**No asynchronous UCI stop/readiness, search-tree resumption or PR1 pending-work
integration is delivered here.** The excluded integration above remains the next
stage: Bend-owned awaiting/draining states, exactly one bestmove, monotonic search
epochs, physical retirement of cancelled work and real selected-leaf model parity.
The existing CUDA implementation is also still awaiting real GPU qualification.
No trained model, live GPU, speed or Elo result is claimed. Nothing was merged,
deployed or changed in live training.
