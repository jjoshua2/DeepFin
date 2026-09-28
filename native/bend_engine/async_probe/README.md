# Native asynchronous lifecycle and search integration

PR4a provides the isolated CPU-singleton execution boundary and component tests.
PR4b composes it with the actual Bend standalone engine, including UCI stop,
movetime, Search.resume and PR1 accounting. **Synchronous execution remains the
default.** See [full engine integration](#full-engine-integration-pr4b) below.
The component probes remain separate qualification scopes, not fallback engines.

The lifecycle foundation was based on #832; the composed application is based on
#836 at `a63409180bc337591a97d8877c5dbc86483381c6`. Lifecycle tests need no model.
The separate model and application checks use the exact trusted CPU checkpoint.

## Ownership and completion contract

`standalone/Async.bend` exposes submit, poll, cancel, drain, and shutdown effects.
`async_call.c` checks the pinned compiler's packed-array classes and marshals the
request identity. `async_model.cpp` adapts the interface to the existing native
`deepfin_model_run` signature. The small Bend probe links a deterministic test implementation of that signature.
The separate model probe links the actual native LibTorch bridge; the two are
explicitly different qualification scopes.

`async_slot.h` provides one worker thread and one bounded request slot. It is an
execution mechanism, not another search scheduler: the worker knows no chess,
selection policy, tree, or legal-move mapping. Admission synchronously copies the
real input into private storage. The worker never retains a pointer into Bend's
heap. There is an intentional additional input copy; this is not zero-copy.

Identity is `(token, epoch, request, node)`. Tokens increase across admissions and
are not recycled within a slot instance. Token exhaustion fails before admission.
Polling with any mismatched identity returns `unknown` and cannot consume a live
result. A matching terminal result is consumed at most once.

| Status | Value | Meaning |
| --- | ---: | --- |
| unknown | 0 | No matching outstanding request; includes duplicate delivery |
| pending | 1 | Matching request still executing or queued |
| complete | 2 | Physical callback finished; real output copied once |
| cancelled | 3 | Physical callback finished; output discarded |
| failed | 4 | Callback failed; no output published and slot permanently poisoned |

Cancellation is logical. It does **not** preempt a callback, refund a reservation,
free a slot, or publish its output. Even a cancelled queued admission executes;
its committed work remains observable in the application accounting. Drain
can release a cancelled slot only after the callback has returned. Failure takes
precedence over cancellation and cannot be hidden by it.

The callback must complete **all** physical accesses before it returns or throws.
The current adapter is CPU-singleton only. A future GPU adapter must synchronize
its own physical completion; substituting a launch-only callback is invalid.
Successful raw transport is not finite-logit validation or search acceptance.

The owner must start the model before the slot is first used, serialize this
adapter against all other calls to that same model bridge, drain outstanding work,
and join the slot before unloading the model. `shutdown()` is a blocking join,
called by one owner, never from the execution callback. Concurrent shutdown calls
are unsupported. Sequential repeated shutdown is harmless; future submissions
fail. This patch makes no shutdown-latency or crash-recovery guarantee.

The C++ admission API returns zero when occupied. The Bend glue treats an attempted
second admission as a fatal owner-contract violation: the owning state machine
must poll/drain the current identity before submitting another request. Pending
polls do not wait for the model, but short mutex contention is possible; this is
not a hard-real-time or wait-free interface.

## Reproduce the checks

Use the repository's existing pinned compiler checkout, Bun 1.4.2, Clang/Clang++,
a C++20 standard library, and Python 3.10 or later. No package installation is needed.
The script verifies the existing 84-file compiler fingerprint and requires a new
output directory. It does not download a compiler or change source files.

```sh
BUN=bun CC=clang CXX=clang++ PYTHON=python \
  bash native/bend_engine/async_probe/qualify.sh \
  /path/to/checked-bend-source /tmp/deepfin-async-check
```

`DEEPFIN_BEND_ASYNC=1` enables this probe and, with PR4b, the CPU-singleton
standalone engine. It is an explicit experimental option, not production adoption. The verifier also tests disabled/malformed
values and a binary without the native implementation. No real model is loaded.

The script runs normal and instrumented checks. C++ worker/adapter code receives
AddressSanitizer plus UndefinedBehaviorSanitizer in the instrumented lane.
Generated Bend C receives UBSan only: local Clang 17 rejects ASan instrumentation
of the pinned musttail code with a backend stack-realignment error. The compiler
and generated source are not patched to suppress that error. This is **not** a
claim of full-stack ASan coverage or a data-race proof; ThreadSanitizer was not run.

The C++ tests use actual threads and deterministic callback barriers, including a
callback held unfinished while the owner polls and cancels. They cover all four
identity fields, competing admission, failure/exception propagation, unchanged
caller outputs on rejection, cancellation without early slot reuse, token wrap,
128 repeated admissions across both input widths, and stable private buffers.
The generated Bend probe cancels and drains its first request, submits a second,
mutates its own input after submission, verifies the original snapshot was used,
and rejects duplicate delivery. This checks the actual generated foreign effect,
not only a separate reimplementation of its behavior.

## Actual CPU-model qualification

The optional `native-async-model-probe` target compares the real native CPU model
executed directly with the same model executed through `async_model.cpp`. It does
not execute the Bend search application. The small Bend probe above separately
checks the actual generated foreign effect and ownership protocol.

```sh
prefix="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
bash native/bend_engine/async_probe/build_model_probe.sh \
  /tmp/deepfin-async-model /path/to/exact-model.pt2 "$prefix"
python -m native.bend_engine.async_probe.verify_model \
  --binary /tmp/deepfin-async-model/build/native-async-model-probe \
  --package /path/to/exact-model.pt2 --checkpoint /path/to/exact-checkpoint.pt \
  --report /tmp/deepfin-async-model-report.json
```

Use matching CPU LibTorch and a trusted CPU-F32 batch-one package/sidecar. The
existing binder enforces shape, encoding and package identity. These commands
refuse to overwrite build/report paths. They do not export a model or modify it.
Only the external verifier uses Python; the probe and worker contain no Python
controller and the build rejects a libpython dependency.

The verifier runs four native processes: direct versus worker, each with raw
tracing off and on. Every process executes six deterministic synthetic inputs,
with input one repeated after intervening work. Worker request three is cancelled
and drained; its physical forward must remain in the trace but its caller output
must remain untouched. Other requests publish once. Host input is overwritten
immediately after submit to verify admission-copy ownership. Wrong identities and
duplicate polls may not change output or consume work. Audits require stable
bridge buffer addresses and one input-tensor allocation in every process.

Quiet/traced publications and direct/worker raw traces must agree bit-for-bit.
Every traced tensor is compared to independently generated input bytes, and all
six model outputs are compared with independent eager singleton evaluations of
the exact checkpoint at the inherited CPU tolerances (2e-6 absolute, 2e-5 relative).
No tolerance is selected from the observed result. Test failures are errors, not
skipped qualification. Raw traces stay in a temporary directory and are not part
of published evidence. This is not a speed or actual-chess-input test.

## Full engine integration (PR4b)

Build the normal bound CPU-F32 batch-one neural product, then opt in at launch:

```sh
bash native/bend_engine/standalone/build_neural.sh /tmp/new-engine \
  /path/to/checkpoint.pt2 /path/to/libtorch/share/cmake /path/to/verified/bend
DEEPFIN_BEND_MODEL_PACKAGE=/path/to/checkpoint.pt2 DEEPFIN_BEND_ASYNC=1 \
  /tmp/new-engine/neural/deepfin-bend-neural --threads 1
```

Without the flag, the synchronous reference path remains active. There is no CUDA
UCI support or multi-root scheduler in this integration. Bend owns encoding,
legal-policy normalization, rules, selection, identities and every Search.resume.
The worker receives a private tensor snapshot, never the tree or played history.

Each valid go receives a process-monotonic search_epoch, with no reset at position
or ucinewgame and no wraparound. While awaiting inference, readiness, stop and
movetime remain active. Stop publishes one decision/bestmove using completed work
or an explicitly marked legal fallback, and logically cancels the pending request.
It retains only identity/accounting metadata for physical retirement, not the old
tree. Replacement roots/searches are allowed after stop, but dispatch waits for
old physical completion. A replacement's own wall clock includes that wait.
Repeated stop cannot duplicate the result. Quit joins the pending callback before
returning; it does not promise to kill a wedged model or emit a bestmove on quit.

`neural_work` is the decision-time snapshot. Dispatched calls are irrevocable
worker callback reservations, not confirmed execution. Neural budgets are charged
at admission and never refunded on cancellation. A cancelled pending row has
cancelled=1 and unconfirmed=1 at decision, not executed=1 or accepted=1.
`neural_retired` is a distinct physical-cleanup record tagged with the OLD epoch.
Its wall time includes drain; do not treat it as time-to-bestmove, sum it with the
snapshot, or attribute it to the next search. Successful cleanup increments
executed/wasted only; failed cleanup records failed-forward work and exits.
The benchmark rejects cancelled/unconfirmed comparisons and ignores the distinct
retirement prefix as a primary decision. Async backend time remains null: polling
elapsed time is not a measurement of GPU or inference service time.

Opt-in whole-application checks link the same generated engine C and actual worker
to `search_gate.cpp` instead of LibTorch. Dedicated pipes block/release physical
completion; neither pipes nor test callback are in the shipped product. Tests
require readiness/stop before release, safe queued replacement, timer checks across
buffered commands, shutdown drain, bad-output failures and a synchronous negative
control. Work/epoch probes verify reconciliation and exhaustion separately.

```sh
bash native/bend_engine/async_probe/qualify_search.sh \
  /path/to/verified/bend /path/to/build/engine.c /tmp/new-async-search-check
```

Real selected-leaf neural parity is checked with the existing standalone neural
verifier in both DEEPFIN_BEND_ASYNC=0 and 1 modes, using the same package/oracles.
[The experiment record](../../../docs/experiments/2026-09-22-native-async-search.md)
contains the completed hosted qualification and limitations. These are CPU fixture
functional checks, not throughput or playing-strength evidence. Encoding, selection,
diagnostics, stdout backpressure and shutdown joins remain synchronous. The
4,096-node arena and production Gumbel gap are unchanged; trained-model and CUDA
qualification remain separate. Self-reviewed, not independently reviewed/proven.
