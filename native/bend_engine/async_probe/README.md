# Native asynchronous request-lifecycle foundation (PR4a)

This is an **isolated CPU-singleton execution boundary**, not an asynchronous
chess engine. It prepares the request/buffer-lifetime part of the neural-search
scaling plan's PR4. It is not wired into `standalone/main.bend`, the engine CMake target, UCI `stop`,
`movetime`, `Search.resume`, or PR1 accounting. Existing engine behavior is unchanged.

The patch is based on PR #832 at
`3e308ec5c491cd9f427e923d63c2f80ac18598b7`. The lifecycle tests need no checkpoint, LibTorch, CUDA device or network access.
A separate opt-in target below executes an actual CPU model through the same
worker adapter. Neither target changes the engine or supplies a fallback evaluator.

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
its committed work must remain observable when accounting is integrated. Drain
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
second admission as a fatal owner-contract violation: its future state machine
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

`DEEPFIN_BEND_ASYNC=1` enables **this probe**, not the standalone engine. Do not use
it as a production-adoption switch. The verifier also tests disabled/malformed
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

## Remaining PR4 integration

Keep this foundation separate from the engine switch until all of the following
are implemented and qualified together:

1. Bend-owned awaiting/draining states carrying both the original leaf ticket and
   the execution token. Only matching, non-cancelled, numerically valid results may
   enter `Search.resume`; backup and reservation release must occur exactly once.
2. UCI `stop`, deadlines, position replacement, readiness and quit handling while
   inference is pending. A stopped search must emit one result promptly, retain
   physical work until drain, and prevent late completion from updating a new tree.
3. PR1 accounting for submitted, pending, completed, accepted, cancelled and failed
   work. A response at stop may have unfinished physical work; do not report it as
   a fully reconciled equal-budget comparison. Retain separate drain reconciliation.
4. Real CPU checkpoint parity and deterministic slow-inference UCI regressions,
   then separately qualified GPU/model execution. Existing arena limits, CUDA
   qualification and production Gumbel parity are not changed by this patch.

An initial full-engine wiring draft did not complete bounded local compiler
qualification and is **excluded** from this deliverable. The included files are
self-reviewed only, not independently reviewed or formally proved.
