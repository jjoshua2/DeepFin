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

## Completed readout — September 22, 2026 (America/New_York)

[Qualification run 35801782528](https://github.com/jjoshua2/DeepFin/actions/runs/35801782528)
passed the complete source/static/build/material/neural/blocked-evaluator gates
before publication. Job 106993537930. Executable-source commit
`552ab2646a0772e33b5690b9d79494e6ed5e48d2` is based directly on #836 at
`a63409180bc337591a97d8877c5dbc86483381c6`. The following documentation commit
changes only this record, the experiment index and the two user-facing READMEs.
It does not recompile or change qualified executable source.

The engine now composes the PR4a worker with the actual Bend search loop.
`EngineRuntime.bend` owns pending identities, original tickets/legal entries,
cache storage and small retirement metadata. The native worker receives only
copied input tensors; it never receives a tree or decides a move. Existing
`Search.resume`, rules, encoders, model bridge, compiler and numerical tolerances
are unchanged. The selected-leaf verifier changes only its expected monotonic
epoch identifiers and the corresponding work-report assertion.

### What actually passed

| Gate | Observation |
| --- | --- |
| Whole-repository Ruff/Basedpyright/Vulture and explicit native verifier lint | Pass, zero type errors/warnings |
| Cheap parser/accounting/benchmark/broker tests | 170 pass, zero skips/failures; 43 new search-report cases |
| Full Bend type check, C generation and native material/neural builds | Pass with unchanged compiler |
| Original material UCI/perft | 10 positions, 137 exact children, 51 searched roots, 23 rejected transactions |
| Perft at existing depths 3/3/4 | 8,902 / 97,862 / 43,238, unchanged |
| Real selected-leaf model, synchronous mode | 18 searches, 50 traced forwards/replies, 1,141 legal priors |
| Real selected-leaf model, asynchronous mode | Same counts; complete report identical to synchronous control |
| Blocked callback with actual generated engine and worker | Normal and UBSan modes pass all lifecycle/deadline/failure controls |
| Concrete Bend accounting and epoch-exhaustion probes | Normal and UBSan modes pass |

The model is the saved **untrained 5,043,005-parameter, 175-plane CPU-F32 batch-one
fixture**. No export or trained model was needed. In each real-model mode, all
inputs and selected paths were checked by the existing independent reference;
maximum raw-logit error is `5.364418029785156e-7` and maximum probability error is
`2.9802322387695312e-8`, within the inherited tolerances. Each mode also retains
six automatic-draw replies, four zero-forward terminal searches, same-board/
different-history input distinctions and three rejected startup-failure cases.
Both JSON model reports have SHA256
`dee5702c6cf32d6f8c3c65b0349181037d6b9065905442c2c06dcaa03128430e`.
This is composition evidence using actual selected leaves, unlike PR4a's separate
worker-model and Bend-interface component tests. It does not compare every tree
field or prove arbitrary positions, and is not a trained-network/GPU test.

### Responsiveness and physical-lifetime evidence

The test-only `search_gate.cpp` replaces only the model callback. The same generated
Bend application and actual worker adapter are linked; test pipes are never linked
into the product. The callback emits a start signal and remains blocked until the
verifier explicitly releases it. In both normal and UBSan modes:

- Partial-line and ordinary readiness return before release. Stop returns one
  legal decision/bestmove before release, and repeated stop returns no duplicate.
- The stop snapshot correctly has dispatched=1, executed=0, accepted=0,
  cancelled=1 and unconfirmed=1. The old request is not refunded or counted useful.
- A replacement black-to-move root waits behind old physical work. Only after
  release can it dispatch. The old epoch's retirement has executed=1/wasted=1/
  accepted=0; the new epoch has its own one-row accepted evaluation and legal move.
- Invalid position rollback and ucinewgame preserve the intended state/storage;
  subsequent searches receive fresh epochs. A terminal root uses zero forwards.
- A pending 200 ms search still expires across a backlog of 400 ready commands.
  The test deliberately suspends the process beyond its deadline before queuing
  them; the decision must precede the end of the backlog. This proves the timer
  is checked between commands, not a real-world latency distribution.
- A queued replacement's 30 ms budget expires with zero new dispatch while the
  old callback is still blocked. Its own clock includes the queue wait.
- A completed infinite search holds its bestmove until stop. Quit waits for
  physical callback completion, emits retirement and no unrequested bestmove.
- Failed callbacks, cancelled failed callbacks and nonfinite live outputs exit 2
  without applying bad output. Cancelled failure has explicit failed-forward
  retirement rather than a fabricated successful result.
- The synchronous negative control cannot return readiness while its callback
  remains blocked. Epoch 4,294,967,295 is issued once, then exhaustion fails
  instead of recycling IDs. Concrete accounting reconciles both successful and
  failed retirement without changing the earlier accepted count.

Readiness/stop tests use a generous 0.75-second functional bound while callback
release is withheld. No fixed-wall strength or throughput comparison is claimed.
The seven primary decision snapshots per mode are retained in the reports;
retirement, shutdown and failure assertions are additional checks in the verifier.
The normal and UBSan modes use identical source and test contracts.

### Environment, failures and limits

Locked Python 3.13 / Torch 2.14.0+cpu, uv 0.12.10, Bun 1.4.2 and Clang/Clang++
18.1.3 on hosted Linux x86-64. Two Torch threads, one compiler job and one Bend
runtime thread. Compiler remains `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
with the unchanged verified 84-file fingerprint. The full-application test lane
uses UBSan on generated C and the C++ worker/test adapter, not LibTorch or compiled
model internals. No full ASan/ThreadSanitizer, chroot or data-race proof is claimed.

Local full type checking exceeded the 4 GiB authoring limit. Initial hosted run
35800772793 passed type checking but found a code-generation name collision:
convenience `Async.cancel`/`Async.drain` names mangled to the same identifiers as
foreign primitives. Renamed the convenience wrappers to `cancel_identity` and
`drain_identity`; primitive signatures and C ABI are unchanged. No compiler or
generated-source patch bypassed the error. The first failed run's evidence is
retained separately. The complete corrected run generated fresh C and passed all
gates. Local 107 focused tests and scoped lint also passed, but the 170-test locked
hosted run is the authoritative regression count for this publication.

**Self-review, not independent review or formal proof.** Async is opt-in and CPU
singleton only; synchronous remains default. Encoding, selection, diagnostics,
output backpressure and shutdown join remain synchronous. A wedged callback is not
preempted. A stopped search may return an explicitly marked legal fallback if no
searched continuation completed. The 4,096-node arena, backend process limits,
CUDA qualification and production Gumbel gap are unchanged. There is no trained
checkpoint, live GPU, throughput/Elo, hard-real-time or production-readiness claim.
Normal stop/quit and finite deadline paths were exercised; no universal claim about
all fatal process exits or the inherited billion-event safety limit follows.

### Evidence

Artifact **10725998132**, `deepfin-pr4b-qualification`, retains exact source patch/
manifest, JUnit/lint, full build log, model/material/normal/UBSan reports and generated
builds for 30 days. It includes generated C and test/engine binaries for reproduction;
these are absent from the PR. No trained weights or raw neural traces are uploaded.
The saved fixture package is referenced by hash and the earlier PR1 artifact.

- Artifact ZIP SHA256: `584dc64eb61d5225585e71bd143b371e1dd3efb98be0535c26ecb2cb982d237b`.
- Qualified source patch SHA256: `18d0ec08b8c8a223e983bf8ab5560dd0ccd006cd3ab2cc242e27a0f10dd44a35`.
- Generated engine C SHA256: `ec5ef7569e0105b390b09b5f9bfbb24e8ceda5317c76afc9a512a84504f9426a`.
- Actual neural executable SHA256: `338dafe19ae5ef95db639f72e5bbdb840ef88c831655be22f75d1330b10d5a33`.
- Model package SHA256: `9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.

The downloaded 13-path qualification manifest matches the locally authored bytes.
Three additional README/index paths are documentation only. Ordinary PR CI is
separate from the completed opt-in qualification. No permanent workflow or staging
payload is part of the feature diff. Nothing merged, deployed or changed live.
