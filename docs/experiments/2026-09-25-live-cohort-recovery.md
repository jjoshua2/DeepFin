# Live cohort recovery and current-stack reconciliation

## Preregistration — September 25, 2026

Continue PR5d from the archived draft on #863. New parent is #874 at
`42d4c78ebfca102e146720c818f5dba9db61fd57`. Preserve its FIFO ordering,
completion notification, arena configuration and source-only tests. No merge,
live training, deployment, trained checkpoint or GPU use.

Hypothesis: persistent generation-tagged roots can safely reuse one bounded model
batch slot across add/replace/remove/cancel/deadline transitions. New generations
must not receive old control actions, values, or accounting; unaffected roots
must match the fixed-cohort serial reference. Invalid replacement must not cancel
its old root. Stop/quit abandon pending lifecycle changes; EOF drains them.

Use the exact pinned compiler without changing generated C. First qualify fresh
full native generation on a hosted runner, then normal/UBSan held-callback tests
at 146/175 planes and batch four, including generation reuse, EOF, failure, shared
batch replacement and complete-tree comparison. Inherit all existing expectations
and reference tolerances. Test nondefault arena capacity through actual live roots
and replacements, not only the environment parser. Ordinary fixed-cohort tests
remain independent. Reuse generated C only after checking its entire source hash.

Bound compute to one compiler job and two Torch threads, one source generation
per unchanged entrypoint plus small registry/mutation probes. No model export,
training or strength run in this initial recovery. CPU-model composition may be a
separate bounded follow-up, never inferred from callback success. Keep exact source,
commands, failures, reports and compact identities. Publish only tested source;
any outstanding gate is stated explicitly. Self-review unless an actual independent
review is obtained. No throughput, Elo or hard-real-time hypothesis is tested.

## Completed readout — September 25, 2026

**[Run 36163025335](https://github.com/jjoshua2/DeepFin/actions/runs/36163025335)
completed every gate successfully**, job `108163973419`, before publishing clean
source commit `5eed9cc30f0352d5c984f53ce96a9369b2a941cf`. It is based on #874
at `42d4c78ebfca102e146720c818f5dba9db61fd57`, not the older #863 draft base.
All fifteen downloaded source hashes match the authored files. Follow-up readout,
protocol documentation, compact evidence and CI changes do not change the tested
Bend/C/C++ implementation. The September 24 draft record is retained historically.

### Reconciliation and implementation

The new, separate `live.bend` entrypoint owns a persistent empty-to-active root
registry. It retains #874's owning FIFO, completion-notification wait and configurable
bounded arena. Every new generation uses the configured capacity; no old fixed
4,096-node initialization is restored. Existing fixed-cohort and UCI entrypoints,
search policy, encoders, worker/model computation and compiler remain unchanged.

Controls address `(slot, generation)`. A replacement validates its full new history
before cancelling its predecessor, then installs only after the whole physical
batch drains. Removal also waits for that barrier. Old generations are reported
exactly once before release, and their accepted/wasted evaluations remain in lifetime
accounting. Timers, counters and report flags reset only for the new generation.
Sixteen occupied slots and one pending lifecycle change bound ownership. Generations
are process-global 1..65,535 reservations, never recycled even when abandoned.

Stop/quit abandon queued lifecycle changes; quit closes intake and joins physical
work. EOF closes intake but lets already queued/admitted work finish. Unaffected
roots in a shared batch retain their own outcomes. This is a headless lifecycle
interface, not UCI, autonomous self-play, replay production or a training worker.

### Completed checks

| Gate | Result |
| --- | --- |
| Locked whole-repository Ruff/Basedpyright/Vulture and explicit verifier checks | Pass; zero type errors/warnings |
| Live/cohort/deadline/accounting/benchmark/broker Python tests | 432 passed, zero failures/errors/skips; 116 live cases |
| Full native live owner, physical batch four | Normal/UBSan at both 146/175 widths: all four configurations pass |
| Actual Bend registry and semantic mutations | Normal/UBSan, 128 replacement cycles per mode; both broken variants detected |
| Existing independent fixed-cohort matrix and cancellation/deadline controls | All 33 expected reports pass |
| Inherited native worker | 1,456 assertions and 160 reuse calls per normal and ASan+UBSan mode pass |

The full live tests include invalid replacement preserving the already-admitted
old result, replacement/removal/re-add while a callback is explicitly blocked,
stale commands, busy transactions, abandonment, capacity/EOF, deadline reset,
queued replacement at EOF, ten startup rejections and failed/nonfinite old output.
The failing callbacks must not install the replacement or emit a successful summary.
The held callbacks are test-only; root/control/search logic is actual generated Bend.
Functional control bounds are unchanged and are not a real-time guarantee.

Seven sequentially installed histories compare **483 complete final nodes per
configuration** with the fixed serial reference, including history, promotion,
en-passant and terminal cases. The shared-forward replacement test preserves the
unaffected root's exact tree while retaining **10 executed rows: nine accepted and
one cancelled/wasted**. Internal tree epochs are slot IDs, but external generation
checks plus the physical barrier prevent a previous generation reaching a new tree.

The live arena test exercises both the initial generation and its replacement:

| Nodes/root | Completed simulations | Used nodes | Outcome |
| --- | ---: | ---: | --- |
| Default 4,096 | 184 | 4,095 | Capacity stop |
| Explicit 8,192 | 256 | 5,663 | Requested budget completed |

Both generations match bit-for-bit within each capacity, at both widths and in
normal/UBSan modes. This is evidence that configuration reaches real replacement
searches, not a speedup or a new arena-memory survey. Repeated modes reuse fixtures,
not independent games. The broader fixed matrix covers batches 1/2/4/8/16 at both
widths, sync/async, coordinator UBSan and held cancellation/deadline cases. Existing
CBoard/Python-chess oracles and numerical tolerances were not relaxed.

### Generation, recovery and review

The full native compiler blocker is cleared. Fresh corrected-source generation
occurred in run 36162016314; the deciding run reused those exact generated bytes
only after checking all runtime-source and C hashes. No generated C or compiler
was edited. The unchanged compiler is
`jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, with the same 84-file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

Reconciliation found two implementation issues. Arena capacity needed explicit
Bend value sharing. The async-only entrypoint also exposed unconditional synchronous
effect registration in `batch_call.c`; open/run handlers now compile/register only
when their generated effect IDs are reachable. The ABI, bounds checks and execution
semantics are unchanged. Full live and existing synchronous/async batch paths pass.

Earlier hosted attempts remain recorded failures: a checked transport mismatch
before source execution; the arena ownership diagnostic; two test parametrize-style
findings; two compound-assertion lint findings; and the new rollback test's event
window including the initial admission. That rollback run had correctly preserved
the old evaluation; its assertion was scoped to events after the invalid command.
The final run changed no runtime source for that test correction. No expectation,
timeout, oracle or numerical threshold was weakened. Local Clang 17 normal-146
full lifecycle execution also passed independently of the hosted four-mode matrix.

Hosted environment: locked Python 3.13/Torch 2.14.0+cpu, uv 0.12.10, Bun 1.4.2,
Clang/Clang++ 18.1.3, two Torch threads, one compiler job and one Bend runtime thread.
UBSan covers generated live/reference C and the worker/test callback; the standalone
worker separately receives ASan+UBSan. No ThreadSanitizer, all-interleaving proof,
full generated-C ASan or chroot isolation is claimed. Local O0 register-allocation
failure and earlier bounded O1/compiler timeouts are not successful tests.

**Self-review only, not independent review or formal proof. No neural model was
executed for PR5d.** All new and regression inference in this run uses deterministic
or explicitly blocked callbacks. The LibTorch build helper is supplied, but actual
live-entrypoint/model composition, trained-network fidelity, CUDA and the 5090 remain
unqualified. Prior PR model evidence does not fill that gap.

### Evidence and continuing CI

Compact results, all 33 fixed-report hashes, live-report hashes and exact source
identity are committed under `docs/experiments/evidence/live-cohort-recovery/`.
The complete artifact **10877841014**, `deepfin-pr5d-live-qualification`, expires
October 25, 2026. It includes logs, JUnit, source patches, reports and generated
C/test binaries for reproduction. Generated code/binaries are not in the feature
source; no model weights or raw neural-model traces are uploaded.

- Artifact ZIP SHA256: `2855d89f3c8d9a16fa71ea5493c088c0e2f3ff4f5079ec8632cca362ae4515a9`.
- Qualified patch SHA256: `9b47a5db68a5f16cbe2750502ad7a1d174ce8967f113a0d6a5eaee38c50c7819`.
- Live generated C SHA256: `0febc12b7a6a8a27ddf894cf7bfdc01115413f2792a3be5c93fa814ba610e471`.
- Fixed generated C SHA256: `c01e88e969d5eb09c81b48173f33e63023b25ee7914f0384bf7db67f86390b87`.

The source-only live workflow regenerates current source and invokes the registry
and complete normal/UBSan live gates; it does not depend on historical artifacts
or measure performance. Its first PR run is separate from this completed dedicated
qualification and must be checked rather than assumed successful. Existing completion/
arena CI remains intact. Usage is in `native/bend_engine/multi_root/LIVE.md`.

One physical batch slot and one queued lifecycle change remain. Validation,
encoding, backup and output can delay commands; a wedged callback is not preempted.
Lifetime timing includes setup, idle input time and drain; useful EPS stays null.
No throughput, Elo, hard-deadline or production-readiness result is claimed.
Service-time-driven dispatch and persistent self-play/data generation remain PR5
follow-up work. Nothing was merged, deployed or changed in live training.
