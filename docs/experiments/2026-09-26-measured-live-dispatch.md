# Opt-in measured live gather limits

## Preregistration — September 26, 2026

Base #887, `19c9cc017be334e3330b8bf4a6c2e488db5845b6`. Continue the neural-search
scaling plan without adopting CPU microbenchmarks as production settings.

Hypothesis: one immutable, exact-package table can change actual live batch row
occupancy while preserving independent-root fixed-work outcomes, lifetime work,
FIFO ownership and cooperative cancellation/deadline/replacement behavior.
Default remains greedy full physical capacity, with no extra root scan. No switch
between model packages, waiting for arrivals, dropped evaluation, or budget refund.

The optional launcher recomputes occupancy cells from raw #887-style reports,
requires compatible CPU/model/encoding/thread metadata and the exact package hash,
and writes a launch receipt before replacing itself with the native executable.
Bend owns all runtime selection, using the largest chunk of each fixed-package
optimal wave for the current number of active roots. This is explicitly a receding
**gather heuristic**, not executing a full optimal wave or predicting a deadline.
Using the lexicographic first chunk of `[1,4]` repeatedly would wrongly request
singletons while five roots remain active. This implementation avoids that trap.

Active roots may yield automatic draws or no neural request. Actual rows can be
below the cap; no invisible leaf reservation or oversubscribed buffer is introduced.
The unchanged post-gather deadline gate still removes expired rows before admission.
A numeric table is an override, not proof of measured provenance. The native model
bridge additionally rejects missing/mismatched package hashes, CUDA, and singleton
(non-batch API) use. The fixed-cohort executable rejects the live-only table rather
than silently ignoring it.

Deciding gates: strict profile/parser/identity negative controls; compiled table
probe and native artifact-binding controls in normal/UBSan; actual live owner at
both input widths under default, one-row, two-row and recorded policies, compared
at the same per-root work to the existing serial tree oracle. Require demonstrably
different physical occupancy, exact unaffected trees and no lost work on combined
cancel/expiry/replacement. Default lifecycle/arena/callback regressions remain gates.

A separate actual-model gate will use the same untrained 5,043,005-parameter,
175-plane CPU-F32 fixture; one batch-four export from the saved PR1 checkpoint.
Require selected-leaf/logit/policy/full-node checks against the unchanged eager
singleton oracle at its inherited tolerances. Synthetic gather overrides test
composition; they are not claimed measured-optimal policies. No in-flight deadline
injection in these real-model checks. No trained model, live GPU or training job.

Budget: one bounded hosted CPU qualification, two Torch threads and one build job,
one fresh live/fixed Bend generation each, existing source-only normal/UBSan panel,
and eight native model sessions (six traced, two quiet). Preserve failures and exact
source/build/report identities. No timing rerolls or external reviewer budget.

Equal-wall-time throughput/strength, calibrated queue latency and target-hardware
service calibration remain separate experiments. Existing hosted CPU profile costs
are not a 5090 recommendation; no speedup/Elo success criterion is declared here.
Recovery: fail closed on incomplete native/model gates, preserve the patch/evidence,
and do not modify the compiler, old oracle, tolerances or live configuration.

## Completed readout — September 26, 2026

**[Qualification run 36219555685](https://github.com/jjoshua2/DeepFin/actions/runs/36219555685),
job 108342098598, passed every stage.** The run applies the exact nineteen-path
candidate to #887 at `19c9cc017be334e3330b8bf4a6c2e488db5845b6`; its workflow
revision is `aa6d541732dfaff61c6adc4df015e5e721360d29`. All downloaded source hashes
match the locally authored candidate. Publication preserves the qualified runtime,
verifier and test bytes, adding compact evidence and this documentation readout.
No parent branch, production configuration or running job is changed.

### What is implemented

`DispatchRun.bend` applies the immutable gather cap in the actual persistent live
owner. It scans both physical FIFO lists without reordering them, excludes inactive
and stopped roots, and calls the existing gather/admission path with the chosen
limit. The default Greedy branch keeps the original full-capacity path without
that extra scan. Pending completion, deadline compaction, root identity, normalization,
backup and physical retirement remain owned by their existing Bend components.
The model bridge only adds an exact-package startup guard, not tensor/search logic.

The Python adapter recomputes cells from retained raw observations and rejects
incompatible model/encoding/CPU/thread identities or inconsistent sample summaries.
For each active-root count it selects the largest chunk of the fixed-package
optimal wave. This avoids repeatedly taking a small first remainder such as the
one in `[1,4]`; it does not execute the complete optimal wave. The receipt is a
preparation record, not proof that the engine ran. Direct environment settings are
manual overrides, not verified measured provenance. Package hashes must match;
a re-export only needs new profiling if its hash differs.

### Completed gates

| Gate | Result |
| --- | --- |
| Focused Python regressions | 574 passed, including 36 new cases; zero failures, errors or skips |
| Whole-repository and explicit native-verifier Ruff/Basedpyright/Vulture | Pass; zero type errors or warnings |
| Fresh pinned Bend generation and full original live lifecycle/arena tests | Pass, normal and UBSan at both 146/175 input widths |
| New actual live gather-policy tests | Four configurations pass: normal/UBSan at both widths, physical batch four |
| Compiled table selection | Four tables and default across counts 0..16, plus 13 malformed-table rejections, per mode |
| Native exact-package binding component | Nine rejection controls and two valid controls per normal/UBSan mode |
| Actual CPU model composition | All six traced and two quiet sessions pass |

The unchanged compiler is `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
with its verified 84-input fingerprint. Both fixed-cohort and live C were generated
fresh from this candidate, not recovered from an older binary or hand-patched.
The original lifecycle panel preserves invalid-replacement rollback, stale-generation
rejection, shared-batch cancellation, timer reset, EOF/stop/quit, backend failures
and default/explicit arena capacities. The full-runner comparison uses the existing
serial reference, not a second newly authored search algorithm.

### The cap demonstrably changes real dispatch

All four new native configurations produced the same following fixed-work counts
for six staged roots, eight simulations each. Every policy compares all 984
populated final nodes bit-for-bit with its serial controls and retains 48 accepted
neural rows. These are repeated fixture searches, not independent games.

| Policy | Forward calls | Physical rows | Padding rows |
| --- | ---: | ---: | ---: |
| Default full-capacity gather | 15 | 60 | 12 |
| One-row cap | 48 | 192 | 144 |
| Two-row cap | 25 | 100 | 52 |
| Table derived from the banked profile | 17 | 68 | 20 |

The recorded table is `[1,2,3,4,3,3,4,4,3,4,4,4,4,4,4,4]`. **It used more
forwards than Greedy in this fixture.** This does not establish a wall-time loss
or gain: the callback is explicitly held by the verifier, not a measured service
workload. It does establish that accepting/logging a table is not the only effect:
the physical batch occupancy actually changes while fixed-work tree results stay
identical. There is no recommendation to enable these recorded CPU caps.

A combined cancellation, immediate deadline and replacement test at cap two
retains seven generations with **42 executed rows, 40 accepted and two wasted**,
22 forward calls and 46 padding rows. Unaffected roots 4–6 and the replacement
match their serial trees. Old cancelled work is retained in lifetime accounting;
the replacement has no inherited deadline. This passes all four configurations.
Test-control pipes and deterministic callbacks do not enter the model product.

### Actual neural model, separately from the held-callback tests

Used the saved untrained **5,043,005-parameter, 175-plane CPU-F32 checkpoint** from
PR1. The singleton package was hash-checked for lineage and one batch-four package
was exported. All model execution here uses that one physical batch-four package.
No architecture, weights, original oracle or numerical tolerance was changed.

Greedy, one-row and two-row policies each execute fourteen completed generations
with replacement/removal/re-admission, followed by a separate two-live-root case.
The two-row policy also repeats both cases without diagnostics: **eight native
model processes, 164 real executed rows**, of which **123 traced rows** pass the
independent feature-bit, raw-logit, legal-policy/WDL and complete-node reference.
There are **2,703 legal-prior and 1,917 final-node comparisons** in the traced cases.
Quiet runs retain the same root outcomes and physical counters. All processes
retain the inherited bridge reuse audit; it does not measure internal allocations.

| Actual-model shared case | Real/accepted rows | Forward calls | Padding |
| --- | ---: | ---: | ---: |
| Greedy | 7 | 4 | 9 |
| One-row cap | 7 | 7 | 21 |
| Two-row cap | 7 | 4 | 9 |

The one-row control reaches the actual model: seven singleton logical calls replace
`[1,2,2,2]`, without changing per-generation outcomes. Sequential reuse intentionally
has one root at a time and uses 34 calls/34 real rows/102 padding rows under every
policy. Maximum raw-logit error is **5.364418029785156e-7**, inside inherited
2e-6 absolute/2e-5 relative limits; legal probability limits remain 2e-7/3e-6.
Full histories and six automatic-draw replies in each reuse case remain intact.
Only validated new dispatch metadata is stripped before the existing identity
projection; original native trace bytes, paths, logits and node fields are untouched.

These model policies are synthetic controls. The profile adapter is tested against
the banked data separately; no freshly calibrated profile-to-exec performance run
is claimed. No in-flight cancellation or deadlines are injected in real-model
sessions. Numerical reference success is not a cross-policy bit-identical-model-tree,
trained-network, GPU, playing-strength or speed claim.

### Recovery, environment and limits

Three earlier hosted attempts stopped before native/model execution. Run
36218967289 accidentally applied Clang globally to the existing Python extension
build, which lacked OpenMP headers; scoped Clang to the native steps and retained
the extension's ordinary compiler selection. Run 36219053531 exposed annotations
for intentionally malformed test inputs. Run 36219287290 exposed an unreachable-code
warning in exact-type shape narrowing. Explicit negative-fixture annotations and
the existing bounded-integer validator resolved these without disabling checks,
removing cases or changing the rejection contract. The final local panel passes
258 parser cases; scoped type/lint checks and normal/UBSan table probes pass.

The original compressed transport had one transcription error. Its correction was
verified against authored part/full-patch hashes before source application. Later
attempts reuse that exact retained patch and explicitly hashed Python corrections.
No compiler, generated C, numerical threshold or original search oracle was changed
to bypass a behavioral failure. Only the successful attempt exported/executed a
model. No service timings were collected or selected in this continuation.

Hosted Python 3.13.15, locked Torch 2.14.0+cpu, uv 0.12.10, Bun 1.4.2,
Clang/Clang++ 18.1.3, two Torch threads and one build job. UBSan covers generated
native runners, workers and test adapters in that lane; it does not instrument
LibTorch or compiled-model internals. No new ASan/ThreadSanitizer, chroot,
independent review, race proof or formal proof is claimed. **Self-review only.**

The extended read-only live CI regenerates source and repeats the new deterministic
panel. Its full workflow is checked separately after PR publication; the YAML and
embedded Python were parsed locally. Normal UCI was not rebuilt by this dedicated
qualification. The fixed-cohort/reference target was rebuilt and remains on its
original default path. A wrong-profile startup fails instead of silently falling
back; the fixed-cohort program explicitly rejects the live-only numeric policy.

The process still owns one physical batch slot and one bound package. There is no
package switching, wait-to-fill, additional forward concurrency, preemption,
automatic production adoption or persistent self-play/data generation. The policy
is a receding heuristic over coarse-compatible historical data. It excludes worker
queueing, encoding/backup, arrival variation and calibrated deadline risk. The
controller-driven fixture wall fields are not equal-wall-time benchmarks. Those
experiments and trained-5090 qualification remain required before recommending
any performance setting. **Default Greedy remains the recommended reference.**

### Retained evidence

Compact original/new native and model reports plus source/build provenance are
committed in `docs/experiments/evidence/measured-live-dispatch/`. Artifact
**10898666782**, `deepfin-measured-dispatch-qualification`, expires October 26, 2026.
It retains the source patch/manifest, JUnit, lint/build logs, reports and compressed
fresh generated C, not model packages, binaries, trained weights or raw traces.
Generated C is absent from Git history. The temporary publishing workflow is not
part of the feature diff. Nothing was merged, deployed or changed in live training.

- Artifact ZIP SHA256: `3bef768eefcf79124319abb4c1c9d95e67e64160d9537f5686516f0e3c82dc10`.
- Qualified patch SHA256: `8a724da71c993929f90baa24b86bf7dbd7f1dc670b27e1fb690becf5815fd14f`.
- Fresh live C SHA256: `a140e5f5755ec4e2b3e256ecbb011f3a01f96c27163481095bac6296008d0c4b`.
- Fresh fixed C SHA256: `9dd546756f6b9ab65f3231bb8fd58f5cd9cd0f613444975208b1f9ea7a3eacd1`.
- Actual model executable SHA256: `5068092a09a32d08faf641ed4d90cbe26cddd5298bab6283d7d1d30ed2768ca8`.
- Model-package SHA256: `f39bea33d07852928e474302107d1922e50ec4155226d555e95cb6f06f49e273`.
- Checkpoint SHA256: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.

### Publication transport follow-up

Initial publication-only run 36220648002 verified all source/report gates but
GitHub rejected its push because the Actions app lacks workflow-write permission.
No branch or runtime source was published by that failed push. The corrected
publication separates the source/evidence commit from the authorized connector
update of `bend-live.yml`; neither step reruns inference or changes the qualified
runtime. The preserved workflow bytes must match the qualification manifest.
