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

## Completed readout — September 22, 2026 (America/New_York)

**[Qualification run 35805850476](https://github.com/jjoshua2/DeepFin/actions/runs/35805850476)
is green**, job `107006408721`. Every source/static/native/matrix/real-model gate
passed before clean publication. Executable-source feature commit:
`f4fa959be5aa628ebb65fb878100b3e616b98659`, directly on #840 at
`31f376752d0baea75a6fbd65bf59f8bd1d92202c`. A following documentation-only commit
records this readout, links the experiment index and adds a README evidence link.
It does not alter qualified executable source.

The exact ten-path qualification manifest downloaded from GitHub matches all
locally authored file hashes. The implementation adds a separate Bend cohort
runner; existing UCI, Search, rules, encoders, model backend, compiler, production
configuration and PR4 worker are unchanged. The experiment index is the only
existing file modified by the final feature diff. No staging workflow, payload,
generated C, model package or binary is in the PR.

### Actual checks

| Gate | Observed result |
| --- | --- |
| Whole-repository Ruff, Basedpyright, Vulture plus explicit verifier lint | Pass, zero type errors/warnings |
| Cheap report tests and existing async/accounting/benchmark/broker tests | 196 passed, zero failures/errors/skips; 64 new cases |
| Full Bend type check and fresh C generation | Pass on unchanged compiler |
| Deterministic callback, batch 1/2/4/8/16 at 146/175 planes | All 10 configurations pass |
| Same generated runner and callback under UBSan, batch four at both widths | Both configurations pass |
| Deterministic final trees versus serial batch one | All populated final node fields are bit-identical across the matrix |
| Actual CPU model at batch one and batch four | Independent singleton-output, selected-leaf and per-root checks pass |
| Quiet versus detailed diagnostic modes | Per-root results and work counters agree in every configuration |

Each configuration exercises a primary 14-root cohort, four terminal-only roots,
and three roots with an exact three-neural-row cap. These are repeated fixture
searches, not distinct games. The matrix checks actual generated Bend Search
preparation, full histories, CBoard feature bits, ticket identities, every real
raw logit, legal priors/WDL, all populated final tree fields and compute accounting
against the existing independent serial reference. Padding input bits are exact
+0. The deterministic callback is row-independent and never linked into the real
LibTorch target.

All ten normal/two UBSan deterministic configurations share the primary complete
node snapshot SHA256:
`593e548adcc43df3cca20bdbc1725fc331274c5e36b33c9a85e74498510a0770`.
Each compares 677 primary final nodes and 663 legal priors; the capped cohort
adds 212 nodes and 209 priors. Terminal roots add four nodes and zero forwards.
The bit identity is a property of these deterministic-control results, not a
claim that arbitrary real-model batching preserves every floating-point bit.

Sixteen malformed argument/history controls reject per configuration, before a
raw model trace is opened. Batch sizes greater than one also pass a failing
callback and nonfinite-final-row control: neither can produce a neural reply or
successful summary in the first batch. A deliberately swapped-row callback is
rejected by the independent oracle. These failure tests run in both normal and
UBSan matrix lanes; no threshold or reference was weakened after a failure.

### Real model composition and measured work counts

The model is the same **untrained 5,043,005-parameter, 175-plane CPU-F32 fixture**
used by earlier PRs. The singleton package is reused byte-for-byte. Exactly one
new CPU batch-four package is exported from the same saved checkpoint and
encoding using the existing exporter. No trained/private model or GPU is used.

| Primary 14-root cohort | Real/accepted rows | Backend calls | Physical rows | Padding | Legal priors / final nodes compared |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual model, batch one | 34 | 34 | 34 | 0 | 670 / 485 |
| Actual model, batch four | 34 | 9 | 36 | 2 | 670 / 485 |

Batch four uses eight full four-row calls and one two-row call. This establishes
that real selected leaves from different roots share a native forward while
preserving their own ticket/result association. It is **not a 3.8x speedup** or
an Elo result. Fewer calls can still cost more time or memory; this slice has no
throughput or strength deciding metric.

The three-root capped cohort consumes nine real/accepted rows: nine singleton
calls versus three batch-four calls with three total padding rows. All three
roots meet their own budget exactly. Four terminal roots consume zero forwards
and correctly retain unmet neural-budget flags. Every case is also rerun in a
separate quiet process. Across the two model batch sizes and both diagnostic modes,
these cases execute 172 real model rows; numerical/trace oracle checks apply to
the 86 traced rows, while quiet runs are checked for matching outcomes/counters.

Every traced real output is compared with independent eager singleton evaluation
under inherited 2e-6 absolute / 2e-5 relative tolerances. The maximum absolute
logit error is `5.960464477539062e-7` in each batch configuration. Legal priors/WDL
use the unchanged 2e-7 absolute / 3e-6 relative tolerances. The full-history,
same-board/different-history, automatic-draw, promotion and en-passant fixtures
remain distinct and pass. Each batch configuration checks 898 legal priors and
720 final nodes over its three traced cohorts.

The primary per-root results, node counts and chosen moves agree exactly across
real batch one/four; all populated node fields pass each independent reference.
Their full bit-snapshot hashes nevertheless differ because real F32 model outputs
can differ within tolerance. No bit-identical real-model tree claim is made.
The deterministic-control bit identity and real-model numerical qualification are
separate evidence.

### Timing and remaining scope

This is synchronous, offline and bounded. Partial batches dispatch at the end of
the available sweep with no wait-to-fill heuristic. Rotating roots prevents the
same prefix from owning every batch, but a live-arrival/deadline scheduler and
measured queue-latency distribution are not implemented here. Normal UCI is not
replaced, and the PR4 responsiveness result does not apply to this separate
blocking executable.

The reported millisecond-resolution cohort clock begins after validation/model
loading, before shared map/buffer setup. It includes the first forward and per-leaf
diagnostics, excludes final tree/report formatting and does not exclude warmup.
It is not an external time-to-decision, fixed-wall, warmed throughput or paired-game
benchmark. Queue/H2D/GPU/D2H timings remain null. A successful summary has no
unresolved work because all calls are synchronous; failure exits have no successful
summary. Per-root and aggregate accepted rows must not be double-counted.

Remaining PR5 work includes asynchronous multi-root polling/cancellation, live
root arrival/removal, persistent self-play integration, wall-time admission and
service-time-driven batch selection. CUDA/trained-checkpoint qualification and
same-tree concurrency stay separate. Root arena capacity remains 4,096, production
Gumbel parity is unchanged, and no production readiness, single-game speed or Elo
claim is made.

### Environment, recovery and review

Locked Python 3.13 / Torch 2.14.0+cpu, uv 0.12.10, Bun 1.4.2 and hosted Linux x86-64;
two Torch threads, one compiler job and one Bend runtime thread. Compiler remains
`jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, with unchanged 84-file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The matrix compiles the same generated C for all backend sizes. Existing native
CMake builds the real target with an explicitly bound header and no CUDA support;
dependency inspection finds no libpython link. The new build wrapper's constituent
commands and shell syntax are checked, not a separate second full build invocation.

UBSan covers generated runner C and the deterministic callback, not LibTorch or
compiled model internals. No full ASan/ThreadSanitizer, chroot isolation, UCI rebuild,
perft or independent review is claimed. **Self-review, not independent review or
formal proof.** Local new-code type checks, 64 report tests and Ruff passed; the
local Basedpyright check lacked a built native extension. The locked hosted
whole-repository and explicit verifier checks passed without suppressions.

Initial run 35805738438 failed its payload-integrity gate before applying or
executing source. The preserved transfer was corrected back to the locally
authored compressed and uncompressed hashes. The complete corrected run applied
that exact patch. No executable source, compiler, oracle or numerical threshold
was changed to bypass a behavior failure.

### Retained evidence

Artifact **10727754120**, `deepfin-pr5-cohort-qualification`, retains source patch/
manifest, JUnit/lint, build identities, matrix and real-model reports, and compressed
generated runner C for 30 days. Generated C is not committed. No model packages,
binaries or raw neural traces are uploaded in this artifact. The package identity
and earlier PR1 fixture artifact provide the model lineage.

- Artifact ZIP SHA256: `430358af9eaaf7442af34d5f13738df14b05af7ca7de82a8f26a1ec89739da09`.
- Qualified source patch SHA256: `0cb52d8c33f06b66bd9adf63f84929b0ca93ec3de09ecc729e34e1960183ca35`.
- Generated runner C SHA256: `79e100a2ab1e9a8da59a2c3ab2b8d1d7ede34a94b33a9010e7cfbe0c068245f5`.
- CPU batch-one executable SHA256: `43324ae86e31b0fda3628286d821a6deb40fcee5b4229a3d1af78d00f66b8172`.
- CPU batch-four executable SHA256: `c3c75ae65c0212ee2e732d46ed8930afd2fd310789fb1e2aba87ad88f16385d4`.
- Checkpoint SHA256: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Batch-one package SHA256: `9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
- Batch-four package SHA256: `f2490e9bec9145462a9e19032ee03b9911119217a5195798119d7eabc64b39be`.

Ordinary PR CI is separate from the completed opt-in qualification above.
Nothing merged, deployed or changed in live training.
