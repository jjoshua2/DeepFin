# Actual affine storage foundations for Bend tables

## Acceptance recorded before aggregate and native qualification

Baseline: PR #820 at `290fc01a10517374518e505f9855c7d4a9a5d956`.
Compiler: unchanged `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`.
The eight source contracts and constructive helper development preceded this
record; this is not a backdated preregistration. This acceptance is recorded
before the aggregate mutation/native/hosted checks.

This increment follows the exact scalar-index results with real affine-array
observations. No production code changes. The intended result is eight checked
source laws, retaining the unchanged 56-law parent and its 72 rejection controls,
plus ordinary rejection of deliberately wrong reads/writes/fill implementations.
Successful exit must be accompanied by exactly `All terms check.`. Crashes,
unsafe warnings and timeouts are not successful proof or semantic rejection.

The source contracts quantify over actual `Array<U64>` values, not merely a
separate copy of an ideal array. A structurally recursive reifier consumes each
affine value once and constructs a duplicable image with an equality certificate.
The image is used only in proofs. The engine does not import or execute it.
Shape preservation means the full binary constructor topology is preserved;
it does not mean values, bounds or non-overlap are correct.

Acceptance is exact agreement with an external independent scalar/tree reference
in four native modes for bounded fixtures, including out-of-range aliases. Actual
Tables.fill is exercised, without increasing perft or launching a model, training,
GPU workload or complete material-engine build. Qualification is opt-in, bounded,
and isolated. Parent suites run once per qualification. No ordinary workflow or
test budget is increased. Whole-repository lint must be run unchanged on the final
candidate; environmental failures must remain visible.

Counterexample retained in the importing consumer: in a two-slot array, writing
slot 2 and reading slot 0 observes the write. Array.get/set mask their addresses.
Therefore distinct U32 indices do not by themselves imply distinct storage.
Prefix bounds, region non-overlap, preservation of different physical slots,
full table contents and independent blocker-ray refinement remain separate work.
Source laws do not prove compiler lowering or physical allocation/lifetime.

Self-review only unless a separate review is obtained. No merge, force push,
deployment, live checkout/process change or new Python-to-Bend migration.

## Local development and bounded evidence

The eight new laws and importing consumer pass the exact checker-output gate.
The initial consumer also imported the entire earlier address proof, redundantly
repeating its expensive finite-domain normalization; a 150-second command timed
out and a second redundant run was stopped. The final aggregate retains the
unchanged parent gate once and checks the new consumer separately. No existing
accepted statement or prior gate is changed. A timeout was not counted as proof
success or semantic rejection.

Local focused mutation development passes ten ordinary affected-refinement
rejections and eight policy/output controls. An initial fixture copier included
only the new directory despite Build importing the earlier layout Spec. The final
copier includes the full existing proof tree. The focused development report is
explicitly not labeled a full aggregate qualification. Hosted qualification must
run the final unchanged-parent aggregate before publication.

All four focused native modes pass **1,096 rows** and **109,354 full-array cell
comparisons per mode**: 705 set/read cases, 384 actual fill cases over all 128 keys,
four partial actual table-builder cases and three extra-table cases. There are
249 distinct-numeric-index alias cases and seven invalid requests rejected per
mode. The same fixtures repeat across modes, not disjoint datasets or exhaustive
arbitrary-U64 native execution. All modes output SHA-256
`1a584b59c30cdf714de9485d86de4be04ffb06e682ad2717c6e6814ad964254d`.
The candidate never calls the structural proof model or external oracle.

Local Bun 1.4.2 / Clang 17 native evidence is committed. The unchanged local
whole-repository lint exits nonzero because Ruff, Basedpyright and Vulture are
missing. The original result is retained; no check is suppressed. Hosted locked
CPU-environment lint remains required for final publication.

The complete parent tree was reconstructed and checked against its Git object,
including existing CI repairs, from source archive run 35709047007. Baseline
commit `290fc01a10517374518e505f9855c7d4a9a5d956`, tree
`a1c46bb0620922e18da3f924e310c49f15888044`; source ZIP artifact 10685747666,
SHA-256 `574a636907a42ab2b5439069f754d50d086668bcb90d97d57461b8cd95f8d928`.
That recovery workflow is isolated from the feature diff.

## Remaining work and application ownership

This is an actual-buffer foundation, not completion of P2. Prove prefix arithmetic
and usable address ranges; prove other-slot preservation for distinct normalized
paths; then connect every enumerated write, final header/offset read and lookup
to initialized, disjoint regions and independent blocker-ray geometry. The exact
scalar-index and ordinal results in the parent are preserved, not repeated here.

No new Python responsibility moved into Bend. Python remains in model export,
external references, production/data orchestration and training; native
C++/LibTorch/AOTI remains transitional inference. No trained-model/GPU or
performance claim. The pinned checker/Base, native lowering/array runtime,
effects/ABI, toolchain, libraries, OS and hardware remain separate trust boundaries.
The compiler fork's previously recorded strict-TypeScript and diagnostic-depth
limitations are not fixed or suppressed. All new source is self-reviewed only.


## Hosted qualification and publication

Hosted run **35711968651**, temporary workflow commit `ae80e948fdd5ef12011a8d9f5d9ed98462df0968`, passes all **64 accepted laws**, **90 rejection controls** (18 new plus 72 retained), four native modes, original compiler source/pin checks and unchanged whole-repository lint on exact parent `290fc01a10517374518e505f9855c7d4a9a5d956`.

The new eight-law importing consumer and all 18 new controls match the focused development results. The aggregate separately runs the unchanged 56-law parent once; no inherited law is omitted or weakened. Ten controls require ordinary affected-refinement failures and eight enforce manifests/imports/exact checker output. A raw zero with unsafe warnings remains rejected.

Each native mode passes 1,096 rows and 109,354 complete-array cell comparisons: 705 set/read, 384 actual fill, four actual table-builder and three extras cases. All 128 chess fill keys are represented, with 249 explicit distinct-index aliases and seven invalid requests rejected. Native source hashes and all outcomes match the saved local report; modes repeat fixtures, not disjoint datasets.

Hosted tools: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1), locked Python 3.13 CPU environment with uv 0.12.10. The unchanged Ruff/Basedpyright/Vulture gate now passes, resolving the local missing-tool lint gap. Original failed local evidence remains retained; no errors were suppressed.

The complete candidate tree and every source identity match the inspected patch, SHA-256 fa871489f0a3b579d66ef19282aa372a292134785ac0e39aad8f3f7b1179b869. The exact compiler fingerprint is checked before and after. Publication only adds hosted evidence and documentation to the tested source. Compact reports are committed; full logs remain in the 30-day artifact bend-affine-storage-qualification.

Fresh branch `feat/bend-affine-storage-laws-20260922` is created only after all qualification steps. No force push, merge, deployment or live-process change; temporary workflow and transport payloads are absent from the feature diff. Self-review only, not independent review.

Source proofs include arbitrary affine U64 arrays and all source Nat loop counts; native fixtures cover complete bounded arrays only. Same-address correctness and topology preservation do not prove safe prefixes, nonaliasing, allocation/lifetime, complete computed table contents or independent blocker-ray geometry. Those are the next P2 obligations. No production code or Python application responsibility moved in this increment. No engine/model/GPU/perft/strength/performance requalification is implied.
