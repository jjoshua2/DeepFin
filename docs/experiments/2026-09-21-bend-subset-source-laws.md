# First source laws for the production subset enumerator

## Preregistration

Base: PR #804, `f1e4037b89314a1f787508d7bd0bb60a1dc6e96c`.
Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`; no pin or
protected-checker changes. The native-leaf work is already validated for its
untrained CPU fixture; do not re-export it merely to continue this task.

Extract the exact `(subset - mask) & mask` step from `Tables.fill` into an
imported Bend module. Prove universal mask membership and preservation through
the mathematical Nat-indexed recurrence. Reuse the existing U64 extraction/deposit
law to reconstruct every recurrence state. Add closed cross-half/bit-63/wrap
witnesses. These are the initial P1 increment, NOT the full ordering, coverage,
slider geometry or table-buffer refinement theorem.

Acceptance: exact source proof output `All terms check.` with successful process
status, explicit import of all eight laws, inherited proofs unchanged, rejection
of missing/false/unsafe/hole proofs and implementation mutations. Compare actual
native subset enumeration against independent compact-bit deposition; preserve
all 128 chess-mask table regions and existing table-reference bytes. No perft
increase, training, model export, GPU or live-process changes. No merge/deploy.

Budget: small local source proofs and CPU-native subset/table probes, one compiler
at a time. A bounded opt-in hosted confirmation may recheck these and the unchanged
compiler regressions; do not add expensive work to ordinary CI. Every test invoked
must record its actual outcome. Self-review unless a separate reviewer is obtained.

## Reconciled parent evidence

Run 35621703547 passed all validation steps; only publication failed. Successful
publication run 35622916175 preserved the old independently existing branch and
published PR #804. Artifact 10649867447 was downloaded and its ZIP hash verified:
`f9f28a829bcddd594610045eadc91799c519c15d68bfe7bf714a98ff5e6d6898`.
The normal, UBSan and no-Python-runtime reports each record 18 searches, 50 native
forwards and 1,141 legal priors compared. These are inherited CPU-fixture results,
not newly executed trained-model/GPU tests or source proofs in this continuation.

## Readout

Local source gate: **PASS** on the exact unchanged 84-input compiler pin. Four
universal contracts and four closed boundary equalities are discharged; the original
16 U64 obligations are imported and checked, not counted as new engine laws.
All eleven new negative controls pass, including missing masking, reversed/truncated
subtraction and a stuck-at-zero implementation. The unsafe-dependency control has
raw CLI status zero but is correctly rejected for its warning. LAWS were not
weakened to obtain a pass. Initial helper Boolean-case and affine-use proof errors
were fixed in proof code, with accepted statements unchanged.

Local native gate: **PASS** in generic, forced-portable, native-target and UBSan C.
Each mode checks 128 chess masks / 107,648 exhaustive chess occupancy states plus
225 synthetic cases / 5,305 states: **112,953 positive state rows per mode**.
Every population 1..64 has bounded synthetic coverage; this is not exhaustive
arbitrary-U64 coverage. Direct compact-bit deposition and independent ray geometry
are the JS reference; the unchanged separate CBoard reference checks all **108,160
logical table entries** from actual `Tables.build`. Five malformed/budget probe
requests are rejected per mode. These are repeated environments, not four distinct
sets. All state digests agree, as do all table digests.

Toolchain: Bun 1.4.2, local Clang 17.0.0, one compiler at a time. The 12 inherited
compiler-pin Bun tests also pass. The original fork source suite was rerun in
workspace preparation run **35623635989** and passed all 16 laws and its seven
negative controls, including the cyclic-template rejection.

The local full `U64_CASES=1024 ... u64_verify.js --regressions` attempt exceeded
this execution environment's 120-second command limit before it emitted a report.
It is recorded as **timeout/uncompleted**, not a pass or a demonstrated assertion
failure. A bounded hosted retry is separate evidence. Whole-repository lint and
full material/static-runtime confirmation are also recorded separately below once
actually run; no local full-engine or trained/CUDA result is inferred.

The checked source and native reports are retained under
[evidence/bend-subset-source-laws](evidence/bend-subset-source-laws/). They include
exact source identities and the verified compiler fingerprint. The native gate's
source hashes also identify its independent references and probe. No generated
binary, model package, training data or private trace is committed.

- Every native subset-output SHA-256:
  `cd647b152c2780a6b18b2edc2cc397c594d1fc233362cf5ebb8e9f1142bdabd2`.
- Every complete logical-table SHA-256 (unchanged from parent evidence):
  `37cf5ae16f1709ef3220c30f60fbd142bc32be6315016dd037a6a8e631c5a663`.

## Interpretation / remaining work

The only production change is factoring the already-Bend `(subset - mask) & mask`
step into `Subsets.bend` and calling it from `Tables.fill`. No new Python application
responsibility, neural model computation or trainer is moved in this increment.
The universal source contracts prove masked-state preservation and lossless
extraction/redeposit, not that extracted indices advance consecutively or visit
all occupancies exactly once. The full P1 index/coverage theorem and P2 table-array
refinement remain open, as itemized in the [durable matrix](../bend_migration_proofs.md).

There is no checker, compiler, fingerprint, inherited proof, existing Chess/Search,
input/neural implementation, old verifier, production configuration or perft-depth
change. Source proofs trust the checker/Base; native lowering/runtime/ABI/toolchain
and hardware remain separate trust boundaries. Self-review only unless a separate
review is explicitly recorded. No merge, deployment, model export, training, GPU
run, strength measurement or benchmark is part of this continuation.


## Hosted qualification and publication recovery

Qualification run **35626785060**, job **106422981932**, development workflow
commit `1e47d4a807c876a2a3192b3ef0c73b47074f5a57`, passed the new source laws and
all eleven negative controls, all four native modes, twelve compiler-pin contracts,
and the original U64 proof/negative-control suite. The full unchanged
`U64_CASES=1024 ... u64_verify.js --regressions` completed: 38 existing regression
files and 45,182 independent-oracle values, JS and four C variants at one/four
threads. This resolves the local timeout without altering the inherited gate.

Whole-repository `./scripts/lint.sh`: **PASS**, including Ruff, Basedpyright
(0 errors/warnings/notes), and Vulture. Linux x86-64, Bun 1.4.2, Clang 18.1.3;
locked Python 3.13 CPU development environment for external checks only.

The actual static material executable was rebuilt with the refactored table step.
Both executions of the unchanged verifier passed: normal runtime and an otherwise
empty chroot containing only that executable. Each reports 10 fixture positions,
137 exact legal children, 51 searched roots, 23 rejected transactions, perft
**8,902 / 97,862 / 43,238 at depths 3/3/4**, readiness/single-stop behavior and the
same eight real-client moves. No perft budget or fixture changed.

**The qualification workflow is nevertheless red:** its final raw `cmp` incorrectly
required a measured timing field to be identical. The sole difference is
`observed_stop_seconds`: **0.010093326999992769** normally versus
**0.010102630000005774** in the empty runtime. Both original reports remain intact.
A publication-only check verifies every other field is exactly equal, confirms
both timings are finite/nonnegative, and records the explicit exclusion from
cross-run equality. The unchanged verifier assertions were not weakened; this
corrects an invalid report-comparison requirement, not an engine implementation.

The static build took **65.30 seconds** with peak RSS **13,411,580 KiB**, about
12.79 GiB. This is build resource usage, not engine runtime memory or a comparative
benchmark. Its normal source compilation warns about 17 definitions depending on
existing foreign effects; this is distinct from the proof suite, which has no
unsafe/foreign proof dependencies and requires exactly `All terms check.`.
The material binary SHA-256 is
`8fa531dccd91fb679931a6c764bb622c861af9cec0681da351d26bcf75cfa61e`.
No neural engine or model was rebuilt/re-exported/requalified in this increment.

Qualification artifact **10652088703**, `bend-subset-laws-confirmation`, has ZIP
SHA-256 `8a0f3be1a16c9a1f4e5197c0a7578b061304e00fb359cebba13bd6fdbe12f7ac`.
Compact hosted reports, original timing reports, lint/build logs, source hashes,
and a separate behavioral comparison are committed under the evidence directory.
The source patch SHA-256 is
`630f5e7d0158cc77ef5a38769559fda2835b9882600eefad742cdf8b5b948c72`.

The first publication-only attempt, run **35627653893**, was rejected as malformed
workflow YAML before any job ran; no engine or proof check failed in that attempt.

Publication recovery run **35628122486**, workflow commit `2e6710638a5f0719afdd4cca863b27bdc8b4b706`, retrieves that exact artifact, checks its digest and recorded step outcomes, reapplies the exact patch to PR #804's parent, matches all candidate source hashes, and rechecks the eight source laws/eleven negative controls. It performs no native rebuild, model export, lint rerun or production operation. Publication is create-only on `feat/bend-subset-source-laws-20260921`; no force push, merge or deployment. Temporary workflows and transport payloads are excluded from the clean branch.

Publication attempt 35627954139 passed evidence validation and source rechecks but stopped before committing because the copied raw build-identity log contains trailing whitespace. The build-identity record is now stored as a JSON string with its original SHA-256, preserving every original byte while leaving the normal Git whitespace check enabled. No implementation, proof, compiler or qualification result was changed.
