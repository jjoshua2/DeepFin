# Bend-owned selected-leaf neural evaluation

## Before execution

Base #803: dfa0cda0498d45494e1c52e8271fbf20ebf37871. Keep verified compiler
 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae; no production or core search change.
Connect actual standalone selected leaves, not only diagnostic input commands,
to the complete Bend input/policy component and a pre-exported CPU native model.
Bend owns history/rules, input, legal-logit gathering, stable softmax/WDL and the
reply consumed by Search.resume. Native code is only bounded tensor transport,
artifact integrity/version validation and LibTorch execution. Python is an external
export/test tool, never the running controller. This is a migration backend, not
Bend-authored transformer computation or a Python-free training toolchain.

Acceptance: preserve material default/regressions; compare actual selected paths,
full input tensors and raw model logits against existing external encoders/eager
reference; compare normalized legal priors/WDL and search traversal against the
existing diagnostic reference. Fail malformed/nonfinite outputs before tree update.
Reject unsupported manifest, version, hash, target, batch or encoding at startup
or binding, never silently fall back to material. Only v3 CPU F32 batch-one compact
policy and corrected root-oriented inputs initially. Use the existing saved untrained
transformer fixture, no downloaded/trained checkpoint or new learning experiment.
Predeclared raw-logit tolerance remains absolute 2e-6 / relative 2e-5. Normalized
probabilities allow absolute 2e-7 / relative 3e-6. Tensor comparison uses exact C
bits and only the previously declared two storm-plane Python rounding exception.

Test isolated deployment with native libraries/package but no Python, Bun, shell
or helper process. Unlike the material build this is not a static one-file engine.
Synchronous model calls are nonpreemptible; readiness/stop can wait for a forward.
Maps rebuilt per leaf and list marshaling are correctness-first, not optimized.
Do not infer strength, throughput, CUDA correctness, or full production Gumbel parity.

Budget: isolated CPU, one compiler at a time and two LibTorch threads. One bounded
hosted confirmation (15-minute cap); follow-ups only for observed faults. Focused
local component checks where memory allows. No deeper perft, new permanent workflow,
training, live-process edits, deployment or merge. Preserve unrelated branches.
Self-review only unless an independent reviewer is actually obtained.

## Actual application and native boundaries

Main now threads an explicit evaluator profile. The ordinary material build opens
profile zero and preserves its old behavior. The separate neural build opens one
bound native package before entering the Bend-owned event loop. Startup failure
exits; it never silently selects the material evaluator.

Selected leaves pass through SearchHistory, full-board validation and Rules in
Bend. Automatic draws bypass input encoding and inference. EvaluationInput creates
complete model input and exact legal entries from the same reconstructed Game.
NativeEvaluation flattens exactly channels*64 values, not Array capacity. A
synchronous foreign effect executes the model; LogitReply validates the raw output
and creates the exact ticket reply consumed by the unchanged Search.resume.
There is no Python controller, encoder, normalizer or reference search in this path.

LogitReply requires exactly 1858 policy logits plus three WDL logits, all finite.
Legal entries must agree with the ticket's exact move order and special flags;
compact slots are checked against the logical bound. Stable softmax subtracts
the maximum of ONLY legal logits and computes exponentials/sum/division in F32.
WDL is normalized separately. Huge finite illegal logits cannot dilute the legal
priors. NaN/infinity anywhere is rejected, including currently illegal slots.
A neural-predicted draw remains an evaluation, not a rule-terminal assertion.

The new model_call.c (47 lines) marshals bounded float lists. model_bridge.cpp
(135 lines) loads LibTorch/AOTI, validates tensor shape/device/dtype, verifies the
exact package bytes and optionally records raw trace IO. Neither contains chess,
input features, policy masking, probability conversion, search or an application
scheduler. This is a transitional native model backend, not Bend-authored
transformer mathematics or training. Foreign effects are not proved by Bend.

The Bun binding tool validates the v3 sidecar, CPU F32 batch one, compact width,
corrected root-oriented encoding, consistent metadata and package hash. It records
the checkpoint identity DECLARED by the trusted sidecar; it does not open the
original checkpoint. The external verifier separately loads that checkpoint and
eager model. Startup hashes the bytes copied into a new private scratch directory
and loads only that verified copy. Hashes identify bytes, not their trustworthiness
or machine-code portability. Use trusted executable packages and sidecars only.
Scratch is removed on normal shutdown; crash/kill cleanup is not established.

## Local evidence, not full local-engine qualification

Torch 2.10.0+cpu, Clang 17 and Bun 1.4.2 pass the logit probe in generic, portable,
native-target and UBSan C: 11 cases per mode. New modules pass component checking,
and the final main parses. Full main/native execution is qualified on the hosted
runner because of the local 4 GiB memory constraint. No unrelated service was
stopped and no compiler or integrity check was bypassed.

A small Bend-to-LibTorch effect smoke executable loads the existing fixture at
batch one, returns 1861 logits, and matches eager inference with maximum absolute
error 4.172325134277344e-7. It also executes inside a native-only chroot without
an interpreter, including as an unprivileged numeric UID, with 0600 traces and
empty scratch after exit. This is a local effect smoke, NOT the full chess engine
or the exact hosted package. Ruff and all 33 Bun contracts pass locally.

## Hosted validation: all test stages PASS; publication recovered separately

[Validation run 35621703547](https://github.com/jjoshua2/DeepFin/actions/runs/35621703547),
job **106406081912**, passes every static, build, numeric, neural, sanitizer,
no-Python-runtime and inherited regression stage. Its final source-publication
step fails because feat/bend-native-leaf-evaluation already exists at
975cc8f6c4399727a0395f321694a3e89b55f31a. The workflow's overall result is therefore
failure, NOT an all-green job; the uploaded JSON evidence still contains all passes.
That existing branch has not been overwritten, reset or used as this run's evidence.

[Publication-only run 35622916175](https://github.com/jjoshua2/DeepFin/actions/runs/35622916175)
is green. It reconstructs the exact validated patch plus corrections, verifies
all 13 executable source blobs and the preregistered record, and publishes a new
branch **feat/bend-bound-native-leaf-20260921**. It neither compiles nor reruns
inference. The clean implementation commit is
**a724ae9752c3c465a7f692e5b2ab168451fc2753**, directly on #803. Commit
61b8aa1aea5f85c1492cea510b55fb7c06dce2f1 adds only usage/index documentation;
this readout is also documentation-only. Executable bytes are unchanged from the
validation run. The clean feature excludes all temporary workflows/patch payloads.

Compiler remains **aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae**, Bend 2.0.21 + U64,
84 verified inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Toolchain: Bun 1.4.2, Clang 18.1.3, locked Torch 2.14.0+cpu, Python 3.13.15 and
NumPy 2.2.6 externally. The engine uses one Bend runtime thread and two LibTorch
threads. No compiler, core Chess/Search, existing input/rules/policy implementation,
old verifier or production configuration was changed.

The network is the existing **untrained 5,043,005-parameter transformer fixture**,
exported once in the final confirmation at CPU F32 batch one, 175 planes,
lc0_root_legacy_meta / v2_threats / repfix=true, 1858 policy and three WDL outputs.
The same package is used in all three engine environments. Other profiles are
binding-test coverage only, not executed-model evidence.

| Engine environment | Searches | Native forwards and checked replies | Legal priors compared | Automatic draw replies | Zero-forward terminal searches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal native CPU executable | 18 | 50 | 1141 | 6 | 4 |
| UBSan on generated Bend C and transport | 18 | 50 | 1141 | 6 | 4 |
| Native executable in no-Python runtime | 18 | 50 | 1141 | 6 | 4 |

These repeat 14 fixture positions plus four successive neural-selected moves,
not 54 distinct games. The four continuation moves are **e2e4 d7d5 d2d4 e7e5**.
Each search is bounded to four simulations and depth two. Nonterminal neural
checks forbid unsearched fallback. The model stays loaded in the same engine
process between searches; every search uses a fresh tree.

Actual traced inputs match external CBoard bit-for-bit. Python comparison retains
only the previously declared two pawn-storm rounding exceptions. Maximum raw-logit
error versus eager singleton inference is **5.960464477539062e-7**, and maximum
legal-prior/WDL error is **2.9802322387695312e-8**, within unchanged predeclared
limits. The reference reconciles actual selected paths, boards, reply identities,
completion counts and best moves; it does NOT inspect every final tree field.
Forty-two distinct oracle positions are encountered. Same-board/different-history
inputs differ as intended; automatic draw notices match the reconstructed history.
Zero-forward roots still load the model at startup; this is not a no-load claim.

Normal and UBSan reports are identical. The isolated report differs only by
explicitly skipping three startup-failure tests, which pass in both normal and
UBSan modes: missing package, changed package, and an existing trace destination.
No evaluator fallback or overwrite is accepted. The isolated run exercises the
same 18 searches, all 50 actual forwards, and all input/output/search checks.

The independent numeric gate passes 11 cases in generic C, forced-portable U64,
native-target C and UBSan C: three accepted conversions and eight rejections.
It covers huge illegal logits, constant shifts, extreme finite numbers, NaN and
both infinities, truncated/extra output, invalid slots, wrong flags and empty legal
maps. All four JSON reports are identical. UBSan instruments generated Bend C and
transport, NOT LibTorch, the C++ bridge or the compiled package internals.

Ruff, Basedpyright (zero errors/warnings), and **33 inexpensive Bun contracts**
pass: 21 new binding cases and 12 inherited compiler-pin cases. No tests, assertions,
numerical tolerances or source fingerprints were weakened.

## Deployment evidence and unchanged material path

The neural runtime initially contains the engine, exact model package, **13 native
ELF libraries**, a read-only /proc/cpuinfo snapshot for native CPU dispatch and
empty writable scratch. No Python interpreter/libpython, Bun, shell, compiler,
application source tree or helper executable is present. Existing reference
encoders and the test driver remain outside, communicating through pipes.
Model packages contain native generated code/metadata; this is not an all-Bend
model or a one-file neural product. Host kernel/stdio/native runtime remain.
This is a deployment dependency test, not a security sandbox.

The isolated engine runs after chroot with the verifier's numeric UID/GID. The
optional actual trace remains **0600**, owned by that UID, and scratch is empty
after shutdown; all are asserted. The trace is created during testing, not an
initial input or executable dependency. Trace data stays private and is not uploaded.

All six unchanged material verifiers pass: policy (2963 legal mappings), paired
requests (236), complete inputs (714 Python / 706 C), history (505 Python / 501 C),
rules (116 facts / 11 searches), and original UCI (137 exact children / 51 searches /
23 invalid transactions / eight standard-client plies). Their numerical digests
remain consistent with the previous component gates. Perft stays **8902 / 97862 /
43238** at the existing depths 3 / 3 / 4. The material executable also passes the
original suite statically linked inside an otherwise empty one-file runtime.

The full build command, including source verification and material/native product
compilation but excluding the earlier model export, takes **103.69 seconds** and
reports **9,664,700 KiB peak RSS (about 9.22 GiB)**. This is build-resource use,
not engine runtime memory or a controlled speed comparison. Local full-engine
compilation remains constrained by available memory. No throughput claim follows.

## Observed failures and corrections

The first attempt stopped before compilation on an unused external-verifier import.
The next caught tuple destructuring inside a Bend IO do block; a pure resume_reply
helper repairs syntax without changing the search transition. A workflow compiler
lookup was corrected to derive the exact verified pin. The local isolation helper
also needed the executable's resolved library directories for package .so discovery
and explicit CPU metadata. The following run passed normal/UBSan execution but its
root-owned 0600 trace was unreadable to the nonroot verifier. Running the engine
with the verifier's UID after chroot fixed this without weakening file permissions.
The final run passes all tests; its publication conflict was recovered separately
as documented above. No expensive test was rerun merely to publish documentation.

Parent #803's ordinary CI was already red at inspection: the lease-watchdog test
could not read its expected lease.log. Its separate old Bend chess-probe job was
also red. These are not diagnosed or repaired by this feature. New PR checks are
reported separately; validation evidence is not a repository-wide green claim.

## Evidence

Validation artifact **bend-native-leaf-confirmation**, ID **10649867447**, 30-day retention.
Downloaded ZIP SHA-256: `f9f28a829bcddd594610045eadc91799c519c15d68bfe7bf714a98ff5e6d6898`.
- Normal and UBSan reports: `6bf97d735cb05abdf7499e0591291f41b01d303f7b848da7689829509400b65d`.
- Isolated report: `30eabd8fdf2e00d47d551199c50c457b3f91cea8ae80c32b6708c7dbafbaeb5a`.
- Runtime inventory: `c7700eda6e1a8eb119a9823e67eb7cf2be729573bbd5ccbe0aadef1f73fedba7`.
- All four numeric reports: `770bc2c0319799d202aeb4a9785c186d42ca29415d180f4229ff0ce328528a31`.
- Build-resource report: `2e4e9ca3109b575176474857a794ca0f1d00819ae8f7aa616f96652078792dec`.
- Package: `176d2f729d8bbebadeae3decdfce2fd6a0cf28575fec4b15df6acd357c01590e`.
- Fixture checkpoint: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Original applied patch: `5c29099945999351f1cacb20eb653e2df7e69a506d1da493081126cdfb4a9e55`.
- Applied correction: `16d26dd93e6f8065bcb005d6dcb9869fd2941fb4d9191b4042627b4b9a646b53`.

Publication run retains a separate complete source-blob manifest and commit IDs.
Only compact reports/build identities are uploaded, not raw traces, model weights,
packages or binaries. These hashes identify tested bytes, not a reproducible-build
or cross-machine portability promise.

## Build, run, limits and next work

```sh
bash native/bend_engine/standalone/build_neural.sh \
  build/bend_neural /path/to/checkpoint.pt2 /path/to/libtorch/share/cmake
DEEPFIN_BEND_MODEL_PACKAGE=/path/to/checkpoint.pt2 \
  ./build/bend_neural/neural/deepfin-bend-neural --threads 1
```

Use the UCI subset, e.g. position startpos moves e2e4 e7e5, then go nodes 4 depth 2.
Build needs Bun/Git/Clang/Clang++/CMake/OpenSSL/compatible LibTorch, but the script
itself invokes no Python. The trusted pre-exported package/sidecar must be v3 CPU
F32 **batch one**; previous batch-four packages are rejected, not silently adapted.
Python export is still a separate migration dependency. Use new output directories.

Synchronous forwards are NOT preemptible. isready, stop, quit and movetime can
wait for encoding, inference or blocked per-forward diagnostics. Policy maps are
rebuilt per leaf and lists are copied; batching, persistent-map optimization and
responsive scheduling remain unfinished. The bridge limits forwards to 65,536 per
process. No trained-model/CUDA, strength, performance, production Gumbel parity,
subtree reuse, all-Bend model math or training is established here.

No new permanent workflow, recurring export/native traversal, ordinary pytest
search execution, deeper perft or production modification. Self-review only, not
independent review or formal proof. No merge/deployment. The next work should test
representative workload costs and Bend-owned scheduling/model computation without
reintroducing a Python controller or confusing native-library execution with
Bend-generated neural kernels.
