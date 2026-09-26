# Bend-owned complete feature inputs

## Predeclared gate

Base PR801, fe658a2680ec7941742c045feff0ccb806a898ad. Port the 34 v1 and
63 v2_threats classical features, then join the existing 112-plane block to real
146/175-plane buffers. No missing channels filled as placeholders. Policy mapping,
model forward/training, and normal material-search activation are separate.

Keep all application feature logic in Bend, no new handwritten C. Compare every
history and binary-feature bit with the unchanged Python and C implementations.
Mobility/count features must be exact against C; v2 storm uses the C float32
subtraction/division sequence, allowing atol 1.2e-7 / rtol 0 against Python's
float64 intermediate on planes 173/174 only. Never relax other planes after a
failure. Exercise pinned/check positions, discovered attacks, pawn heuristics,
promotion/special moves, and high attacker counts. Original rules/history/perft
and UCI tests remain unchanged. Verify no Python runtime with a static executable
in an otherwise empty chroot; Python/C oracles run outside it.

The fork head has advanced to aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae (2.0.21).
Test the old manifest-pinned compiler first and separately requalify the new pin;
do not label prior tests as execution of the new compiler. Record the exact result.

Budget: isolated CPU, one runtime thread, at most two C compilers; one 15-minute
hosted confirmation, with retries only for observed findings. No GPU/model export,
benchmark, production/live change, merge or deployment. Native tests remain opt-in.
No deeper routine perft. Rollback discards this isolated branch. Self-review only.

## Implementation

ClassicalEncoding.bend owns all 34/63 feature formulas, using the existing Bend
move-generator attacks and threaded attack-table owner. ModelInput.bend constructs
complete contiguous float32 buffers with explicit channels, and a separate Bend
read-only diagnostic exposes those bits after validating any hypothetical moves.
No new foreign code or generated feature tables. Existing history encoder,
Chess/Search/rule cores and raw IO are unchanged; material search remains the
normal path and does not yet call the new full-input function.

King zones and attacks, full pin rays, discovered attacks, pawn structure,
pseudo-mobility, outposts and space form v1. v2 additionally constructs piece-type
attack maps, saturated counts, hanging/cheaper-attacker masks, safe-check squares,
control, pawn tension and storms. These preserve the existing training semantics:
pinned attacks are not silently removed and pawn mobility is not a legal-move count.
Discovery also preserves the reference behavior for already-checked enemy kings.

ModelInput copies the logical 112-plane block and computes the remaining real
features into one Array<F32>. Explicit channels are 146 or 175, logical lengths
9344 or 11200. Physical capacity 16384 is not an inference shape. No placeholder
channels, policy mapping or model call is introduced. The diagnostic is read-only,
limited to 32 legal hypothetical plies, rejects busy or malformed requests, and
emits no partial input after a late invalid move. Its synchronous output may block.

## Local readout

Clang17 / NumPy2.3.5 / chess1.11.2: initial complete-input validation passed on the
prior fd1df817 compiler. After adding an explicit eight-versus-four attacker
saturation fixture and its mirror, all 714 Python / 706 C tensor comparisons pass,
with eight explicitly Python-only clock cases. Every C bit matches. Python also
matches exactly except the predeclared storm planes, whose maximum difference is
5.960464477539063e-8. All 63 feature planes become nonzero somewhere in the suite;
this is not an unnoticed zero-placeholder implementation.

The newer aaeb9bc compiler then passed the SAME expanded suite without changing
application code, with identical tensor digest
5e038b78e6cf9ae46422bad14d2d2144922a9cac25cd0c6260cbf80db6c881c6.
The new manifest fingerprints 84 source files, SHA-256
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4,
upstream6018e28 / Bend2.0.21. An intentionally supplied old compiler is rejected
before output creation and its sources are not changed. The pin test's explicit
expected revision/version/fingerprint is updated, not removed or made permissive.
All 12 pin contracts and Ruff pass locally. Basedpyright is a hosted check.

The unchanged history, rule and original UCI/perft suites also pass on the new
generic build. The same four suites pass in local UBSan and static-only empty-
chroot runtimes. Complete-input/history/rule reports are identical across those
three environments, and the full-input report also matches the old compiler run.
Normalizer/typechecking is not substituted for native execution. No Torch, model
forward, policy mapping, training or throughput test occurs here.

Local checker findings during implementation required helper boundaries for
computed destructuring/match values and explicit affine reuse; they were corrected
without compiler changes. Ruff required splitting a compound test assertion. The
new exact pin naturally required updating its explicit expected test constants.
No tensor comparison, tolerance, existing test or source-integrity check was relaxed.

## Hosted confirmation: PASS

[Run 35556531816](https://github.com/jjoshua2/DeepFin/actions/runs/35556531816),
job **106201120735**, passed every stage on the first full hosted confirmation:
exact patch integrity, reference build, Ruff, Basedpyright, 12 compiler-pin tests,
five executable environments, all inherited verifiers and clean source publication.
The exact tested implementation is
**2cfe732bce6f660b4e308c8f18bd166dc73f8088**, directly on PR801's head. This later
follow-up changes only this experiment record. New Bend modules, Protocol and
verifier blob identities match the locally tested files. No development workflow
or patch transport is included in the feature branch.

Bun1.4.2, Clang18.1.3, Linux x86-64, one engine thread; flags remain -std=c11 -O1
-ffp-contract=off -Werror=shift-count-overflow. External references use Python3.13,
NumPy2.2.6 and chess1.11.2. The C reference uses the project's unchanged fast-slider
macros from setup.py and is built before type checking its import. No Torch/model
package is installed or executed. build.txt verifies aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae,
all 84 source inputs and the new fingerprint above.

| Executable environment | Complete Python tensor comparisons | Complete C tensor comparisons | Python-only clock comparisons | Result |
| --- | ---: | ---: | ---: | --- |
| Generic C | 714 | 706 | 8 | PASS |
| Portable U64 helpers | 714 | 706 | 8 | PASS |
| Native CPU target | 714 | 706 | 8 | PASS |
| UndefinedBehaviorSanitizer C | 714 | 706 | 8 | PASS |
| Static, otherwise empty chroot | 714 | 706 | 8 | PASS |

Each lane repeats the same **176 position fixtures**, two hypothetical continuations
across both layouts/versions and two reset checks; these are not 3570 unique
positions. Each lane checks **7,334,208 float32 values** against Python. Eight
layout/version comparisons have clocks outside CBoard's uint8 representation and
are explicitly checked only against Python, never against wrapped C clocks.

All C values match exact F32 bits, including the graded features. Python matches
exactly except planes 173/174, where the predeclared C-F32/Python-F64 rounding
allowance is used. Maximum observed difference is **5.960464477539063e-8**, below
the fixed absolute1.2e-7/relative0 criterion. No new tolerance was chosen after a
failure. All five complete-input reports are byte-identical to each other AND
to the local old/new-compiler reports.

The explicit high-count fixture has eight friendly and four opposing attackers
on d4 (plus its color mirror). It tests saturating at seven rather than wrapping
three bits; the reference control margin is 0.75. Every one of the 63 classical
planes is nonzero in at least one case. v1's entire 146 planes must equal the
prefix of v2 for the same history/layout. Nine malformed or unsupported commands,
busy requests and reset behavior preserve the accepted root and never emit a
partial tensor. Hypothetical histories reach the 32-ply bound without becoming
played moves.

Every environment also passes the UNCHANGED inherited suites:
- history block: 505 Python / 501 C comparisons, exact F32 bits throughout;
- rule facts/history: 116 comparisons and 11 search cases;
- original engine: 137 exact legal children, 51 searched roots, 23 invalid
  transactions, partial-line readiness/repeated-stop recovery and eight legal
  plies through the independent UCI client.

Perft stays 8902 / 97862 / 43238 at the previous startpos3 / Kiwipete3 / endgame4
fixtures, and drawn-position perft remains unpruned. The eight material-evaluator
plies are unchanged. These are regression results, not new neural playing strength.
No elapsed encoding/engine throughput acceptance gate was used; observed stop
telemetry in the old verifier is not an encoding or end-to-end speed measurement.

## Interpreter-free runtime

The static ELF has no INTERP segment. The complete runtime filesystem inventory is:

```
deepfin-bend
```

All four suites execute through external clients over pipes to
sudo chroot ROOT /deepfin-bend --threads1 (the actual argument is `--threads 1`).
No Python, Bun, shell, dynamic loader, shared-library file, repository, reference
plane data or helper executable is inside that root. Both reference encoders are
outside it. Host kernel/stdio and statically linked native runtime infrastructure
remain. This tests deployment dependencies, not security isolation or a universal
correctness theorem. No new handwritten C feature/encoding helper exists.

## Evidence

Artifact **bend-complete-input-confirmation**, ID **10621260257**, 30-day retention.
ZIP SHA-256: bcf137a20ca992f29b7796940fde5443781157dbaf14ba1a7fba434e8fa51352.
All five full-input reports: 477f45e4a73225011679ee4077fc1107f8d3e2e8b0504671542f38b8043c79a4.
Ordered full-input tensors: 5e038b78e6cf9ae46422bad14d2d2144922a9cac25cd0c6260cbf80db6c881c6.
All five old history reports: 72b4466260668b3ed5b05879720e49b1a5000cf4dd4fde4fb0e933760a810bff.
All five old rule reports: 002f1c059bc5d4caee872ecef691573d0126e1294a21d3301e22367b84105c1f.
Original engine report SHA-256 values:
- generic: 2498d60cbc63252a538f1214fd4106d2bc216d55bfc5e03839f0b932b4333b1c;
- portable: 6c23626311058e6d9309508efe2cb7ecfc3b6c16a2470b282574ba768c35c167;
- native: 404bb2aa7e2cfbde938cea7438db5c901bc729fcdf578326f22c1697ca5a6596;
- UBSan: a77d8ab3666abac973d4256f72a38a3b5c2d38dbd7450e425cd5baa834dea6a1;
- isolated: ee4967873d5e12d00d4aa4e8ca14a54f85f77d54153a4c0679aec53aecd85989.

Static executable: 5ede086915746344257523d4503f733042666f00c5d8851dd57f4452bfe6fb3c.
Generated C: 4ead79cd6fe66eeaba347f0a2e34767cbabc8fbaea30d5c79707b044654a44ce.
Applied patch: bed77d6ab2503be34da43ac8592ec1041a5e912042f913a55d418d278d52bd04.
Only compact reports/build/commit identities are uploaded, not binaries, model
packages or generated tensors. Hashes identify tested bytes, not a reproducible
archive or hardware-portability guarantee.

## Review and remaining limits

Self-review only; no independent reviewer or universal proof. Reviewed the
root-side square transform, attack-map ownership and pseudo-versus-legal semantics,
full pin-line geometry, discovery with an already checked king, saturation,
pawn direction/wrap guards, channels versus capacity, read-only replay/error
behavior and exact old/new compiler identity. Matching the existing encoders is
not a proof of chess strength or of every possible input position.

The standalone remains a material-evaluator engine. This is a component gate on
played roots and hypothetical legal descendants: search-ticket input construction,
policy-index mapping, model execution and batching are not connected here yet.
v3 feature variants, legacy per-frame perspective and repfix=false are unsupported.
No inference, training, CUDA or speed measurement occurred. The synchronous input
diagnostic can block and has no stop-latency guarantee.

Only the standalone compiler pin changes. Older bitboard/session/neural probes
retain their explicit prior compiler. The newer fork commit records source-budget
and U64-law/oracle passes but separate inherited strict-TypeScript diagnostics;
this engine test does not certify the whole compiler repository. Old checkouts and
binaries are never reset or silently rebuilt. Rebuilding is necessary for adoption.

## Reproduce

```sh
bash native/bend_engine/standalone/build.sh build/bend_complete_input
./build/bend_complete_input/deepfin-bend --threads 1
```

At idle, use `encode_input lc0_root v1` or
`encode_input lc0_root_legacy_meta v2_threats`, optionally followed by `moves` and
at most 32 legal hypothetical moves. The accepted root/history is preserved.
External tests only:

```sh
python -m native.bend_engine.standalone.verify_classical --require-c \
  --report artifacts/bend-complete-input.json \
  --command ./build/bend_complete_input/deepfin-bend --threads 1
```

A new output directory protects previous builds. The external test environment
needs Python, NumPy, chess and the existing CBoard extension. Build/run itself
needs no Python. No new permanent workflow, pytest-discovered native traversal,
model export or benchmark; existing perft depths remain unchanged. No production
paths, prior verifiers, core chess/search code or handwritten C are modified.
The next work is policy mapping and a direct evaluator connection under Bend
control, not another Python controller. No merge/deployment is performed. Broader
PR checks are separate from the passing dedicated confirmation.
