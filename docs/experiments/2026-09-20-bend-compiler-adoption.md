# September 20 Bend U64 compiler adoption

## Decision and baseline (recorded before hosted execution)

The owner asks whether the Python-free engine uses today's updated U64 fork.
At base #798 / 43bfad24787bd94f37b4884bf7b8617b02a3ce3d it uses d9b9bce9...
(the widening fix), not today's upstream-synced fd1df817... . Older probes use
57bc84ed... . The fork's main is upstream-only, not the U64 deployment branch.

Adopt exact commit fd1df81707fd758f749a9570ccb5b12b1bb2fea3 for the standalone
engine. This contains upstream 2.0.20 (7561656155a4285c1e4ccfcb3505ab59524de973),
U64 native lowering and the widening fix. No compiler/core/chess/controller
source changes are planned. Do not claim untested older probes are migrated.
Keep Python out of the build and runtime; external oracles remain independent.

Acceptance: unchanged standalone verifier passes on generic, forced-portable,
native, UBSan and static-in-empty-root builds. Preserve 137 legal-child checks,
51 searched roots, 23 rejected transactions and 8 real UCI-client plies per mode;
perft counts stay 8902/97862/43238. All 108160 attack entries must still match the
separately linked C reference. Compile with shift-count-overflow as an error.
An explicit stale compiler directory must fail before creating output. Pin tests
must detect changed/extra/missing/symlinked source without mutating the checkout.
Run the fork's existing U64/oracle/upstream regression harness without benchmarks.
No performance comparison or strengthened playing-strength inference from counts.

Budget: isolated Linux CPU, one compilation at a time, engine threads=1, a
10-minute hosted confirmation bound. No GPU, training, checkpoints, production
changes, deployment or merge. No increased perft depth, ordinary pytest native
work or new permanent CI workflow. Self-review only; preserve prior branches.

## Pin and cache contract

One toolchain.json controls fetch and source validation. Hash algorithm remains
sorted relative file names + NUL + binary SHA-256(contents), then SHA-256 of that
stream, for the four compiler/Base files and every recursive effect file.
Updated input count: 84. Fingerprint:
88f7505294c77f8187396aaeefd4d1845a6194d7e64b0ab9a982455bf2d8d38b.
Core/effect symlinks are rejected. The default compiler cache contains the exact
revision in its path, rather than reusing/resetting an old source tree. Explicit
source paths still have to match. Output overwrite protection is retained.
Build metadata includes the verified revision and fingerprint. Build-time
JavaScript is not linked or launched by the compiled engine.

## Hosted readout: PASS

[Run 35512798097](https://github.com/jjoshua2/DeepFin/actions/runs/35512798097),
job **106083566129**, passed all stages on the first hosted confirmation.
The tested source is be872720a2bfc2fbf898821b7e9fd38d266c2c3a. Its clean feature
commit **379154a2598a5d18edd85d123b1fc3c33cd7e75a** is directly on #798 and
contains no temporary workflow. A subsequent documentation-only commit records
this readout, updates the standalone README and adds the experiment index row.
The compiler manifest, build script, verifier and contract tests are unchanged
from the passing run. No application .bend or runtime .c file changes.

The build fetched the exact fd1df817... U64 commit using the new default cache
path and reported Bend 2.0.20. Bun 1.4.2, Clang 18.1.3, Linux x86-64,
-O1 -ffp-contract=off -Werror=shift-count-overflow; engine threads=1.
No Torch or NumPy installation/model work was needed. Python/python-chess ran
only as external verification clients, not build.sh or executable dependencies.

| Executed environment | Exact legal children | Searched roots | Rejected transactions | Standard UCI-client plies | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Generic C | 137 | 51 | 23 | 8 | PASS |
| Forced-portable U64 | 137 | 51 | 23 | 8 | PASS |
| Native CPU target | 137 | 51 | 23 | 8 | PASS |
| UndefinedBehaviorSanitizer | 137 | 51 | 23 | 8 | PASS |
| Static, otherwise empty chroot | 137 | 51 | 23 | 8 | PASS |

These are the same deterministic ten-root fixtures repeated in each build, not
five disjoint datasets. Every mode preserves all three canonical counts:
startpos depth 3 = 8902, Kiwipete depth 3 = 97862, and canonical endgame depth 4
= 43238. Full boards/clocks/history, partial-line readiness, repeated stop and
subsequent recovery all pass. The real UCI client produces the same eight legal
material-evaluator plies in each environment:
b1a3 a7a5 a1b1 a5a4 b1a1 a8a5 a1b1 a5b5.
These are not learned-model or playing-strength observations.

The static executable has no ELF INTERP segment. The entire isolated runtime
filesystem contains one file, deepfin-bend. No Python, Bun, dynamic loader,
shared libraries, repository, model or attack-data file is copied inside.
The external verifier remains outside; the host kernel and open stdio are
provided, and standard runtime/library code is statically linked. This is a
runtime-dependency check, not a security sandbox or proof that no C exists.

All **108160 used attack-table words** generated by Bend match the separate C
reference exactly. Serialized table SHA-256 remains
37cf5ae16f1709ef3220c30f60fbd142bc32be6315016dd037a6a8e631c5a663.
The reference is not linked into the candidate executable.

The updated fork's existing harness also passes without a benchmark:
- Six U64 fixtures through normalization, JS, generic/portable/native/UBSan C.
- Twenty-nine upstream regression fixtures through normalization, JS and C.
- **13694 independent BigInt-oracle values per lane**: 256 runtime-fed random
  pairs, 64 single-bit cases, 14 boundary pairs and 41 expressions per pair.
  Generated JS and all four C variants agree; C runs use one and four threads.
- Instruction checks find PEXT/PDEP in generic/native code and neither in the
  forced-portable output. This does not qualify non-x86 or GPU hardware.

All **12 build-time Bun contracts** pass. A real checkout of old compiler
revision d9b9bce9... was deliberately supplied to build.sh: it failed with
compiler source mismatch before creating output, kept the same HEAD, and remained
clean. No auto-reset or fallback to a different compiler was allowed.

Local Clang 17/Bun 1.4.2 additionally passed the unchanged generic engine verifier
and the 12 pin contracts. The five-mode/table/oracle evidence above is hosted;
no five-mode local result or engine-speed comparison is inferred.

## Evidence

Artifact **bend-september20-adoption**, ID **10605324452**, 30-day retention.
ZIP SHA-256: fd6069c55a3b7e658d9e08d9d95b3c9a328b412fe71d66deb25be35eb1c2cacb.
- Generic report: fbf26e3d153aeb6c9693bb252b92645f38310ba21616e3311ce2ae04137466f0.
- Portable report: 394d96200f81c6b634fd1e1156888a2b4b5c5bcdaacea8672c9f3d6bd7ef1072.
- Native report: 0e2943d005fc1d4c90c546aacfa48cb48415dcece4244a970530a6f25e4cb51a.
- UBSan report: 188de6935bf9ce0ab0e405c96a0d5c64e77dc15aa290eaa505dd84c166ec7c33.
- Isolated report: 5b41fcd75f5e35c8c49d6052580bfe77cc9eba40ccabb19f8cc24c88419f5bf6.
- U64 harness: 9535f1dfadb941871e0a35f95595b10fb01ff33b1bcc4488b6b7745377f63a86.
- Generated C: bf59f96f78f21d1556f45b44443dced5c45c3d0679474bec24a7fec1b3548a71.
- Static binary: c4f5ae3b67cb0f61eb0cbf80ed43231cab91e36b65d40cd811ec62c7eadc984c.

Only compact reports and build/commit identities were uploaded, not compiler
sources, model weights or binaries. Hashes identify executed artifacts, not a
promise of reproducible binary bytes or cross-machine ISA compatibility.

## Reproduce / limits / next decision

From the feature branch in an isolated checkout, use a NEW output directory:

```sh
bash native/bend_engine/standalone/build.sh build/bend_standalone_20260920
./build/bend_standalone_20260920/deepfin-bend --threads 1
cat build/bend_standalone_20260920/build.txt
bun test native/bend_engine/standalone/verify_compiler.test.js
```

Existing binaries are not updated by changing this source pin. Rebuild to use
this compiler. The new source cache path is revision-specific. Earlier probes
keep their own explicit 57bc84ed... pin/release installer; they are neither
silently migrated nor claimed requalified by this standalone run.

Fork PR #2 remains draft. Its compiler source-budget gate and inherited upstream
strict TypeScript diagnostic still fail as recorded there. Targeted U64 and
standalone runtime passes do not turn those gates green. No kernel change,
source-budget relaxation or compiler feature merge into main is performed here.

No permanent CI workflow or recurring native test/benchmark was added. Perft
depths, engine behavior and production paths are unchanged; new Bun contracts
are opt-in build tests, not interpreter dependencies. Self-review only, not an
independent review or universal proof. No merge/deployment, CUDA, trained model,
strength or throughput result. The next migration work remains Bend-owned
history rules and neural encoding, now on the current U64 compiler.
