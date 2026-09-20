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
51 search roots, 23 rejected transactions and 8 real UCI-client plies per mode;
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

## Evidence and readout

Local preliminary: unchanged application builds under the new compiler and passes
the full generic verifier on Clang 17/Bun 1.4.2. All three perft counts and eight
client moves match the previous baseline. Hosted/multi-mode readout pending.

The fork's compiler source-budget and inherited strict TypeScript gates still
fail as recorded in fork PR #2. Adoption here qualifies only the enumerated
runtime integration, not universal compiler correctness or all upstream gates.
