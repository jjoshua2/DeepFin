# Bend-owned neural history encoding

## Predeclared scope and acceptance

Base #800 / a3a2333e13fb9176bd31a3dbb3c8d5e18284d36a. Compiler stays at
fd1df81707fd758f749a9570ccb5b12b1bb2fea3 (Bend 2.0.20 + U64), source verified.
Move the common 112-plane input block into Bend: eight root-oriented history
frames, per-frame repetition flags and layout-specific metadata. Support exactly
lc0_root and lc0_root_legacy_meta with the corrected repetition semantics.
Do NOT describe this as a complete 146/175-plane model input. Extra classical
features, neural execution, policy mapping and training remain separate work.
Do not zero-fill absent classical features or claim support for repfix=false.

Acceptance before native execution: compare every F32 bit for both layouts to the
existing pure-Python encoder and the current C encoder where its clock storage is
representable, with repetition fix set before construction. Verify full prior
history, missing-frame zeros, both perspectives, all metadata, legal versus pinned
EP identity, irreversible moves after repetition, promotions, castling, rewinds,
and read-only hypothetical paths. Invalid layout/path requests must emit no partial
plane block and preserve the root. Existing perft/rule/UCI tests remain unchanged.

Compile generic/portable/native/UBSan and a static executable tested in an otherwise
empty chroot, with external Python oracles outside it. Only raw IO/native runtime
infrastructure may remain foreign; no added C encoder/helper or tables. Budget:
CPU only, at most two compiler processes, one engine thread, one ten-minute hosted
confirmation; repeat only for actual fixes. No new recurring native tests/deeper
perft, production changes, training, model export, GPU, merge or deployment.
Self-review only; no independent reviewer is currently assigned.

## Implementation and local readout

HistoryEncoding.bend owns the 112-plane float32 buffer, all frame orientation,
per-frame repetition lookback and metadata. It uses existing Bend board/history
and legally normalized EP helpers. EncodingDiagnostic.bend exposes a read-only
encode_history command, including up to 32 legality-checked hypothetical moves.
Invalid commands produce no partial block; the accepted root is never replaced.
Normal material search does not call the encoder. This does not yet feed a neural
model or verify a new search-ticket-to-model path; it qualifies the shared input
component on played roots and legal hypothetical descendants.

Local Clang17, NumPy2.3.5 and python-chess1.11.2 pass generic, UBSan and static
empty-chroot runs. Each has 248 position fixtures, 505 complete Python comparisons
(3,619,840 F32 entries) and 501 CBoard comparisons. Four cases exceed CBoard's
uint8 clock range and are explicitly Python-only, not silently compared to wrapped
C clocks. All values match exact float32 bits, with no tolerance. The digest of
the ordered tensors is cb7a7e6688a73c5bfbf734769cd6c5f1d3a04609167bf019a4cbfa79d5eb6bbf.

All three modes also pass the unchanged rule suite (116 positions / 11 search
cases) and original standalone suite (137 exact children / 51 searches / 23
invalid transactions / 8 independent UCI plies). Canonical perft3/3/4 remains
8902/97862/43238. Ruff and all 12 existing compiler-pin contracts pass. Portable,
native-target and Basedpyright were subsequently confirmed in the hosted run.

Two local Bend-checker findings (computed match scrutinee and a forward reference)
were corrected before compilation, without compiler changes. The C oracle was
built using the project's existing setup.py fast-slider macros after a direct
macro-less build correctly failed. Test-only assertions were split for Ruff.
No numerical assertion, case, runtime source pin or legacy test was relaxed.

## Hosted readout: PASS

[Run 35518630156](https://github.com/jjoshua2/DeepFin/actions/runs/35518630156),
job **106098736247**, passed every stage, including source hashes, static checks,
five runtime environments, original regressions and clean source publication.
Exact tested implementation: `adef0e2cb61fd42e133a38e06c29cb598202c4e9`, directly
on #800. This later follow-up changes only the experiment readout. Published
HistoryEncoding, EncodingDiagnostic, Protocol, main and verifier blob identities
match the locally tested bytes. The final feature excludes temporary workflow
and patch-transport history.

Bun 1.4.2, Clang 18.1.3, Python 3.13.15 and NumPy 2.2.6 for the external tests.
Engine flags: -std=c11 -O1 -ffp-contract=off -Werror=shift-count-overflow, one runtime
thread. The unchanged build verifies the current U64 fork revision fd1df817...,
all 84 compiler/Base/effect files and combined fingerprint
88f7505294c77f8187396aaeefd4d1845a6194d7e64b0ab9a982455bf2d8d38b.
Neither Torch nor a model package is installed or executed by this confirmation.

| Executable environment | Complete Python tensor comparisons | Complete C tensor comparisons | Python-only clock cases | Exact equality |
| --- | ---: | ---: | ---: | --- |
| Generic C | 505 | 501 | 4 | PASS |
| Portable U64 helpers | 505 | 501 | 4 | PASS |
| Native CPU target | 505 | 501 | 4 | PASS |
| UndefinedBehaviorSanitizer C | 505 | 501 | 4 | PASS |
| Static, otherwise empty chroot | 505 | 501 | 4 | PASS |

Each lane uses the same 248 position fixtures, eight hypothetical-path/layout
comparisons and one reset comparison. These are repeated test cases across builds,
not 2525 unique positions. Each Python comparison checks all 7168 F32 bit patterns:
**3,619,840 values per lane**, without any numerical tolerance. CBoard cannot
represent every extreme current/prior clock, so four layout comparisons are
explicitly counted as Python-only; they are not tested against wrapped C clocks.
All five encoding JSON reports are byte-identical. Their ordered tensor digest
also equals the local digest despite different NumPy/compiler versions.

Coverage includes every normalized clock fraction from 0 through 100, raw large
clocks, both sides to move, single/mixed castling rights, all promotions for both
colors, empty and long history, repetition before/after irreversible moves, legal
and pinned-illegal EP, missing-frame zeros, and 32 hypothetical plies. Same-board
positions reached in different orders must produce different history tensors.
Each frame's repetition flag uses the history that preceded that frame; a later
pawn move cannot erase older visible flags. All frames use the evaluated position's
perspective, never alternating perspectives with each historical board.

Eight malformed/unsupported requests are rejected with no partial plane block and
no root/history mutation. The busy-command rejection and new-game reset checks
also pass. The data buffer is a Bend Array<F32> with 8192 capacity, but exactly
7168 elements are the logical [112,8,8] block. Storage capacity must not become
an inference shape. Optional diagnostic output is synchronous and can be blocked
by stdout backpressure; no stop-latency or performance guarantee is established.

Every lane also passes the unchanged rule verifier (116 position/history facts,
11 search cases) and unchanged original standalone verifier (137 exact children,
51 searches, 23 rejected transactions and eight standard UCI-client plies).
Perft counts remain 8902/97862/43238; draw adjudication does not prune perft.
No new neural moves or playing-strength claim is made. Ruff, Basedpyright and
all **12 existing compiler-pin contract tests** pass, without suppressions.

The first hosted attempt stopped before native compilation: type checking ran
before the external _lc0_ext reference binary was built. Reordering the reference
build before its typed import check fixed the test environment. No engine source,
verifier assertion, compiler pin, oracle or numerical criterion changed.

## Runtime dependency evidence

The static ELF has no INTERP segment. A fresh runtime filesystem contains exactly:

```
deepfin-bend
```

Both the encoding verifier and inherited clients run externally through
sudo chroot ROOT /deepfin-bend --threads 1. No Python/Bun, shell, dynamic loader,
shared-library file, repository, reference planes or helper executable exists in
that runtime. The C oracle extension is built/loaded only outside it. Host kernel,
stdio and the statically linked native runtime remain; this is a dependency test,
not a security-isolation or formal correctness proof. No C encoding helper or
foreign effect was added to the engine.

## Evidence

Artifact **bend-owned-encoding-confirmation**, ID **10607737267**, 30-day retention.
ZIP SHA-256: 3617a8e1c166f3a2ce4f5a31da51bbd036d23c1e75b1f414d80c38d7aa2179dd.
All five encoding reports: 72b4466260668b3ed5b05879720e49b1a5000cf4dd4fde4fb0e933760a810bff.
Ordered tensor payloads: cb7a7e6688a73c5bfbf734769cd6c5f1d3a04609167bf019a4cbfa79d5eb6bbf.
All five rule reports: 002f1c059bc5d4caee872ecef691573d0126e1294a21d3301e22367b84105c1f.
Original regression report identities:
- generic: ff0e1e7dfd0476c7780b2e359c53d355589214c8ddd8568c23337260caf839f1;
- portable: e49597891b404f843d658ca863be2a175dd0bc67dff5698b7f86b3d6cb940315;
- native: 149c2ddc05d4947abe77da18daf8408c3172421ca06ac726f812982094bf7c45;
- UBSan: 56973a24569a0ba8fb42016ceebf4188352f07c4b509a9a3bab787b0042ca685;
- isolated: 6a6db5647680812e7870db5a858413412924d8522101e90929d5d4effba914fc.

Static executable: 598a5fcf8339b82f5d8fe9f5e45a580178d03038fc40dc42eac83304370a88c0.
Generated C: 70744c6c57ad1668c735e010af7ca442b0678962910791c59529abc4d13b6243.
Build record: 6528a07f3c504835a3cce41368d52e572799b554c74af3b9dfed003959e95b08.
Only compact reports and build/commit identities were uploaded, not engine binaries,
model packages or generated tensors. Hashes identify executed bytes, not guaranteed
reproducible binary archives or cross-machine portability.

## Reproduce / limits / next step

```sh
bash native/bend_engine/standalone/build.sh build/bend_owned_encoding
./build/bend_owned_encoding/deepfin-bend --threads 1
```

At the idle engine prompt, use encode_history lc0_root or
encode_history lc0_root_legacy_meta, optionally followed by moves and up to 32
legal hypothetical moves. The README documents the explicit partial_input header,
bitwise output and external verifier invocation. Build/run itself needs no Python.

This implements the common 112 planes only, **not complete 146/175-plane input**.
The missing 34 v1 / 63 v2_threats classical planes are not silently zero-filled.
Policy mapping, inference, batching and training remain future Bend migration work,
not permanent Python exceptions. Only corrected repetition semantics are supported;
legacy per-frame POV and repfix=false compatibility are explicitly absent. The
normal material-search hot path does not yet use the encoder, and no actual neural
checkpoint, GPU execution or new search-ticket-to-model integration is qualified.

No perft-depth change, new permanent workflow, recurring native encoding test,
ordinary pytest traversal, model export, benchmark or production modification.
The existing search/move/history-rule cores, C transport and compiler inputs are
unchanged. Self-reviewed, not independently reviewed or formally proven. No merge
or deployment. The next substantial migration is the version-specific classical
feature planes and policy mapping, then connecting a complete verified input to
native inference without a Python controller. Broader PR checks remain separate
from the passing dedicated confirmation.
