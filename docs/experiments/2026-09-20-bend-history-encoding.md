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
native-target and Basedpyright results will be recorded after hosted execution.

Two local Bend-checker findings (computed match scrutinee and a forward reference)
were corrected before compilation, without compiler changes. The C oracle was
built using the project's existing setup.py fast-slider macros after a direct
macro-less build correctly failed. Test-only assertions were split for Ruff.
No numerical assertion, case, runtime source pin or legacy test was relaxed.

## Hosted readout

Pending. A 112-plane block is a partial input, not a loaded/evaluated network.
No test here establishes throughput, playing strength or trained-model/CUDA support.
