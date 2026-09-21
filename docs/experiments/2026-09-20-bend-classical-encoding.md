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


## Implementation / local preliminary readout

ClassicalEncoding.bend owns all 34/63 feature formulas, using the existing Bend
move-generator attacks and threaded attack-table owner. ModelInput.bend constructs
complete contiguous float32 buffers with explicit channels, and a separate Bend
read-only diagnostic exposes those bits after validating any hypothetical moves.
No new foreign code or generated feature tables. Existing history encoder,
Chess/Search/rule cores and raw IO are unchanged; material search remains the
normal path and does not yet call the new full-input function.

Local Clang17 / NumPy2.3.5 / chess1.11.2: initial complete-input validation passed
on the prior fd1df817 compiler. After adding an explicit eight-versus-four attacker
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
All 12 pin contracts and Ruff pass locally. Basedpyright remains a hosted check.

The unchanged history, rule and original UCI/perft suites also pass on the new
generic build. UBSan and static-isolated confirmation are recorded below when
complete. Normalizer/typechecking is not substituted for native execution. No
Torch, model forward, policy mapping, training or throughput test occurs here.

## Hosted confirmation

Pending. Record actual results and identities; do not infer them from local runs.
