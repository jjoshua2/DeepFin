# Castling king-count preservation: qualified C002

C002 passed the pinned Bend component qualification. The public API is:

`CastleCount.preserved(b,ks,rows,guard) -> Actual.preserved(b,castle_emission.Spec.move(b,ks))`

Its only public premises are `board.Spec.valid(b) == True` and
`castle_emission.Spec.guard(b,ks) == True`. It preserves both actual
fixed-white/king and fixed-black/king Nat intersection counts for either
castling route, with arbitrary raw turn values and arbitrary parent king
counts. Extra kings, nonzero rights, and unrelated metadata are retained.
This is a castling component theorem for a later full-ply bridge; it does not
prove general legal-move king-count preservation.

The fixed-color theorem derives source-color ownership, destination clearance,
and rook-source clearance internally. The general color-plane equation still
requires its source-color premise. Fixtures include extra kings, raw turn 2,
both routes, occupied destinations, and overlapping king/rook rows.

## Qualified evidence

The importing consumer and fixtures passed separately in 176.129 seconds and
294.207 seconds. Each exited 0 with exactly `All terms check.\n` on stdout
and empty stderr. Four controls exited 1 with unequal types at their intended
declarations: `reject_missing_source_color`, `reject_occupied_destination`,
`reject_overlapping_rook`, and `reject_count_one` (expected 2n, observed 1n).
Parser, inference, affine, quantity, timeout, and resource failures receive no
semantic-rejection credit.

C001 remains a failed predecessor: its consumer hit a parser rejection at
`CastleCount.bend:67` after 2.005 seconds; no negative controls ran. C002
repairs computed proof-pair destructuring through parameter helpers and
retains the intended count-one contradiction. It adds no stronger public
premise. The preserved C001 evidence receives no qualification credit.

Compiler revision: `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`.
The 84-file compiler fingerprint is
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`;
runtime is Bun 1.4.2. All 14 guards bind 878 source inputs, 84 compiler
files, and sealed prelaunch producers. All six checks used attempt 1,
with no retries and zero remaining owned processes.

Checks ran serially on CPU 29 at nice 19, with 600 seconds per check,
a sampled aggregate-RSS threshold of 6 GiB, and 16 MiB stream limits.
Address space was unlimited. Sampled RSS monitoring is not a hard RSS ceiling.
No GPU qualification is claimed.

## Immutable identities and current-main reuse

Original base: `beb7fd59b08524594bf1fdb87cdc1f92e8e4250b`.
Qualified C002 source tree: `460d21c6cab22c80cda8c161f215ee943af28a84`.
Canonical final C002 patch SHA-256:
`e03e7ac0aaf7637371a9c86846034fc3fda805285960a2d0818affd9558e1a53`.
Raw C002 qualification report SHA-256:
`887ef5f00f286c02e63181f3ca1f7414fe7f29ecfc40c77b25f922a5afa74569`.
Prior independent review SHA-256:
`731f15ac5044beb38090e939b3d75ef6bc454e34ad8b98c1822eb6350f29e528`.

The qualification/materialization records retain the original recovered C001
input patch hash
`142104dfe6d17d426f15d6b230829f1359d6e2c1f3585efa629b9eeb746233dd`
as a legacy lineage field. That field does not identify the final C002 patch.

Composition target main is
`f8394dd067204e02b3c8810174505eccda61e553`, tree
`a36340f995371dfff6afb818b258e85de823c961`. All 867 existing qualified
source inputs preserve their Git blobs and modes. The intervening Bend
additions are confined to `standalone/proofs/castle_valid/`, which C002
does not import. `CastleFrame.bend`, `Frame.bend`, and `Actual.bend`
prerequisites are already on main.

This document replaces the frozen candidate's pre-qualification status.
The ten qualified Bend additions are unchanged; only this non-Bend prose
differs from the original guard snapshot. Reuse follows exact proof-input
and compiler identities. The entire composed tree has not been freshly
compiler-qualified, and no new compiler execution is claimed here.
