# Bend castling update contracts

## Scope

This increment stacks on DeepFin PR #886 at `a458a7f855625c65cb0a8132529781e15f8c1608`.
The qualified source head is `105b361a4e4a7cc5b1a1a14308c74c310d5a5b9a`. It promotes the exact checked castling
helpers archived by #886 and adds a public LAWS/PROOF gate, durable negative controls,
and a four-mode native verifier. No production source changes.

Compiler pin remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
Bend 2.0.21 plus U64, 84 source inputs and fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

## Two public laws

`castling_make_move_exact_update` proves that for any actual Board and raw U32
source/destination, `Chess.make_move(b,Ply{src,dst,0,2})` equals the explicit
complete update used by the checked refinement. All eight bitboards and three metadata
fields are included.

`castling_make_move_preserves_representation` proves representation consistency for
one of the four routes 4→6, 4→2, 60→62, 60→58 when the input Board is consistent and
the initial rook landing square is empty (5,3,61,59 respectively).

The second premise is intentional and non-vacuous. The actual raw branch inserts the
rook without first clearing its landing square. The preserved consumer exhibits a
consistent white e1/h1 plus black bishop f1 board whose raw e1→g1 result overlaps on
f1 and is inconsistent. This is a counterexample to unconditional raw-update
preservation, not evidence of a legal-move-generator production defect.

No theorem here establishes source king/rook existence, side to move, castling rights,
other path-square clearance, attack-free king travel, king safety, legal reachability,
clock/history correctness, or move-generation soundness/completeness. Metadata follows
the current raw helpers rather than an independently proved chess-metadata specification.

## Fresh hosted qualification

Final run **36218760990** completed successfully.

- Public source gate: two laws and importing consumer PASS.
- Rejection gate: **17 controls PASS**: eight semantic/refinement, eight
  manifest/import-policy, and one synthetic warning-output unit (not another compiler
  execution).
- Native generic, forced-portable, native-target and UBSan: **4,360 complete Boards /
  82,840 U32 fields per mode**, nine malformed requests rejected per mode.
- Fixtures: 4,096 raw coordinate pairs, 256 route cases, four blocked-rook-target
  cases and four empty-source cases. All 4,360 inputs are representation-consistent;
  264 satisfy the preservation law's route/freshness premises and all264 outputs are
  consistent. Across unrestricted raw calls,185 outputs are intentionally
  outside-premise inconsistent cases.
- Both actual corruptions (omitted rook and wrong rook source) compile/run and fail
  independent returned values. Dropping the rook-target-freshness premise is rejected
  in the source proof.
- Original compiler16-law/seven-control source gate and all12 pin tests PASS.
- Unchanged repository Ruff/Basedpyright/Vulture lint PASS.

Native modes repeat fixtures and are not exhaustive Boards or legal-game coverage.
The external reference uses a 64-square kind/color-set model and compares every
bitboard and metadata field. It supplies no candidate answers.

Exact focused report SHA-256:
`ac66f13579617401d18c9d57e09a675b88703730c3c890c0acd3ed3ae70da303`.

Exact native report SHA-256:
`2d5585a996f8bf102b195260f2d9a221a7d5b62145179566d8044ce56d3544c9`.

Artifact **10899180020**, ZIP SHA-256
`f7422f561bb20b838d41b419114ce57a55c5a0fd2deedc4666502f2db616b4a8`.

Evidence is modular **154 laws/470 controls**: exact parent152/453 plus this executed
2/17 gate. The complete154-law wrapper was not executed.

## Retained harness failures

The first development workflow file was malformed and produced no qualification job.
A later durable-focused run correctly reached the intended proof failures but its
diagnostic-location assertion used the wrong capitalization. After fixing that
harness-only expectation, the source gate passed. The next run exposed a stale
`verify_review.js` filename in the promoted native verifier's self-manifest. That
harness path was corrected to `verify_native.js`; no law, production function or
proof statement changed. Final run36218760990 is the qualifying result. Earlier
successful direct source/native evidence remains historical and is not substituted
for the final durable gate.

## Trust and next work

Self-review only, not independent review. Source proofs trust the pinned checker/Base;
native lowering/storage, ABI, C/C++ compiler, libraries, OS and hardware remain
separate trust boundaries. Fork PR #2 still has its documented strict-TypeScript
diagnostics; structural-snapshot lowering and literal closed-builder equality remain
separately unqualified.

No production runtime, model/GPU, search, training, benchmark or perft workload changed.
No application responsibility newly moved from Python in this proof increment.

Next decisive P3 work is to derive the explicit castling premises from the legal-move
producer and prove king-safety and independent metadata semantics. Exact raw update
and representation preservation do not establish legal-move soundness/completeness.
