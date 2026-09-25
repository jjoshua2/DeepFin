# Scalar deletion and ordinary move-update contracts

This opt-in P3 suite composes deletion with the previously proved typed insertion
invariant, then connects that composition to actual `Chess.make_move`. It changes
no production function and does not implement another move application API.

```sh
bun native/bend_engine/standalone/proofs/move_update/focused.js /path/to/pinned/bend --report /tmp/move-source.json
bun native/bend_engine/standalone/proofs/move_update/verify_native.js /path/to/pinned/bend --report /tmp/move-native.json
# Expensive complete inherited chain; not implied by focused/modular evidence:
bun native/bend_engine/standalone/proofs/move_update/verify.js /path/to/pinned/bend --report /tmp/move-aggregate.json
```

Use unchanged compiler `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, whose 84-input
fingerprint is checked before and after every gate. This suite is not added to
ordinary CI, perft, search, inference or training execution.

## Public claims

`clear_mask_exact_update` equates the complete Board returned by `ClearActual.run`
with removal of the mask from every piece/color plane, preserving metadata.
`ClearActual.run` is a proof-interface lift of the real `Chess.update_piece`
scalar kernel across all eight planes, not an existing production clear-Board API.

`clear_mask_preserves_representation` proves this lift preserves the independent
board partition when the input satisfies it. `cleared_target_is_fresh` proves
clearing the source/target removal mask makes the target empty, even on arbitrary
inconsistent input Boards. Both source and target may be arbitrary U64 masks;
no expected output or assumed fresh-target certificate is required.

`ordinary_make_move_exact_update` connects actual `Chess.make_move` with
`Ply{src,dst,0,0}` to clearing the source/destination, inserting the selected piece
and updating the complete Board including metadata. `Selection` is explicitly a
typed image of the actual decoder, not an independent chess-piece specification.
The result's metadata expressions retain the current `corner_right` helper and
raw arithmetic; this is not independent FIDE metadata correctness.

`ordinary_make_move_preserves_representation` then proves the output's per-square
partition from input consistency. Its source domain includes arbitrary U32
source/destination and metadata values. It follows actual bit-constructor
semantics and makes no modulo-64 claim. Native move tests restrict squares to
0..63. The production function's intended caller supplies legal moves; this
structural theorem does not license arbitrary external calls.

The move claims exclude promotion and all special flags. They do not establish
occupied source, correct moving color, destination admissibility, movement rules,
king safety, legal rights/EP metadata, clocks/history or legal reachability.
The consumer demonstrates why: an empty-source raw move can manufacture the
king fallback while retaining the representation invariant. Same-square calls,
raw metadata and inconsistent-input diagnostics are explicitly separated in tests.

## Proof structure and checks

`Rows` proves finite Boolean deletion facts inside Bend; `Words` lifts these by
structural induction to arbitrary words. `Clear` handles both limbs and reuses the
existing insertion invariant. `ClearActual` proves its connection to the real
scalar helper. `Core` removes harmless scalar-zero alternatives, and `Ordinary`
connects the entire actual non-special move expression to that composition.
A no-op can preserve consistency, so the exact-update laws and behavioral tests
are independently necessary.

The importing consumer, exact law/proof manifests and mutation checks are all
required. Nine controls require ordinary semantic/refinement failures at named
locations, eight check source/import policy, and one synthetic warning-output
unit is not a compiler execution. Crashes, missing imports, parser/linearity
errors and timeouts are not accepted semantic rejection. Safe success is status
zero AND exactly `All terms check.`. The complete wrapper preserves the unchanged
141-law/401-control parent; a modular result must not be called its execution.

The native probe imports only real Chess/Position/Text operations, not the proof
specification. It exercises actual ordinary `make_move`, a probe-side lift of the
scalar clearing kernel, and arbitrary scalar updates. An independent 64-square
set model compares all 19 emitted U32 fields. Modes repeat fixtures; counts are
not exhaustive boards or legal-game trees. Invalid request rejection belongs to
the bounded probe, not a raw `make_move` validation theorem.

## Status and remaining work

See the dated readout and compact receipts for exact executions. Local source and
native gates pass; repository lint remains unqualified for missing tools. No new
hosted qualification or independent review is claimed. The entire inherited
aggregate was not rerun. Promotion, en passant, castling, occupied/legal source
conditions, king safety and move-generation soundness/completeness remain P3 work.
The existing P2 closed-builder and structural-snapshot lowering limits remain.
