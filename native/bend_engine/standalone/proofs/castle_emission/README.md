# Actual castling emission and guard provenance

This opt-in suite imports the unchanged `Chess.castle_side` implementation. It closes
the producer-output-to-update-premise step left open by the castling geometry suite.
It does not replace the producer or claim complete legal-move correctness.

## Public contracts

| Law | Domain and guarantee |
| --- | --- |
| `castle_side_guarded_extension` | Any actual Board, side choice, affine attack table and caller tail: the returned list is that exact tail, or exactly one computed flag-two/promotion-zero move prepended to it with the producer guard true. |
| `new_castle_member_has_guard` | A complete Ply present in the returned list but absent from the caller tail entails the input producer guard. |
| `new_castle_member_matches_route` | That same new member equals the complete computed route Ply, including its special flag and promotion field. |
| `producer_guard_supplies_rook_freshness` | A true producer guard entails the existing independent board-invariant predicate for an empty rook landing square. No board consistency premise is needed for this implication. |
| `new_castle_member_preserves_representation` | Applying such a new member to a consistent input Board yields a consistent Board, by composing the real producer result with the existing public castling-update law. No output-validity or target-freshness assumption is supplied by the caller. |
| `rejected_guard_preserves_pair` | A false input guard preserves the entire actual `(table, tail)` pair. |

`Flow.side` checks the actual producer's guard/coordinates by source equality and
control-flow refinement, not a textual anchor. `Membership` derives output provenance
by eliminating the checked sum and list-membership witnesses. `Fresh` proves the
connection between actual occupancy-zero tests and the independent per-bit freshness
predicate by structural Word induction. `Guard` supplies route inclusion certificates
and applies the previously checked public castling preservation law unchanged.

## Deliberate boundaries

The arbitrary input tail can contain unrelated moves, including flag-two moves and
exact duplicates of the new candidate. No theorem claims that inherited tail members
passed the current guard. The consumer preserves an explicit false-guard/preexisting-
castling-tail counterexample. No list deduplication or arbitrary-tail legality follows.
The stronger extension contract still describes a newly prepended occurrence even
when its value duplicates the tail; the membership corollaries require value absence.

The affine table is an input to the actual producer and remains threaded through
actual calls. The positive extension contract observes the returned move list; it
**does not** prove the returned table equals the original or prove native lifetime.
The false-guard contract separately proves equality of the complete returned pair.

No attack-table correctness assumption is needed for output provenance or representation
preservation: even arbitrary actual attack-table results cannot bypass the input guard.
One source non-vacuity witness uses a zero-filled table and is explicitly not a legal
castling or correct-attack-table certificate. Native tests instead build the actual
current `Tables.build()` once per bounded batch and use real `castle_side` checks.

Source contracts follow the producer's raw turn convention (exactly 1 means white;
other values select black) and arbitrary metadata. They do not certify valid turn,
rights history, king counts, FIDE legality or reachability. `castle_side` checks start
and transit with the original king square vacated; the later `legal_moves` filter
handles destination safety. This suite does not prove that later filter, all output
members of complete `legal_moves`, semantic attack correctness, or legal-generation
soundness/completeness.

## Reproduction

Use the unchanged compiler recorded by `../../toolchain.json`:

```sh
bun native/bend_engine/standalone/proofs/castle_emission/focused.js /path/to/pinned/bend --report /tmp/castle-emission-focused.json
CC=clang bun native/bend_engine/standalone/proofs/castle_emission/verify_native.js /path/to/pinned/bend --report /tmp/castle-emission-native.json
```

The focused gate executes the six-law importing consumer, eight intended semantic
mutations, eight manifest/import checks and one synthetic warning-output unit. It
requires exact safe source output, preserves the compiler identity, and rejects
crashes, parser/affine errors and timeouts as invalid semantic-control outcomes.

The native verifier uses an independent square-set model for rights, ownership,
clearance and geometric attacks, and compares ordered output lists and all 19 raw
U32 Board fields for every returned move, including inherited tail moves. Those raw
inherited updates may be illegal or inconsistent; they are compared, not certified.
All four modes repeat the same 852 distinct requests. Four destination-attacked cases
are deliberately retained by `castle_side`, preventing a false final-safety claim.
No full perft run, model, training, performance measurement or older-pin probe is used.
