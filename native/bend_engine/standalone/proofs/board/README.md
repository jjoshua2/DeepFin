# Board partition and fresh insertion (P3 foundation)

This opt-in suite starts P3 representation work after #877. It does not alter
production code, prior proofs, compiler inputs, perft or ordinary test budgets.

```sh
bun native/bend_engine/standalone/proofs/board/focused.js /path/to/pinned/bend --report /tmp/board-source.json
bun native/bend_engine/standalone/proofs/board/verify_native.js /path/to/pinned/bend --report /tmp/board-native.json
# Expensive complete inherited chain; focused/modular receipts do not imply this ran:
bun native/bend_engine/standalone/proofs/board/verify.js /path/to/pinned/bend --report /tmp/board-aggregate.json
```

## Representation is deliberately weaker than legal position

`Spec.row_valid` independently describes the allowed per-square states: all
planes empty, or exactly one of six piece-kind planes and exactly one of two
color planes. `Spec.valid` checks these Boolean rows across both 32-bit limbs of
the eight actual U64 bitboards. This excludes overlapping colors, multiple kinds,
uncolored pieces and color-only occupancy. It ignores metadata.

The empty board satisfies this representation invariant although it is not a
legal orthodox-chess position. So do boards with extra kings or pawns on end
ranks. No king count, turn validity, castling/EP validity, reachability, occupied
piece-decoder correctness, move application or legal generation theorem is
claimed. Metadata preservation below does not certify the metadata's validity.

The chosen normative anchor for future P3 chess semantics is **FIDE Laws of Chess
applied from 1 January 2023**, English text, deliberately pinned rather than
claimed to be the latest edition: https://handbook.fide.com/chapter/e012023
Article 2.1 supplies the 64-square domain and Article 2.2 the two colors/six kinds.
Engine movement/king safety and special-move rules in Article 3 remain future
obligations. Physical piece-touching, clock operation and arbiter procedures are
not board-bitboard predicates. This suite does not claim FIDE compliance.

## Six source contracts: four universal, two closed

- `empty_representation` and `initial_representation` check actual `Position.empty`
  and the board extracted from actual `Position.start`. These are two closed
  consistency facts, not universal theorems or a proof of all initial-rule details.
- `fresh_put_preserves_representation` proves actual `Position.put` preserves
  consistency for an arbitrary consistent board, typed piece and color when the
  actual inserted bit mask avoids both color planes. Under consistency that is
  an empty target. The premise examines input occupancy, not the desired output.
- `put_exact_bitboard_update` gives the entire actual Board result, including all
  eight planes and metadata, as the explicit mask-addition specification. It does
  not require freshness; overlapping insertion is described, not rejected.
- `put_preserves_metadata` preserves all three fields (turn, rights, EP) for
  arbitrary input boards and raw U32 kind/square values.
- `metadata_preserves_representation` proves that actual `Position.metadata`
  leaves the partition predicate unchanged, even on inconsistent inputs.

The `Piece` type has exactly six constructors; `tag` supplies the actual 0..5
encoding. Fresh insertion is sufficient, not necessary: repeating the exact same
piece/color on an occupied square can also preserve consistency. No freshness-
checking runtime API or replacement insertion semantics is added.

The formal insertion statements quantify over arbitrary U32 square arguments
using the actual `U64.bit(U32.to_nat(sq))` mask. This broader algebraic domain is
not a promise of meaningful out-of-board squares. Native insertion requests are
bounded to 0..63. Metadata-only requests deliberately permit arbitrary U32 fields.

## Proof construction and implementation link

`Rows.keep` exhausts the finite eight-Boolean row domain (256 rows, thirteen
consistent states); its branches are checked by Bend, not imported oracle answers.
The explicit cases keep invalid-row contradictions visible to this checker.
`Words.insert` uses structural induction on arbitrary word width and mask bits.
`Actual.mask_valid` combines both limbs; `Actual.bridge` mentions the imported
real insertion and complete Board result. Final laws transport the structural
invariant through that implementation equality. This is not enumeration of
possible 64-square boards or reliance on sampled native states.

The exact-update specification shares the trusted U64 Boolean primitives with
the implementation; the partition specification and native mailbox reference do
not use the production piece decoder. A no-op insertion preserves consistency
but violates the separately checked exact-update bridge; both source and native
controls detect it. Do not describe invariant preservation alone as value fidelity.

The consumer includes symbolic board use, every typed piece/color at bit 63,
nonempty two-piece construction, and an explicit overlapping-insertion example
outside the freshness premise. Prior P2 proofs and tests remain unchanged.

## Checks and limits

The focused gate runs all six public obligations and importing use, then eight
ordinary semantic/refinement controls, eight manifest/import safety controls and
one synthetic exact-output warning test. The synthetic test is not a compiler
execution. Missing files, malformed terms, affine-use errors, crashes and timeouts
are not successful semantic rejection. Safe success requires exit zero **and
exactly `All terms check.`**; the pinned compiler is verified before and after.

The native gate executes actual Position operations on independent test inputs.
An external 64-square model of sets of piece kinds/colors encodes each complete
expected Board. All eight bitboards and all three metadata fields are compared;
the candidate does not execute the proof predicate or receive expected answers.
There are 768 fresh square/kind/color combinations, 64 repeat insertions, 64
conflicting insertions, 64 valid and 64 invalid metadata cases, empty and start.
Collision cases characterize behavior outside the premise; they are not runtime
rejection tests. Nine malformed requests are rejected by the bounded probe.
Generic, portable-U64, native-target and UBSan modes repeat the same fixtures.
Two actual no-op/wrong-color mutants must compile and execute before the original
reference rejects incorrect Board fields. No search/table build/model is needed.

Source correctness still trusts the pinned checker/Base and its Boolean-word
semantics. Native lowering, physical storage, ABI, toolchain, OS and hardware
remain separate boundaries. No performance, strength, complete migration or
bug-free-engine claim follows. Self-review only.
