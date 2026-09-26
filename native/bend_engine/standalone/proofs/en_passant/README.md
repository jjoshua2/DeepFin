# En-passant raw-update contracts

This opt-in suite promotes the checked `Actual.bend` and `Mask.bend` from #885's
supplementary en-passant archive unchanged. The probe is also byte-identical.
Only the importing consumer, public law registration and dedicated test gates
are new. No production move function, earlier proof or compiler input changes.

```sh
bun native/bend_engine/standalone/proofs/en_passant/focused.js /path/to/pinned/bend --report /tmp/ep-source.json
bun native/bend_engine/standalone/proofs/en_passant/verify_native.js /path/to/pinned/bend --report /tmp/ep-native.json
# Full inherited chain; NOT implied by focused or modular receipts:
bun native/bend_engine/standalone/proofs/en_passant/verify.js /path/to/pinned/bend --report /tmp/ep-aggregate.json
```

## Formal interface

`en_passant_make_move_exact_update` equates actual
`Chess.make_move(board, Chess.Ply{src,dst,0,1})` to the complete specified update.
It removes the source, destination and `dst XOR 8` masks, inserts the actual
selected source kind at the destination, and retains the existing metadata
helper semantics. Both square inputs are arbitrary U32, and all Board fields
are arbitrary. It is not an independent chess-metadata specification.

`en_passant_make_move_preserves_representation` additionally assumes the input
Board partition invariant and proves it for the result. The invariant distinguishes
empty squares and a single kind of a single color, not chess legality. It does
not assume source occupancy, a pawn, correct rank, the previous double push,
rights to capture, or king safety. Source fallback and raw metadata follow the
actual implementation. Source laws do not assert wraparound modulo 64 for square
arguments. Native probes restrict them to 0..63.

The local source producers and inherited clearing/fresh-insertion theorems supply
the mask, update and freshness facts; the caller does not assume the desired
result. The consumer checks both pawn directions and the explicit empty-source
boundary. The specification functions live beside the preserved proof producers,
as in the existing ordinary-update suite; no desired equality is imported as an
axiom.

## Qualification layers

The focused gate checks both registered laws and importing consumer and then
requires eight intended source-refinement failures, eight manifest/import-safety
failures, and a synthetic output-wrapper rejection. The synthetic check is not a
compiler execution. Crashes, missing dependencies, malformed terms, affine-use
failures and timeouts are not valid semantic controls. Source success means exit
zero and exactly `All terms check.`.

The native reference uses independent coordinate-derived capture locations and
square sets. The probe receives no expected answers and imports no proof model.
It compares all eight actual bitboards and three metadata fields for 4,316 rows
per mode (4,188 consistent and 128 diagnostic inputs), all raw square pairs and
selected pawn/metadata cases. Generic, forced-portable, native-target and UBSan
repeat the fixtures; they are not disjoint or exhaustive legal-game coverage.
Two actual capture-selection corruptions must compile and execute before failing
the reference. Nine malformed probe requests are rejected per mode; this is not
a claim that raw `Chess.make_move` validates arbitrary moves.

The full wrapper preserves the previous 150-law chain. A modular qualification
retains that exact-source evidence with the new 2-law/17-control gate; it must
not describe a full 152-law/453-control aggregate as newly executed.

Checker/Base, native lowering/storage, ABI, toolchain, OS and hardware remain
trust boundaries. Legal en passant and producer/king-safety connections remain
separate obligations. No production logic moves from Python in this proof-only
increment.
