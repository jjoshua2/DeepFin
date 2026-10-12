# Actual multi-bit scan_after full-Ply count

This increment is based on the verified PR1000 head
`1690865e2fc59091d0b8947364fb2a5e64ea40b9` in the isolated branch
`proof/bend-scan-after-20261003`. The official GitHub archive URL is
`https://codeload.github.com/jjoshua2/DeepFin/tar.gz/1690865e2fc59091d0b8947364fb2a5e64ea40b9`.
Its SHA-256 is `f041a81c1eb9f545fe1da622b9d33e0c58a2deb0180f20d4e7033f2b7e177d3a`;
the extracted Git tree hash is `c4c5898a2f34bf10cecaaaedd4e1a353db014fdb`,
matching the exact tree reported by GitHub for that commit. The reconstructed
unsigned commit object hash matched the same exact commit SHA.

## Checked result

The new `AfterCount.bend` module exposes the scan count factorization and PR1000 key frequency/range results as separate guarantees:

- `Scan.exact` proves the actual `Chess.scan_after(src,pawn,ep,tail,(table,targets))`
  returns the identical input table paired with the exact expanded move list.
- `Count.destinations` factors full-`Ply` occurrences through the actual
  `Chess.destinations`, including source, destination, promotion and flag fields,
  and adds the count of the caller tail.
- PR1000's `AllEmittedSound.actual_bit_squares_frequency_all_u32` and
  `actual_bit_squares_below64` establish exact frequency and range for the actual
  64-step `Chess.bit_squares` output for every U32 query.

The importing consumer states that for arbitrary U64 target masks, arbitrary U32
source/EP/query fields, either pawn flag, any input table, and any Ply tail,
`pair_count(actual_scan_after, query)` equals `Spec.tally(actual_bit_squares,
src,pawn,ep,query,Spec.count(tail,query))`. A repeated identical Ply in the tail
has count two. No uniqueness premise on the tail is used.

The field cases were checked against `legal_probe/Chess.bend`: only an actual
emitted destination is counted; promotion is selected for pawn destinations whose
computed rank is 0 or 7 and `put_move(True,...)` emits promotion values 1, 2, 3,
4 with flag 0. Other destinations emit promotion 0 and
`Bool.to_u32(pawn && dst == ep_sq)`. The accepted promotion consumer
`Promotion.promotion_choices_exact` supplies the checked expansion body. The
proof does not claim that arbitrary target masks are legal board targets or that
the full legal-move generator is complete.

The count runner pins Bend 2.0.21 + U64 at revision
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, checks all 84 compiler files against
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`,
and uses Bun 1.4.2. Run receipts, raw checker streams, resource logs, hashes, and
semantic negative controls are in
[`evidence/2026-10-03/scan-after-count`](../../native/bend_engine/standalone/proofs/destination_factorization/evidence/2026-10-03/scan-after-count/).

## Remaining boundary

The all-U32 frequency result says each queried set bit appears once in the actual
square list and an unset/out-of-range query appears zero times. The count theorem
keeps that actual list as the fold in `Spec.tally`; the frequency/range guarantees
are separate companion results, not premises used to rewrite the tally to a
closed bit-predicate form. The work introduces no board-validity or target-geometry
premise and makes no complete legal-move correctness claim. Move filtering, attack geometry,
history, and full legal-move correctness remain separate.
