# Attack witnesses and single-king query selection

Opt-in source/refinement suite on the existing pinned compiler. No production code changes.

## Public contracts

1. `mask_reduction_has_square_witness`: for any raw Board, attacker-side U32 and
   five supplied reverse masks, the arithmetic reducer is exactly a per-square
   existence scan over owned pieces of the matching kinds. Queens may witness
   either slider mask; no Board consistency or non-overlap premise is needed.
2. `actual_attacked_uses_witness_reduction`: actual `Chess.attacked` equals
   `Queries.run`, including the complete returned array and Boolean. Each actual
   pawn/knight/king/rook/bishop query and callback is connected in `Flow.bend`.
3. `single_king_in_check_uses_witness_reduction`: when the selected side's king
   bits equal exactly one bounded square bit, actual `in_check` equals the witness
   query at that square for the other side. The singleton and square bounds are
   explicit premises, not facts inferred from representation consistency.

`Spec.scan` is structural existential reduction across the 32 bits of each limb;
`Spec.row` asks whether the selected color has a matching piece/mask witness there.
It is not a priority decoder and does not use production `attacked_*` callbacks.
The bit arithmetic is justified by structural Word induction, not sample enumeration.

## Important boundary

`Queries.run` still calls actual `Chess.attack` to retrieve masks. This increment
proves their aggregation/query plumbing, **not that those masks have universally
correct coordinate geometry**. No correct-table premise is hidden: arbitrary affine
arrays are allowed because the two paths make the same sequence of actual queries.
Their final arrays are equal to each other; neither is proved equal to the input array.

Raw U32 side handling follows the engine's white-if-one/otherwise-black convention.
Native requests restrict the attacker to 0/1 and the queried square to 0..63.
No generated-castle singleton-king invariant, king-safety theorem, historical rights,
metadata legality, reachability or complete generator correctness follows.

## Execution

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/attack_witness/consumer.bend
bun native/bend_engine/standalone/proofs/attack_witness/focused.js /path/to/pinned/bend
BUN=bun CC=clang python native/bend_engine/standalone/proofs/attack_witness/verify_native.py \
  /path/to/pinned/bend --report /tmp/attack-witness-native.json
```

The focused command runs all three laws, an importing consumer, ten semantic
mutations, eight import/manifest controls, and a synthetic warning-output test.
Missing inputs, parse/ownership errors, timeouts and crashes are not semantic rejection.

The native probe builds current actual tables and observes five masks, `attacked`,
and bounded `in_check`. An independent forward coordinate/set oracle supplies
expected results externally. The proof-only structural scan is not compiled by
this probe. Native modes repeat fixtures rather than contributing disjoint datasets.
A missing king prints diagnostic sentinel 2 without calling out-of-domain `in_check`;
that sentinel is probe behavior, not a production API return value.

The consumer explicitly preserves no-king and two-king selection boundaries and
wrong-color/diagonal-queen witnesses. The native fixture set additionally retains a
pinned-knight attack and a higher attacked king that `ctz` does not select when two
same-side kings exist. These are domain/regression cases, not legal-game claims.
