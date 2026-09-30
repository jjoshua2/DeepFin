# Actual final-generator castling provenance

This opt-in suite extends the single `castle_side` result from PR #891 through the
actual ordinary piece scan, both castling-side calls, and both branches of the final
legal-move filter. It does not replace the production generator.

## Public contracts

| Law | Actual function and guarantee | Premises |
| --- | --- | --- |
| `ordinary_scan_excludes_castling` | Every move from `Chess.scan(keys,b,(table,Nil{}))` has a flag unequal to 2. | Arbitrary keys, Board and affine table. |
| `final_filter_preserves_member_origin` | A member of the actual `Chess.filter_prepare` output occurs in its input list. | Actual result membership; arbitrary Board/table/list. |
| `filtered_castling_suffix_origin` | After the two actual side calls and final filter, a member is from the original list or is an exact guarded castling move. | Actual suffix-result membership; arbitrary input list. |
| `legal_castle_member_has_producer` | A flag-2 member of actual `Chess.legal_moves` has the selected producer guard and exact complete Ply for one of the two sides. | Actual final-result membership and flag equal to 2. No caller-tail-absence or assumed scan property. |
| `legal_castle_member_preserves_representation` | Applying that actual final-generator castling member preserves the Board partition. | The preceding membership/tag premises and input Board consistency. Rook-target freshness is derived. |

`Spec.castle` is a disjunction of two explicit certificates. Each contains the actual
producer's input guard and equality to its computed complete Ply, including promotion
zero and flag two. The guard contains the selected rights bit, owned source-king and
corner-rook bits, and empty between path. Existing `castle_emission/Guard.bend`
derives rook-target freshness and invokes the previously accepted castling update law.

The generic helper statements preserve any pointwise list predicate. Dependent
continuations keep actual affine arrays single-owned while composing actual calls.
`Chain.legal` is indexed by imported `Chess.legal_moves`; the result is not established
only for a model generator. The source certificate does not prove returned-table
identity, physical allocation or native lifetime.

## Boundaries and examples

The initialized ordinary scan starts from Nil and cannot introduce flag 2. Therefore
the full-generator laws need no assumption that an arbitrary caller's tail lacks the
queried move. In contrast, the arbitrary-tail suffix law retains the explicit
old-member alternative. A final filter alone can retain an invalid preexisting
castling entry; the consumer preserves that counterexample.

Source satisfiability examples use a small arbitrary zero table and show both castling
routes, their order, and a derived update-preservation instance. That table is not
presented as correct attack data or a legal chess position. Native tests instead build
the actual current `Tables.build()`.

These are provenance and representation results, not semantic king-safety, historical
rights, independent metadata, full legal-move soundness/completeness or no-duplicates
proofs. Input Boards, metadata and source tables are arbitrary in the source laws.
The native reference uses consistent Boards with exactly one king per color; those
extra test-domain conditions are not hidden formal premises.

## Reproduction

Use the unchanged compiler root at
`jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, with Bun 1.4.2.
Its 84-file source fingerprint is
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Run from the repository root:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/castle_chain/consumer.bend
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/castle_chain/focused.js \
  /path/to/pinned/bend --report /tmp/castle-chain-focused.json
BEND_NO_TELEMETRY=1 CC=clang bun native/bend_engine/standalone/proofs/castle_chain/verify_native.js \
  /path/to/pinned/bend --report /tmp/castle-chain-native.json
```

The focused gate requires exactly `All terms check.` and status zero. Its 18 controls
comprise nine intended semantic/refinement failures, eight manifest/import policy
checks and one synthetic warning-output unit, which is not a compiler execution.
Crashes, missing imports, parser/affine failures and timeouts do not count as semantic
rejection. Every production mutation is made in a disposable copy.

The native driver builds generic, forced-portable, native-target and UBSan executables
from the real tables and generator. It checks 1,062 distinct requests per mode:
386 final-filter, 386 filtered-two-side suffix, and 290 full-generator requests.
Exact ordered lists are checked for the first two operations. For full `legal_moves`,
the complete output is inspected but only the ordered castling subset has an
independent move-set oracle. Every returned move's complete child Board is separately
compared with a square-set raw-update reference, including noncastling moves and old
input-tail entries. This does not qualify the full noncastling legal move set.

Each mode compares 9,241 complete child Boards / 175,579 U32 Board fields and rejects
seven malformed batches. Four destination-only attacked castling examples must be
rejected by the final generator, unlike the earlier single producer. Three actual-code
regressions test forged scan-tail castling, duplicated kingside production, and disabled
destination rejection. They must compile and execute normally, then fail independent
output comparisons. Modes repeat fixtures, not disjoint or exhaustive legal games.

No permanent CI, model/GPU, search, perft, training or performance work is added.
See the dated experiment readout and committed local receipts for executed results,
lint/publishing limitations, provenance and trust boundaries. Self-review only.
