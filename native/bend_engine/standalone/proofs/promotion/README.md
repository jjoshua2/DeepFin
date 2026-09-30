# Typed promotion update and enumeration

This opt-in suite extends the unchanged `move_update` proofs. It uses the actual
`Chess.make_move` and `Chess.put_move` functions; no production code is replaced.

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/promotion/focused.js /path/to/pinned/bend --report /tmp/promotion-source.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/promotion/verify_native.js /path/to/pinned/bend --report /tmp/promotion-native.json
# Optional complete inherited chain; not implied by the focused checks:
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/promotion/verify.js /path/to/pinned/bend --report /tmp/promotion-aggregate.json
```

Use compiler aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae and its unchanged manifest.
The tests default to Clang; CC may select a compatible C compiler. Do not suppress
unexpected diagnostics. Native commands are bounded and preserve all four modes.

## Public statements

`promotion_choice_roundtrip` connects typed Knight/Bishop/Rook/Queen choices to
raw tags 1/2/3/4. Neither pawn nor king is a promotion choice.
`promotion_choices_exact` relates actual `put_move(True,...)` to exactly these
four entries in order, retaining the arbitrary caller tail and flag-zero encoding.
It is not completeness or nonduplication of the entire legal move generator: the
input tail may already contain duplicates and legality filtering is separate.

`promotion_make_move_exact_update` proves the complete actual flag-zero promotion
result equals clearing the source and destination planes, inserting the selected
piece of the side-to-move color, and applying the existing metadata update.
`promotion_make_move_preserves_representation` proves the output partition when
the input Board has the established independent partition invariant.

All four statements are universal in their declared domains. Board/occupancy and
metadata are arbitrary; source/destination are raw U32 values following the actual
bit constructor, not modulo-64 assumptions. The native probe restricts squares to
0..63. Promotion tag validity is provided by a four-constructor type. The exact
update law does not require input consistency; preservation does.

## Proof and boundary

`Actual` unfolds the real promotion dispatch, proves the chosen tag overrides the
source kind, and reuses the existing Core bridge, erase/insert proof and metadata
frame. It does not copy those proofs or postulate freshness. Rights and EP behavior
use the original source piece kind and the current helpers; this is not an
independent legal-metadata theorem. There are no new axioms, unsafe/foreign proofs
or holes. Every previous law and gate remains unchanged.

The raw API is not a move validator. Empty sources, wrong ranks, mismatched side,
or identical source/destination can be representation-consistent updates. The
consumer preserves an empty-source example and a real-shaped capture-promotion
example. King safety, legal pawn movement, last-rank requirements, en passant,
castling, clocks/history and reachability are not established here.

## Validation

The source gate checks all four LAWS obligations, PROOF implementations, importing
consumer and 17 controls. Eight controls require ordinary semantic/refinement
failures, eight enforce manifests/import safety, and one is a synthetic unsafe-
warning output-wrapper test (not a compiler execution). Missing imports, crashes,
affine errors and timeouts do not count as valid semantic rejection. Success
requires status zero and exactly `All terms check.`.

The independent native reference uses 64 square sets of piece kinds and colors;
it compares all 19 raw Board fields. Each build mode checks 16,944 distinct Board
inputs, including all 4,096 source/destination pairs for all four promotion tags.
These are raw updates, not all legal promotions. Other cases include 176 final-rank
pawn geometries, 256 arbitrary-metadata cases and 128 inconsistent diagnostics.
The actual generator probe checks 256 request tuples with a nonempty sentinel tail,
returning five list entries per request. Ten malformed Board requests and six
malformed generator requests are rejected per mode.

Generic, forced-portable, native-target and UBSan repeat fixtures. They are not
disjoint datasets or exhaustive arbitrary Board/U64 coverage. No proof predicate,
shadow engine, oracle output, model or attack table runs inside either candidate.
Three disposable production mutations ignore promotion, force every choice to a
queen, or duplicate a knight in the choice list. They must compile and execute,
then fail as value mismatches. Source checks additionally reject list truncation,
dropped tails, malformed choice encoding, turn corruption and missing premises.

The complete aggregate remains opt-in. Local focused and native success does not
imply hosted qualification, independent review or a passing repository lint gate.
See the dated readout for exact receipts, inherited evidence and unresolved limits.
