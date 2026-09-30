# Whole-board abstraction and reconstruction

This opt-in P3 continuation is based on occupied-decoder PR #879 at
`d8629e31a55ceda25edef836f8f34a895416dba4`. No engine representation is replaced.
The new proof interface observes every square and retains all metadata.

```sh
bun native/bend_engine/standalone/proofs/abstraction/focused.js /path/to/pinned/bend --report /tmp/abstraction-proofs.json
bun native/bend_engine/standalone/proofs/abstraction/verify_native.js /path/to/pinned/bend --report /tmp/abstraction-native.json
# Full inherited chain is optional and significantly more expensive:
bun native/bend_engine/standalone/proofs/abstraction/verify.js /path/to/pinned/bend --report /tmp/abstraction-aggregate.json
```

## Public contracts and actual correspondence

`whole_board_roundtrip` proves `restore(observe(board)) == board` for every
actual Chess.Board satisfying the existing partition invariant. The equality
contains all eight U64 planes and all three raw U32 metadata values, not just
occupied squares. No expected reconstruction is supplied by the caller.

`Snapshot` is two ordered 32-cell lists (low limb first, bit zero first) plus
turn, castling rights and en-passant metadata. Each cell uses the previously
independent Empty/Occupant/Invalid classifier. `snapshot_has_64_squares` proves
each list has length 32 for every input board, even an inconsistent one.

`snapshot_square_correspondence` independently links a bounded abstract query
to the actual eight bit observations used by the existing square classifier.
It requires only square < 64, not consistency. This detects orientation errors
that could otherwise be hidden by matching wrong encoders and decoders.

`equal_snapshots_identify_board` derives injectivity on consistent actual Boards
from the complete round trip. Arbitrary metadata participates in snapshot equality.
It is not an unrestricted two-way bijection on arbitrary Snapshot lists or raw
Occupant tags. `restore` is total on malformed lists, but no inverse, legal
position, or validation claim is made for those lists.

`empty_square_has_empty_observation` recovers the explicit empty-square result
from the saved local decoding candidate using #879's published producers. It
requires global consistency, square <64, and actual occupancy false; then all
eight actual observations are false. It does not equate the raw king fallback
with an empty tag or change Chess.piece.

The proof inducts over structural words and transfers the already-proved valid
row encode/classify inverse to all columns. At the Board boundary it handles
both actual U64 limbs and the full Board constructor. The square-order proof
connects list indexing through the existing actual test_bit refinement. No new
axiom, unsafe dependency, foreign equality witness, or altered prior law.

## Native scope and a compiler limitation

The native verifier executes actual Board operations and observes **every square**
with real Chess.occupied/piece/getters. It compares all 19 raw fields and 128
kind/color observations with an independent 64-square-array reference, and also
reconstructs all eight bitboards from the observed cells. Four build modes repeat
the same 1,026 boards; this is 65,664 square observations and 150,822 output-field
comparisons per mode, not exhaustive board/game coverage. All 768 fresh insertion
combinations, 192 mixed boards, 64 metadata updates, empty and start are included.

The structural `Spec.observe/restore` proof interface is **not executed by that
native probe**. A direct lowering probe failed with `an arity over 255` on the
pinned compiler. Its exact source and diagnostic are archived in the dated
record's evidence. Source correctness is established; native compilation of the
structural snapshot remains unsupported by this tested path. No checker or
compiler was changed to disguise the limit.

Malformed requests are rejected by the bounded test probe, not by a proved raw
Chess API validator. Two disposable programs compile and run before an actual
wrong pawn tag and skipped observation square are rejected. No model, table build,
search, training or increased perft work is added.

## Qualification and limits

The focused gate checks all five laws and an importing consumer, then eight
ordinary source/refinement controls, eight manifest/import controls and one
synthetic unsafe-warning wrapper unit. Crashes, timeouts and malformed imports
are not semantic rejection; exact `All terms check.` is required for source
success. `verify.js` retains the unchanged full #879 chain. Modular qualification
may retain its exact-source 127-law/350-control evidence, but must not claim a
fresh full 132-law/367-control wrapper run.

These results advance whole-board correspondence under representation consistency.
They do not prove metadata legality, king counts, reachability, parser-derived
freshness, move updates, special moves, or legal-generation completeness. The
pinned checker/Base and native lowering/runtime/toolchain remain separate trust
boundaries. No Python application responsibility moves in this proof increment.
