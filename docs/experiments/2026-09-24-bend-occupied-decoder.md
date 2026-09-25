# Actual occupied-square decoding from the board partition

## Baseline and acceptance

Base is #878, `1337d5a30e42a1b2999819a36e3b538ace3e7071`, complete tree
`f4bf89c8bb0cfd266d3438a3707795c5fa5d5281`. Repository instructions,
development guidance, branch lifecycle and experiment index were read. The full
baseline was reconstructed from its retained archive plus the exact final review
file; its original commit and all 391 native-source manifest entries match.
The refreshed branch and compiler pin are unchanged. No review comments or
existing occupied-decoder branch were found at refresh.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 plus
U64, 84 inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

This record follows constructive development and precedes hosted qualification;
it is not a backdated preregistration. Acceptance requires all four public laws,
importing consumer and eighteen classified controls, four native modes and three
actual wrong-behavior mutations, original compiler source/pin gates, unchanged
repository lint, and verified exact-source parent receipts. Full-chain execution
must not be inferred from retained modular evidence. Work is isolated, bounded
and opt-in; no model/GPU, search, perft, training, deployment or benchmark action.

## Four universal source contracts

| Law | Actual guarantee |
| --- | --- |
| `square_projection_valid` | A globally consistent Board yields a valid local row at every square below64. |
| `abstract_square_roundtrip` | Independently classify and encode the square to reconstruct all eight real bit observations. |
| `guarded_decoder_matches_square` | The explicit occupancy-guarded observation using real Chess operations equals the independent Empty/Occupant classification. |
| `occupied_decoder_matches_square` | On an actually occupied square, real Chess.piece and the white bit identify its independent occupant. |

The invariant and bounded square are explicit input conditions. The raw decoder
law additionally requires actual occupancy. Metadata is arbitrary; no desired tag,
color, decoded cell or output-equality premise is assumed. The independent
classifier uses the thirteen valid states, never the optimized decoder's fallback.
The proof-only guarded wrapper is not a new production function. Raw empty-square
Chess.piece behavior is unchanged and lies outside its occupied contract.

The proof projects the global bitboard invariant through a structural Word
observation. A new lemma relates those observations to the actual U64.test_bit,
including indices in both limbs. The row proof checks all Boolean combinations
inside Bend and returns both reconstruction and decoder certificates. A shared
finite proof replaces duplicate row-case proofs without changing their statements.
The consumer composes the laws to reconstruct all eight bits from actual occupied
kind/color outputs, using actual initial-board and fresh-insertion examples.

This is pointwise abstract-square correspondence, not yet a full Board/abstract
board bijection, legal position, metadata validity, parser freshness or move theorem.
The inherited FIDE-2023 reference remains unchanged; no new normative-rule claim.

## Local execution and development record

The final four-law consumer passes with exact safe checker output. Native generic,
portable, native-target and UBSan builds pass 1,920 square observations and 21,120
fields per mode. Fixtures contain 832 square/local-state combinations, all768
fresh square/kind/color insertions, 64 initial-board queries and 256 additional
random consistent-board queries: 1,751 occupied rows and169 empty rows. Every
occupied decoder result reconstructs the observed eight-bit row. All nine malformed
probe requests are rejected per mode. Three actual pawn-tag, king-fallback and
occupancy-combination corruptions compile/run before rejection on occupied inputs.

The native reference is an independent array of optional (kind,color) values.
Candidate code executes only real Board construction, Chess reads and Position
insertion/start; it receives no expected decoder label or proof-model output.
Modes repeat fixtures, not disjoint or exhaustive arbitrary-board coverage.
Square observations do not compare every complete Board or prove nonmutation
of unrelated fields; they are independent runtime checks of the stated decoder
behavior, not whole-game or parser acceptance checks.

The final focused gate and its exact controls are recorded in the hosted/local
receipts. Earlier enclosing command timeouts left only partial control results;
these are not passes. The control harness now retains concrete empty and
inconsistent-board false-premise witnesses rather than demanding expanded
symbolic error messages. No law or intended domain was changed. Seven controls
mutate implementation/specification behavior; two reject those false concrete
premises; eight enforce manifests/imports; one synthetic warning-output unit is
not a compiler execution. Intended semantic diagnostics remain mandatory.

Initial bit-projection drafts failed explicit duplication/pattern and empty-domain
elimination details. A native-probe draft omitted explicit square duplication and
was rejected before C generation. These were corrected without modifying the
compiler or accepted parent sources. Original failure logs remain in the review
package. Local original compiler16-law/seven-control and12 pin checks pass.
Local unchanged lint failed because Ruff/Basedpyright/Vulture are unavailable;
that nonzero result remains historical and requires fresh hosted qualification.

## Remaining obligations and trust

Next decisive P3 work is parser-derived fresh-square insertion and whole-board
abstraction, followed by removal, move application and special-move invariants.
King safety, legal-move generation soundness/completeness/no duplicates, reachability
and perft remain separate. Prior P2 results and the direct closed-builder equality
limitation remain unchanged.

Self-review only, not independent review. Pinned checker/Base, native lowering,
storage, ABI, C/C++ toolchain, libraries, OS and hardware remain trusted boundaries.
No production code, earlier accepted law, compiler input, permanent workflow or
routine test/perft budget changes. No Python application responsibility moved;
export/references/data/control/training and transitional C++/LibTorch/AOTI remain.
No merge, force push, deployment or live-process action.
