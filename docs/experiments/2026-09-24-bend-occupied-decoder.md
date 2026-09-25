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


## Completed hosted qualification

Hosted run **36083420058** passes all four universal laws, importing consumer, eighteen classified controls, four native modes and three actual corruption witnesses, original compiler checks and unchanged whole-repository lint on source `4c5232cc1e6fceb8adc66dcc2dd5479b373ead3f`.

The focused command executed all four public laws, importing consumer and eighteen controls. Nine are ordinary semantic/refinement rejections: seven implementation/specification corruptions and two concrete false-premise consumers. Eight enforce manifests/import policies; one synthetic unsafe-warning output unit is not another compiler execution. Missing files, malformed terms, affine-use errors, crashes and timeouts are not counted as semantic rejection.

The unchanged exact-source123-law/332-control parent is retained, giving modular127-law/350-control evidence. The combined127-law wrapper was not executed. All391 parent manifest entries and its successful qualification job/report were checked. All404 candidate native-source hashes still match after execution.

Every native mode passes1,920 observations and21,120 fields:832 square/local-state combinations,768 fresh insertions,64 initial-board queries and256 random consistent-board queries. The1,751 occupied cases also reconstruct all eight plane bits from actual decoded kind/color. The169 empty cases record raw fallback5 outside the occupied contract. Nine malformed requests are rejected by the bounded probe. Three actual decoder/occupancy mutations compile/run before failing occupied inputs; no empty-only witness is used to claim occupied-law sensitivity. Modes repeat fixtures, not disjoint or exhaustive boards.

The reference uses an independent64-square optional kind/color model. Candidate code uses actual Chess reads and Position.put/start, never proof classifiers or expected output labels. This is selected-square observation, not full-Board mutation checks, metadata validity, parser acceptance or legal moves. The guarded total observation lives only in proof code; no production empty-square behavior changed.

Hosted Bun1.4.2/Clang18.1.3; Clang is scoped to native checks only. Locked Python3.13/uv0.12.10 CPU tools use the normal build compiler. Ruff/Basedpyright/Vulture passes unchanged, resolving the retained local missing-tools gap. Original compiler16-laws/seven-controls and12 pin tests pass. The final local focused command passed in49.437seconds; prior enclosing tool-command timeouts are incomplete runs, not proof rejections. No public law changed during that harness recovery.

Only documentation, index and evidence are added after qualification. Four universal statements cover arbitrary globally consistent Boards and all bounded squares, with actual occupancy required only for the raw-decoder claim. Independent classification and encode/decode reconstruction are pointwise, not a global abstract-board bijection. Metadata and legality remain unconstrained.

Self-review only, not independent review. Pinned checker/Base, native lowering/storage,ABI,toolchain,libraries,OS and hardware remain trust boundaries. No production runtime, earlier proof, compiler input, permanent workflow or routine perft budget changes. No model/GPU, search, training or benchmark added. No Python application responsibility moved; export/references/data/control/training and transitional C++/LibTorch/AOTI remain dependencies. Nothing merged,force-pushed,deployed or changed in a live process.

Next P3 acceptance is parser-derived freshness and whole-board abstraction, followed by invariant-preserving removal/move/special-move operations. King safety, legal generation soundness/completeness/no duplicates, reachability and perft remain separate. Previous P2 results and the direct closed-builder equality limitation remain unchanged.
