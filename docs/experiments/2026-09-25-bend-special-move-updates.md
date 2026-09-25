# Ordinary, promotion and en-passant update qualification

## Scope before hosted qualification

Base is PR #882 at 3ddb2be9d49d46d3848a35e15732adf1ec370c14. This publication
preserves all sixteen primary files from saved ordinary-move commit
06879a1dfc004eee143e623eafce3d2edddd1928 byte-for-byte. That local commit and its
complete historical review package remain preserved; this source commit is not
claimed to be the same Git object. The archived FEN consistency consequence and
its previous receipts remain separate from move-law counts.

Four new universal contracts add exact typed-promotion (tags1..4,flag0) and
en-passant (promotion0,flag1) updates and representation preservation. Actual
make_move is used, all bitboards/metadata are included, and current rights/EP
metadata helpers remain explicit dependencies. These are not legal-move rules,
king-safety, valid-metadata or castling proofs. Source indices/metadata are raw
U32; native endpoints are restricted to0..63. Input consistency is needed for
preservation, not for the exact-update equalities.

The new target-freshness lemma derives emptiness after the actual removal mask,
including the en-passant victim at destination XOR8. Typed promotion labels have
a separate closed witness. Tag6 is outside the domain and has a counterexample
leaving color without any kind. Both results reuse the previous ordinary clear,
scalar-update and board-partition proofs without changing those files.

Before native execution, the new four-law consumer passed with exact safe output.
The first freshness-bit draft failed because Bool.and on symbolic inputs did not
reduce to the proposed equality. Splitting all sixteen Boolean input cases
resolved that proof without changing its statement or the checker. A streaming
execution facility failed before running any command; a bounded file-logged
supervisor then completed the new focused and native checks. Original receipts
are preserved in the conversation review package.

Acceptance is a fresh hosted run of the unchanged five-law/18-control ordinary
gate, new four-law/17-control gate, both four-mode native gates, original compiler
source and pin tests, and unchanged repository lint. The exact141-law/401-control
parent is retained only after manifest and successful receipt checks. The full
150-law wrapper is opt-in and will not be implied by these modular checks.

No production runtime, prior accepted law, compiler input, permanent workflow or
perft budget changes. No model/GPU, search, training or benchmark run. No merge
or deployment. Source/checker/Base, native lowering/storage, ABI, toolchain and
hardware remain trust boundaries. Self-review only, not independent review.
Python export/references/data/control/training and transitional C++/LibTorch/AOTI
remain dependencies; no new application responsibility moves into Bend.

Next separate work: castling representation under explicit target/rook
conditions, then legal-move producer and king-safety connections. Ordinary,
promotion and EP representation consistency alone never establishes legality.


## Completed hosted qualification

Hosted run **36160654909** passed both focused source gates, all35 controls across them, both four-mode native suites and four behavioral mutations, original compiler checks and unchanged repository lint on source `1c45eae81922a6364cc98b01163e279036d799b9`.

The five ordinary laws and eighteen controls are preserved from saved commit06879a1dfc004eee143e623eafce3d2edddd1928 and freshly reexecuted, not five new mathematical results. Four new universal laws and seventeen controls cover typed promotion tags1..4 with flag0 and en-passant flag1/promotion0. Every result concerns the actual complete Board, including metadata. The saved ordinary README status paragraph is historical local-only status, superseded by this dated hosted record; its primary bytes were deliberately retained unchanged.

The new exact-update and partition-preservation theorems derive clearing and target freshness from the actual source/destination/victim masks and reuse the earlier insertion theorem. En-passant removes the actual destination XOR8 victim. Typed promotion rejects the invalid rawtag6 domain; the consumer pins all four type labels independently. No caller assumption supplies desired output, fresh target or correct captured piece.

These are representation results, not move legality. Source occupancy, pawn identity, promotion ranks, legal victim, turn ownership, king safety and independent castling/EP-right semantics remain separate. Metadata expressions retain the existing implementation-linked helpers. Castling is not covered, and arbitrary rawU32 source endpoints do not become legal squares. The native domain restricts endpoints to0..63.

Exact-source parent141/401 is retained after444 manifest checks and the separately failed-source/successful-recovery job receipts. New execution of saved5/18 plus new4/17 gives modular150 laws/436 controls; the complete150-law wrapper was not executed. The two focused commands each executed their own importing consumer and controls. Semantic failures, manifest/import policies and synthetic warning units remain separately classified; crashes and timeouts are not semantic rejection.

The ordinary native suite passes6,656 complete Board cases/126,464 U32 fields per mode and nine invalid requests. The new suite passes1,612 complete Board cases/30,628 fields per mode and eleven invalid requests, including176 promotion-shaped,28 en-passant-shaped,512 raw promotion,512 raw en-passant,256 metadata and128 inconsistent-input cases. Modes repeat fixtures; these are not disjoint or exhaustive legal-game datasets. Candidates execute production code, not proof models or supplied reference decisions.

Actual ignored promotion and wrong en-passant capture-square mutations compile and run before incorrect Board values are rejected. The saved no-op and retained-bit regressions pass again. These are deliberate corruption controls, not production bugs. The independent external set-based reference checks every bitboard and metadata field but does not constitute a universal FIDE metadata proof.

The pinned compiler and84-input fingerprint are unchanged. Hosted Bun1.4.2 uses the installed system Clang only for native tests. Locked Python3.13/uv0.12.10 CPU tools use the normal build compiler. Original16-laws/seven-controls and12 pin tests pass. Fresh unchanged Ruff/Basedpyright/Vulture resolves the saved ordinary local missing-tools gap without rewriting the old failed receipt.

All472 candidate hashes are checked before publication. Only evidence, index and inventory are added afterward. The saved complete local package, its supplementary FEN consistency adaptation and historical logs remain separately preserved; that adaptation is not newly counted or requalified by these move suites. New local Fresh.bit first failed Boolean reduction and was corrected with a complete finite Boolean proof without changing its claim or compiler.

Self-review only. No production runtime, earlier accepted law, protected checker, permanent workflow, routine perft, search, model/GPU, training or benchmark change. No additional Python application responsibility moves into Bend; export,references,data/control/training and transitional C++/LibTorch/AOTI remain dependencies. Native lowering/storage,ABI,toolchain,libraries,OS and hardware remain trust boundaries. Existing compiler/snapshot/closed-builder limitations remain unchanged. No merge,force push,deployment or live-process action.

Next substantive P3 acceptance is castling under explicit source/destination/rook-storage conditions, then connection to legal-move producers, correct chess metadata and king safety. The present promotion and en-passant partition results do not establish legal special moves.
