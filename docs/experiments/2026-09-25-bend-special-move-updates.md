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
