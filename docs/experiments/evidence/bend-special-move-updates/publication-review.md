# Ordinary and typed special-update publication review

## Exact qualified source and reconstruction

Hosted run **36160654909**, job **108156093911**, completed every stage on its first attempt. The saved ordinary five-law/eighteen-control gate and new special four-law/seventeen-control gate were freshly executed, followed by both four-mode native suites, original compiler source/pin checks, unchanged repository lint and publication.

Qualified implementation: `1c45eae81922a6364cc98b01163e279036d799b9`, complete tree `e403edc54009abf5ff0165ebb835e47fa3aea967`.
Evidence head: `dc5aa5f3212d9bfd20e299191bb916c81b701aba`, tree `00cd23dd4c0fbe8533c2b141272a9be86ce63701`.
Parent: #882 at `3ddb2be9d49d46d3848a35e15732adf1ec370c14`, tree `dad729630fe773c081e1276cbcb4311ffdb8450e`.

Artifact **10875079609**, ZIP SHA-256 `e4f86b6b907ed485996c5077e1446b0f25b6b01f409b8a7475cc5af36fe4f4f4`, was downloaded and verified. Its complete archive reconstructs all **3,765 tracked paths**, the evidence tree and original commit object. The complete original parent was separately reconstructed from its prior archive plus its unchanged final documentation file: **3,724 paths**, exact parent tree and original commit object. Applying the verified 29-file publication payload in a separate index of that exact parent reproduces the entire qualified source tree. Its original source commit object was also reproduced exactly.

All **472 candidate native-source hashes** match the inspected local candidate. All16 ordinary-suite files are byte-identical to saved local commit `06879a1dfc004eee143e623eafce3d2edddd1928`; the12 special-suite files are the newly checked implementation. The saved full local patch, bundle, historical readout and supplementary FEN adaptation remain preserved in the original review package. This publication does not pretend its different source commit is the original local commit, or count the five saved laws as new mathematical discoveries.

The transmitted UTF-8 JSON payload was153,359 bytes, SHA-256 `90945c78e7fe21b2d41779700e4a1075857dc5d2369b9374000bf3956812db99`. Four transport chunks were verified against their Git object identities before application. It contains only the28 primary proof/test files and the initial dated record. No workflow or transport payload is in the feature diff.

The new focused report agrees locally except measured consumer seconds. Both complete native reports agree with their respective local reports except the recorded compiler version. Local Clang17 and hosted Ubuntu Clang18.1.3 are separately identified; original reports were not rewritten. The saved ordinary source consumer took1.784 seconds hosted, the new special consumer3.723 seconds. These are execution receipts, not performance claims.

## Formal domains and actual behavior

`typed_promotion_exact_update` and `typed_promotion_preserves_representation` cover actual Chess.make_move with flag0 and typed Knight/Bishop/Rook/Queen tags1..4. `en_passant_exact_update` and `en_passant_preserves_representation` cover flag1/promotion0 and the actual destination XOR8 victim square. The exact statements return the complete Board, including metadata. Partition-preservation requires only the existing input representation invariant; it derives clearing and target freshness, not a desired output premise.

This broad raw-call domain is not chess legality. An empty source can still invoke the decoder fallback. Source/destination ranks, pawn identity, moving-side ownership, victim identity, legal en-passant metadata and king safety are not proved here. The formal endpoints are raw U32 values; native probes restrict them to0..63. Metadata expressions retain current implementation-linked rights and EP helpers rather than establishing independent FIDE metadata correctness. Castling is excluded.

The typed promotion labels are independently pinned in the consumer. Raw promotion6 can create an invalid partition, so the type restriction is essential. The en-passant clearing proof includes source, off-destination victim and destination; it does not require a caller-supplied empty destination. A no-op would preserve representation, which is why exact-update and behavioral rejection are separate requirements.

The two focused commands passed nine registered laws and35 controls across them. Exact-source parent141/401 is retained, producing **modular150-law/436-control coverage**, not a newly executed full150-law wrapper. The saved ordinary gate has nine semantic failures, eight policy checks and one synthetic warning unit. The new gate has eight semantic/refinement or closed-witness failures, eight policy checks and one synthetic warning unit. Synthetic tests are not compiler executions; crashes, missing imports and timeouts are not accepted semantic rejection. Safe checking requires status0 and exactly `All terms check.`.

## Native and lint qualification

Both suites execute actual Chess/Position functions, not the proof model. Independent external64-square sets produce expected complete Boards. Each ordinary mode passes6,656 records/126,464 U32 fields and nine malformed requests. Each special mode passes1,612 records/30,628 fields and eleven malformed requests. Generic, portable, native-target and UBSan repeat the same fixtures; the datasets are not claimed disjoint or exhaustive legal games.

Special fixtures include176 promotion-shaped and28 en-passant-shaped updates,512 raw promotion cases,512 raw en-passant cases,256 metadata cases and128 inconsistent-input diagnostics. The1,484 consistent inputs and128 inconsistent diagnostics are separated; raw calls need not be legal moves.

The actual ignored-promotion mutation compiles/runs and fails promotion-shaped row0/field1:2255873 rather than2255872. The actual wrong-victim mutation captures at destination instead of destination XOR8 and fails en-passant-shaped row0/field1:1108000768 rather than1074446336. The saved ordinary no-op and uncleared-bit regressions also passed again. These are deliberate test corruptions, not production bugs.

Original compiler16-law/seven-control checks, including cyclic-template rejection, and all12 compiler-pin tests passed hosted. Unchanged Ruff/Basedpyright/Vulture passed with locked Python3.13 CPU tools and uv0.12.10. Clang was scoped to native probes, not Python extension compilation. This resolves the saved ordinary missing-tools qualification gap but does not rewrite its historical local failure. The unchanged ordinary README's local-only status paragraph is historical; the current dated hosted record supersedes it.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,84 inputs and fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. Fork PR#2's strict-TypeScript limitations were refreshed and remain unresolved.

## Supplementary castling premise review

The following exact source was separately checked locally. A consistent input with a king on e1, rook on h1 and knight on f1 becomes inconsistent if a caller directly requests castle e1-g1: the raw update adds a rook at f1 without clearing its existing knight. This is a counterexample to an unconditional castling representation theorem, **not a claim that the legal move producer accepts this move**.

```bend
# Supplementary raw-call counterexample, not a new legality theorem.
import Base
import ./Position.bend as P
import ../legal_probe/Chess.bend as C
import ./proofs/board/Spec.bend as B

# e1 king, h1 rook, and a knight occupying the rook's f1 destination.
def blocked() -> C.Board:
  P.put(1,True{},5,P.put(3,True{},7,P.put(5,True{},4,P.empty())))

def consistent_input() -> {B.valid(blocked()) == True{} : Bool}:
  {==}

# Raw castle flag2 does not erase the rook's intermediate destination.
def blocked_raw_castle_is_inconsistent() ->
  {B.valid(C.make_move(blocked(),C.Ply{4,6,0,2})) == False{} : Bool}:
  {==}

def overlapping_rook_and_knight() ->
  {U64.and(C.get_rooks(C.make_move(blocked(),C.Ply{4,6,0,2})),
    C.get_knights(C.make_move(blocked(),C.Ply{4,6,0,2}))) == U64.bit(5n) : U64}:
  {==}
```

Copy the block with one trailing newline into a disposable qualified checkout as `native/bend_engine/standalone/CastlingDomainReview.bend` and run the pinned bend2/main.ts checker. Source SHA-256: `b5c995eabda4e0f37023b406e3b30bc3a35c9519ae7edbe2554ed87f13fdeb19`. It passed with exact safe output in0.812 seconds. Changing the inconsistent result to True, or the overlapping bitboard to zero, was rejected ordinarily at the respective statement. Complete local receipts are in the review package. These three closed facts and two diagnostics do not increase registered law/control counts, and no native execution of this supplementary module is claimed.

This identifies a concrete next precondition: control the rook landing square and distinguish the two inserted locations. Additional source/rook identity, movement, rights and king-safety conditions are still needed for a legal castling theorem.

## Failures, scope and remaining work

An initial new Fresh.bit proof failed Boolean reduction; a complete finite Boolean split corrected the proof without changing its theorem or the compiler. The final local special focused gate completed in16.005 seconds and native gate in13.333 seconds. The disabled streaming-container invocation never executed; ordinary bounded synchronous/file-supervised checks completed instead. The original failed proof diagnostic and successful final receipts are preserved.

This final review adds documentation only. Qualified primary sources remain unchanged; no expensive full chain or lint was rerun for it. The supplementary castling checks were separately executed and labeled. Self-review only, not independent review. Source/hash agreement establishes provenance, not a second reviewer.

No production runtime, earlier accepted law, protected checker, permanent workflow, routine perft, search, model/GPU, training or benchmark changed. No additional Python application responsibility moved into Bend; export,references,data/control/training and transitional C++/LibTorch/AOTI remain dependencies. Checker/Base,native lowering/storage,ABI,toolchain,libraries,OS and hardware remain trust boundaries. Snapshot-lowering and closed-builder limitations are unchanged. No merge,force push,deployment or live-process change.

Next substantive P3 work is castling with explicit storage and piece-location premises, followed by legal-move producer guarantees, independently specified metadata and king safety. Promotion/en-passant representation preservation does not establish their legality or generator soundness/completeness.
