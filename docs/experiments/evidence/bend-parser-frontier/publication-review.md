# Accepted parser-frontier publication review

## Exact source, qualification and reconstruction

Base: #881 at `c3f9dd30cf58c90f91dbac6a7f74ec0d8ee18241`, complete tree `2bacec89fa8f1eefbe6b743529a9dffd4f20ff56`.
Published source: `be5afc6308d159d52f42b2fcccb59a9c20109b4c`, complete tree `efa431703a1d47208655595f816a3008d0c4a27a`.
Evidence successor: `429e3428df5186ec054400b7ce9fed1ecb0843fb`, tree `504d5752c464e898f862c75b9b7ddb6be11e7213`.
This final review changes documentation only. No qualified primary source changes.

The source-only 120,461-byte patch has SHA-256 `d7c124b631758bf5ee68ba39a3be2ff9bb193d1cb27478dd93e7f60af004272f`. A separate index initialized from the exact complete parent applies it cleanly and reproduces the entire qualified source tree. Sixteen paths change: fifteen new frontier proof/test/README files and the dated readout. Every inherited native-source entry remains unchanged.

Final qualification artifact **10871287850**, ZIP SHA-256 `bb8a9cbe916a26d8aebe9a3d06e47daaa4d293ede55ee8813cedc079c08d7493`, was downloaded and verified. Its archive reconstructs all **3,723 tracked paths**, the exact evidence-head tree and original commit object. All **444 candidate native-source hashes** match both the artifact and final local candidate. The complete focused report agrees locally except consumer seconds; the complete native report agrees except the separately recorded C compiler version.

The compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 source inputs, fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. No compiler/checker source, accepted earlier theorem, production parser or native runtime was modified.

## Qualification required two exact-source executions

**Run 36150116903, job 108120960214**, executed the complete four-law focused consumer and all seventeen controls successfully on source tree `efa431703a1d47208655595f816a3008d0c4a27a` (its ephemeral source commit was `0b5041a600d7dbbaec2e4d4ffa068945595a3a3d`). It then passed generic and portable native fixtures but stopped because native-target C compilation emitted an AVX10 feature-combination warning. Strict empty-stderr acceptance correctly rejected that compilation. The remaining native modes, compiler suite and lint are not credited to that failed run.

Its original artifact **10870783483**, SHA-256 `f91ff0c2ab7fa7bdba0dbd7b04124821dc7373e480c1e68fc6b46bad5847a664`, is retained, and the native failure text is committed losslessly. No warning was ignored, filtered or treated as a valid semantic rejection.

**Recovery run 36150660570, job 108122790012**, recovered the exact patch and successful proof receipts, verified every source hash and the first job's individual step outcomes, and reran the complete native gate. Its fresh runner selected `/usr/bin/clang`, reporting **Ubuntu Clang 18.1.3**, after a warning-free compatibility probe with the unchanged native-target flags. Thus no claim is made that a newer compiler major fixed the earlier problem: the later host/compiler combination accepted the same flags. No warning-suppression option or candidate build flag was introduced. Compiler selection is scoped to native tests, not Python extension builds.

All four native modes and both behavioral mutants then passed, as did the original compiler source suite, twelve pin tests and unchanged whole-repository Ruff/Basedpyright/Vulture. The locked Python 3.13 CPU environment used uv 0.12.10 and the normal extension compiler. The successful source gate was retained on its exact tree, not unnecessarily rerun or falsely attributed to the recovery job. The first run remains failed; this is successful qualification across two recorded exact-source executions, not an all-green first attempt.

The local cleaned focused command passed in 91.661 seconds, including a 17.603-second consumer. The first hosted consumer took 15.087 seconds. The local four-mode native command passed in 16.774 seconds with Clang 17.0.0. Differences in these measured runtimes are not performance claims.

## What the four universal contracts establish

`layout_preserves_frontier` follows arbitrary actual String traversal from a state satisfying the explicit safe-frontier predicate. A live state has file at most eight, rank below eight, a representation-consistent Board, and no occupied color bits in lower ranks or in the current rank at/after the cursor. Once invalid, temporary Board/cursor values are intentionally unconstrained by this invariant.

`accepted_prefix_has_frontier` derives that useful frontier state at every chosen prefix of an accepted initialized placement. A caller-supplied true flag alone does not establish the invariant. The real empty Board and file0/rank7 initialization supply it.

`live_typed_insertion_is_fresh` derives an actual target below64 and fresh in the pre-insertion Board whenever a typed piece/color step after an initialized prefix remains live. The caller does not supply a desired target bound, emptiness certificate or correct resulting Board.

`accepted_placement_has_consistent_board` states that the actual optional placement result has a consistent Board whenever it is Some. None is an explicit allowed result; the theorem does not assert every String is accepted. A populated complete placement, valid final-square insertion and 899 accepted native final strings prevent an always-rejecting interpretation of the candidate.

The independent unvisited mask is built from Nat rank/file predicates and structural words. Bounded scalar cases connect it to actual U32 cursor/address arithmetic inside the checker, while structural word proofs cover arbitrary Boards and masks. The proof uses actual Position.layout/layout_char/digit/put and existing exact transition/partition producers, not a replacement parser or caller-supplied fresh-square trace.

This does not yet establish exact abstract placement contents, full six-field Position.fen semantics, metadata legality, reachability, move legality or rollback of a raw invalid Layout. An invalid character can alter a temporary Board before finalization returns None; the consumer preserves that distinction. The earlier structural-snapshot lowering and closed literal builder limitations remain unchanged.

## Controls and native observations

The seventeen controls contain eight ordinary semantic/refinement failures, eight manifest/import-policy checks and one synthetic warning-output unit. The two actual cursor corruptions reject in unchanged parser transition dependencies; their receipts are not mislabeled as isolated failures of new frontier lemmas. Missing files/imports, malformed terms, affine errors, crashes and timeouts never count as semantic rejection. Safe source success requires status zero and exactly `All terms check.`.

The native verifier runs actual layout_char and layout_result at every character boundary from the real initialized state. The source String-induction theorem connects this traversal to actual layout. An independent square-set/cursor interpreter supplies only external expected observations, never candidate decisions or Boards.

Each generic, portable, native-target and UBSan execution checks **1,614 distinct placement strings, 29,526 complete Layout states and 679,098 numeric fields**. These include 26,879 live states, 2,647 invalid states, 899 accepted final strings and 7,197 live typed insertions. All 768 single-piece square/kind/color combinations are represented. The actual live-state observations are checked for bounds, consistency, empty unvisited regions and fresh pre-insertion targets, in addition to complete expected-state comparisons.

Two malformed bounded-probe requests are rejected per mode. Source statements cover arbitrary Strings; native input is Unicode without embedded NUL, at most256 characters per string and32 strings per invocation. Modes repeat fixtures, not disjoint or exhaustive String/Board/game datasets. This is not a native execution of a proof predicate or a whole six-field FEN validation test.

The actual piece-file mutation compiles/runs then fails case0/prefix1/field0 (observed0 instead of1). The actual slash-rank mutation compiles/runs then fails case0/prefix4/field1 (observed7 instead of6). Both are deliberate regressions, not production bugs. Common output SHA-256: `cc12b044ec805bec1984979a11c9cbcda52cad76d91d457f096b918d9bb6349f`.

Exact-source parent137-law/384-control evidence is retained after checking all429 parent manifest entries and its successful job. The new4/17 evidence gives **modular141-law/401-control coverage**; the complete141-law wrapper was not executed. Original compiler16-law/seven-control checks, including cyclic-template rejection, and all12 pin tests passed in recovery. Source and native qualification remain distinct from benchmarks.

## Saved candidate and failure preservation

The supplied local parser-rejection candidate `71f391281bb0e6b6da0ba11b6a47f4d9873f4999` and published #881 are different five-law implementations sharing the same directory name. Their patches must not be applied together indiscriminately. Both are preserved; this continuation uses #881's qualified exact transitions and prefix-safety producers.

The saved candidate's explicit public-FEN rejection theorem is not falsely described as one of #881's registered laws or as newly requalified here. Its prior local evidence remains in the original saved patch/review. The new frontier result is additional mathematics on the published branch, not a relabeling of that saved candidate.

Early construction failures involved explicit proof binders, pattern order, a False/Nat mismatch and native do-block parsing. One trailing blank line in Finite.bend was removed before a full final local focused rerun and hosted source checking. Two compressed transport transcription mismatches were caught by Git-object hashes and discarded before source application; corrected chunks reproduce the complete expected tree. The local missing-tools lint failure remains a historical nonzero receipt. These attempts and final cleaned reports are retained in the downloadable review package; no failed draft is credited as a pass.

## Scope, review and next work

This final publication record adds documentation only, so no source/native/lint suite was rerun for it. Self-review only, not independent review. Exact hashes prove provenance rather than providing a second reviewer. The pinned checker/Base, native lowering/storage, ABI, C/C++ toolchain, libraries, OS and hardware remain trust boundaries. The compiler fork's separately documented strict-TypeScript diagnostics are not fixed or suppressed.

No production runtime code, earlier accepted law, permanent workflow, routine perft, search, model/GPU, training or benchmark changed. No additional Python application responsibility moved into Bend; export, references, data/control orchestration and training remain dependencies, with C++/LibTorch/AOTI transitional inference. No merge, force push, deployment or live-process action.

Next substantive P3 work is invariant-preserving removal and move application, alongside separately scoped exact placement and full FEN/metadata semantics. King safety, special moves, legal reachability and move-generation soundness/completeness remain distinct. Initialized accepted-placement freshness and partition preservation should no longer be described as wholly unproved.
