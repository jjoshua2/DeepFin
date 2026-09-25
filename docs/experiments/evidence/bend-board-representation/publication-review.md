# Board-representation publication review

## Exact qualification and publication

Hosted run **36080761492**, job **107901963017**, completed every stage on its first attempt: exact source recovery, retained-parent evidence checks, six-law focused gate, four native modes and behavioral mutants, original compiler source/pin checks, locked CPU development tools, unchanged repository lint and publication.

Qualified source: `ad0c9e379a0b1bfbb91b7c5d7755e22668bf1004`, tree `5210ff4c5300a710f4b679b3619bba2ccfb33a1d`.
Evidence successor: `a508d00fccb0f39f5565d3416cf61ab55432e5ab`, tree `52a09d4195b9e7331b863bd826692db76b565b76`.
Parent: #877 at `e486cf04bd04f6fd33204bca8d7ebcebfb45b3a2`, tree `022afa4ae9e1236435c741c671c46fd6430d453a`.

This final review file changes documentation only. No primary source changed after successful qualification. No merge, force push, deployment or live-process action. The temporary workflow lives only on the separate development branch, not in this feature's diff.

## Download and reconstruction

Artifact **10840884138**, `bend-board-representation-qualification`, downloaded with ZIP SHA-256 `ce6c9a01174419c41d28f9b583963d301a97ebdc20d9d93654d1716e1c270e79`.

The complete archive reconstructs all **3,620 tracked paths**, the exact evidence-head Git tree and original commit object. Every one of the **391 candidate native-source manifest entries** matches the locally checked candidate. All fields of the focused proof/control report match the local final report except measured consumer seconds. The complete native report matches except the C compiler identity (local Clang17, hosted Clang18.1.3). Original reports are preserved rather than edited to force byte identity.

The original 177,760-byte source patch has SHA-256 `d9ead50873ed2792135b7af29baa14a8fd8eade7b37dd88e286078edb6c9dc20`. Applying it to a separate worktree/index of the exact complete parent reproduces source tree `5210ff4c5300a710f4b679b3619bba2ccfb33a1d` with clean Git whitespace checks. Publication rechecked all source hashes before adding only documentation/evidence.

The pinned compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs with fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. Its separate fork PR #2 still records unresolved strict-TypeScript diagnostics; this increment does not suppress or fix that gate.

## What the six contracts do and do not establish

Four universal laws establish actual fresh typed insertion preserves the independent per-square partition, exact complete-Board insertion behavior, insertion's metadata frame, and partition preservation through arbitrary metadata updates. Two closed facts establish consistency of actual empty and initial boards. Empty consistency does not imply a legal chess position.

The independent predicate accepts exactly thirteen single-square states: empty or one of six kinds paired with one of two colors. The finite eight-Boolean row case split is checked inside Bend; structural Word induction covers both limbs and arbitrary mask patterns. It is not enumeration of possible 64-square boards or inference from native samples. No optimized piece-decoder fallback supplies the predicate.

Freshness observes only the input color planes at the actual inserted mask, together with the separate input-consistency premise. Typed piece constructors supply the valid 0..5 encoding. The public insertion statement intentionally supports arbitrary U32 square arguments via actual mask semantics; this does not make out-of-board squares meaningful. Native insertion is bounded to0..63. Metadata fields remain arbitrary and are not certified valid. Repeated identical insertion can remain consistent without freshness, so the condition is sufficient, not claimed necessary.

A no-op insertion also preserves consistency. The separate exact-update bridge is therefore essential: it rejects no-op and wrong kind/color/metadata behavior, while the independent partition theorem supplies the stronger input/output relationship. The native per-square set reference checks every one of eight bitboards and three metadata fields, not only the Boolean invariant.

The chosen future P3 semantic anchor is the English FIDE Laws applied1January2023, linked in the source README. This suite does not yet formalize Article3 movement, king safety or reachability, and makes no FIDE-compliance claim. Physical tournament procedures are distinct from these representation predicates.

## Execution distinctions

The final focused command executed the six-law consumer and all seventeen controls. Eight controls are ordinary intended semantic/refinement failures; eight enforce manifests/imports/no holes/unsafe/symlinks; one synthetic warning-output test is not a compiler execution. Missing files, malformed terms, affine-use errors, crashes and timeouts are not semantic proof rejection. Safe success requires exit0 and exactly `All terms check.`.

Exact-source117-law/315-control parent receipts are retained with the new6/17 result, giving **modular123-law/332-control coverage**. The expensive combined123-law wrapper was not executed. The inherited native/source gate budgets are unchanged; no new routine perft workload is added.

Each native mode passed1,026 complete-board rows and19,494 U32 field comparisons:768 fresh insertions covering every square/kind/color combination,64 identical repeats,64 conflicts,64 consistent and64 inconsistent metadata inputs, actual empty and actual start. Nine malformed requests are rejected by the bounded probe; this is not a transactional-rejection theorem for Position.put. Collisions are executed and compared outside the freshness premise, not falsely described as runtime rejection. Modes repeat the same fixtures and are not exhaustive board/game coverage.

Both actual no-op and wrong-color mutations compile and execute, then the independent reference rejects row0: the no-op differs at field1 (131328 versus131329), and the wrong-color insertion differs at field13 (272961305 versus272961304). They are deliberate regressions, not production bugs. No proof predicate, ray table builder, search, model or oracle answer executes in the candidate probe.

Original compiler16-law/seven-control checks, including cyclic-template rejection, and all12 compiler-pin tests passed locally and hosted. Local whole-repository lint failed because Ruff, Basedpyright and Vulture were absent. The original failure remains in the review package. Hosted locked Python3.13 CPU tools with uv0.12.10 passed the unchanged Ruff/Basedpyright/Vulture command. Clang18 is scoped to native probes rather than Python extension compilation.

## Retained construction failures

Early Word proof drafts failed because of pattern-order/explicit-duplication requirements and Boolean reduction orientation. A compact wildcard row proof failed, so the complete finite Boolean case split was retained. The first metadata frame needed explicit U64 limb destructuring. The first focused run passed the laws and seven controls, then correctly stopped on a nonunique mutation-site assertion; the final targeted metadata mutation passes. Probe drafts failed a Type/Data list distinction and a forward recursive helper, then the final Maybe-based parser compiled and passed malformed-input tests.

One trailing blank line in Spec.bend was removed before a full final local focused rerun and hosted qualification; no source term or contract changed. The cleaned local consumer completed in3.023 seconds. Full original logs, final JSON reports and cleanup identities are in the downloadable review package. No failed draft, timeout or missing-tool receipt is relabeled as a successful result.

## Next work and trust boundaries

Next decisive P3 acceptance is occupied-square decoding and abstract-square correspondence under this partition, followed by freshness from actual parser traversal and invariant-preserving removal/move/special-move operations. King counts, pawn ranks, turn/castling/EP validity, legal reachability, legal-move soundness/completeness/no duplicates and perft remain separate. The P2 closed literal public-builder equality is still separately unqualified as recorded in #877; no earlier result was weakened or silently upgraded here.

Self-review only, not independent review. Source/hash/report agreement is provenance rather than a second reviewer. The pinned checker/Base, native lowering/storage, ABI, C/C++ toolchain, libraries, OS and hardware remain trust boundaries. No production runtime code, old accepted law, compiler input, permanent workflow or routine perft budget changed. No additional Python application responsibility moved into Bend; export, references, data/control/training and transitional C++/LibTorch/AOTI remain dependencies. No model/GPU, training, strength or benchmark result is added.
