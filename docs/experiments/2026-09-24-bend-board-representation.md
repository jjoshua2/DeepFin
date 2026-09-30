# P3 board partition and actual fresh insertion

## Baseline, scope and acceptance

Continue #877 at `e486cf04bd04f6fd33204bca8d7ebcebfb45b3a2`, complete tree
`022afa4ae9e1236435c741c671c46fd6430d453a`. The mounted qualified full-source
archive plus the exact review overlay reconstruct that entire tree. All 379
parent native-source manifest entries match. Remote head and compiler branch
were refreshed; no competing board-invariant branch or review comment was found.
Repository instructions, development guide, branch lifecycle and experiment index
were read before edits. Parallel map/scheduler work remains untouched.

Compiler is unchanged `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21
plus U64, 84 source inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

This record follows constructive local development and precedes hosted
qualification; it is not a backdated preregistration. Acceptance requires the
exact six-law safe source consumer, seventeen classified controls, four native
modes with independent complete-Board comparisons and compiled/executed wrong-
behavior mutations, original compiler source/pin gates and unchanged whole-repo
lint. Inherited 117-law/315-control qualification may be retained only with exact
source and successful report/job verification. No new full-chain execution is
implied by that modular method. All work is bounded and opt-in; no live process,
production runtime, prior law, checker, perft budget, model or training changes.

## Independent representation and rules boundary

At each of the 64 positions, the six kind bits and two color bits must describe
an empty square or exactly one kind of one color. The specification is a Boolean
row partition, not the optimized decoder's fallback. Both U64 limbs are checked
structurally. It is deliberately weaker than orthodox position legality and does
not include metadata validation. In particular the empty board is consistent,
and king counts, pawn ranks, castling/EP history, check and reachability are not
established. No move generator or FEN acceptance guarantee follows.

The chosen future P3 rules anchor is the authentic English FIDE Laws applied from
1 January 2023, https://handbook.fide.com/chapter/e012023 (accessed for this task).
Articles 2.1 and 2.2 anchor squares, colors and piece categories; future engine
movement/special-move semantics must map Article 3 explicitly. Physical touching,
clock operation and arbiter procedures are separate from board representation.
Pinning this edition is not a claim of current-edition completeness or compliance.

## Public laws

| Contract | Scope |
| --- | --- |
| `empty_representation` | Closed consistency fact about actual Position.empty. |
| `initial_representation` | Closed consistency fact about actual Position.start board. |
| `fresh_put_preserves_representation` | Universal actual insertion preserves the partition when the input is consistent and the inserted mask avoids both colors. |
| `put_exact_bitboard_update` | Universal complete Board equality for actual insertion of a typed piece/color, independent of freshness. |
| `put_preserves_metadata` | Universal equality of turn/rights/EP before and after actual insertion, even for arbitrary raw kind/square values. |
| `metadata_preserves_representation` | Universal partition-predicate equality through actual metadata updates, including inconsistent input boards. |

The consumer and manifest use a stable declared order; there are four universal
and two closed public contracts. The insertion premise does not assume the desired
output. Typed piece constructors supply valid raw tags 0..5. Formal square inputs
are arbitrary U32 values interpreted using the actual bit-mask operation; native
insertion tests use only meaningful 0..63. This is not a claim that out-of-range
squares are valid chess inputs. Same-piece repeated insertion may be consistent
without freshness; freshness is sufficient, not claimed necessary.

The finite row proof checks all 256 Boolean rows, with thirteen allowed states;
Word induction handles all board bits and arbitrary masks. Actual.bridge connects
the specification to imported Position.put, and the other actual-operation
lemmas preserve metadata or transport the invariant. No production function,
existing law, trusted checker or compiler source was altered. A no-op can preserve
the invariant, so the separate exact-update law and behavioral controls are essential.

## Local execution before hosted qualification

All six public laws and their importing consumer pass with exactly `All terms
check.`. All seventeen focused controls pass: eight intended semantic/refinement
failures, eight policy/manifest/import failures and one synthetic warning-output
unit check. The synthetic check is not a new compiler execution. Removing the
freshness premise, corrupting real insertion/no-op/color/kind/turn, corrupting the
real initial board, or deleting a plane during metadata update is rejected at the
intended affected bridge or contract. No crash or malformed dependency is counted.

Four native modes each pass 1,026 complete Board rows / 19,494 U32 field comparisons.
Coverage: 768 fresh insertions across every square, six kinds and two colors;
64 same-piece repeats; 64 conflicting insertions; 64 consistent and 64 inconsistent
metadata inputs; actual empty and initial boards. An independent per-square set
model supplies full expected planes and metadata, not candidate answers. Nine
malformed requests reject per mode. Two disposable actual no-op/wrong-color
implementations compile and execute, then fail the original reference at row zero.
All four outputs SHA-256: `514e7686bce93cfeaa4a65d4a8ae0c11927a9826e536246d2225f37c6b27b59d`.
Modes repeat fixtures; these are not exhaustive arbitrary board values or games.
No proof predicates, table construction, search, model, perft or benchmark execute
in the candidate native probe. Native metadata inputs deliberately include values
that are not legal chess metadata; the theorem only frames representation bits.

Earlier constructive drafts failed pattern ordering, explicit duplication,
Boolean-reduction orientation and metadata-destructuring checks. An attempted
shorter row proof failed and the explicit complete Boolean case split was retained.
The first focused gate passed the laws and seven controls, then stopped on a
nonunique mutation-site assertion; the final targeted metadata mutation passed.
The native probe's first drafts failed a Type/Data list distinction and a forward
recursive helper definition. The final Maybe-based parser compiles and passes all
malformed-input tests. Draft failures are retained and not counted as completed
qualification. None required weakening a public law or changing compiler semantics.

## Remaining acceptance

Hosted qualification and repository lint are pending in this initial source
record. Local/hosted receipts will be appended without rewriting failed history.
No combined 123-law wrapper execution is implied by the focused result.

Next P3 targets: derive occupied-square decoding and abstract square contents
from the partition invariant, prove freshness through actual parser traversal,
then invariant-preserving removal/move/special-move operations. Full move legality,
completeness/no duplicates, legal-tree/perft reasoning and metadata/history rules
remain separate. The P2 closed literal public-builder equality is still separately
unqualified as recorded in #877; this increment does not silently resolve it.

Self-review only; checker/Base, native lowering/storage, ABI, toolchain, OS and
hardware remain trust boundaries. No application responsibility moved from
Python to Bend: export, external references, data/control/training and transitional
C++/LibTorch/AOTI remain dependencies. No merge, force push or deployment.


## Completed hosted qualification

Hosted run **36080761492** passes all six public laws, importing consumer, seventeen controls, four native modes with full-Board comparisons, two compiled/executed corruption controls, original compiler checks and unchanged whole-repository lint on source `ad0c9e379a0b1bfbb91b7c5d7755e22668bf1004`.

This is the first P3 partition-invariant suite, not another P2 table proof. Four statements are universal and two are closed empty/initial consistency facts. Metadata values, king counts, pawn ranks, occupied decoding, reachability and move legality remain separate. Every earlier accepted law and compiler input is unchanged.

The focused source gate executes its six-law consumer and all17 controls in one successful command. Eight controls are ordinary implementation/refinement failures; eight enforce proof manifests and imports; one synthetic zero-exit warning checks the wrapper and is not another compiler execution. No crash, malformed import, affine-use error or timeout is counted as semantic rejection.

Exact-source117-law/315-control parent evidence is retained, yielding modular123-law/332-control coverage. The complete123-law wrapper did not run. All379 parent manifest entries and successful source/control/runtime job identities were checked before application; all391 candidate native-source entries match before publication.

Each native mode passes1,026 complete Board rows and19,494 U32 field comparisons:768 fresh square/kind/color insertions,64 repeated insertions,64 deliberate conflicts,64 valid and64 invalid metadata inputs,empty andstart. Nine malformed probe requests are rejected. The real no-op and wrong-color insertions compile and execute before failing the independent per-square-set reference. Modes repeat fixtures; this is not exhaustive arbitrary-board or legal-game coverage. Collision cases describe behavior outside freshness, not rejection by Position.put.

The source insertion law intentionally covers raw U32 square inputs via the actual bit-mask semantics; native insertion is restricted to0..63. Metadata framing permits arbitrary U32 field values and does not certify them as valid chess metadata. Freshness is sufficient, not necessary: identical repeated insertion can remain consistent. A no-op preserves consistency, so the separate exact-update equality is essential.

Hosted Bun1.4.2/Clang18.1.3; Clang is scoped to native probes. Locked Python3.13/uv0.12.10 CPU tools use the normal build compiler. Ruff/Basedpyright/Vulture passes unchanged, resolving the preserved local missing-tools gap for this candidate. Original compiler16-laws/seven-controls, including cyclic-template rejection, and12 pin tests pass. A trailing blank line in new Spec.bend was removed before final local and hosted rechecks without changing any term.

Only documentation,index and evidence are added after checking. The full source tree matches the recovered local candidate. No permanent workflow, production function, earlier proof, compiler input or routine perft budget changed. No search/table build, model/GPU, training or performance run is added. Nothing merged, force-pushed,deployed or changed in a live process.

FIDE English rules applied1January2023 are the explicitly pinned future P3 semantic reference, not a claim of current-edition or full-rule compliance. The current independent Boolean-row partition supplies only square/color/kind consistency. Next: occupied decoding and abstract-board correspondence, parser freshness, and invariant-preserving removal/move/special-move semantics.

Self-review only. Checker/Base,native lowering/storage,ABI,toolchain,OS and hardware remain trust boundaries. Existing compiler limitations are unchanged. No Python application responsibility moved into Bend; export,references,data/control/training and transitional C++/LibTorch/AOTI remain dependencies.
