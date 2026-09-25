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
