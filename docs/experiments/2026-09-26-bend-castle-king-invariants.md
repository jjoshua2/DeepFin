# Castling mover-king and valid-side invariants

## Scope and acceptance plan

Exact parent is PR #897 at `1b27a4d75744167fc0e365f7f331d2cd80881b21`,
tree `c7afa4b4f843a3f7dc6c1b6619541160967e5cf4`. Compiler remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
No production function or prior accepted law changes. The parent's alternative
implementation/archive and table-state regressions remain untouched.

Acceptance requires the complete four-law importing consumer, exact safe output,
ten semantic/refinement rejections, eight policy controls and one synthetic output
check; four actual native modes and three compiled/executed corruptions; the original
compiler source/pin tests; and a separately reported unchanged repository lint run.
Failures, inherited receipts and fresh executions must remain distinguished.

## Four public laws

| Law | Guarantee |
| --- | --- |
| `transit_preserves_single_moving_king` | The actual ordinary king step to the transit square preserves exactly one moving-side king there. |
| `castling_preserves_single_moving_king` | The actual flag-two move preserves exactly one moving-side king at the castling destination. |
| `move_flips_valid_side` | A raw turn equal to a Boolean color becomes its opposite after actual make_move, for arbitrary Ply. |
| `castling_stage_check_targets_moving_king` | Actual in_check at each starting/transit/final Board equals actual attacked at the intended stage square against the opposite color, including the complete returned array/Boolean pair. |

All four orthodox routes are covered. Transit, final and stage results explicitly
require an initially representation-consistent Board, one selected-side king at its
home square, and raw turn equal to the Boolean selected side. The final and common
stage law additionally require the actual producer guard. The transit singleton law
alone does not require a guard and does not claim its intermediate move is legal.
The side-flip theorem needs only the valid input-side premise.

No caller assumes the output king plane, a correct selected decoder tag, or empty
rook landing. Initial singleton and valid metadata are not derived from arbitrary
FEN parsing: that separate global invariant still needs its own origin/propagation.

## Proof structure

`Rows` checks all 256 local Boolean kind/color assignments inside Bend. From the
independent partition it proves that a king bit decodes as king and that fresh
storage contains no hidden/unowned king bit. This is a finite checked case split,
not an externally supplied expected bitboard table.

`Fresh` lifts the local fact by structural Word induction. `Kind` connects the
initial singleton through actual bit selection and the accepted per-square decoder
refinement to actual Chess.piece == king. `Algebra` structurally proves the selected
king-plane projection across clearing and the two insertions. `Projection` applies
these identities to the imported actual ordinary/castling update bridges. `Stages`
derives rook-landing freshness from the existing producer guard, proves exact
transit/destination singletons, and follows actual XOR turn arithmetic. `Check`
combines those facts with the existing bounded singleton-index proof and threads one
affine table through the actual in_check/attacked pair equality.

The final law does not call a proof-only attack oracle. It routes to actual attacked.
The independently initialized attack semantics from #897 is unchanged, not rerun or
reproved as part of this smaller stage/index gate. Composition into full generated
castling safety and universal forward/reverse attack correspondence remain separate.

## Non-vacuity and boundary counterexamples

The importing consumer applies every public law symbolically, and checks both colors
and wings on concrete nonempty Boards with a true producer guard. It also preserves
an explicit consistent input with one white king on e1 and an opposing king on f1.
Raw e1-to-g1 castling leaves both f1 and g1 in the white king plane. The producer guard
is false, so this does not contradict the theorem or identify a legal-generator bug.
It explains why the final singleton result cannot discard its guard condition.

Raw turn3 becomes2, not a Boolean color. XOR toggling preserves valid side metadata
but does not sanitize arbitrary U32 metadata. Opposing-side singleton preservation,
historical castling rights, legal reachability and complete generation correctness
are not claimed.

## Native scope

The probe builds actual current Tables.build and performs actual start, ordinary
transit, and flag-two final operations from the same original Board. It reports all
19 Board fields, two selected-king limbs, lowest index, bounded in_check and actual
attacked at the stage square: 24 numbers per stage. The three stages are not a
simulation of making transit and then castling; both children derive from the start.

The external reference reuses the existing independent forward-coordinate attack
interpreter and a separate square-set update model. It compares full child Boards
as well as king/check observations. Raw rights/EP arithmetic mirrors the existing
raw operation and is not an independent chess-metadata theorem. Candidate code
imports no proof specification and receives no expected decisions.

The final fixture bank contains 1,148 distinct request tuples and 3,444 complete
stage Boards / 82,656 observed fields per mode. It includes 256 guarded backgrounds,
256 opposing-king placements, 256 extra-moving-king placements, 116 kind/raw-side
cases, 256 arbitrary consistent Boards and four each of blocked-king and inconsistent
priority diagnostics. Four duplicate request candidates were removed. There are 451
transit-premise cases and 437 full castling-premise cases; their required singleton
and stage-target relationships are checked, while excluded cases document raw behavior.

Missing kings print probe sentinel2 and skip out-of-domain in_check. The sentinel is
not a production answer. Generic, portable, native-target and UBSan repeat the same
fixtures rather than independent or exhaustive positions. Eight malformed requests
per mode cover numeric, side/wing, tuple and bounded-budget errors. Three generic
actual-code mutations retain the old king, add the rook into the king plane or omit
the side flip; each must compile and execute before a wrong value counts as detection.

## Retained construction and harness history

An initial row helper required an explicit Boolean reduction; a raw ordinary projection
required the accepted OR-zero identity rather than definitional equality. An initial
check rewrite had the wrong orientation and was replaced by explicit transitivity.
These are proof construction failures, not weakened theorem domains. An initial
native pattern duplicated affine side/wing binders; those binders were marked for
duplication. A Python driver initially iterated a dictionary as pairs rather than
items; this stopped before the mode checks and was corrected.

The first source-control expectation omitted the imported module's diagnostic prefix.
It was corrected to the actual intended implementation bridge. A later enclosing
invocation timed out after three completed rejections; no complete gate result is
credited. The check-color mutation now targets the unchanged accepted King module
directly instead of traversing the larger consumer first. Final complete receipts
are appended only after completion. No timeout or syntax/ownership error counts as a
semantic negative control, and no protected compiler/checker input was modified.

## Status and trust

Local four-law consumer and native checks have completed; the complete final source
control and hosted qualification status will be recorded below after execution.
Self-review only. Source checking trusts the pinned checker/Base; native lowering,
ownership/lifetime, ABI, toolchain, libraries, OS and hardware remain boundaries.
No full-aggregate, native-lifetime, semantic castling safety, strength or benchmark
claim is made. Existing compiler/closed-builder/snapshot-lowering limitations remain.
No model/GPU, search, training, additional perft or Python responsibility change.

## Completed local checks before publication

The complete importing consumer passed in 19.910 seconds with
exact safe output, followed by all 19 controls in the same successful command.
The final focused command completed with exit0 in 123.681 seconds. Four native modes
and all three behavioral corruptions completed successfully; the later additions
are the source-control harness and scope README, not changes to native executable
source or its expected outputs. Hosted qualification will rerun both complete gates
on the final published source snapshot. Original compiler16-law/seven-control and
12 pin tests passed. Local configured lint failed for missing tools, retained as such.


## Completed hosted qualification

Hosted run **36291285103** passes four public laws, the importing consumer, all nineteen controls, all four native modes and three compiled/executed corruptions, original compiler tests and unchanged repository lint on source `bc8d5f83dbd9f3e77492763644ffe92478be1a16`.

All586 inherited native-source entries remain unchanged and all601 candidate hashes match. Exact parent186/581 plus the new4/19 gives modular190 laws/600 controls; no full190-law aggregate was run. The complete new consumer checks all imported proof producers with the unchanged pinned CLI.

Native modes each pass1,148 distinct requests,3,444 complete stage Boards and82,656 observed fields. The451 transit-premise and437 final-premise cases satisfy their claimed singletons and check targets. Eight malformed batches are rejected per mode. Three actual mutations retain the old king, add a rook to the king plane or omit side flipping; all compile/run and fail independent values. Modes repeat fixtures, not exhaustive legal games.

Ten source controls require intended semantic/refinement failures; four actual-code mutations deliberately reuse accepted imported implementation bridges. Eight enforce manifests/import policy. One synthetic output-wrapper unit is not a compiler execution. Syntax, ownership, missing imports, crashes and timeouts receive no semantic-pass credit.

The original missing-tools local lint remains a separate failed receipt. Fresh locked CPU tools pass the unchanged repository lint command at its configured scope, not a claim of additional all-native-Python type checking. C compiler selection is scoped only to native probes. Original compiler16-laws/seven-controls and12 pin tests pass.

The initial selected-side singleton and valid Boolean raw turn remain public input conditions. Their transit/final consequences are now proved rather than assumed. The final stage requires the actual producer guard; the blocked enemy-king example shows why its absence can recolor a second king. No opposing-side singleton, rights-history, legal-reachability or attack-reversal result is inferred.

The stage theorem compares complete actual in_check and attacked pairs on an arbitrary affine array. Combining this routing with initialized independent attack semantics and accepted legal_moves membership remains a later composition step, not a newly executed full-chain theorem here. No production code, prior law, compiler source, permanent workflow, model/GPU, search, training, perft or Python responsibility changed. Self-review only; no merge or deployment.
