# Moving-side singleton preservation through actual castling

## Scope and acceptance

The exact parent is PR #897 at `1b27a4d75744167fc0e365f7f331d2cd80881b21`,
tree `c7afa4b4f843a3f7dc6c1b6619541160967e5cf4`. The parent already proves
initialized target-centred attack/check geometry and retained table state. This
increment addresses a different missing premise: the king selected for the input
side must remain a singleton at the correct starting, transit and castled square.

The compiler stays `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` with 84 source
inputs and fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
No production source, earlier law, historical probe pin or running process changes.
This readout records proof development and completed checks; it is not represented
as a precommitted training experiment. No training or performance claim is made.

Acceptance requires the complete five-law importing consumer with exact safe output,
all 18 classified controls, four native build modes and three actual mutations,
original compiler source/pin checks, and an explicitly reported unchanged repository
lint attempt. Local execution, hosted qualification and publication are separate
statuses. A missing executable, timeout, signal or harness failure is not a pass.

## Five public contracts

| Contract | Result |
| --- | --- |
| `transit_preserves_moving_king_singleton` | A consistent input with exactly one moving-side king at a route source has exactly that side's singleton at the actual ordinary transit destination. |
| `castle_preserves_moving_king_singleton` | The actual castling move relocates that singleton to its destination when the initial rook landing square is fresh. |
| `guard_supplies_castling_singleton` | The true implementation-linked producer guard derives the fresh rook target before applying the exact producer move. |
| `guarded_checks_target_original_side_king` | Actual in_check at every Start/Transit/Castled stage equals actual attacked at the certified original-side king square; both returned pair components are equal. |
| `move_flips_valid_side` | A valid Boolean side-to-move value becomes the opposite value under any actual move. |

Every route is one of the four orthodox coordinate patterns, not an unrestricted
source/destination pair. Input consistency and singleton king location remain real
premises. The raw route results use the original Board's color selector and do not
implicitly assume valid turn metadata. The guarded version uses the route selected
by the actual producer; valid 0/1 propagation is stated separately.

The original side is essential. After castling, the child's side to move is the
opponent. Querying that side instead would inspect the opponent's king. The consumer
checks a concrete white-castling example where child-turn king selection is black
square 60 rather than moving white square 6.

## Structural argument

The independent king-plane projection keeps both the kind and selected-color planes.
For removal mask m, king target t and rook landing mask r, the relevant algebra is:

```text
((k & ~m) | t) & ((own & ~m) | t | r)
 = (((k & own) & ~m) | t) | ((k & ~m) & r)
```

The last term is important: a king-kind bit already on the rook landing square can
acquire the moving color when the rook is inserted. The proof does not drop that
term by assumption. It derives its emptiness from existing partition consistency
and initial rook-target freshness. The producer guard supplies that freshness in
the guard corollary. Word induction, both U64 limbs, actual decoder refinement and
unchanged actual make_move bridges establish the result for arbitrary bitboards in
the stated domain, not by extrapolating native fixtures.

The actual starting singleton clears at source. The ordinary transit operation has
no rook insertion; castling adds a rook at a proven king-free landing square. The
final selected plane therefore equals the one target bit. Twelve route/stage index
cases establish actual ctz selection and substitute that index into in_check.

## Boundary examples retained

Representation consistency alone permits additional moving-side kings; an example
with another king at square 0 retains both 0 and 6 after raw castling. Another
consistent input places an opposing king at f1, blocking the rook landing square.
The raw e1→g1 update then has moving-side king bits f1 and g1. That example fails
the theorem's freshness premise; it does not demonstrate a legal-generator bug.

The stronger guard corollary derives freshness, but **not** initial singleton king
count. Opposing king count is unrestricted and is not newly preserved here. No
king/rook identity or target freshness is fabricated as an expected result.

The query theorem establishes where the actual check runs. It does not prove the
attack result false, assume initialized tables, validate historical castling rights,
or newly compose the complete initialized-geometry theorem. Positive pair equality
is not a native allocation/lifetime claim. These results are prerequisites for the
next semantic castling-safety composition, not that whole theorem.

## Native reference and coverage

The candidate runs actual Position/Chess/Text functions and emits the complete Board,
original-side king plane and selected index. It does not import the proof model or
execute in_check. The independent reference is a 64-square representation of kind
and color sets; raw metadata expectations follow the current update semantics, not
an independently proved legal castling-rights specification.

512 distinct starting Boards are tested at three stages. Main fixtures allow one
or more opposing kings, emphasizing that only the moving-side singleton is proved.
20 additional cases cover a missing king, another own king, a blocked rook landing
square, overlapping source kinds and invalid turn metadata. All 1,556 request tuples
are distinct; there are 532 distinct original Board field tuples. Modes repeat the
fixtures, not independent or exhaustive legal-game datasets.

Each mode compares 34,232 numeric fields (1,556 × 22). Three generic-only mutations
misplace the king, omit source removal or skip the side flip; rejection must follow
successful C generation, compilation and normal execution. They fail first at the
transit observation. Nine malformed requests test only the bounded probe interface.

## Control accounting and development history

18 controls comprise 9 source semantic/refinement failures, 8 manifest/import checks,
and 1 synthetic warning-output unit, not 18 independent compiler executions. Two
actual-code mutations are intentionally caught by the inherited Ordinary.unfold
bridge; seven target new law conclusions/premises. Missing singleton, consistency,
rook freshness or producer guard, a retained source king, wrong attacker color and
unchanged turn all receive ordinary intended type/refinement failures.

Early proof construction required explicit duplicate binders, constructor/match
ordering and Boolean reduction orientation. Their available draft logs are retained.
One outer consumer call hit its execution timeout without a complete receipt. The
first native probe mishandled an affine Board binder and failed before C generation;
the corrected helper consumes the Board once. The first complete focused attempt
passed its consumer but stopped because a diagnostic-location regex expected the
local unfold name rather than the actual inherited Ordinary.unfold location.
The correction changes only that expected location, not a law or production source.
The symlink policy control points to the correctly renamed Raw.bend file.

Only completed final gates receive qualification credit. Old failures are not
relabelled or averaged into a passing result. The first completed native report is
retained; the final repeat has an explicit enclosing-process exit receipt.

## Publication and trust boundaries

At development time the connector exposes GitHub reads only, and direct Git transport
cannot resolve GitHub. A read of the live PR precedes work; no available write action
is inferred from historical sessions. A guarded recovery/publishing script and exact
patch/bundle will be packaged, but packaging them is not remote publication.

Self-review only. The unchanged pinned checker/Base, native lowering/storage,
C/C++ toolchain, ABI, libraries, OS and hardware remain trust boundaries. Existing
snapshot-lowering and literal closed-builder limitations are unchanged. No compiler
source or earlier accepted contract was weakened. No Python application responsibility,
model/GPU, search, training, performance benchmark or perft budget changed.

Next: compose these derived singleton and valid-side conditions with initialized
attack geometry and the established generated-castling checked path, retaining the
separate universal forward/reverse correspondence and legal-rights obligations.

## Completed execution

Final numerical receipts and exact commit identities are appended after their
commands finish. No hosted result is inferred from local success.


## Completed local qualification and exact source

Qualified source commit `4e1787f0750ca72c8aa7308f1d814841ee750a75`, tree `50ffd3ab0f9be9aaa6d79c0e60bce3bf6af2166f`, includes the final cleaned proof/test files and scope documentation. The complete public consumer passed with exact safe output in 34.866 seconds; the same focused command executed all18 controls and completed with exit zero in 165.811 seconds. The native verifier completed with exit zero in 62.213 seconds and all four modes passed. Both commands have enclosing-process exit receipts, not just partial output reports.

Native counts are1,556 distinct request tuples/34,232 fields per mode. The512 original positive Boards each have Start/Transit/Castled observations;20 other input Boards are off-premise diagnostics. All1,536 selected-side singleton observations agree with the independent reference. The common fixture SHA-256 is `b3afeb8b97c4ce49c6ba71432f406f3f1d46c3dadbe0a7364caa61aa1ed994ff`; every mode's output SHA-256 is `da6312e429d10689e7ac0a054ebe227608e60e5aab17c169d6487a2255c3c02d`. All three actual mutations compile and execute before a transit-stage mismatch; they are not claimed as independent per-stage detections.

Original compiler16-law/seven-control checks and all12 pin tests pass. The unchanged repository lint command exits1 because Ruff, Basedpyright and Vulture are absent; the exact failure is committed. No hosted qualification or independent review occurred. One trailing blank line was removed from each of LAWS.bend and PROOF.bend; both complete source and native commands were then rerun on the cleaned files. Earlier completed receipts remain available separately.

All586 inherited native-source files are unchanged, and all600 candidate native entries plus the focused/native manifests match the exact source commit. Modular evidence is191 laws/599 controls, retaining the186/581 parent; the complete191-law aggregate wrapper was not executed. No prior law, production function or compiler input changed.

Direct Git transport again exits128 with `Could not resolve host: github.com`. This session's GitHub tools provide reads only. The new local branch and portable patch/bundle are preserved, but **no remote PR update, PR creation, merge or deployment is claimed**.
