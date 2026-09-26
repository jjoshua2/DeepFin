# Attack witnesses and singleton king selection

## Result on the exact PR #894 tree

Parent is `991e4937bbeeedbf9e9a3ddb40a15417c86a4cd6`, complete tree
`abae23517afaef0efa93fbaf6b7b47e20389aacf`. The local saved safety commit
`0d9c426fb560ec9fc35761adac68075a551df3da` has that identical tree; it was
recovered from the retained incremental bundle, not substituted for another PR's
implementation. The publication parent will be the actual #894 head. Overlapping
#892 remains untouched. No existing file under `native/bend_engine` changed.

## Three new universal contracts

The actual five attack-query callbacks now refine a per-square witness scan:
a selected-color piece whose corresponding reverse mask contains its square.
The structural Word proof connects nonzero masked arithmetic to existential
Boolean scan over both limbs. The intermediate mask-reduction law accepts
arbitrary Boards and masks, including overlapping raw piece/color planes.

The complete actual `Chess.attacked` table-and-Boolean result equals the new
witness-query interface. Its query sequence includes reverse pawn direction,
knights, kings, rooks/queens and bishops/queens. `Flow.bend` checks each actual
callback and preserves the final array component, not just the Boolean.

The singleton-king corollary connects `Chess.in_check` to that same query at the
specified king square. It requires the chosen side's king mask to equal exactly
one bounded bit. Representation consistency alone does not supply this fact.
No kings gives ctz64; two kings select only the lower bit. The consumer preserves
both boundaries, plus a diagonal queen and rejection of a wrong-color witness.

**Not yet a complete attack-geometry theorem.** `Queries.run` intentionally still
retrieves masks through actual `Chess.attack`. The new source result discharges
aggregation, side/pawn query wiring and singleton selection. Universal correctness
of all table masks against independent coordinates must still be connected to
existing table/slider/extras results. Actual-vs-query array equality does not imply
input-array identity or native memory/lifetime correctness.

## Completed local execution

The final importing consumer, three registered laws and all19 controls passed
with exactly `All terms check.`. Controls: ten intended semantic/refinement
rejections, eight manifest/import checks, one synthetic warning-output unit.
No syntax/ownership errors, missing imports, crashes or timeouts count as semantic
rejection. Ordinary compiler16-law/seven-control suite and all12 pin tests passed.

Generic, portable, native-target and UBSan each passed14,090 distinct requests
and183,170 output fields: five actual masks, attacked Boolean, chosen king index,
and bounded in_check Boolean. Every query square/piece kind/color is exercised;
origin sampling is not exhaustive all-square pairs.13,697 requests have exactly
one defending king at the query square.80 have no defending king,232 have more
than one, and81 have a single king elsewhere. Out-of-domain no-king in_check is
skipped with a probe-only sentinel. Native modes repeat identical fixtures.

The external oracle uses forward coordinate attacks from each piece, independent
of the reverse-mask accumulator. It includes first blockers, pinned-knight attacks,
wrong-color pieces, queens on both slider classes, arbitrary metadata, and raw
king-count diagnostics. All three actual-code mutations (wrong attacker color,
omitted knights, unreversed pawn direction) compile/run then fail returned values.
The native probe does not execute proof-only Spec.scan.

The unchanged local lint command failed because Ruff/Basedpyright/Vulture were
missing. That receipt is preserved. Hosted qualification is not yet completed.
The first enclosing local proof/native launches wrote their successful final JSON
reports but did not retain outer-shell status receipts; hosted reruns must complete
normally and record their own status, not infer it from those launches.

## Development history and remaining scope

Initial drafts were rejected for a term-position match and constructor inference.
Factoring the match into a function parameter and using an explicitly typed Masks
constructor helper resolved them without changing any intended theorem. The full
parent archive initially omitted ignored tracked files from the Git index; adding
all archived paths with their original modes reproduced the exact parent tree.
No bad tree or failed source draft was published as qualified evidence.

Modular180-law/543-control evidence retains177/524 from the exact #894 parent;
the complete180-law wrapper was not executed. All receipts remain separately
labeled local or hosted. Self-review only; no independent review.

Next: universally connect the five queried masks to independent geometric attacks
under explicit initialized-table and square conditions, and derive single-king
conditions for the castling start/transit/child Boards. Do not call aggregation
alone semantic king safety. Historical rights, metadata validity, legal reachability,
whole-generator soundness/completeness and native lowering remain separate.
No production code, compiler pin, prior law, permanent workflow, model/GPU,
training, search, performance benchmark or perft budget changed. Python remains
external oracle/control/training infrastructure; no application ownership moved.
