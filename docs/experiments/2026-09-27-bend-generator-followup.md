# Post-generation geometric check and saved-proof reconciliation

## Scope before qualification

Continue existing PR#910 at83045b38c499ecaf945d6df56da68dd1b8ee2e80, rather than
create a conflicting table-preservation implementation on its parent#904. The active
five-law implementation remains byte-identical. Preserve the saved three-law version
as an explicitly inactive archive and carry the existing whole-generator state
certificate into a real subsequent check operation.

## New contracts

The actual adapter runs `Chess.legal_moves`, then `Chess.in_check` using its returned
array and a separately selected query Board/side. Both public results retain the
complete ordered generated move list, final array and check Boolean. The first is
generic array/check transport. The second composes the initialized geometric check
under explicit depth17/128/64, bounded query square and selected singleton premises.
The generation Board is not equated with the query Board or assumed legal.

This closes post-generation query composition, not correctness of the generator's
internal checks or the safety/completeness of accepted moves. A correct query after
an incorrect move list would still be possible. Universal attack-direction equivalence
and historical rights remain separate.

## Acceptance

Require a fresh complete importing-consumer run and all15 classified source controls,
four native modes with independent follow-up Boolean answers and differential complete
move lists, all four executable adapter corruptions, original compiler/pin tests,
unchanged configured repository lint, and exact source/archive identity checks.
Run the restored saved three-law source gate separately without counting it as new
active laws. Retain the active existing source/native qualification on exact hashes.
No whole aggregate wrapper, model/GPU, training, search or benchmark is requested.

## Development receipts

The first expensive consumer invocation was deliberately stopped after source review
found the use of nonexistent `Equal.apply`; the fixed source uses existing
`Equal.cong`. The stopped invocation received no pass credit. A draft mutation changed
generation to consume the query Board twice and failed ownership checking; the harness
correctly refused it. The corrected mutation instead changes the generation Board
through an ordinary move and fails the intended transport equality.

The exact saved three-law sources and old failed lint receipts are archived without
replacing the active five-law implementation. No formal equivalence between those
implementations or duplicate discovery/count is claimed. Publication uses a new
commit; no claim that this continuation is the original saved local tree is made.

The corrected local full consumer was later killed with exit -9 and no output after450.933 seconds while native work overlapped. This is not a proof pass or an intended semantic rejection. Hosted acceptance will run the expensive consumer serially. All four local native modes and four compiled/executed adapter corruptions completed successfully afterward.

## Status

Fresh qualification pending; no result follows from this candidate record alone.
Self-review only. No independent review, merge, deployment, production source change,
previous-law change, compiler change or permanent workflow change.
