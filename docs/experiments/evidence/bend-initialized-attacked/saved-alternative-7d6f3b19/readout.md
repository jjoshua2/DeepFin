# Initialized attack and single-king check composition

## Scope and acceptance plan

Parent is PR #896 at `68fee266244d7b6210d004a78c865a29266b76e3`, tree
`5511e6e4fa908a4f70454dcbdb09074e505df50d`. This increment composes existing
initialized slider and non-slider contracts with the actual attack-witness reducer.
It does not replace production code, broaden a historical probe's compiler pin, or
change the accepted parent proofs. The compiler remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` (84 inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`).

Acceptance requires the complete three-law importing consumer with exact safe output,
all 19 classified controls, actual current-table native probes in generic, portable,
native-target and UBSan modes, original compiler source/pin checks, and a separately
reported unchanged repository lint attempt. Source acceptance, native observations,
lint and publication are separate statuses; missing tooling or a killed process is
not a passing check. No whole-aggregate, independent-review, legal-move or native
lifetime result follows from these gates.

## Public contracts

The common initialization is the actual expression:

```text
Tables.extras(extra, 0,
  Tables.tables(n, 0, 512, Array.new(U64, d, seed)))
```

The public conditions are `d == 17`, `n == 128`, `extra == 64`, and the query square
below 64. Seed is arbitrary. Side/attacker is a Boolean mapped explicitly to raw 0/1,
not an unrestricted raw metadata selector. No caller supplies a correct stored mask,
expected query result, returned-array equality, or pre-proved attack answer.

| Law | Result |
| --- | --- |
| `initialized_attack_query_matches_geometry` | Every typed pawn, knight, bishop, rook, queen or king query returns the independently specified mask and the entire initialized array, for arbitrary occupancy. |
| `initialized_attacked_matches_coordinate_witness` | Actual `Chess.attacked` returns the same array and a per-square piece/color witness over independent reverse-query geometry, for arbitrary actual Boards. |
| `initialized_single_king_check_matches_coordinate_witness` | Actual `Chess.in_check` returns the same array and the opposing-side witness at the selected king square, when its side's king intersection is exactly that singleton. |

The attack theorem does not require representation consistency or nonoverlapping
bitboards. The check theorem does require exactly one selected-side king at the
bounded square. Representation consistency alone is not that king-count premise.

## Proof structure and assumption audit

`Spec.mask` reuses independent bounded-coordinate leaper geometry and independent
first-blocker-inclusive rays. `Spec.attacked` uses the independent word-by-word
piece/color witness from the previous reduction suite. None of these expected
answers calls actual `Chess.attack`, `Chess.attacked` or `Chess.in_check`.

`Index` checks 64 scalar index certificates inside Bend, providing the rook/bishop
key bounds and the nonoverflowing bishop offset. It is not a host-provided table of
expected attack bitboards. `Query` applies both previously checked initialized
interfaces to the same full construction. Queens compose two sequential actual
slider lookups with their complete pair equalities.

`Evidence`, `Queen`, `Flow` and `Check` use exact query certificates as INTERNAL proof
parameters. `Initialized.bundle` discharges every one with `Query.actual` and the
public allocation/count/bounds conditions. The complete public `PROOF` consumer
imports and checks all those producers. The small-lemma split does not turn these
certificates into public caller assumptions; it avoids rechecking the expensive
inherited builder proof graph for every individual mutation.

`Flow` composes each state-threaded pawn/knight/king/rook/bishop query and the existing
actual reducer law. `Check` composes the bounded single-king selection law with that
initialized result. The pawn query reverses the attacker direction explicitly.
Thus the final equality preserves BOTH returned components, not only the Boolean.

The consumer uses all three public laws on symbolic arguments. Additional closed
examples cover a black knight attack and wrong-color rejection, both pawn directions,
a blocked queen, the singleton premise, and an arbitrary uninitialized table that
returns the wrong knight mask. These examples do not independently normalize a huge
literal full builder term or replace the symbolic proofs.

## Native scope

The probe executes real `Tables.build()` or full actual tables-plus-extras from
all-ones and patterned seeds. Each batch threads the returned table through six mask
queries (including the two-step queen query), actual `attacked` and bounded `in_check`.
It then reads unused slot 131071. The sentinel verifies one observed location, not
every native array cell. Source-level whole-array equality does not prove native
allocation/ownership/lifetime.

An independent forward-coordinate/set reference supplies expected masks, attacking
piece membership, lowest-king index and check Boolean. The probe receives raw boards
and query parameters, not expected decisions, and imports no proof-only geometry.
It compares 15 numeric query fields: 12 mask limbs, attacked, selected king index,
and check. Missing kings cause the PROBE to print diagnostic 2 and skip `in_check`;
this is not claimed to be a production return value. Multiple kings exercise the raw
lowest-bit behavior, not safety of all kings.

Fixtures include all bounded query squares, both sides and six kinds at sampled
origins, first blockers of both colors, arbitrary metadata, overlapping/unowned
planes, pinned attackers and missing/multiple-king boundaries. They do not enumerate
all source/target pairs, occupancies, Boards or legal games. The same base fixtures
repeat across initialization contexts and compiler modes; counts must not be summed
as independent datasets.

Three actual-code mutations omit the bishop part of a queen, fail to reverse pawn
direction, or replace the returned table. Each must compile and execute before a
wrong value is accepted as behavioral detection. A returned-table mutation can leave
the current attack Boolean correct but corrupt the following `in_check`; checking
only the first Boolean would miss that state error.

## Control accounting

Nineteen controls comprise ten intended source semantic/refinement rejections, eight
manifest/import-policy checks and one synthetic zero-exit warning-output unit. The
synthetic unit is not a compiler execution. Source controls run small composition
lemmas or the inherited implementation link they import; the complete initializer
producers are checked separately by the public consumer, not assumed correct by it.

Controls include queen union versus intersection, wrong/lost masks, lost returned
array, wrong attacker side, and omitted singleton or bounded-square premises.
Previously qualified initialization-premise controls remain parent evidence; they
are not silently counted as newly executed controls here. Missing imports, malformed
terms, affine errors, crashes, signals and timeouts never count as semantic rejection.
Controls-only output deliberately keeps `consumer` and `focused_gate` as `NOT_RUN`.

## Retained development failures

Early source drafts rejected a reserved type name, direct matching of a computed
certificate and a forward helper reference. The fixes changed construction syntax,
not theorem domains or existing source. An outer short-timeout invocation left no
complete result and gets no pass credit. A later earlier-version query check passed
in 587.077 seconds; it is development evidence because the queen composition was
subsequently factored into a separately checked core lemma.

A disposable draft without imported proof producers was rejected for missing law
implementations and is explicitly unqualified. No proof-import omission was used to
accept the final consumer. Core `Check` and `Queen` checks passed before the final
combined consumer; neither alone is called the public initialized theorem.

The first control run reached the intended inherited failure but the harness expected
a different diagnostic path. That assertion was corrected to the observed intended
path. The second run was killed by SIGKILL while checking a mutation returning a
literal depth-17 zero array. No reason beyond the observed signal is inferred and no
semantic success is credited. The final source control uses a depth-zero replacement
array, still a wrong returned-table mutation, and rejects ordinarily. The native
returned-table corruption retains its own full-size wrong-array behavior.

The original missing-tools lint result remains separate from source/native results.
No old law, production code, protected checker or compiler fingerprint was changed.

## Remaining mathematical and operational boundary

The result is a complete initialized **reverse-query coordinate witness**. A general
formal forward/reverse ray-membership equivalence is not newly proved. Native tests
use forward attacks, but agreement on finite fixtures is not that theorem. Deriving
single kings through castling's start/transit/final Boards, historical rights and
metadata validity, semantic king safety for generated moves, and move-generation
soundness/completeness remain separate.

No new direct equality for the fully expanded closed `Tables.build()` term is
claimed; the formal construction stays symbolic with explicit equalities. Existing
public-builder guards and their limitation are retained. This is self-review only.
The pinned checker/Base, native lowering/storage, ABI, C/C++ compiler, libraries,
OS and hardware remain trust boundaries, including existing strict-TypeScript and
snapshot-lowering issues.

No application responsibility moved from Python. Export, external references,
data/control orchestration and training remain dependencies, with transitional
C++/LibTorch/AOTI inference unchanged. No model/GPU, search, training, benchmark or
extra perft workload was added. No merge, deployment or live-process action occurred.

## Final execution status

The final public consumer, controls, native probes, source-identity audit and recovery
checks are recorded below only after their commands complete. Publication is local
because the current connector offers read actions only and direct Git DNS failed.


## Completed local qualification

Source commit `af9f7c6322fe724721b72b6203b86b59abe98c8c`, tree `8afb5ba12db0db43d9b7ac80b1be350ebcb99b6c`, is the exact
primary-source snapshot. The complete importing consumer passed with exactly
`All terms check.` in 659.660 seconds. The same final focused
command then executed all 19 controls and completed with exit zero in
709.657 seconds. Its complete proof dependency/source identities
match the committed files. This is not a controls-only or missing-producer substitute.

The final native verifier completed with exit zero in 62.114
seconds. Each of generic, portable, native-target and UBSan passed **14,052 distinct
initialization/Board/query requests and 210,780 numeric query fields**. There are
4,684 base Board/query inputs repeated under three initializers. All modes perform
30 successful sentinel reads and reject nine malformed batches. All three actual-code
mutations compile/run and then fail the independent reference.

Within the base fixtures there are 4,420 singleton-at-target cases, 59 missing-king
diagnostics, 155 multiple-king diagnostics and 50 singleton-elsewhere cases. These
base counts repeat with the initializers; they are not independent datasets. Common
fixture hash is `24d28b875245d9930e6c009d1a41d688ab88e70740718b9cf9ece464c9872851`; each mode's output hash is
`a0ece625e57caf77598559156991bea2a8dbd81b53a1bb73bcf087c0776a0ab5`.

Original compiler source tests pass 16 laws/seven controls, and all 12 pin tests pass.
The unchanged repository lint attempt fails for absent Ruff, Basedpyright and Vulture.
That remains an explicit qualification gap, not a successful inherited lint result.
There is no new hosted run and no independent reviewer.

All **571 inherited native-source entries** remain unchanged; all **586 candidate
entries** match. Exact-source parent183/562 plus the executed3/19 yields **modular
186 laws/581 controls**. The complete186-law aggregate wrapper was not executed.
Final evidence is local; current GitHub actions are read-only and direct Git transport
failed DNS. No remote branch, PR, merge or deployment is claimed.

One trailing empty line in `Queen.bend` was removed before this final complete gate.
The older in-flight consumer was deliberately stopped with SIGTERM, without a result,
when the exact cleaned snapshot was rechecked. The source hash difference from that
superseded invocation is recorded, not concealed. The final source AND native commands
both confirm unchanged `.bend` source bytes throughout their executions. All original
available development logs and full receipts remain in the recovery package.

The final documentation/evidence successor changes no primary source. Its complete
patch and incremental bundle have explicit #896 prerequisites; exact recovery checks
and final identities are included in the package's publication manifest.
