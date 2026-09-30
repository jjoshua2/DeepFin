# Mandatory king-safety path for generated castling

## Preregistered scope

Continue the saved complete-generator provenance at fc86363cfc4363ea2d5dc93548f7c8231b923638,
on published PR #891 b3bbfd5c38f8647cece0ce1d4d541e26fbb6e29f. Prove that its generated
castling member cannot take the optimized no-check shortcut. Derive source king occupancy
and a bounded source from actual producer provenance; prove the sensitive mask includes
that king regardless of the ray result; link the actual blocker stage and fast step to
the actual make_move/in_check/filter_after path. Separately check that a true check result
rejects the current move while preserving the preexisting accumulator.

No production code or old statement changes. No assumption that attack tables or in_check
compute independent chess semantics. No king-count, rights-history, move-completeness or
semantic king-safety claim. Affine tables in generation and filtering remain actual inputs.

Acceptance: new public consumer, important semantic mutations, bounded current-pin native
step/reference checks, saved chain gate, original compiler gates and unchanged repository
lint. Native builds sequential; no routine perft, GPU, model, search, training or benchmark.
Failures remain recorded. Independent review and hosted publication are separate statuses.


## Scope extension before paired native qualification

The source proof now also establishes complete array-and-list equality between actual Chess.legal_moves and the audit-only Forced.legal_moves variant that explicitly adds flag2 to mandatory fast-path checks. It has no input-consistency or correct-table premise. This is shared-code observational equivalence, not independent chess-attack correctness. The planned paired native check uses unchanged parent chain fixtures and preserves that oracle's narrower full-generator castling-subset scope.


## Completed local source and native qualification

Local branch: `feat/bend-castle-safety-20260926`, starting at the saved chain commit
`fc86363cfc4363ea2d5dc93548f7c8231b923638`. Its published prerequisite remains PR #891
at `b3bbfd5c38f8647cece0ce1d4d541e26fbb6e29f`, unchanged at the final connector refresh.
The original saved chain commit/patch/bundle remain preserved rather than overwritten.
This increment is locally committed and packaged; no new GitHub PR, hosted run, merge,
deployment or live-process change. Available GitHub actions are read-only and direct
Git transport is unavailable in this runtime.

### The mandatory-check result

The actual optimized filter constructs `own & (rays | kings)`. Structural Word proofs
show that an owned king source is included for **every** U64 ray mask. The saved full
producer proof supplies an owned king at the exact bounded source for a generated
castling move, without assuming Board consistency or correct attack tables. Therefore
that move cannot take the unchecked fast-path branch.

`generated_castle_takes_checked_step` equates the complete actual fast-step result to
actual `make_move`, `in_check` for the **input side**, and `filter_after`. Generation
and filter arrays are separate affine inputs in this reusable statement. No table is
silently duplicated and no already-correct check result is assumed. A true check result
preserves the table/accumulator pair without adding the current move; an equal old
entry in that accumulator is not removed.

### The complete generator comparison

The additional sixth law strengthens the local branch argument:

```text
Chess.legal_moves(table, board) = Forced.legal_moves(table, board)
```

This is equality of **the complete returned array and ordered move list**, for arbitrary
actual Boards and affine input arrays. No caller consistency, metadata validity,
source bound, correct ray/table, membership, or desired check-result premise is needed.
The audit variant explicitly ORs flag 2 into mandatory fast filtering. The proof derives
the ordinary-or-guarded-castling classification through actual scan and both castling
producers, then proves the original and forced choices coincide for all candidates.
Dependent continuations preserve the actual array through all stages.

This equality does not say the returned array equals the original array. It compares
two actual-array computations. `Forced` reuses the actual check and metadata functions:
a shared wrong attack checker could satisfy this equivalence, so independent attack
semantics is still a separate obligation. It is a proof-interface audit, not a new
production generator or a claimed performance improvement.

### All six public contracts

| Law | Guarantee and domain |
| --- | --- |
| `owned_king_source_requires_check` | Owned king at source below64 requires checking for arbitrary ray mask, destination, promotion and flag. |
| `blocker_stage_uses_sensitive_mask` | Actual blocker-stage complete pair equals fast filtering using the specified sensitive mask. |
| `generated_castle_requires_check` | Actual full-generator membership and flag2 entail checking for every ray mask. |
| `generated_castle_takes_checked_step` | Such a move's actual fast step equals the complete current-side child-check operation. |
| `positive_check_rejects_current_move` | True check preserves the existing table/accumulator pair. |
| `complete_generator_equals_forced_castle_checks` | Actual full generator equals the explicit mandatory-castling-check variant, including array and ordered list, without extra premises. |

The consumer includes a nonempty actual generated castle from a small zero-filled
table, a bit63 king case, an empty-source raw flag2 counterexample to unconditional
checking, and rejection with an already-equal accumulator move. The zero-table example
proves satisfiability, not table correctness or legal chess. No false positive is
removed by silently assuming a valid raw flag2 source.

### Executed gates

The **final six-law/19-control focused gate** completed normally with exit0 in
47.553 seconds; its consumer completed in 8.605 seconds and
returned exactly `All terms check.`. All six public laws and controls ran together on
the final source closure of 59 entries. Ten controls require
ordinary semantic/refinement failures at the intended source locations, eight enforce
manifest/import integrity, and one synthetic warning-output unit is not a compiler
execution. Crashes, missing imports, syntax/affine errors and timeouts do not count as
semantic rejection.

The unchanged **saved five-law/18-control chain gate** also reran and passed. These are
rechecks of retained results, not another five added laws. The original compiler's
16-law/seven-control suite, including cyclic-template rejection, and all12 pin tests
passed. Compiler remains Bun1.4.2 with source revision
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,84 inputs and unchanged fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

### Direct optimized-step native qualification

The direct gate completed normally in 28.742 seconds. Generic,
forced-portable, native-target and UBSan each pass **1,360 distinct requests**,
**4,184 complete child Boards /79,496 Board fields**, and5,440 mask/required/check
fields. There are1,216 owned-king cases covering all64 source squares; all require
checking. In total1,224 require checks and136 take the raw shortcut.

Eight raw nonking cases intentionally differ between the shortcut and a forced full
check. They document behavior outside the actual generated-castle conditions, not a
production bug or a failed theorem. Native inputs contain consistent Boards with one
king per color; source laws do not require that stronger general condition. Squares
are bounded to0..63 in native probes. Eight malformed batches are rejected per mode.

The independent reference uses square/color sets and coordinate attacks. The candidate
builds real current tables and executes actual sensitive-mask, blocker-stage, fast-step,
full-step and check operations. It does not run the proof model or receive expected
decisions. All resulting ordered lists and every child Board field are compared.
Actual mutations omitting kings from the sensitive mask, bypassing required checks or
checking the opponent all compile/run and fail the reference in the first batch.
Common output SHA-256:
`32d399b72af27b73c13ba76aa20676f444d2a6eac18fae18149b9315d6fa3f24`.

### Paired complete-generator native qualification

The paired gate completed normally in 43.169 seconds. Both compiled
programs pass **1,062 identical requests per mode**:386 filter,386 two-side suffix and
290 full-generator requests. The original uses unchanged actual `Chess.legal_moves`;
the audit executes the source-proved `Forced.legal_moves`. Their **entire ordered output
is identical**, including9,241 complete child Boards /175,579 Board fields per program
per mode. Both reject seven malformed requests per mode. Four destination-only attacked
castles are rejected and39 full-generator fixtures retain both castling choices.

The inherited independent oracle still validates only the castling move-set subset for
full generation, exact filter/suffix lists, and all returned raw child Boards. Equality
of the complete noncastling output between the two programs is differential evidence,
not an independent proof of noncastling legality or completeness. Both programs share
actual attack/metadata helpers. Output is identical to the saved chain result:
`6dde6b9e3ad7b9ec0b4c27c4acb823bd06ec6c0d6bc2fb2d15cc0ecbecd02d10`.

An audit-only mutation disables required checking. It compiles and executes, breaks
complete parity at batch256 and then independently fails the castling reference at
batch448. This is a comparison-program corruption, separately labeled from the three
actual-production-code controls. The first complete driver attempt stopped because it
incorrectly expected the narrower castling oracle to reject the first noncastling
parity difference. The correction records both rejection boundaries and requires both;
no fixture, oracle decision, production source or theorem was weakened.

All four modes and paired programs repeat the fixtures. Counts are not disjoint Boards
or exhaustive legal-game coverage; the native transcript does not inspect complete
table contents or prove storage lifetime.

### Source identities, lint and retained failures

All524 inherited native-source entries remain unchanged. All540 final candidate entries
and the new focused/native dependency hashes match final source bytes. Only documentation
and compact execution evidence are added after the completed source/native checks.
Coverage is **modular177 laws/524 controls** with the retained171/505 parent; the complete
177-law aggregate wrapper was not executed.

The unchanged whole-repository lint command ran and failed with status1 because Ruff,
Basedpyright and Vulture are absent. Its exact output and failed receipt are committed.
No newly hosted lint pass or independent review is claimed. Parent lint receipts are
not reused as qualification for new files.

Earlier exploratory and incomplete invocations remain recorded. A nonexistent helper
reference in the first Pipeline draft was replaced with an existing checked Boolean
lemma. A driver invoked without required arguments failed its usage assertion. Two
outer container invocations timed out without complete receipts; later supervised runs
completed normally. One valid-symlink control correction improved the test target.
Two trailing blank lines in new Mask/Generated files were removed before the final
complete source/control rerun; hashes and the earlier six-law run remain preserved.
No production statement, earlier accepted law, compiler or intended domain changed.

### Remaining obligations and trust

The optimized final-filter bypass question for actual generated castling is now resolved
at source level. The next substantive target is to connect the **actual** start, transit
and child-destination check predicates to independent attack geometry under explicit
Board/king/table conditions. Correct historical rights, independent metadata rules,
legal reachability, whole-generator soundness/completeness and duplicate freedom remain
separate. Executing an obligatory check is not proof that the check's chess semantics
are correct.

Self-review only. Pinned checker/Base, native lowering/storage, ABI, C/C++ compiler,
libraries, OS and hardware remain trust boundaries. Existing strict-TypeScript,
snapshot-lowering and closed-builder limitations are unchanged. No production runtime,
old accepted law, compiler input, permanent workflow, routine perft, search, model/GPU,
training or benchmark changed. No additional Python application responsibility moved
into Bend; export, external references, data/control/training and transitional
C++/LibTorch/AOTI remain dependencies.
