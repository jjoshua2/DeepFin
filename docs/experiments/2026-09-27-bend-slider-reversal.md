# Blocker-aware slider reversal

## Scope and deciding checks

Base: PR #910 at `ec9997838788ea5de34c81a7f97037ef47753723`.
Compiler: `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` (unchanged).

The two source contracts connect independent geometric masks and actual unmasked
`Tables.slider` computation to endpoint reversal under a single arbitrary occupancy.
Rook, bishop and their queen union are included. Source and target must be below64.
Occupied endpoints are allowed, and the first blocker is included. Only the strict
interior matters. A changed occupancy on reversal is deliberately outside the law.

The generic `Bits.path` proof reflects first-blocker-inclusive list scanning into
strict-prefix visibility. Structural Boolean/list induction preserves emptiness
under prefix reversal. Finite coordinate certificates cover64 sources/eight
directions and1,456 on-ray targets; they compare prefixes, never occupancy or
expected bitboard answers. Positive visibility supplies certificate membership.
Bidirectional implication yields Boolean equality, including false/disconnected and
same-endpoint cases. Existing checked slider refinement connects actual computation.

Before the native execution, the complete two-law consumer and the coordinate-only
certificate closure have each checked locally. The combined gate with all controls
is still in progress. Qualification requires that final combined gate, fail-closed
negative controls, actual native values, original compiler gates and configured lint.
No final qualification follows merely from this plan.

## Native experiment plan

Use unchanged `ray/probe.bend` to call actual unmasked slider computation. Independently
walk signed coordinates and include the first occupied target. For every728 aligned
unordered rook/bishop pair, enumerate all5,322 strict-interior subsets across pairs,
four endpoint-bit settings, and two off-segment-noise settings; also check queen union.
Add six global occupancies covering all ordered source/target pairs and all three
families. Deduplicate query inputs and relation cases, preserving exact counts.

These yield86,072 distinct actual rook/bishop queries and155,852 distinct relation
cases per clean build. Full masks must match geometry and every reversed relation
must agree. Generic, portable, native-target and UBSan repeat the same fixtures.
Queen native observations union actual rook/bishop values; they do not qualify a
new initialized queen dispatch. This is not a new initialized table lookup test.

Two actual-code corruptions must compile and execute normally: ignoring blockers
must still pass reciprocity but fail independent geometry; excluding the first
blocker must fail the endpoint-asymmetric relation and geometry. Nine malformed
bounded batches per build must reject without being credited for sanitizer crashes.
Host validation must remain enabled under optimized Python. Reports start
NOT_COMPLETED; a partial output transcript cannot qualify a pass.

Budget is one bounded source gate at a time, then serialized native builds, using
this isolated checkout and at most two worker threads. No training, GPU, search,
benchmark, perft escalation or live process is involved. Stop on any unexpected
acceptance, wrong output, source drift or timeout; retain the failed receipt.

## Boundaries

This is slider computation and coordinate reciprocity, not a new universal initialized
`attacked` forward-witness composition, full-generator starting/transit safety,
historical castling rights, all move classes or completeness theorem. Source array
and native allocation/lifetime correctness remain separately established boundaries.
The two contracts are related interfaces to a shared argument, not independent
mathematical discoveries. No earlier proof or production function is edited.

Initial local drafts exposed reserved `Kind` syntax and affine pattern-match ordering;
a first reflection draft used a non-decreasing helper self-call and was replaced by
explicit rook/bishop sub-compositions. Two incomplete exploratory reflection checks
were stopped without success credit. The first negative-control draft reached the
intended changed-prefix semantics at `Booleans.append_clear`, earlier than the asserted
`Bits.path` location; the diagnostic expectation was corrected, not a theorem weakened.


## Completed hosted qualification

Run **36343736223** passes both complete slider-reversal laws, the importing consumer and all17 controls, reproducible finite certificates,24 host methods in three optimization modes, four native builds under optimized Python, original compiler/pin checks and configured locked-environment lint on source `053f3d490d0fd774169e4b7fa38faea7519aafc0`.

The source laws quantify one shared arbitrary U64 occupancy, not a finite occupancy sample. They cover independent rook/bishop geometric masks and actual unmasked Tables.slider computations, plus the explicit queen union. Both endpoints are bounded below64. Either endpoint can be occupied: the first blocker is included, and only the strict interior must be clear.

Bits.path reflects generic list scanning into strict-prefix visibility. Boolean/list induction proves clear-prefix reversal. Nine generated source files certify64 source squares/eight directions and1,456 on-ray coordinate-prefix equalities, without occupancy terms or expected bitboard answers. Positive membership supplies the required finite certificate. Opposite-direction implications establish Boolean equality even when neither direction attacks. Computed reuses the unchanged actual slider refinement.

All17 controls pass: eight semantic/refinement failures, eight manifest/import guards and one synthetic warning-output check. The controls reject a wrong opposite direction, hit inclusion in the strict interior, ignored blockers, unreversed prefixes, omitted endpoint bounds, actual missing stop-at-blocker behavior and a different reverse occupancy. Some target supporting prefix or actual-refinement lemmas, while three mutate public law statements; not all replay the complete outer consumer. Crashes, parser/ownership errors, missing imports and timeouts do not count as valid semantic rejection.

Each native mode checks86,072 distinct actual unmasked slider queries,172,144 U32 mask fields and155,852 distinct relation cases. All728 aligned unordered pairs exercise every strict-interior subset (5,322 subsets summed over pairs), all four endpoint occupancies and two off-segment contexts. Queen cases union actual rook/bishop results. Six additional global occupancies cover all ordered endpoint pairs and all three families, including unaligned/equal endpoints.3,028 duplicate relation cases are removed. Modes repeat the same fixtures; native occupancies remain finite.

Complete masks must agree with an independent signed-coordinate forward walk, separately from reciprocity. Ignoring actual blockers passes all tested reciprocal relations yet fails the geometry oracle. Excluding the first occupied square fails geometry and reciprocity with asymmetric endpoint occupancy. Both actual-code corruptions compile and execute with generic flags before rejection. Nine malformed bounded batches are rejected per clean mode. The reused production-only ray probe performs computation, not initialized Chess.attack lookup, and never imports the new proof model or receives expected answers.

The native driver actually runs under PYTHONOPTIMIZE=2; its mandatory framing and result checks are tested in all three host optimization modes. A partial transcript cannot count as a complete comparison. The wrappers replace a writable requested report with NOT_COMPLETED before external checks. The24 host methods are not new formal laws or registered compiler controls.

The entire local consumer and all17 controls also completed together; all native modes and both mutations completed locally, as did original compiler and12 pin checks. Local lint failed only because Ruff,Basedpyright,Vulture were missing; its receipt remains in the review package. Fresh hosted configured lint passes with locked CPU dependencies and does not relabel that earlier failure. The 53-entry proof/test closure and all705 candidate native entries are fixed; all679 inherited entries remain unchanged.

Active evidence is modular207 laws/706 controls: retained205/689 plus newly executed2/17. The complete aggregate wrapper was not run. These two contracts are related interfaces to a shared proof, not two independently discovered mathematical facts. No complete initialized forward-attack witness, full-generator initial/transit safety or historical castling-right result is inferred.

Source/checker, native lowering/storage, ABI, C toolchain, OS and hardware remain trust boundaries. Self-review only; no independent reviewer is claimed. No production function, old proof, compiler source, permanent workflow, Python application responsibility, model/GPU, training, search, benchmark, perft-budget increase, merge or deployment changed. The temporary workflow remains outside the feature diff.

Next composition is to convert the existing initialized target-centred attack witness into an attacker-origin witness using both non-slider and blocker-aware slider reversal, then connect the remaining accepted castling stage checks. This increment closes the slider reversal prerequisite, not those final compositions.
