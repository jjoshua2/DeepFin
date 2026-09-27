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
