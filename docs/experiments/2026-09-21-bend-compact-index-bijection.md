# Compact-index range and bounded PEXT/PDEP bijection

## Scope and hosted acceptance

Continue PR #805 at `d6503d8d6a0c2b8fa938477b755e8303c0af49f1`, without
merging or deploying. Compiler remains the exact `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`
Bend 2.0.21 + U64 pin and its unchanged 84-input source fingerprint.

This record follows constructive proof development and the first local checks;
its hosted acceptance is fixed before hosted qualification. It is not represented
as a preregistration preceding those local checks. No learning experiment or live
change is being conducted.

The increment is the missing bounded reverse PEXT/PDEP law and extraction bounds
identified in P1. Prove actual public U64 operations, keep the strict numeric bound
and connect its capacity to actual popcount with mathematical Nat exponentiation.
Derive masked/bounded injectivity and a PDEP witness for every valid compact word.
The existing `Subsets.next` order/coverage target remains an accepted obligation,
not silently replaced by these representation lemmas.

Acceptance: unchanged prior source gate and its eleven negative controls; all
nine new laws, a non-vacuous importing consumer and fifteen new negative controls;
four-mode raw native agreement with independent BigInt scatter/gather, all mask
populations 0..64, both valid and out-of-range compact inputs; unchanged compiler
and application hashes. Exact equality, no floating-point tolerance. Run the
whole-repository lint gate once on the final hosted candidate. Publish only clean
source and compact evidence on a new branch stacked on #805.

Budget: one compiler at a time, small source/native probes; one bounded hosted
confirmation (ten-minute limit) plus correction only for an observed failure.
No full engine/model rebuild, model export, perft, training, GPU work, live changes,
merge or deployment. No permanent workflow or ordinary-test cost increase.

## Changes and laws

All new executable test/proof sources are under `standalone/proofs/index/`.
Production `Subsets.bend`, `Tables.bend`, Chess/Search, neural code, the previous
accepted LAWS/PROOF files, their verifier, and the inherited U64 proof files are
unchanged. **No additional application responsibility moves from Python to Bend**.
The new mathematical `Bits.clip` is a specification, not an implementation swap.

Nine new universally quantified contracts:

| Law | Actual guarantee |
| --- | --- |
| `capacity_matches_popcount` | Capacity equals mathematical `2^toNat(popcount(mask))`, including 64 set bits. |
| `extraction_bound` | The actual PEXT result is strictly within that capacity. |
| `compact_projection` | Unbounded PDEP then PEXT discards exactly the bits above the mask population. |
| `compact_roundtrip` | Strictly bounded compact inputs survive actual PDEP then PEXT unchanged. |
| `deposit_in_mask` | Actual PDEP never creates bits outside its mask. |
| `deposit_injective` | Equal deposits of bounded compact values imply equal compact values. |
| `masked_extract_injective` | Equal extractions of mask-contained bitboards imply equal bitboards. |
| `compact_coverage` | Every bounded compact U64 value has an actual PDEP witness with membership and inverse certificates. |
| `sequence_index_bound` | Every state of the imported production-step recurrence has an in-range compact index. |

The aggregate source gate checks **33 accepted laws: nine new, eight prior engine,
and 16 inherited U64**, not 33 new laws. The bijection/coverage here is between the
bounded compact representation and masked bitboards; **it is not a proof that the
carry-rippler sequence visits all those bitboards**.

Generic Word induction covers arbitrary width. A separate 65-count-value lemma
bridges the existing popcount's U32 representation without enumerating masks.
`consumer.bend` establishes non-vacuity for zero under arbitrary masks and reuses
the bounded theorem at cross-half/high-bit examples. It records the empty-mask
counterexample to the unrestricted inverse. No holes, unsafe/foreign witnesses,
new axioms, compiler edits or weakened accepted statements.

## Local readout

Source gate: **PASS**, nine new laws, importing consumer and all **15 new negative
controls**. The unchanged parent gate also passes its eight laws, 16 pinned U64
laws and 11 controls. Separately reran the original compiler's `u64_proofs.js`:
all 16 laws and seven negative controls pass, including the cyclic-template case.
All 12 unchanged compiler-pin Bun contracts pass.

New controls cover omitted/missing laws and proofs, missing parent proof, holes,
zero or incorrect capacity growth, inclusive/unrestricted inverse preconditions,
wrong projection, unsafe/foreign dependencies and three Base/public-operation
mutations in disposable copies. The unsafe helper returns raw CLI status zero
with a warning and is correctly rejected. No mutation edits the real checker,
compiler or inherited proofs. Proof development corrected local rewrite direction,
helper order, binder-use and function-quantity errors without altering the accepted
nine law statements.

Native gate: **PASS**, Clang 17.0.0 and Bun 1.4.2, generic/portable/native/UBSan.
Each mode checks the same **1,590 distinct operand pairs**, including **702 bounded
and 888 out-of-range inputs**. All populations 0..64 are represented, including
empty masks, bit63, cross-half layouts, full-width masks, boundary compact values
and seeded arbitrary masks. These are bounded cases, not exhaustive U64 pairs.
All outputs match the independent BigInt scatter/gather reference exactly.
Four malformed/budget requests are rejected in each mode. Shared output SHA-256:
`3193588bdbf593f75cd7b43844808823003adf86f775f2681295ce76754d2afb`.

The native probe observes real U64 operations; it imports neither proof
specifications nor reference answers. Native testing is separate from the source
proof, and still trusts the compiler/runtime/ABI/toolchain/hardware. No full engine
or neural execution is inferred from this small probe. The engine's executable
sources are unchanged from #805, whose existing static-runtime evidence remains
historical evidence, not a newly executed test here.

## Refreshed parent and compiler status

At continuation start #805 remains open/unmerged at the recorded head and has no
review comments. Its ordinary CI run **35628265140** has now passed lint, PEXT
sliders and the ordinary/capped CPU suites. Separate old probe run **35628265160**
is red at its `Install latest Bend` step; native parity never ran there. This
increment does not change or diagnose that older installation workflow.

Compiler fork PR #2 remains draft at `aaeb9bc...`: recorded source/native/budget
passes and separately failing inherited strict TypeScript diagnostics. Those
checks are not suppressed, and fork main is not substituted for the U64 pin.
Self-review only; proof checking is not an independent human/model code review.

## Next decisive acceptance and remaining migration

Still prove for mathematical Nat `i < 2^k`:

```text
value(pext(SubsetLaws.at(i, mask), mask)) = i
```

The next required bridge is actual two-limb `U64.sub` borrow semantics to the
masked compact successor, then induction on the actual recurrence. These new
bijection/range lemmas support deriving uniqueness and coverage; they do not
assume the successor/order result. P2 must separately relate real affine table
writes/offsets and Chess lookup to initialized logical regions and ray geometry.

Python remains in model export, external references, production/data orchestration
and training. The neural backend is still transitional C++/LibTorch/AOTI, not a
Bend-authored model. Trained-checkpoint/GPU qualification, scheduling/batching,
production Gumbel semantics, selfplay and Bend model/trainer conversion remain.
No speedup, playing-strength or end-to-end verification claim.


## Hosted qualification

Run **35643399130**, temporary workflow commit `c15d2ff7a32f5bd14f694d6a5083a9d2856b67a4`, passed all qualification stages before publication. The clean implementation is based directly on #805's `d6503d8d6a0c2b8fa938477b755e8303c0af49f1`. Every candidate source digest matched the locally tested bytes before and after qualification, and the unchanged compiler fingerprint passed again.

All nine new universal laws and 15 new negative controls pass; the aggregate also retains eight prior engine laws, 16 original U64 laws and 11 prior controls. The new source report is byte-identical to its local report. The original fork source gate (16 laws/seven controls) and all 12 compiler-pin contracts pass separately.

The four native modes pass the same 1,590 distinct operand pairs each, with 702 bounded and 888 out-of-range compact inputs and four malformed/budget rejections. All masks populations 0..64 are represented. Results and source hashes match the local gate; hosted compiler: `Ubuntu clang version 18.1.3 (1ubuntu1)`, Bun `1.4.2`. The four output digests match exactly. These are repeated environments, not disjoint data sets.

The unchanged whole-repository lint gate passes Ruff, Basedpyright and Vulture in the locked CPU development environment. No lint failure was ignored. Original lint and pin logs are stored losslessly as JSON strings with SHA-256 identities so trailing log whitespace does not alter the Git diff gate.

No production executable source changed. No full engine build, perft, model export/forward, GPU, training, performance measurement or full inherited native suite was rerun. The small new native probe and source-law gates are the newly executed evidence; #805's static-engine and broader native results remain historical. Source proofs are conditional on the pinned checker/Base, and the representation bijection is not the still-missing carry-rippler ordinal theorem.

Reports and exact candidate identities are committed under this record's evidence directory and retained in the 30-day workflow artifact `bend-compact-index-qualification`. The source patch SHA-256 is `f4fa71f0969e94298e7a10663c26d1c61544d86004ea891db08654a26750da3a`. Publication is create-only on `feat/bend-compact-index-bijection-20260921`. No force push, merge or deployment. The temporary workflow and transport blobs are absent from the clean branch. Self-review only; no independent review is claimed.
