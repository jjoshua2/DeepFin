# Slider-mask populations, block sizes and lookup-index bounds

## Historical local scope and source basis

Targets PR #807 at `33c363ba6715956fd9323a6032b1a6c4a2b916fe`. Initial refresh found
`7aad3d0dd3fe20ba1e5524ce99f3ed89b4fef2c1`; the subsequent head changes only
`tests/test_derive_parallel.py`. No production source, existing proof, compiler,
configuration, live process, branch or test budget is changed by this patch.
No remote branch/commit/PR was created: the available GitHub connection has read
operations but no publish action. These sources and this record are delivered in
an apply-ready patch, not described as a published pull request.

Local verification used the archived #806 source snapshot plus byte-for-byte
restored current #807 proof dependencies. Restored files were checked against
GitHub blob IDs; comparisons confirm the required older inputs are unchanged.
This is not a claim of a complete checkout or whole-repository qualification of
the latest CI repairs. The modified documentation was reconstructed and checked
against its current base blob before creating the patch.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`;
all 84 inputs retain `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The protected checker and Base are untouched. Existing fork strict-TypeScript
failures remain separately documented; no passing strict check is inferred.

## Accepted contracts and implementation link

Eight new universally quantified contracts cover the actual 128 slider keys,
with satisfiable domain `Nat.is_lt(U32.to_nat(key),128n) == True`.
Rook keys are 0..63; bishop keys 64..127. Occupancies are arbitrary U64 values.

| Law | Guarantee |
| --- | --- |
| `mask_population` | Actual `Tables.slider` mask population equals independent file/rank ray-length geometry. |
| `tight_population_bound` | At most 12 rook bits or 9 bishop bits; consumer witnesses attain both bounds. |
| `shift_in_range` | Actual population used as U32 shift exponent is below 32. |
| `size_positive` | Actual U32 shift result is nonzero. |
| `tight_size_bound` | Size is at most 4096 for rooks or 512 for bishops. |
| `size_matches_capacity` | U32 size observed as Nat equals the existing proved mathematical capacity. |
| `full_index_bound` | Full PEXT value is below the block size for every occupancy. |
| `lookup_index_bound` | Actual imported `Sliders.pext_index` is below the block size for every occupancy. |

The scalar observers call the real mask implementation and reproduce the exact
inline size expression in `Tables.tables`; the source gate checks that link.
They are not a replacement table builder. No claim is made that all Array
operations have been refined merely because the scalar expressions agree.

`Cases.bend` checks all 128 cases inside the source checker. Its answers are an
independent arithmetic geometry expression, not an injected table of host values.
`Domain.recover` connects the Nat split back to arbitrary U32 keys. Generic
structural lemmas and the inherited extraction theorem establish arbitrary-U64
occupancy bounds. This is a finite-domain source proof, not a native sample
presented as a universal theorem.

## Executed local results

The final source gate passes **48 accepted laws: eight new plus all 40 prior**.
It runs the unchanged inherited gate and **42 prior rejection controls**, followed
by **13 new controls**. It requires status zero and exactly `All terms check.`;
an unsafe-dependency warning with raw exit zero is rejected. The importing
consumer includes the prior ordinal proof and valid corner/center/last-key cases.

New semantic controls reject too-small rook/bishop limits, changed actual ray
endpoint handling, an always-empty runtime mask, a size shifted by 32, a lookup
index of 64 (out of range for the 32-entry b2 bishop block), and an off-by-one
independent count. Five missing-obligation/import/hole controls are explicitly
manifest/policy checks, not semantic counterexamples. The remaining unsafe
control exercises both import policy and the exact checker-output guard.

Native generic, forced-portable, native-target and UBSan all pass **2302
distinct operand pairs per mode** across all 128 keys. The independent reference
uses signed-coordinate ray walks and direct BigInt bit gathering. It compares
full masks, populations, actual sizes, full PEXT and actual U32 lookup values.
Each mode rejects five malformed/out-of-domain requests. Modes repeat the same
fixtures; neither arbitrary occupancies nor complete table arrays are exhaustively
executed here. Numeric prefix totals in the external oracle are labelled
reference-only, not a source proof about runtime offsets.

The original fork source gate separately passes its 16 laws/seven controls,
including cyclic-template rejection. All 12 compiler-pin tests pass. Local
Bun 1.4.2 and Clang 17 are used. The final source gate took 374.04 seconds;
this is proof-checking cost, not engine runtime performance.

The repository lint command was attempted but could not execute Ruff,
Basedpyright or Vulture because those tools are missing from this environment.
Its nonzero result and full log are retained. No passing whole-repository lint,
new hosted CI, full-engine build, perft, model/GPU execution or benchmark is
claimed. No permanent workflow or routine-test integration is added.

## Construction failures and corrections

An initial combined check rejected a proof parameter consumed twice; marking
that equality witness copyable repaired the affine use without changing the
proposition or restricting its domain. The complete finite mask-count lemma
had already passed independently.

The first aggregate attempt passed all 48 laws, then failed a negative control:
changing the lookup to the largest U32 caused a `term_higher` stack overflow
while the checker rendered its diagnostic. A crash was **not** accepted as a
semantic rejection. The control now uses 64, which still violates the real b2
bishop bound of 32 and produces an ordinary expected/observed rejection. The
checker was not modified; extreme diagnostic rendering remains an observed
limitation. The final complete gate, including the revised control, passed.

## Proof limits and next acceptance

This closes the mask-population and block-size part of P1, plus scalar index
bounds. It does not prove prefix-offset arithmetic, disjoint/initialized Array
regions, the actual `Tables.fill` write trace, or the full blocker-ray lookup
specification. Population equality is not full mask-geometry equality. Exact
universal agreement of the low U32 projection with the full PEXT value is not
one of these new laws; native fixtures compare it separately.

Next connect the existing ordinal law to lossless actual indices, exact prefix
offsets, and initialized/disjoint writes and reads against an independent ray
specification. Keep these obligations explicit instead of treating bounds or
native table comparisons as complete refinement.

No application responsibility has newly moved into Bend. Python remains in
export, external references, production/data orchestration and training;
C++/LibTorch/AOTI remains the transitional neural backend. Source proofs trust
the pinned checker/Base, with native lowering, affine runtime, effects/ABI,
C toolchain and hardware separate. Self-review only, not independent review.
The evidence directory retains reports, exact source identities and actual
failures/limitations. No binaries, compiler payloads, model data or private
traces are included.


## Hosted qualification and clean publication

Run **35681951103**, development workflow commit `2201ded911519896f89f5a23f6e5b7fa524dfae4`, qualified the exact saved patch against refreshed #807 `ef3ded0b652a09918abb1d79467b1ac56a8e1ce5`. The earlier no-publication and missing-lint-tool statements above are retained as historical local outcomes, not the current publication status.

The original patch was 103,636 bytes, SHA-256 `569a67a46c60aea0c5bc11f82886d4fc80eef629c83c4b33c2678c3c79f5ce52`. Its original 33c363ba baseline differs from the refreshed ef3ded0b baseline only by the canonical executable mode on resolve_latest.py, which is preserved. No patch source or accepted law changed.

All 48 accepted source laws and 55 rejection controls (13 new plus 42 inherited) pass. The hosted source report is byte-identical to the saved local report. The original compiler source suite and its seven controls, including cyclic-template rejection, pass separately; all 12 compiler-pin contracts pass.

The four native modes pass the same 2,302 distinct key/occupancy pairs and five invalid requests per mode. All 128 keys are covered; mode results and source identities match saved local evidence. Toolchain: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1). Repeated environments are not disjoint datasets or exhaustive native U64 testing.

The unchanged whole-repository ./scripts/lint.sh passes Ruff, Basedpyright and Vulture in the locked Python 3.13 CPU development environment. This resolves the earlier missing-tools gap for this exact candidate without ignoring failures or weakening checks. Original source hashes and lossless tool logs are committed under the evidence directory.

Publication recovery failures are retained: run 35681173158 rejected the earlier truncated/bloated unpublished payload before source application. Transfer-readback run 35681825701 detected one duplicated base64 character in the replacement fragment. The one-character transport repair is followed by exact fragment, full transport, decompressed patch and candidate source hash checks. No damaged payload was applied or used as proof evidence.

No production source, compiler/checker, old proof, existing test, workflow or perft budget changes are in the feature diff. No full-engine rebuild, model export/forward, GPU, training or benchmark was rerun. Source-checker diagnostic overflow observed by the historical extreme mutation remains documented; the checker itself was not edited. Fork strict TypeScript remains a separate inherited unresolved check.

Publication creates `feat/bend-slider-mask-bounds-20260921` only after the above checks, with parent `ef3ded0b652a09918abb1d79467b1ac56a8e1ce5`. No force push, PR merge, deployment or live-process change. Temporary development workflows and transport blobs are excluded. Qualification reports are retained in artifact `bend-slider-mask-bounds-qualification` for 30 days and in the clean commit. Self-review only; no independent review is claimed.

The next acceptance remains exact U32 compact-index correspondence, prefix arithmetic and affine writes/reads, then independent blocker-ray lookup refinement. Current size/range proofs do not establish those. Python export, production/data orchestration and training remain dependencies; C++/LibTorch/AOTI remains transitional inference.
