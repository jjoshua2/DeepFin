# Complete-tree routes and actual other-index writes

## Scope and acceptance

Parent #827: `e680c126908d1e816f31c38b7e3255e196e6f174`, entire tree
`b9a31c6971f4eda33a5c486b3c2b6dacade3ea54`. Read-only source recovery run
35762619330 provided the exact archive and commit object; the reconstructed Git
tree and commit were verified. The separate earlier six-law interior-address
package is retained, not substituted or applied. Current repository instructions,
branch lifecycle, development guidance and experiment index were read.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 + U64,
84 inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
No production runtime, existing proof/test, compiler or permanent workflow change.

This acceptance record follows constructive local development and precedes hosted
qualification; it is not a backdated preregistration. Hosted acceptance requires
all 79 inherited laws/140 controls, four new laws/16 controls, four native modes,
original compiler source/pin checks and unchanged whole-repository lint. Checks
are bounded and opt-in. No model, GPU, training or perft workload is introduced.

## Four public source contracts

| Law | Guarantee and premise |
| --- | --- |
| `allocation_complete` | Actual Array.new has the complete constructor shape, for arbitrary source depth and seed. |
| `table_pipeline_complete` | Actual tables followed by extras preserve that initialized complete shape, for arbitrary source counts and scalar parameters. |
| `bounded_complete_routes` | An actual depth-17 complete array and two distinct bounded indices imply the existing normalized path-separation predicate. |
| `bounded_other_index_write` | Under those shape/bound/inequality premises, an actual write preserves the other queried value and returns the complete updated array. |

The last two laws do not assume separation or the desired read equality. The
consumer constructs the shape certificate from the actual initialized pipeline.
Internal arithmetic proofs establish child bounds after actual half subtraction
and injectivity of actual ripple subtraction. Depth induction then proves distinct
routes for complete shapes through depth 17. The only finite case splits are the
17 half-capacity constants and a Boolean full-adder table, not U32 address pairs.
Existing normalization and affine representation/frame proofs are reused unchanged.

## Local qualification

The four-law consumer and all 16 new controls pass. Eight ordinary semantic
rejections cover incomplete shape specifications, broken cross-branch/right-route
semantics, equality instead of inequality, inclusive bounds, replacing shape by
capacity alone, a ragged actual allocation and a destructive actual table base
case. Eight policy controls preserve obligations/imports and reject holes, foreign
witnesses, symlinks and unsafe output even with raw status zero. Missing-file
errors, crashes and timeouts are not accepted as semantic rejection.

All four native modes pass 152 fixtures using actual 131072-cell public allocations:
125 different in-range pairs, 17 same-index pairs and ten out-of-range wrapping
cases. Every divergence bit 0..16 is exercised. Six malformed requests reject per
mode. The independent reference predicts query values using unsigned remainder
and seed/write expectations. Modes repeat the same fixtures, not disjoint datasets
or exhaustive U64 coverage. Query values/capacity/normalization are observed, not
all cells or internal raw helper execution. No fresh full table-builder native
qualification is attributed to these focused tests. Local Bun 1.4.2, Clang 17.

Local development used the focused gate, not a claimed full 83-law aggregate.
The unchanged parent aggregate and repository lint are reserved for hosted
qualification on the exact source. Their prior successes are not new passes.

## Development failures and corrections

Early constructive drafts had binder-use/type/definition-order errors; final
accepted functions retain the original intended complete-tree and strict-bound
domains. A direct depth-17 route composition exceeded a 20-second local command
limit. The final proof keeps depth symbolic until closed scalar specialization;
no checker change or proof-domain narrowing was made. An initial mutation driver
looked for the expanded Array.new spelling rather than the actual `[v : T^p]`
notation. That run failed its mutation assertion, not the theorem or compiler.
The driver now mutates the actual source bytes and requires ordinary proof failure.
Its original failed log is retained with the local evidence package.

## Remaining scope and trust

Complete-tree route injectivity is now source-proved, not inferred from reported
capacity. The consumer retains the ragged capacity-four alias at indices two/three
and an excluded-endpoint alias. No native qualification of ragged shapes is claimed.
The parent report's raw-internal C-lowering limitation remains unchanged.

Actual prefix-plus-relative-index no-overflow bounds and derivation of every fill
clear-write certificate remain open. Combining these with enumeration/frame laws
to prove final computed contents and independent blocker-ray lookup is still the
P2 acceptance target. The shape theorem is not a claim that arbitrary loop counts
are chess-correct, that every initialized value is computed, or that huge native
allocations terminate successfully.

Self-review only, not independent review. Source laws trust the pinned checker and
Base. Native lowering, array allocation/ownership/lifetime, effects/ABI, toolchain,
OS/hardware and model libraries remain separate trust boundaries. Compiler-fork
strict-TypeScript and diagnostic-depth issues are not fixed or suppressed.
No Python application responsibility moves into Bend; export, external references,
data/control orchestration and training remain dependencies. C++/LibTorch/AOTI
remains transitional inference. No engine/model/GPU/perft/strength/speedup claim.


## Completed qualification with separate failure provenance

Qualification is complete on exact source `868335129a501b677ca916d2bbdc779a1b2e22e9`: source run **35764846623** passed all **83 laws and 156 rejection controls**; run **35766677592** passed all four native modes with explicit Clang 18 and original compiler source/pin checks; final run **35767511451** verifies those retained exact-source results and passes unchanged whole-repository lint. The first two overall workflow verdicts remain red and their failures are preserved.

First run 35764846623 passed the full inherited/source aggregate, then default clang with -march=native emitted six invalid-feature-combination warnings involving AVX10.1. The strict empty-stderr check rejected them. Generic and portable modes had passed; native-target execution and later steps did not receive successful verdicts. The original failure is preserved, not classified as an engine-value mismatch or relabeled green.

Second run 35766677592 retained and verified the full source report, rechecked the focused four-law/16-control gate, and passed every native mode with explicit installed clang-18, without altering flags, output expectations or warning policy. Original 16 compiler laws/seven controls and all 12 pin contracts passed. However, setting CC=clang-18 job-wide also changed the external Python editable-build compiler; that installation failed because omp.h was absent. Lint and publication were skipped. This environment-scoping mistake is retained separately; no compiler or source theorem was changed.

Lint-publication attempt 35767241042 stopped before lint when gh refused ANSI-bearing historical job-log output, even though it was redirected. The recovery saves the unchanged raw log directly to a file using curl --output; it never renders escape sequences to a terminal or disables the gh terminal guard. That attempt performed no new proof/native/lint execution.

Final run 35767511451, workflow 09de2f84f304d4bbe522db310ed9aaacf408984a, downloads artifact 10712168622 with ZIP SHA-256 c91f200ed814f33886da5ed3eab7da579198f8e42de1ac8ab16657b81a69332b, validates both recorded job/step outcomes, the source/native results and all 239 source hashes. It does not redundantly rerun completed proofs or native checks. The Python development install uses its normal compiler with CC/CXX unset; locked Python 3.13 CPU dependencies and uv 0.12.10 are retained. Unchanged Ruff, Basedpyright and Vulture now pass. No error or warning is suppressed.

The first full-source report is from artifact 10711688170, verified ZIP SHA-256 4ef8329318f900a341c2f3ce4663e1788918c1d7286c4aeda487174a3c95195a. The focused local and hosted reports agree exactly; the full native report agrees except C compiler identity. Candidate tree remains 2e6428e87a347679b2f39b5eacc0a0ea58e421fc. Subsequent changes are evidence/documents only.

Native results: 152 rows per mode, 125 distinct bounded pairs, 17 same-index and ten deliberate out-of-range cases, all 17 divergence bits and six malformed rejections. The candidate uses supported public APIs with complete 131072-cell allocations. Query values/capacity/normalization are observed, not all cells, direct raw routines or the full table builder. Repeated fixtures across environments are not disjoint or exhaustive coverage.

Compact reports, lossless lint/pin logs, source identities and separate-run provenance are committed. Full final/install/earlier environment logs are retained in the 30-day artifact bend-complete-routes-final-qualification. Publication fast-forwards feat/bend-complete-routes-20260922 only. Temporary workflows are excluded; no merge, force push, deployment or live-process change.

Complete shape now derives distinct normalized paths for distinct bounded addresses, and the actual initialization/table/extras pipeline supplies shape. Prefix-plus-relative-index arithmetic, every fill-clear certificate, final computed contents and independent blocker-ray lookup remain separate P2 obligations. Shape preservation for arbitrary source depths/counts does not prove native feasibility, allocation/lifetime or physical pointer identity.

Self-review only, not independent review. No production code, old accepted law, compiler input or ordinary test budget changes. The earlier six-law local candidate, raw-internal C-lowering limitation and compiler TypeScript/diagnostic issues remain separate and unchanged. No Python application logic moved; export, external references, data/control and training remain, with C++/LibTorch/AOTI transitional inference. No engine/model/GPU/perft/training/strength/performance result is added.
