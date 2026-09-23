# Actual fill-loop interval certificates and frames

## Baseline and acceptance record

Continue #833 at `d8de6e46fd04c4b34385d754224df37722a6a118`, complete tree
`99f19d5ef816a238e89da12ec19529088943012d`. Refresh discovered that #833 had
already qualified the relative-address continuation of #829; it was not redone.
Repository guidance/development/branch lifecycle and the experiment index were
read. The complete source archive and original commit were verified before
creating an isolated worktree. Earlier overlapping branches are untouched.

Compiler pin stays `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 + U64,
84 inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

This record follows constructive development and local checks, and precedes
hosted qualification. It is not a backdated preregistration. Acceptance requires
all six public contracts and their consumer, 20 reviewed controls, the unchanged
89-law/176-control parent, four native modes, original compiler/pin checks and
unchanged whole-repository lint. No timeout, malformed mutation or missing tool
is a passing result. No merge, deployment, live process or model work is authorized.

## Six public contracts

| Law | Actual result / explicit preconditions |
| --- | --- |
| `interval_clear` | Complete depth-17 shape, bounded query/end, mathematical count plus start <= end, and query outside the interval derive the existing clear-write-path predicate for every increment. |
| `interval_fill_preserves_query` | Those numeric premises imply query-value preservation through actual Tables.fill, with arbitrary slider parameters and initial contents. |
| `chess_fill_clear` | Valid key and bounded outside query imply the clear certificate for the actual full block count; count/end certificates are produced internally. |
| `chess_fill_preserves_query` | Actual full slider-fill invocation preserves every bounded outside query. Header writes are excluded. |
| `fill_preserves_later_lookup` | A full earlier block fill preserves a later block's actual PEXT-address query, for arbitrary occupancy; callers supply no index/separation certificate. |
| `fill_preserves_reserved_query` | A full actual slider fill preserves any query below 512. |

New structural Word induction proves the actual bounded U32 increment's Nat
observation and the Nat observation of actual ripple addition with no final carry.
The existing widened prefix endpoint theorem supplies the carry condition.
Nat induction then follows the real incrementing fill schedule and constructs
all separation premises. The actual affine implementation is connected through
existing certified reification/frame results, not replaced by a shadow engine.
No new axiom, unsafe dependency, hole or foreign equality witness is introduced.

## Recorded local checks and development failures

The primitive increment, addition, numeric-order, clear-induction and real-array
frame modules checked successfully. The first full public proof command failed
at `later`: its interval tuple was incorrectly marked duplicable (`+range`),
although the inferred tuple is Type rather than Data. Removing that binder marker
preserves the theorem statements and accepted domains. The corrected full consumer
is checked separately; no success for that command is presumed here.

All 20 mutation controls pass in the final controls-only development run. The
first control run correctly rejected a malformed mutation which duplicated an
unmarked U32 argument. The mutation now explicitly duplicates that argument in
its disposable copy. A second run stopped because two real fill-body mutations
failed at the inherited storage/Build.fill refinement rather than the predicted
model helper. The final gate requires that exact observed body-refinement layer;
these two controls are not claimed to exercise a new theorem first. Eight other
semantic controls fail at the new arithmetic/schedule refinements. Ten policy
controls cover manifests, certificate producers, unsafe/foreign/hole/symlink cases.
Original nonzero development logs are retained separately from final evidence.

The new native verifier passed all four modes locally: 1024 rows each, 768 full
block executions, 128 one-write cases and 128 zero-write cases. It observed 768
protected query results and 256 intentional overwritten-query results per mode,
performing 646016 actual fill writes per mode in addition to metadata builds.
All 128 keys are covered. Seven malformed/out-of-domain requests are rejected.
Every output SHA-256 is
`b87290fe69a8cc25fe4cc9f7a53688d17a22f983c9b309fbf560d4961a1a389b`.
The native report observes selected values/capacity/metadata, not every array cell.
Build modes repeat cases; their counts are not disjoint datasets. No proof model
or expected oracle data is executed inside the candidate.

The original compiler source gate passed (16 laws, seven controls), as did all
12 compiler-pin tests. Local whole-repository lint failed because Ruff,
Basedpyright and Vulture are absent; hosted locked-environment lint is required
to close this gap. Neither the old parent's lint pass nor local native success is
relabeled as qualification of the new full source aggregate.

## Remaining P2 work and trust

These contracts protect values through a full **slider-fill component**, not
through all header writes and the whole final table/extras pipeline. Prefixes
are certified expressions; complete correctness of every stored header remains
separate. Prove the computed value at each enumerated in-block address, metadata
initialization/preservation, and independent blocker-ray lookup equality next.
Generic interval endpoints are strictly below allocation capacity, and complete
shape is required. Source query equality is not physical pointer/lifetime safety.

Self-review only, not independent review. The pinned checker/Base, native lowering,
allocation/ownership/lifetime, effects/ABI, toolchain, libraries, OS and hardware
remain trust boundaries. Compiler TypeScript, diagnostic-depth and raw-internal
C-lowering limitations are unchanged. No production code or additional application
responsibility moved from Python to Bend. Export, external references, data/control
orchestration and training remain dependencies; C++/LibTorch/AOTI is transitional
inference. No full-engine/model/GPU, perft, training, benchmark or strength result.


## Completed hosted qualification

Hosted run **35888318491**, workflow commit `2a611c4f4580e0feb046b114eff48ccece60348a`, qualifies exact source `cf0dea478acaf43c9f1bf95d1b8d7ccc1a4d4925` and tree `e3b5dc68e284190bbad06b7a1740aede33b34add`. Fresh unchanged parent and new focused jobs together pass **95 laws and 196 rejection controls**. Four native modes, original compiler source/pin checks and unchanged whole-repository lint pass.

The proof jobs are intentionally parallel and separate: unchanged relative/verify.js checks all 89 prior laws and 176 controls; fill_frame/focused.js checks six new contracts, their importing consumer and 20 controls. The dependent publication job verifies both reports and records the aggregate. The sequential fill_frame/verify.js convenience wrapper was not additionally run. No inherited obligation is dropped and no historical report substitutes for these fresh executions.

The corrected complete local consumer and all 20 controls passed in 615.93 seconds after removal of the invalid +range binder. Hosted focused data matches its local canonical identity exactly. The native report matches locally except the independently recorded C compiler version. Every one of 268 native-source manifest entries is unchanged; all proof/native recorded source hashes match.

Native: 1024 rows per mode, all 128 chess blocks, 768 full-count, 128 one-write and 128 zero-write calls; 768 protected and 256 deliberately overwritten queries. Each mode executes 646016 fill writes in addition to metadata builds and rejects seven invalid requests. Tools: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1). Clang selection is native-step-only; the locked Python 3.13 CPU environment uses its normal compiler with uv 0.12.10. Ruff/Basedpyright/Vulture pass unchanged, resolving the historical local missing-tools gap.

The candidate reads actual metadata and calls the real fill body, while the reference uses independent coordinates, BigInt deposition and integer prefix sums. Observations include selected values, first/last block entries, metadata and capacity, not every returned array cell. Modes repeat fixtures, not disjoint datasets or exhaustive native U64 coverage.

This establishes interval-derived clear-path certificates and actual full slider-fill frames, including later-lookup and reserved-query preservation. It does not prove the preceding header writes, all stored metadata, every computed in-block result, the final all-block/extras state or independent blocker-ray lookup equality. Complete shape and the stated strict bounds remain essential.

Postqualification edits add reports, the experiment index and migration inventory only. Full logs are retained for 30 days in the fill-frame artifacts and bend-fill-frame-qualification; compact reports are committed. Self-review only, not independent review. No production code, earlier accepted law, compiler input, existing test, permanent workflow or routine perft-budget changes. No Python application responsibility moved, model/GPU/training/strength/performance result or native pointer/lifetime theorem is implied.

Publication fast-forwards `feat/bend-fill-frame-20260923` only after all needed jobs succeed and both parent/source heads still match. No merge, force push, deployment or live-process operation. Earlier stalled/overlapping candidates and compiler TypeScript/diagnostic/raw-internal-lowering limitations remain separate and unchanged.
