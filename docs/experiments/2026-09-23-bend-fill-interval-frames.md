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
