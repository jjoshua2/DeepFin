# Initialized castling producer and checked-filter sequence

## Exact baseline and intended acceptance

This increment starts at PR #899, `22aa6449cf4fc69506442e9396837f342cf061e6`,
tree `fbae01e969299855cd99aeb9a1ed38d37d912f8c`. All 607 inherited native-source
entries and all earlier public laws are retained unchanged. The compiler remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

This is a local continuation. No remote push, PR creation, merge or deployment has
occurred. Qualification below is pending until actual completed receipts are added.

## Public contracts and the remaining boundary

`initialized_castle_producer_matches_geometry` states that actual `castle_side`
returns the original initialized array and the caller's exact list, adding the
computed castle move at its head exactly when both starting and transit geometric
checks are false. The input list may already contain that move or other unsafe moves;
it is not silently treated as safe or deduplicated.

`initialized_filtered_castle_matches_three_stage_geometry` states that the same
actual producer, from an empty list, followed by actual `filter_legal` returns the
original array and exactly one computed castle move if all three geometric check
Booleans are false, otherwise the empty list. `Runtime.run` is the tiny proof/test
adapter expressing these real calls. It does not substitute for production
`legal_moves`, which has an ordinary scan, both castling calls and optimized final
filtering. The whole-generator initialized-safety theorem remains separate.

Both public statements require the actual input producer guard, initial partition
consistency, exactly one moving-side king at home, valid Boolean turn, and explicit
initialization depth17/128 slider blocks/64 extras. The initial seed is arbitrary.
The source input and output table values are included in the equality. Callers do
not assume correct query answers, stored masks, intermediate table equality or
output king singletons. The public proofs derive the internal query certificates
from #899's checked initialized-stage result. Nonempty satisfying examples for all
four side/route choices are imported from the earlier king suite.

The stage Boards are initial, source-to-transit ordinary update, and source-to-final
castling update, each child made directly from the original Board. Queries always
use the original moving side, not the child's flipped turn. The actual producer
checks both initial and transit; final checking happens only when it retained a
candidate. The proof is about that real sequential table threading, not a tuple of
three unrelated calls each allowed to reset or lose its table.

The independent geometric answer is the existing target-centred coordinate/ray
witness. Universal equivalence to attacker-origin geometry, legal rights/history,
king counts from arbitrary FEN and whole-generator soundness/completeness remain
separate. No blanket safe-castling claim is made for arbitrary inputs.

## Source-gate design

The full importing consumer checks both new law bodies and the complete retained
initialized-stage proof graph. Eight deliberately small mutation executions target
`Wire.bend`'s actual-code continuation equalities: ignore starting or transit checks,
check the unmoved transit Board, check the flipped side at transit or final stage,
ignore the final check result, erase a returned-table slot, or bypass the final
filter in the adapter. Those are certificate-wiring/control-flow rejection checks,
not eight re-executions of the expensive initialization proof. Eight manifest/import
policies and one synthetic warning-output unit make 17 planned controls. Missing
files, parser/ownership failures, signals and timeouts are not semantic rejection.

The new import scanner normalizes parent-directory components without resolving a
symlink away before comparing lexical and canonical paths. Its symlink control
must be rejected by that policy. No prior gate is edited or its earlier pass
reclassified by this addition.

## Native verification design

The candidate builds actual `Tables.build()`, plants a nonzero marker at unused
slot131071, and threads the same array through actual single-producer and producer-
plus-filter calls. Complete ordered lists and all19 raw fields of every returned
move's child Board are compared with the existing independent square/set and
forward-coordinate attack reference. Two marker reads per request observe retained
state. The sentinel instrumentation means this is not asserted to be a literal
uniform-seed instance of the symbolic source initializer; referenced attack slots
are those from the actual builder. Source full-array equality and native sampled
state observations are separate evidence layers.

Fixtures contain clean routes, rights-disabled tails, enemy pieces at every allowed
square for every piece kind, and deterministic mixed Boards. All input Boards have a
singleton moving-side king at home and Boolean turn. Some guards deliberately fail;
those native diagnostics are outside the public source laws' true-guard premise.
The verifier explicitly requires four discriminating guarded patterns: safe,
starting-only, transit-only and destination-only attacks. The reports separately
record all eight patterns actually observed in the final fixture set.

Mutant builds execute actual code which ignores initial checks, ignores transit
checks or erases only the unused table marker. The marker-only mutant is required
to preserve every output move list and Board value: only retained-state reads may
reject it. That is evidence the state observation adds information beyond answer
comparisons. The marker is not a native whole-array or lifetime proof. Modes repeat
fixtures; mutation builds are generic only. No perft budget is increased.

## Execution history so far

An initial Wire draft attempted an inappropriate match after equality substitution;
its parser rejection was retained, corrected by a separate Boolean helper, and not
counted as a proof/control pass. The lightweight Wire and actual-producer Bind
modules then returned exactly `All terms check.`.

The first complete source consumer was terminated with signal9 after400.577 seconds
while native compilation was also running. It produced no diagnostic or success
output. The container reported one OOM kill; this is recorded as a resource failure,
not a proof pass. Heavy checks were then serialized and the unchanged pinned compiler
was invoked with Bun's documented installed `--smol` smaller-heap mode. The theorem
statements and compiler source were not modified to accommodate the resource limit.

The first complete native run passed all four modes and all three mutations. Its
report is retained as an earlier completed execution, not relabelled as a final
source-gate result. Final unchanged-source rechecks and exact statuses follow below.

## Trust and application scope

Self-review only. The pinned checker/Base, native lowering/storage, ABI, C/C++
toolchain, libraries, OS and hardware remain trust boundaries. Existing compiler,
snapshot-lowering and literal closed-builder limitations are unchanged. No production
runtime, old law, compiler input, permanent workflow, routine perft, model/GPU,
training, search or benchmark changes. No application responsibility moves from
Python: external references, export, orchestration and training remain dependencies,
with C++/LibTorch/AOTI transitional inference.

## Source acceptance before final evidence commit

The complete source consumer passed in657.312 seconds using the installed Bun
--smol option. Both new public laws and all17 classified controls then passed
together in the final focused invocation, covering225 hashed proof/test dependencies.
All eight intended semantic errors were rejected at the specified Wire functions;
all eight policy corruptions and the synthetic warning check also failed closed.
The earlier native run passed; final native/compiler reruns are separate receipts
and are not inferred from this source result. Repository lint exits1 because
Ruff,Basedpyright and Vulture are missing locally.

## Completed local qualification on exact source

Source commit `199ce3bda33e87b0432d153400bdc6f7730b6eb8`, tree `d6acde447e39f51cddbdc947889a97c4a6334353`: both public laws and the complete
importing consumer passed, followed by all17 controls in the same focused command.
The225-entry source/test closure matches after execution. The Bun --smol option
changes garbage-collection scheduling, not the pinned compiler source or proof
statements. The complete consumer took657.312 seconds; that timing is an execution
receipt, not a performance comparison or benchmark.

Final native rerun passed all four modes and all three actual-code mutations on
unchanged sources. Each mode executed1,776 distinct input tuples, observed4,466
complete returned-move child Boards/84,854 U32 Board fields, and read the retained
marker3,552 times. The producer added1,182 moves; the checked adapter retained1,054.
The1,597 true-guard inputs include all eight check patterns, including128
initial/transit-safe but destination-attacked cases. Those128 survive the single producer
but are rejected by the final filter. The179 false-guard inputs are diagnostics
outside the new public laws' domain. Tail values may still describe unsafe moves;
they are compared exactly, not certified safe. Fixture modes are repeated, not
independent datasets or distinct Board counts.

The marker-only mutant preserves every output move list and Board field but loses
the unused slot. Native state reads reject it, and the source mutation at Wire.finish
rejects the corresponding complete-pair equality. Ignoring initial or transit
checks also compiles/runs normally before ordered-list mismatches reject the code.
The final report includes mutation evidence and separately labeled generic builds.

Original compiler16-law/seven-control tests and all12 pin tests passed. Unchanged
whole-repository lint remains UNQUALIFIED because all three tools are absent. No
new hosted run or independent reviewer is claimed. The earlier failed full source
invocation and initial Wire draft are not included in pass counts.

All607 inherited native-source files and all620 candidate files match their hashes.
The active modular total is193 laws/623 controls (191/606 retained plus2/17 freshly
executed). The complete193-law aggregate was not executed. Documentation/evidence
following the source commit does not change any qualified native/proof file.

## Publication status and next acceptance

No GitHub write action is exposed in this session, and direct Git transport failed
DNS. This continuation has not updated #899 or created a PR. The recovery package
contains the exact patch, incremental Git bundle and a prepared draft-PR publisher.
The publisher is syntax-checked but has NOT been executed; it checks the exact
parent and refuses to overwrite a differing target branch or to merge/deploy.

Next, carry initialized-table preservation through ordinary generation and the
optimized final filter so this single-side checked-filter result can be composed
into complete legal_moves. Universal attack-direction correspondence and historical
castling rights remain separate. The new exact returned-array/list equality closes
sequential check wiring here without claiming those remaining obligations.
