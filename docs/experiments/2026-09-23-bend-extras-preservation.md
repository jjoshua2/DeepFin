# Final extras-stage preservation

## Result and publication status — September 23, 2026

**Two public source laws and the importing consumer pass the pinned checker.**
The final focused gate passes all **14 new rejection controls**. Four native modes
pass five complete 131072-cell buffers each, **655360 cell comparisons per mode**,
and seven invalid requests per mode. These are newly executed local checks.

This is a local patch targeting PR #861 at
`6f2c71fa9bd1d1537c1f5ae0ff6e70e31e495c53`. Proposed branch:
`feat/bend-extras-preservation-20260923`. No new remote branch, commit or PR was
created; available GitHub actions are read-only and no authenticated CLI was
available. No merge, deployment, force push or live-process operation.

**The complete combined aggregate and hosted qualification were not executed.**
The new wrapper preserves the entire unchanged 98-law/212-control parent and is
syntax-checked only. Do not label its expected 100-law/226-control total as a run.
Whole-repository lint was attempted and failed because Ruff, Basedpyright and
Vulture are absent. The parent lint pass does not qualify the new candidate.
Self-review only, not independent review.

## Source basis and integrity

Refreshed #861, its comments, current CI and the compiler branch. #861 remains
open/unmerged and has no review comments. Its ordinary CI 35912865132 and Bend
chess probe 35912865194 both completed successfully. The read branch contains no
new executable changes beyond its qualified source, as verified by GitHub compare.
Repository instructions, development/branch guidance and the experiment index
were read. Existing feature branches and the running system were left alone.

The complete qualified source checkout at
`94e380d218d1713828fd192427282588bf7df910`, tree
`731bf2a90704324e1dd34727d3f1aaad07162a67`, was restored from the exact full parent
archive and saved patch. Every one of the **283 inherited native-source manifest
entries** matches. The current target matrix was reconstructed exactly and its
Git blob matches `12494a9efc781f685eb215c5f4a7067c6e341484`; the experiment index is
unchanged between qualified source and current head. Other later documentation
and evidence additions were not reconstructed locally. This is **not a claim of
reconstructing the entire current #861 tree**. The patch changes only new paths
and those two exact current documentation files, preserving absent unrelated files.

Compiler remains `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
Bend 2.0.21 + U64, 84 inputs and fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Fork PR #2 still records separate inherited strict-TypeScript failures. Neither
that checker nor its source/native policies were modified or suppressed.

## Proof increment

| Law | Actual guarantee and premises |
| --- | --- |
| `bounded_extras_read` | For actual complete depth-17 array a, n+k <=64, q<131072, and q<256 or q>=512, actual get after `Tables.extras(n,from_nat(k),a)` returns the complete updated array paired with the original value at q. |
| `table_pipeline_extras_read` | The same preservation for actual `Tables.tables` over actual `Array.new(d,seed)`, with d=17 and a bounded extras range. Actual prior operations supply the shape; no shape or expected-value premise is imposed on the caller. |

All table-loop parameters and seed values remain arbitrary in the pipeline law.
The theorem does not establish that those prior values are correct chess headers:
it proves that the final extras loop preserves whatever values are there. Both
protected regions include every mask and prefix header [0,256), every logical
slider address [512,108160), and the remaining allocation slack up to 131072.
The actual four write regions are [256,320), [320,384), [384,448), [448,512).

The independent numeric specification `outside` contains only address comparisons.
`Facts.square` proves bounds and increment identities for all 64 square cases in
the original checker. The proof derives normalized routes using the existing
complete-tree theorem, composes actual writes via certified storage helpers,
inducts over the actual loop, and lifts scalar preservation to the complete
returned affine pair. No axiom, hole, foreign equality witness, unsafe dependency,
shadow-only correctness claim, or replacement implementation is introduced.

Depth and counts are kept symbolic during checking. The public pipeline law's
explicit depth equality is satisfiable and fixes depth17. The importing consumer
uses both laws and verifies positive full/last-square budgets and boundary-domain
certificates. Fully closed pipeline instances are not claimed as separate checker
runs. The earlier literal `Tables.build` and partially concrete consumers remained
unqualified drafts; the final universal statement has a more general count/start
interface and an explicit depth equality, not a weaker accepted law.

## Newly executed checks

| Check | Result |
| --- | --- |
| Public two-law proof module | PASS, exact `All terms check.` |
| Final importing consumer | PASS, exact `All terms check.` |
| Final focused gate | Two laws, 14 controls, source/compiler identity rechecks PASS |
| Generic / portable / native-target / UBSan | Five complete buffers and 655360 cell comparisons per mode PASS |
| Malformed native inputs | Seven rejected per mode |
| Original compiler source suite | 16 laws and seven controls PASS, including cyclic-template rejection |
| Compiler-pin contracts | 12 pass, zero fail |
| Final JS wrapper/driver syntax | PASS; full aggregate wrapper not executed |
| Whole-repository lint | Nonzero: missing Ruff/Basedpyright/Vulture |

Five new semantic controls catch inclusion of the first/last extras slot, an
inclusive query bound, overlong square range, and returning the original rather
than final buffer. Two actual-code mutations fail deliberately in the imported
`storage/Build.extras` implementation bridge. They are not claimed to fail at a
new-law location. Seven policy/output controls preserve manifests, required proof
imports, no holes/foreign/symlinked inputs and unsafe-warning rejection. Expected
failures require status1, ordinary expected/observed diagnostics and the exact
intended location; malformed imports, affine/termination failures, crashes and
timeouts do not count. The final focused report records diagnostic lengths,
SHA-256 values and excerpts, avoiding expanded multi-megabyte error terms in Git.

Native fixtures exercise zero count, first square, last square, a two-square suffix
and the entire 64-square run. Each begins with a fresh complete array and distinct
protected sentinels. Full-cell comparison checks both preserved regions and actual
extras results against independent signed-coordinate knight/king/pawn generation.
It does not rerun `Tables.tables` or a full engine. Same fixtures repeat across modes.
Shared output SHA-256:
`0cf635bddb79e02ebb6a9d64e42284129bc699f1214f1a01e7c01b58eb85eef3`.
Tools: Bun1.4.2 and local Clang17.0.0. No nonlocal CPU/GPU claim.

A separate disposable mutation moves the real knight writes from 256+sq to sq.
The unchanged native driver compiles and runs it, then rejects **case1, cell0**:
observed132096 rather than protected9305357568037403289. This is a behavioral
negative diagnostic, not a candidate failure or an extra source-control count.
The original runtime source remains untouched.

## Development failures retained

A draft marked a proof-pair Type as duplicable and another referred to an as-yet
undefined helper. A recursion draft placed the changing array before its decreasing
Nat argument and correctly failed termination checking. These were fixed in new
proof construction before acceptance, not by changing the checker or old laws.

Fixed-depth routes expanded the concrete complete tree and were stopped. Symbolic
depth reduced the checked core to a small proof while retaining the depth17 case.
Later literal-build/concrete-count consumers timed out or were intentionally
stopped; one outer tool timeout returned no completed checker status. Final public
laws and their fully symbolic importing consumer pass. No timeout or interruption
is counted as proof rejection, and no public statement that passed was weakened.

The first final harness attempted to match `Laws.bounded_extras_read` in diagnostics;
the compiler uses `LAWS.bounded_extras_read`. It therefore correctly failed the
gate despite an ordinary proof rejection. A later actual-code control expected an
unqualified function name, but the diagnostic identified `../storage/Build.extras`.
The final gate matches these exact intended locations. Both failed gate receipts
are retained; no proof, mutation or semantic assertion was relaxed. Formatting now
limits error excerpts and retains hashes/lengths instead of printing entire trees.

Full local diagnostics and unaccepted drafts are retained in the external review
archive. Compact command results and failed-log digests/excerpts are in Git evidence.
The local missing-tool lint error is unchanged and unresolved for this candidate.
Constructive proof development preceded this record; this is not backdated
preregistration or a training experiment.

## Remaining P2 and migration work

Extras cannot destroy correct incoming headers or slider entries under these
contracts. Still prove the actual stored mask/prefix values and persistence through
later table-header and other-block writes. Compose those with computed-fill contents,
then connect actual lookup to independently specified blocker-ray geometry. No
source correctness theorem for the extras region's own attack values is asserted
here; the native oracle provides separate bounded test evidence.

Source laws trust the pinned checker/Base. Native lowering, affine allocation and
physical lifetime, effects/ABI, C toolchain, libraries, OS and hardware remain trust
boundaries. Historical compiler TypeScript, diagnostic-depth and raw-internal-call
limitations are not addressed or suppressed. No new production application logic
moves into Bend: Python export, external references, data/control orchestration and
training remain dependencies; C++/LibTorch/AOTI remains transitional inference.
No full-engine/model build, perft increase, GPU run, training, benchmark, strength or
speedup claim is added. All new verification is opt-in.
