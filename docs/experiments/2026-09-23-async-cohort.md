# Asynchronous bounded cohorts and per-root cancellation

## Preregistration — September 23, 2026

PR5b, on #847 at dbe12bd862bbe04573f0e4d3aeb26c910406f74c. No merge,
deployment, live training or GPU use. CPU-F32 only, existing fixed-batch model.
Synchronous cohort execution remains the default and its output contract is preserved.

Hypothesis: a single batch execution worker lets the Bend cohort owner service
control input while a forward runs, without allowing a cancelled row into a tree
or cancelling unaffected roots that share the same forward. No speed/Elo claim.

Scope: opt-in async bounded cohorts; readiness, per-root cancellation, stop-all,
and graceful quit. This is not dynamic root admission/removal, persistent self-play,
a measured batch-selector, or GPU qualification. A worker callback must finish all
physical accesses before returning. The native worker knows batches, not roots;
only Bend associates tickets and rows with trees and cancels individual roots.

Controls: unchanged synchronous cohort runner, deterministic row-independent
callback, actual selected-leaf CPU model, explicit blocked callback before release.
Require same no-cancel trees/results and counts as sync; exact admission/physical/
accepted/cancelled reconciliation; no output on pending, duplicate or wrong-token
poll; no slot reuse before physical completion; shutdown join; input snapshots;
full, partial and repeated batches at supported sizes and both widths; malformed
commands leave state unchanged. Cancellation must not refund computation. Backend
failure/nonfinite rows still fail even when cancelled; no successful summary.

Budget: one bounded hosted CPU qualification, two Torch threads and one compiler
job; fresh cohort C generation, deterministic normal/UBSan checks, and the existing
untrained CPU fixture. Reuse singleton export; at most one batch-four export from
the identical weights. No training, games, live hardware or new model architecture.
No tolerance relaxation: inherit 2e-6 absolute/2e-5 relative logit and existing
policy/WDL tolerances. CPU time summaries are descriptive, not warmed benchmarks.

Recovery: preserve failed evidence, fix the affected layer, retain compiler and
oracle contracts. Publish only a checked source diff. Independent review is not
available; self-review must be labeled. Component checks are not substituted for
full Bend application/worker/model composition checks.

## Completed qualification and publication recovery — September 23, 2026

[Run 35888265043](https://github.com/jjoshua2/DeepFin/actions/runs/35888265043),
job `107273910581`, completed every static, native, serial-oracle, control and
actual CPU-model gate before publishing executable-source commit
`ccfd616f4d7508b4f195366c9eb5ccdcf92e02a9`. The response was interrupted before
opening its PR. The continuation recovered the existing branch and evidence,
reviewed the code and results, and added this readout without replacing that work.
The exact base remains #847 at `dbe12bd862bbe04573f0e4d3aeb26c910406f74c`.

The final qualification reused hash-verified generated C from successful build
[35887078702](https://github.com/jjoshua2/DeepFin/actions/runs/35887078702).
Its reconstruction step verifies that the test-only delta leaves every prior
source-manifest entry unchanged. The final 17-path patch and manifest are in the
qualification artifact. Publication checks those same source hashes before
changing only this record, the experiment index and the cohort README. No new
model export, full code generation or repeated full qualification was needed to
recover and publish the completed work.

### Implementation and ownership

The existing cohort executable has an opt-in `DEEPFIN_COHORT_ASYNC=1` mode.
The build wrapper links the CPU batch worker; runtime defaults remain synchronous.
A single native worker snapshots the real input rows and owns its output until
physical callback completion. A complete batch token can be consumed once; pending,
wrong-token and duplicate polls cannot publish output. Failed callbacks poison
the slot and do not expose partial private output. Shutdown joins physical work.

Bend retains every root, ticket, row mapping, legal entry and cancellation flag.
The C++ worker has no root IDs or chess/search decisions. While a batch runs,
Bend services readiness, `cancel ROOT`, stop-all and graceful quit. Cancelling one
root does not cancel its batch peers. Cancelled pending rows are removed only
from the result-routing step after all real logits have passed the whole-batch
finite check. Already accepted tree work is retained. A cancelled root that has
not dispatched a row creates no fictitious cancelled evaluation.

A successful final report follows physical drain and reconciles dispatched =
executed = accepted + executed-wasted rows. Per-root and aggregate counts agree;
no budget is refunded. Async reports use `deepfin.multi-root-async-work.v1`, not
the synchronous schema. Failed execution/nonfinite output exits without a
successful cohort summary; it is not reported as a completed zero-error run.
Control acknowledgments are immediate when processed, but final root summaries
wait for drain. This differs intentionally from PR4's early UCI bestmove snapshot.

### Tests actually completed

| Check | Result |
| --- | --- |
| Locked whole-repository Ruff, Basedpyright, Vulture plus explicit verifier checks | Pass; zero type errors or warnings |
| Cheap cohort/report and existing regression tests | 267 passed, zero failures/errors/skips; 71 new cases |
| Deterministic callback at five batch sizes and two input widths | 10 configurations, synchronous and asynchronous modes each, pass |
| Same generated coordinator under UBSan at batch four, both widths | Both pass against the serial reference |
| Blocked-forward control tests at both widths, normal and UBSan | Four configurations pass |
| Native worker in normal and ASan+UBSan modes | 1,456 assertions, 10 configurations and 160 reuse calls per mode pass |
| Actual CPU model at batch one and four, both execution modes | Selected-leaf, input/history, numerical, tree and work checks pass |

The deterministic matrix compares every populated primary final tree field
against serial batch-one execution. The common tree snapshot SHA256 is
`593e548adcc43df3cca20bdbc1725fc331274c5e36b33c9a85e74498510a0770`.
The fixture has 14 primary roots, four terminal roots and three exact-budget
roots; repetitions across configurations are not independent games. Quiet runs
also reproduce the detailed runs' root outcomes and accounting. Existing invalid
input, swapped-row, backend-failure and nonfinite-output controls remain active.

The recovered worker source was additionally compiled and executed during this
publication continuation with local Clang 17.0.0: both normal and ASan+UBSan modes
again passed 1,456 assertions and 160 reuse calls. This local recheck is distinct
from the earlier locked hosted coordinator/model qualification. No fresh model
or GPU run is claimed for this documentation/publication continuation.

### Selective cancellation and control evidence

The tests link the actual generated Bend coordinator and worker to a callback
held by explicit test-only pipes. No test pipes/callback enter the model product.
Readiness and cancel/stop/quit acknowledgments must arrive before releasing the
held callback. The synchronous negative control cannot satisfy readiness-before-
release. Tests check that no result summary or neural backup appears while the
first batch is still held.

In the six-root selective test, root 1 is cancelled twice while its row is already
in the first four-row batch; root 6 is cancelled before dispatch. Only root 5
needs the later one-row batch. Final work is five executed real rows, four accepted
rows, one cancelled/wasted row and three padding rows across two physical batches.
Roots 2–5 have bit-identical final trees to the uncancelled control. Roots 1 and 6
have no accepted values/visits; only root 1 has a cancelled physical row.

A second test cancels root 1 after an earlier sweep was accepted: its banked tree
is preserved, its second row is discarded, and all eight admitted rows remain
charged. Seven results are accepted and one is cancelled. Stop-all and quit tests
acknowledge before release but cannot finish until the batch retires; all four
in-flight rows remain executed/wasted, and undispatched roots add no rows.

Six malformed commands are rejected without altering cancellation state. Three
invalid async option values fail explicitly. Backend failure and a nonfinite
final row fail even when the affected row's root has been cancelled: no successful
summary or first-batch neural reply is permitted. These controls run at both
input widths in normal and UBSan modes. EOF disables further controls but does
not implicitly cancel the cohort, preserving ordinary batch/no-stdin usage.

### Actual selected-leaf model evidence

Used the same untrained 5,043,005-parameter, 175-plane CPU-F32 checkpoint as prior
PRs. The original singleton package was reused exactly; the qualification exported
one new batch-four package from identical weights/encoding. This is not a trained
checkpoint, GPU run or new model architecture.

| Primary 14-root cohort, in either execution mode | Real / accepted rows | Calls | Physical rows | Padding |
| --- | ---: | ---: | ---: | ---: |
| Batch one | 34 | 34 | 34 | 0 |
| Batch four | 34 | 9 | 36 | 2 |

The capped three-root case has nine accepted rows; the four terminal roots use
zero forwards. Each configuration compares 898 legal priors and 720 populated
final nodes across its three traced cases. Four configurations (two batch sizes
by two execution modes), each run with and without diagnostics, execute 344 real
rows. Detailed numerical/trace comparison applies to the 172 traced rows; quiet
runs are checked for matching results and work counts.

All real traced outputs match independent eager singleton evaluations within the
unchanged 2e-6 absolute / 2e-5 relative logit tolerances. Maximum absolute error is
`5.960464477539062e-7`. Existing legal-policy/WDL tolerances are unchanged. Full
histories, same-board/different-history cases, promotions, en passant and automatic
draw distinctions remain covered.

At EACH fixed model batch size, synchronous versus asynchronous primary tree
snapshots are bit-identical and root outcomes/work counts match. This is not a
claim that batch one and batch four have identical floating-point tree bits;
their snapshot hashes differ. No cancellation was injected into the actual-model
runs; selective cancellation is demonstrated by the blocked deterministic callback
in the same generated coordinator/worker, as described separately above.

### Limits and next work

One worker and one physical batch slot only. This makes control processing
asynchronous relative to inference; it does not overlap several model forwards or
prepare the next cohort batch during the current forward. Gathering/encoding,
normalization, reporting and output backpressure remain synchronous. Stop/quit
cannot preempt a wedged callback. The command timeout is a functional test bound,
not a latency guarantee or benchmark.

The final cohort clock includes setup, first inference, controls and physical
drain, with millisecond resolution; no warmup is excluded. Async phase/service and
queue/GPU/transfer timings remain null. Its useful EPS is not an external UCI
decision-time or steady-state performance result. Fewer forward calls are not a
measured throughput or Elo improvement.

Still absent: live root admission/replacement/removal, persistent self-play,
per-root deadlines/wall-time admission, measured batch-bucket selection, multiple
slots, CUDA/trained-5090 qualification and same-tree concurrency. Root arenas stay
at 4,096 nodes; production Gumbel parity is unchanged. Normal UCI is untouched.
CMake rejects enabling this async cohort worker together with CUDA support.

Self-review only, not independent review or formal proof. The normal/UBSan
coordinator matrix covers generated C and test callback/adapter. The standalone
worker gets ASan+UBSan; LibTorch and compiled model internals are not instrumented.
No full ASan/ThreadSanitizer, new UCI/perft, chroot, race proof or production-
readiness claim follows. Hosted dependencies were locked Python 3.13 / Torch
2.14.0+cpu, uv 0.12.10 and Bun 1.4.2, with two Torch threads and one compiler job.
The verified compiler remains aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, with the
unchanged 84-file fingerprint. Dependency inspection found no libpython link;
that is not a process-isolation test.

### Retained evidence

Artifact 10763219294, `deepfin-pr5b-qualification`, expires October 23, 2026 and
retains source patch/manifest, JUnit/lint, matrix/control/model reports and build
identities. It contains no model packages, binaries or raw neural traces.
The generated source is retained separately by the corrected-build run.

- Artifact ZIP SHA256: `0b955e289aaf85519ded37b29965a9c192f388c4da465e0fe839a67926fd39cd`.
- Qualified source patch SHA256: `f6ae0f9fc8aeb98eb737af2e366fa56f22a50c281dce8c045e20c8c8e0220abc`.
- Generated C SHA256: `f0c958201e4b057919d5b24a31649a110d9720982784f2454ce28aac99273cb0`.
- CPU batch-one executable SHA256: `4b102558eacad8aab7e9b62194fce1450d0f3010d222ce554c0a489ecf912f55`.
- CPU batch-four executable SHA256: `8e789600cff1abfc40479ef9925a54228916197831a1b67d28cb4682d9f9f7af`.
- Checkpoint SHA256: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Singleton package SHA256: `9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
- Batch-four package SHA256: `36b1f6a8da1456e942e5047e99eedd0d2eaa28fef896b34129fc5331c774c4e9`.

Ordinary PR CI is separate from this completed qualification. Nothing was merged,
deployed or run on live training hardware. The final README documents controls,
EOF/stop semantics and reporting differences rather than claiming all PR5 is done.
