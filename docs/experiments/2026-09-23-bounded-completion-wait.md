# Bounded completion notification in the cohort runner

## Preregistration

Parent: PR #866, d90f9cfbeb64b9d83f4296543307c377c5049917. Its callback
full-runner screen found no consistent >5% benefit from replacing the ready
list with a FIFO. Hypothesis: a fixed IO.sleep(1) after a pending poll adds
avoidable latency even when the worker completes during that sleep.

Intervention: replace only that pending sleep with a native condition-variable
wait. The C ABI requests at most one millisecond; an already-complete or newly
completed callback can wake it earlier. The next Bend loop still checks control
and deadlines before taking/retiring output. No busy spin, unbounded application
wait, root ownership transfer, tensor copy, budget change or model logic moves
into this operation. take remains nonblocking and the sole retirement path.
The ordinary OS scheduling/oversleep caveat still applies; this is a requested
wait budget, not a guaranteed one-millisecond response deadline.

Correctness gates: unchanged native worker tests; gated completion waits (ready
before wait, completion during wait, pending timeout, failure/exception, token
mismatch and reuse, no output copy or early slot release); a compiled no-notify
mutation must fail the notification test. Existing actual coordinator serial
oracle and cancellation/stop/quit/deadline controls remain unchanged. Generate
new C from the unchanged compiler pin, never patch generated C. Compare the
actual parent FIFO and new-wait runner's final trees/events/non-time work.

Performance: batch four, 146 channels, one Bend worker, same single/opening-16/
mixed-16 cases as the previous full-runner screen, 256 requested simulations,
depth eight. Compare actual realized work; arena exhaustion may stop early.
Primary metric is high-resolution child-process wall time with diagnostics off.
Record child user+system CPU as well so a spin-based apparent gain is visible.
Use one warmup per arm then a common repeat count (target 400ms for the faster
arm, at most 16 repeats), six alternating paired groups per case and mode.
Retain raw invocations and output hashes. Ratios require every group >=200ms;
all six ratios >1.05 is a practical improvement screen, all <1/1.05 a regression,
otherwise inconclusive. CPU is descriptive; inspect any >20% median increase.
Sync is the unchanged-path control, not an assumed exact speed equality. No
p-value, confidence interval, GPU, neural-model or playing-strength claim.

Budget: isolated hosted CPU qualification, two Torch threads, one compiler job;
240-second timed-panel budget, bounded subprocesses, no live jobs/config/model
exports. Negative and failed results stay recorded. No rerolls to find a faster
panel. Self-review only. Source-only stacked PR; no merge or deployment.

## Readout

The completed hosted qualification is recorded below. A passing measurement run is not itself a speedup verdict.


### Preserved first hosted failure and stream synchronization correction

Run 35951329817 passed all 236 focused Python cases, the 12-case/326-assertion native wait tests in normal and ASan+UBSan builds, and the compiled no-notification negative control. Fresh C generation and the first five 146-channel batch configurations passed in both sync/async modes. The first held-control suite then timed out waiting for the quit acknowledgement. Later configurations, deadlines, performance and whole-repository lint did not run; this is not a completed qualification.

A local diagnostic using that exact retained generated C reproduced the distinction: stop was acknowledged while the callback remained held, but quit was only visible after callback release. The unchanged sleep-parent control acknowledged quit while held. The quit flag disables subsequent stdin polls, which previously helped flush stdout; IO.sleep also parks through the pinned runtime's stream synchronization. The synchronous foreign wait did neither. The corrective source now calls the runtime's checked io_sync() before waiting, preserving that existing visible-I/O behavior. The control verifier, deadlines, expectations and timeout thresholds are unchanged. No generated C is patched; full generation and all gates are repeated from the corrected source. No performance panel had executed before this correction.

Failed-run artifact: 10788469658. Downloaded ZIP SHA-256: 2dbd944a5a442b7a6e99d5360c66abcef0ad42a29f13e3390744843294d13ee4. This failure is retained rather than counted as passing cancellation coverage.


### Completed hosted qualification

Run https://github.com/jjoshua2/DeepFin/actions/runs/35952107892 completed all preceding static, build, behavior and timing stages. 236 Python cases passed without skips; whole-repository Ruff/Basedpyright/Vulture passed. Native completion checks passed 12 cases / 326 assertions in each normal and ASan+UBSan build. Removing the notification was rejected by the compiled test. The unchanged serial-reference matrix passed ten configurations in both modes, two coordinator UBSan configurations, held cancellation/stop/quit checks and deadline checks at both widths. The original worker ownership tests also passed normal and ASan+UBSan. Twenty additional exact parent/candidate pairs matched all trees/events/non-time work. Repeated modes reuse fixtures, not independent games.

Six diagnostic comparisons and all 180 timed child executions matched realized semantic work. There were 72 measured arm-groups and 12 excluded warmups. Per-child wall/CPU observations and output hashes are retained in evidence/bounded-completion-wait/samples.csv.

| Case/mode | Old process ms | New process ms | Median paired old/new | New/old CPU | Verdict |
|---|---:|---:|---:|---:|---|
| single/sync | 68.90 | 69.02 | 0.9982 | 1.0010 | inconclusive_at_5pct |
| single/async | 271.68 | 81.38 | 3.3209 | 0.9737 | consistent_over_5pct_improvement |
| openings-16/sync | 794.89 | 788.83 | 1.0077 | 0.9924 | inconclusive_at_5pct |
| openings-16/async | 1486.00 | 831.87 | 1.7857 | 0.9826 | consistent_over_5pct_improvement |
| mixed-16/sync | 599.51 | 594.73 | 1.0086 | 0.9920 | inconclusive_at_5pct |
| mixed-16/async | 1098.54 | 634.19 | 1.7325 | 0.9682 | consistent_over_5pct_improvement |

Ratios use paired groups; displayed times are unpaired medians per invocation. These callback-only single-host results do not establish neural-model/GPU speed, playing strength, all interleavings or an OS response-time guarantee. Waiting requests at most one millisecond before control service; callback completion can end the wait early. Shutdown still joins physical work. Self-review only; no independent review or formal proof. Nothing merged or deployed.

Reproduce with `BUN=bun CC=clang-18 CXX=clang++-18 bash native/bend_engine/multi_root/qualify_wait.sh COMPILER PARENT.c PARENT_SOURCE NEW_OUTPUT`. Use the source and parent C identities in the committed summary; the historical parent C artifact has finite retention.
