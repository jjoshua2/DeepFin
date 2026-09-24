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

Pending qualification. Local native checks do not qualify the full application.
