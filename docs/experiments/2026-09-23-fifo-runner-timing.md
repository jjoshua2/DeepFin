# Matched full-coordinator FIFO timing

## Preregistration

Continue #866 at f45ae893 against its #863 original-list parent bb1847d2.
Question: do the earlier owning-collection loop gains survive actual position
parsing, rule checks, encoding, dispatch, normalization and search backup?
The exact retained generated C from both qualified sources is recompiled with
identical Clang 18 -O3 -ffp-contract=off flags and the unchanged deterministic
callback/async worker. No regenerated or patched C, compiler update, scheduler
change, real neural model, GPU, production UCI or live training is involved.

Fixed scope: 146 input channels, physical batch 4, one Bend worker; one opening,
16 opening-history roots, and 16 roots with every fourth root checkmated, in both
synchronous and asynchronous mode. Each root requests 256 simulations, depth 8,
no additional neural cap. Existing 4096-node capacity can stop search early;
compare realized complete trees, events and work rather than treating requested
simulations as completed work. Chronological events/full trees must agree in
an untimed diagnostic pair per case. All timed outputs must match those root
summaries and all non-time work fields with diagnostics disabled.

First execute one diagnostics-off warmup per arm. Select a common repetition
count from the faster warmup's whole-process duration, targeting 400 ms and
capping at eight. Do not pool warmups with measurements or recalibrate after
seeing results. Execute six paired groups per case, alternating arm order.
Retain every individual subprocess duration, internal wall duration and output
hash; keep pairs as the descriptive comparison unit. Six pairs on one host do
not establish a broad population effect or a latency bound.

Primary metric: median paired list/FIFO ratio of summed coordinator wall time.
Also report high-resolution subprocess elapsed time, including initialization,
reporting and exit but excluding Python verification between invocations. The
internal clock excludes initial root loading and final reporting; it includes
policy/buffer setup and reports warmup_excluded=false. Diagnostics and binary
trace writing are disabled for timing. Async phase durations remain unmeasured.

A metric is usable only if all groups exceed 200 ms. Internal timing additionally
requires at least 50 ms per invocation on average within every group: repeating
zero/short millisecond observations does not magically remove quantization.
Classify improvement only if every paired ratio exceeds 1.05; classify regression
only if every ratio is below 1/1.05. Otherwise report inconclusive at 5%, not proof
of equality. Below-floor results get no ratio claim. These are practical screen
rules, not p-values or multiple-comparison-adjusted confidence statements.
No speed threshold is a CI correctness gate; a completed negative benchmark
must be retained. An incomplete panel or semantic mismatch fails qualification.

Budget: one hosted CPU screen, one compiler at a time, at most 240 seconds of
benchmark execution plus build/static checks; child calls have 60-second bounds.
No automatic retries or expansion to a more favorable fixture. Local checks
qualify script behavior only, not the authoritative hosted performance result.
Recovery: leave #866 unmerged and preserve both implementations' identities.
Self-review only unless a separately identified reviewer supplies a review.

## Readout

Pending execution. No full-runner speedup has been established by this record.
