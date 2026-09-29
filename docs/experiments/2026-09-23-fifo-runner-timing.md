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

The complete, source-qualified panel is recorded below. A passed run means valid measurements, not a speedup verdict.


### Setup failure and local checks

Run 35945422686 stopped before native builds or measurement on three new-test Ruff findings: an intentional regex needed a raw string and two combined assertions needed splitting. All expectations and measurement rules are unchanged. The 26 new admission tests passed locally. A local extended O3 semantic smoke reached five completed configurations before the command time limit; it is not a completed timing panel and is not used as performance evidence.


### Completed hosted panel

Run [35945654847](https://github.com/jjoshua2/DeepFin/actions/runs/35945654847) on `7d2beda3fca4284fae622052537e0e85ead032b1` passed all stages. All 41 focused Python cases passed without skips (26 new and 15 existing), explicit static checks and whole-repository Ruff/Basedpyright/Vulture passed, and source reconciliation found no changes.

The six diagnostic pairs matched full trees/events and work. All 144 measured child executions across 72 arm-groups matched diagnostics-off root summaries and non-time accounting. Twelve warmup observations were retained but excluded. All raw per-child timings and output hashes are in samples.csv; source/build identities, configuration, warmups, diagnostic work counts and recomputed summaries are retained beside it.

Times below are coordinator medians per invocation (group medians divided by the common repetition count). Ratios are medians of paired group ratios, not ratios of unpaired medians. Above 1 favors FIFO.

| Case/mode | Repeats | List ms | FIFO ms | Paired list/FIFO | Observed paired range | Screen decision |
|---|---:|---:|---:|---:|---|---|
| single/sync | 6 | 15.00 | 14.75 | unresolved | below floor | below_measurement_floor |
| single/async | 2 | 214.75 | 214.50 | 1.0000 | 0.9953..1.0047 | inconclusive_at_5pct |
| openings-16/sync | 1 | 177.00 | 177.00 | unresolved | below floor | below_measurement_floor |
| openings-16/async | 1 | 868.00 | 866.50 | 1.0006 | 0.9954..1.0116 | inconclusive_at_5pct |
| mixed-16/sync | 1 | 131.00 | 131.00 | unresolved | below floor | below_measurement_floor |
| mixed-16/async | 1 | 617.00 | 617.00 | 1.0000 | 0.9935..1.0292 | inconclusive_at_5pct |

Whole-process timing (initialization/reporting/exit included):

| Case/mode | Median paired list/FIFO | Screen decision |
|---|---:|---|
| single/sync | 0.9909 | inconclusive_at_5pct |
| single/async | 0.9968 | inconclusive_at_5pct |
| openings-16/sync | 0.9821 | inconclusive_at_5pct |
| openings-16/async | 0.9947 | inconclusive_at_5pct |
| mixed-16/sync | 0.9899 | inconclusive_at_5pct |
| mixed-16/async | 0.9938 | inconclusive_at_5pct |

Interpret each result against the preregistered all-six-pairs/5% rule. Inconclusive means this screen did not establish that practical gain or regression; it is not equivalence proof. The older 3.65x/7.90x owning-loop measurements cannot be substituted for these full-coordinator measurements. This fixed batch-4, 146-channel callback screen does not establish actual model/GPU speed, playing strength, other batch sizes or performance beyond the 16-root cap. No scheduler, compiler, model, live setting or arena size was changed during this continuation. Self-review only. Nothing merged or deployed.

Artifact `10787165583`, ZIP SHA-256 `1b2d87308893e6e3beabaa07116d41ca4c590e93a062e9c65bdab6da863948e8`. Source hashes describe the tested preregistration before this documentation-only readout. Historical generated C inputs remain in the two previously qualified artifacts; their ordinary retention may expire. Recreating a historical input requires separately qualifying code generation, not silently accepting a different hash.

### Reproduction

```sh
CC=clang-18 CXX=clang++-18 bash native/bend_engine/multi_root/build_fifo_benchmark.sh PARENT.c FIFO.c NEW_BUILD
python -m native.bend_engine.multi_root.benchmark_fifo --reference NEW_BUILD/list --candidate NEW_BUILD/fifo --report NEW_REPORT.json
```
