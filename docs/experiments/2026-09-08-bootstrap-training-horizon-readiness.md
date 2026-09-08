# Bootstrap training-horizon readiness

The next horizon comparison should train two selected target families from scratch
for two uninterrupted epochs, retaining each trajectory's epoch-one checkpoint.
Which families earn that compute remains conditional on the broad target screens.
This record qualifies part of the implementation; it does not launch that comparison
or establish that longer training improves strength.

## Completed CPU check

Main at `ee46d0f1139211be15d84e4f9391c240dc51dbc7` ran five existing focused
cases successfully under Python 3.10.12, Torch 2.11.0+cu128, NumPy 1.26.2 and
Zarr 2.18.3. CUDA was hidden and uninitialized; work used two threads on CPU cores
6 and 7 at low priority. The complete bounded qualification took 384.43 seconds.
No active runtime changed and no native extension was rebuilt.

The positive test uses the real trainer on a tiny 20-row corpus. It verifies two
complete passes with sampler seeds 0 and 1, continued trainer/optimizer/augmentation
RNG objects, monotonic update counts, and epoch-one weights bit-identical to a
standalone default one-epoch run. Negative cases inject an incomplete second epoch
or underreported update count and verify that completion is refused. Two cases
check ragged checkpoint-boundary arithmetic; these are not training runs.

A separate scheduler diagnostic shortened warmup to two steps so the tiny run could
exercise the existing release schedule. Its first attempt was correctly refused by
the trainer configuration pin. The explicitly waived diagnostic then completed two
epochs and saved checkpoints at steps 5 and 10. Assertions comparing adjacent
step, scheduler, learning-rate and RNG states executed successfully, but a final
assertion requiring Torch RNG state to *change* failed: this fixture is deterministic.
Both unsuccessful diagnostic logs are preserved; this was not an all-green probe.
The normal tests provide guard evidence, while the waived probe does not.

Unchanged RNG state cannot distinguish continuity from reseeding to that same state.
Optimizer evidence covers object continuity, actual updates and retained nonempty
checkpoint state; it does not independently compare the full moment trajectory.
The diagnostic's observer records were not written after its assertion failure, so
the executed continuity assertions are supported by source and traceback. These
limits are recorded in the independent review; no training was repeated to polish
the result.

The [evidence manifest](evidence/bt4-bootstrap/training-horizon-runtime-manifest.json)
binds original commands, logs, producer and independent reviews, probe sources and
completed summaries. Tiny generated weights and fixture payloads remain host-local.

## What remains before a scientific comparison

Use a fresh pinned runtime for **both** families. The historical B100/H20 runtime
predates multi-epoch support; copying only the newer training script would mix
incompatible replay, fingerprinting and loss-normalization contracts. Comparing a
new two-epoch model directly with historical one-epoch weights would confound
training horizon with implementation changes.

The existing schedule retains one global 1,000-step warmup and per-window
sqrt-release cycles. Two epochs do not stretch a whole-run decay schedule. At the
historical corpus geometry, each pass has 419 windows of 88 steps and one of 63,
for 73,870 total updates across two passes. The optimizer and scheduler persist;
there is no qualified old-checkpoint continuation path.

Remaining work is a bounded CUDA forward/backward and native-compatibility check,
both corpora's full planner/resource and logical-schedule qualification, and a
coordinator that accepts the multi-epoch receipts and corresponding time budget.
The current one-epoch coordinator deliberately rejects those receipts. This CPU
exercise did not qualify the 61M-parameter model, compilation, the main search
backend or full-corpus memory use.

H20's measured 9,851 seconds per epoch projects to about **5.47 hours per family**
or **10.95 hours for two families**, before planning and matches. This is a cost
baseline, not measured throughput for the newer runtime. Freeze the selected
families, schedule, deciding checkpoint, match protocol and resource limits before
launch; evaluate each fresh trajectory's own epoch-one and final checkpoints.
