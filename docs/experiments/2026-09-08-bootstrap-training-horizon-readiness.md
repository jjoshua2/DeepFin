# Bootstrap training-horizon readiness

The next horizon comparison should train two selected target families from scratch
for two uninterrupted epochs, retaining each trajectory's epoch-one checkpoint.
Which families earn that compute remains conditional on the broad target screens.
This record qualifies part of the implementation; it does not launch that comparison
or establish that longer training improves strength.

**Later 2026-09-08 update:** the [bounded compiled-CUDA probe and full SF/B100
planner completed](2026-09-08-value-collection-and-horizon-readiness.md#two-epoch-runtime-qualification).
The earlier preparation and failed diagnostic below remain historical evidence;
no full two-epoch training comparison has launched.

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

## Further preparation and the next scientific question

An isolated runtime at `0ff96f006e7cb1a72278c4ccc533abecce37ce86` passes a
CPU-only training import check in the same Python 3.10 / Torch 2.11 environment.
All imported repository Python sources match the earlier CPU qualification;
20 native source/binary identities also match. The existing tiny two-epoch tests
were not repeated. This is import compatibility evidence, not CUDA qualification.

A small hardware-check corpus is now prepared from the already qualified training
bank: 1,024 rows with distinct stored game IDs, selected in sorted bank-part and
original row order, keeping the first occurrence of each ID. Its original SF
policy and all 16 non-policy columns survive storage/readback unchanged. No label
was fabricated, position duplicated or original corpus rescanned. This selection
is for trainer plumbing, not a new statistical sample or recipe comparison.

The [registered CUDA check](https://github.com/jjoshua2/DeepFin/pull/530#issuecomment-5588657902)
uses the unchanged 61,444,448-parameter configuration,
compiled training, batch size 512 and two uninterrupted passes: two updates per
pass, four total. It has a 15-minute execution ceiling, including cleanup, and
must wait for the shared GPU lease after the registered G50 matches. It has not
launched. Finite updates, no skips/retries, complete per-pass sampling receipts,
and epoch-one/final checkpoint identities are required. Four updates remain
inside warmup; one small shard and two loaders do not qualify full-corpus memory
or the full release schedule. The new loader's stricter memory accounting must
be checked before choosing the worker count for both full training runs.

The scientific priority remains finishing the broad target comparisons. SoftSF10
still tests a distinct raw-score construction. If B100 remains strongest and
SoftSF10 loses, a raw-score SoftSF40 follow-up would test a broader SF target
before concluding that SF supervision is uncompetitive. Its previously measured
entropy is closer to B100's, but those reported means use different data supports;
this is not an exact entropy-matching claim. Construct any later temperature from
raw scores, since the stored float16 SoftSF10 target has already lost tail values.
Neither this follow-up nor a temperature grid is a mandatory queue.

For the horizon question, compare two competitive families freshly trained in the
same runtime. A focused option is **128 opening pairs at 400 simulations** for
A1 versus B1, then the same opening panel for A2 versus B2: 512 games total. The
paired change estimates their relative response to a second epoch. It does not
establish either family's absolute improvement; that requires within-family
comparisons. Another 100-simulation cell needs a specific depth-interaction
question to justify it. Freeze the actual families, deciding contrast, match
settings and budgets before launch; these remain conditional design choices.
