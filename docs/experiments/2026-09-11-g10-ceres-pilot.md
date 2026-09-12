# G10 Ceres first-four-shard convenience pilot — prepared, not launched

Collect C3-768-30-pre8-I8 policy and both raw value-logit heads for the first four whole original run06 G10 derived shards (0..3, 32,768 rows, 1,024 batch-32 calls, zero padding). This is a convenience prefix, **not representative G10**. Same-row teacher disagreement is a diagnostic, not an Elo estimate or a new training queue.

Schedule only after the original18.91M Ceres collection is terminal, during final saved-output qualification or CPU target preparation, and without delaying the registered Ceres training anchors. Parent reviews this plan and publishes the preregistration before GPU spending. There is no automatic launch, wait daemon, resume or retry.

The original complete admitted run06 batch remains 262,079 rows/32 shards. Source summary `ab214d52665ee4ad6ccf756c0515344dc9653c3b51de15965fc74323198b7fd9`, G10 qualification `d8d71e5e1c73f193710757c36db3e07b36fbc6a9962ed28c318d0cd3742d5c1c`, and adapter `8b81f9633050f8e2f286259671e3cefb39bfb30dffa29c7d76608723d36505c8` are pinned in `plan.json`. Exact selected specs include each original row-provenance hash. No synthetic smaller source manifest, B100 substitute or adaptive-value source is used.

## Execution and reuse

`run_chunk.py` is a fresh narrow copy of the accepted `chunk_adapter_v2/run_chunk.py`. Its differences are fixed G10 range0..3 with exact original full shard specs, two G10 qualification flags, and a 1200-second deadline. No collector/numerical/provider implementation was changed. The mapped-library helper is copied byte-identically. `integration.diff` records the entire adapter change.

The unchanged existing `ceres_collection_batches.py` receives a one-chunk `driver.plan.json`. It owns a new process group and TERM/KILL cleanup; the collector retains its shared GPU lease and own child cleanup. `command.prepared.txt` adds an independently surviving external 1170-second TERM plus30-second KILL bound. Driver and wrapper default invocations only validate; the prepared launch command contains explicit `--execute` and is not executed here.

Runtime `/tmp/deepfin-ceres-selected-bank-runtime`, commit `e9c1b74bc44c45f9ac6d70195f1422c6d24a65f9`; Python `/tmp/deepfin-ceres-collector-ort129/bin/python` (CPython3.13.15, **ONNX Runtime1.29.0**, NumPy2.2.6), accepted CUDA13/cuDNN package and loaded-library pins. Model `data/ceres/C3-768-30-pre8-I8/C3-768-30-pre8-I8.onnx`, SHA `44aa02c775456f18ed464e33fc37b8e4abf58d7bf8f4cfb3ff19492e32e56df3`. Profile `ceres-c3-fixed32-compact-v2`, outputs policy/value/value2, `--retain-value2 --pad-final-batch`, primary WDL output value/logits, native float16. Exact package, runtime source, native-extension and mapped-library evidence is retained in the plan; large libraries/model were not rehashed during preparation.

Fresh output: `/home/josh/projects/chess/data/lc0/ceres_compact_sidecars/g10_run06_first4_v1`. The collector generates a fresh timestamped invocation directory under this output; freshness is enforced before execution. No existing output is relabeled or modified.

Bounds: CPUs4,5 with all numeric threads2 and nice19/ionice3; GPU0 arena8GiB; sampled device memory<=12GiB; host MemAvailable>=16GiB; per-process sampled RSS<=12GiB; SSD reserve150GiB; combined output/state128MiB; STOP at pilot, driver, Ceres readiness and output paths. The 1200-second inclusive external/driver deadline covers metadata admission, lease wait, model/library verification, calls, writes and cleanup. Existing telemetry retries only rc3/timeouts, with failure receipts and bounded logical-query deadline; inference is never retried. These are inherited measured sampling guards, not hard process-RSS quotas.

Cost basis: a completed131,072-row original-corpus recovery chunk took532.3155seconds. A simple quarter-row ratio is133.08seconds, but fixed session setup/library hashes and full G10 metadata admission do not scale with row count. Use the20-minute ceiling and report actual elapsed/runtime/call counts rather than claim a measured G10 speedup.

Fresh host preflight must establish original collection is done, the GPU lease is available, CPU affinity is appropriate, model/library stat and WSL boot/mount identities remain valid, and resources are free. Historical GPU/boot identities in this prepared plan are not a new host observation. If changed, stop and review the evidence instead of weakening checks.

## Completion and prospective diagnostic

Successful collection requires driver terminal0, wrapper completion, complete actual invocation, all4 complete saved shards/32,768 rows, exact original source bindings, both logits heads, fixed32 call accounting, first-call CUDA provider proof and first/final loaded-library/project evidence. Independently qualify saved outputs before creating the small audit-specific Ceres manifest from actual accepted bindings. A terminal collection alone is not training-corpus qualification.

Then reuse the existing policy audit with the same four source shards and the already pinned BT4 policy adapter. Retain per-row qualified source/game/shard/row identities and source-qualified game clustering. Descriptively report Ceres/BT4 top-move agreement, Jensen–Shannon divergence and each teacher's entropy; compare prespecified existing arithmetic and geometric mixture coverage and common-paired deeper-SF regret, with explicit conditional-regret versus all-row coverage denominators and mate/invalid/unscored exclusions. Preserve fixed exact-T300 preview semantics if included. No outcome-driven mixture/temperature grid or promotion criterion is introduced here. Confirm the actual audit's temperatures/weights from its frozen implementation before a later separately reviewed diagnostic invocation; this collection does not choose them or silently qualify a new downstream runtime.

Bank both raw Ceres WDL heads for a later explicitly authenticated native-BT4-WDL join. The present G10 policy adapter is policy-only; current audit cannot make the intended three-way value comparison from these inputs. Do not claim value agreement/calibration is available merely because both Ceres heads were stored. No new native-WDL join or value-audit code is included in this pilot.

## Publication and scheduling record

This pilot follows the [completed SF constraint diagnostic](2026-09-11-sf-negative-constraints-screen.md) and preserves the [registered Ceres training anchors](2026-09-11-ceres-weighted-bootstrap.md). To avoid CPU contention if collection overlaps final original-corpus qualification, run the existing CPU-only qualification on cores 2,3 and this pilot on its pinned cores 4,5. Target materialization retains cores 0,1. These are resource allocations, not changes to either teacher or verifier.

Preparation and independent review passed without reading payloads or loading a model. Actual collection, output qualification and policy diagnostics remain unexecuted. The [preparation receipt](evidence/g10-ceres-first4-prepared-20260911.json) pins the reviewed plan, adapter and driver. Both Ceres heads will be banked for future value work; their presence alone does not complete the missing native-BT4 join.
