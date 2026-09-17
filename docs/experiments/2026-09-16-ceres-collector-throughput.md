# Ceres collection throughput: orchestration before backend changes

Status: saved-data analysis and CPU-tested scheduler change; no GPU benchmark launched.

The completed 113-shard evening block09 took 6753.61 seconds. Its chunk receipts total 3389.07 seconds; synchronous session.run timings total 2035.41 seconds. Inter-chunk gaps total 3364.47 seconds (median30.039seconds). The prescribed112 thirty-second pauses account for3360seconds, almost half the wall time. Subtracting only these fixed sleeps projects3393.61seconds, or1.99x throughput; this is a counterfactual estimate, not a benchmark.

The collector already opens one ONNX session before iterating selected shards. The next configuration should group four shards into one invocation, retain fixed32 inputs, and specify zero inter-chunk pause. This reuses the already supported session lifetime. Existing per-process memory, disk, GPU-provider proof, source identity, final-output verification, and owned-process cleanup remain necessary. A pause is not a memory bound. Larger neural batches change the currently fixed32 backend contract and need separate qualification.

## Bounded next comparison

GPU ownership: only after the current collector terminates and the parent confirms the shared GPU lease is free. Preserve all existing plans and outputs. Use a fresh benchmark namespace and the exact qualified four-shard selection. Recomputing these selected rows is solely for output-equivalence validation, not another strength experiment.

Compare a four-shard shared session against the saved one-shard receipt baseline. First validate all outputs against the saved qualified banks: ordered physical rows, source/feed hashes, compact legal indices, both raw value logits and policy logits. Require exact integer/source identity equality; report floating maximum absolute and relative errors and require bitwise-equal float16 logits for initial adoption. A mismatch blocks numerical-equivalence claims and requires diagnosis.

Bounds:10minutes inclusive for the grouped pilot, existing48GiB startup/32GiB running available RAM,150GiB free-disk floor,8GiB GPU arena, two CPU numeric threads, fresh bounded output, TERM/KILL cleanup. Report wall rows/sec, session.run fraction, source/conversion/gather/write timing and GPU activity over the entire invocation. Target>=80% GPU utilization is aspirational, not established by a CUDA kernel event count.

If grouping and zero pauses still leave substantial host-side gaps, compare a bounded one-batch-ahead source/TPG prefetch against the grouped synchronous control at identical fixed32 inference. At most one producer thread and two live32-row batches; preserve source order, propagate producer errors, and join the producer before cleanup. Keep this as a separate implementation/benchmark so output drift or memory regressions can be attributed. Increasing model batch size comes afterward, with explicit backend requalification.

Stop after one complete bounded comparison per orchestration variant. Adopt only configurations with complete guards and output-equivalence evidence; choose by measured wall rows/sec, not utilization alone.

## Registered pilot and continuation

The prepared pilot compares two four-shard sessions with one eight-shard session on the same eight previously qualified block09 shards (65,536 rows per arm), at fixed32 and zero pause. It records NVML utilization samples and requires bitwise equality of every saved payload array to the existing reference, including both value heads and gathered legal policy. Total bound900seconds; per-arm collection bounds380 and300seconds. Parent owns scheduling after the current cohort finishes; the active plans remain immutable.

Pinned descriptor: `/home/josh/projects/chess/scratchpad/bt4_joint20/takeover_20260916/ceres_throughput/pilot_v1/registered_command.json`, SHA256 `c8d4b979d56854f4b5ede827ff525efd678a58201548edfed9de88242ead81d1`.

The corresponding eight-shard reference spans442.562849seconds including seven pauses; active chunk time232.301631seconds and session.run133.268036seconds. Compare against both the wall-time baseline and the active-time baseline to separate sleep removal from grouping.

Conditional continuation is prepared for untouched cohorts11–12 only:2,111,278rows,34 grouped invocations total, final groups of one and two shards. Full consumer qualification checks the complete physical shard roster and all original source/payload hashes. Group8 is provisional until the pilot passes numerical checks and its measured throughput supports adoption. Two cohort caps sum10,200seconds; outer limit10,800seconds and aggregate output cap4GiB.

## Results

Pending: actual pilot terminal receipt, numerical equivalence, rows/sec, utilization samples, selected grouping, and continuation adoption. No performance gain or80% utilization claim is established by this preregistration.
