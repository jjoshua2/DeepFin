# External storage and BT4 labeling efficiency — 2026-09-20

## Question

Can the external drive supply the exact-epoch training sampler fast enough, or
should future 500M-position training use a bounded NVMe staging cache? Separately,
does increasing BT4 one-node batch size from 128 to 256/512 improve end-to-end
labeling throughput without changing accepted output semantics?

The existing 58M factorial experiment remains unchanged. Benchmarks write to
separate artifact namespaces and do not move pinned training inputs.

## Storage pilot

`scripts/benchmark_storage_loader.py` uses 16 real 8,192-row V50 shards, copied
onto NVMe and `/mnt/e`. It compares the original directory Zarr representation
with lossless single-file compressed NPZ supported by the existing shard loader.
The pilot preserves extended corpus metadata; the generic NPZ writer rejects
that extended metadata, so the benchmark writes the supported arrays plus
`meta_json` representation directly. Every packed shard must reproduce the
original decoded tensor digest.

The actual `GameAwareEpochBuffer` plans and consumes a complete epoch at batch256,
seed121, with one planning/loading worker. NVMe and external directory variants
must return identical ordered batch tensors. The first16shards cannot support
batch512 under the sampler's one-position-per-game rule; its refusal was retained.
Packed NPZ uses the actual validated shard decoder in a separate measurement,
with full tensors checked against source digests. The exact sampler currently
discovers only directory shards; packed training support is not implemented. Reverse-order repeated traversal provides a warm/cache-
affected comparison, not independent replicates. Initialization, consumer wall
clock, digest overhead and batch wait percentiles are recorded separately.

Limits: two CPU affinity cores, nice19, 16GiB process RSS, 32GiB available RAM,
80GiB free storage and one hour wall time. No GPU use, cache dropping, or production
writes. Monitor failures terminate the pilot. Copies deliberately omit POSIX
metadata preservation because the Windows drive mount rejects those operations.

This is a small cache-affected sampler pilot. It does not prove cold external-disk
bandwidth, multi-terabyte throughput, full trainer augmentation/collation speed,
or GPU saturation. A storage cache should be justified by a larger measured
working set and a short training confirmation before adoption. Packing does not
change how rows are shuffled; the sampler's resolved path identities do change,
so a storage migration still needs a newly qualified epoch plan.

## BT4 batch screen

`scripts/bt4_batch_benchmark.py` measures batches128/256/512 on the same closed raw
shard, checking identity equality and bounded policy/value differences. It records
inference time and full producer-stage time separately. The placement validator
reuses the strict Ceres contract: CUDA must execute neural kernels; CPU execution
is restricted to small integer/bool shape work. Merely registering CUDA or seeing
a CUDA memory-copy event is not accepted as GPU computation.

The original benchmark missed CPU FusedMatMul/FusedGemm fallback. Thirteen
regressions now cover neural fallback, floating-point shape operations, unknown
providers, missing/large/dynamic shape metadata, and a valid mixed-placement graph.
Batch size is recorded in the proof. This qualifies measurement, not automatic
production adoption of a larger batch.

## Status

Storage preparation finished for all16shards. The initial external copy attempt
failed on POSIX metadata; byte-only copies succeeded. The first loader attempt
correctly rejected an incorrect history_rep_fix=False setting. The retry uses the
source's actual True contract and retains the preparation receipt. Subsequent
preflight correctly refused the small sample at batch512 and exposed the sampler's
directory-only discovery contract; the bounded retry uses batch256 and separates
packed decoder measurements from sampler measurements.

Bulk artifacts:
- NVMe: `/home/josh/chess-artifacts/operations/storage-loader-pilot-v2-20260920/`
- External: `/mnt/e/chess_storage_loader_pilot_v2_20260920/`

Results will be appended after the complete sequence-equivalence check. Early copy
timings are not a training-throughput verdict.

## Validation

Independent review covered resource failure handling, copied tensor identity,
prepared-roster verification, and provider placement. Seventeen focused tests pass.
Ruff passes. The host full lint run reports14 type errors in six unchanged baseline
test files; no new benchmark/test files appear among those findings. The initial
restricted-environment lint could not discover installed dependencies; the reported
host findings are from the actual interpreter environment, not that failed discovery.

## Compact queue readout

`python scripts/bootstrap_queue_status.py --loop <operator-directory>` reports
active/queued jobs, known duration estimates separately from timeout caps, and
recorded recovery states. `--json` includes five compact completed records in
queue order. It takes the existing scheduler read lock, performs no dispatch or
polling, and deliberately makes no process/GPU-health claim. Historical failed
experiments remain in status counts but are not all treated as current incidents.
