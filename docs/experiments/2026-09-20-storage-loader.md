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
discovers only directory shards; packed training support was not available to this pilot. Reverse-order repeated
traversal provides a warm/cache-affected comparison, not independent replicates. Initialization, consumer wall
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

The complete readout below passed sequence-equivalence checks. Early copy
timings remain separate from training-throughput measurements.

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

## Completed sampler/NPZ pilot

All16shards (131,072rows) completed. All four exact-sampler passes produced the
same tensor-sequence digest, and all packed NPZ decodes matched original shard
array digests. [Full receipts](evidence/storage-loader-20260920/sampler-and-npz.json).

| Medium | Pass | Planning seconds | Consume seconds including digest | Positions/second including digest |
| --- | --- | ---: | ---: | ---: |
| NVMe directory | first | 0.72 | 11.19 | 11,715 |
| External directory | first | 72.40 | 127.74 | 1,026 |
| External directory | repeat | 87.88 | 139.63 | 939 |
| NVMe directory | repeat | 0.79 | 11.65 | 11,253 |

Maximum batch waits were43.39s on the first external traversal versus3.17s locally.
The completed58M continuation processed58,090,688rows in29,211.84s (~1,989rows/s),
including its run overhead. This is a planning comparison across different workloads,
not a measured GPU slowdown, but current direct external-directory loading lacks
headroom. A larger cold-working-set test would not rescue that observed metadata cost.

Packed NPZ validated decoding of the same131,072rows took10.46s on NVMe and15.11s
externally (16.58/20.03s including digest verification). This strongly motivates
packed storage but does not establish exact-sampler performance: NPZ ignores lazy
loading and expands full arrays during planning. The next candidate is ZIP_STORED
Zarr, retaining compressed chunk bytes and lazy metadata/game-column reads.

## Capacity sample

Twelve sampled derived shards across four cohorts used635–707bytes/row. A closed
raw G10 shard used2,569bytes/row. These are selected samples, not corpus-wide bounds.
At unchanged density,500M derived rows would occupy roughly318–354GB decimal;
the raw corpus would add roughly1.28TB before other teachers, variants, metadata
allocation, and checkpoint retention. [Samples](evidence/storage-loader-20260920/size-samples.json).

This supports keeping raw history on the external drive while potentially fitting
an entire selected500M training representation in a1TB NVMe budget. The exact
required capacity still depends on teacher representation and retained variants;
there is no reason to assume every raw artifact must be staged for training.

## Lossless Zarr ZIP follow-up

[PR #793](https://github.com/jjoshua2/DeepFin/pull/793) preserves every original
Zarr metadata and compressed chunk byte inside one ZIP_STORED file per shard.
All six validated decoder arms matched the source tensor digests. On the same
131,072 rows, external ZIP decoding took 6.86 seconds versus 56.18 seconds for
external directories; local ZIP decoding took 6.50 seconds. These measurements
exclude digest time and use a different harness from the sampler table above.

This isolates a practical metadata-access improvement while preserving lazy reads.
It remains a fixed-order, cache-affected decoder pilot. Opt-in integration into the
exact-epoch sampler is being tested separately; no current training inputs have
been migrated and these numbers are not a measured training speedup.
