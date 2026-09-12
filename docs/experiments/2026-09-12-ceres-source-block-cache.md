# Ceres source-read cache

A bounded CPU benchmark supports caching decoded source blocks before future
Ceres collection. It does not measure end-to-end GPU throughput or qualify a
replacement for a running collector.

## Observed bottleneck and comparison

One completed original-corpus chunk (shards 1008–1023, 131,072 rows) recorded
579.52 seconds elapsed, with 341.49 seconds inside 4,096 synchronous `session.run`
calls. The remaining 238.03 seconds includes CPU preparation, source reads,
verification, output and guards; existing logs do not separate those costs.
`session.run` includes copies/synchronization and is not pure GPU kernel time.
Its saved ORT profile covers only the first call, so it cannot rank steady-state
CUDA Graph or I/O-binding opportunities.

The source stores 512-row compressed chunks, but the collector reads 32 rows per
call. The registered microbenchmark compared one complete 8,192-row shard using
those repeated 32-row reads against one decoded 512-row block sliced into the same
batches. Both paths read the same five source columns in original order. The
preregistered acceptance criterion required every per-column/per-batch digest to
match, with no repeat, inference or end-to-end speed claim.

| Measurement | Direct 32-row reads | Cached 512-row blocks |
| --- | ---: | ---: |
| Read/decode/slice time |1.742816 s |0.123424 s |
| Digest/compare time |0.157714 s |0.152701 s |
| Total loop time |1.901140 s |0.276438 s |

All 256 batches across all five columns matched. Read/decode was 14.12× faster,
saving 1.619392 seconds on this shard. The whole bounded process took 2.223 seconds
and peaked at 69,812 KiB RSS, under two CPU threads with the GPU hidden.

Baseline ran first and may have warmed filesystem pages. This single ordered
comparison retains that confound. Multiplying its saving by 16 suggests 25.9 seconds
against the historical 579.5-second chunk, but that is an unmatched extrapolation,
not a measured collector speedup. It does not explain the entire residual or
justify claims of 14× faster labeling.

[Compact evidence](evidence/ceres-source-cache-20260912.json) records the plan,
source/chunk metadata, identical decoded digest, code hash, terminal receipt and
local artifact paths. No source data or model was rewritten.

## Future collector behavior

The collector caches one block aligned to the source `x` row chunk for its five
input columns. Chunk sizes 1–1024 rows are cached, bounding the main cached
feature block to 21.875 MiB at the qualified float16 layout, plus small identity
columns and crossing-batch views/copies. Batches crossing a storage boundary and
partial final batches preserve original order. Larger source chunks retain direct
batch reads, so this optimization does not reject previously supported layouts.
The selected-row path already holds arrays in memory and keeps direct slicing. Source hashes, conversion,
fixed 32 inference, padding, guards and target arrays retain their existing
semantics. The changed producer hash requires a fresh qualified namespace.

New invocation completion metadata reports aggregate source-read, CPU
preparation/postprocessing, synchronous session-run and output/readback times
for newly written shards. A residual covers shard setup and guards; shard totals
exclude model/session creation and final corpus admission. CPU postprocessing
includes first-call provider qualification. These coarse timers identify the next
bottleneck without claiming GPU utilization. Cached-only invocations report an
empty timing map; existing shard attributes and teacher arrays are unchanged.

Tests compare storage-boundary reads with direct reads, verify one block fetch
per column, retain direct reads for oversized chunks, and run the actual writer
with fake inference against its direct-read equivalent. They compare all feed batches, source hashes
and output arrays, including a padded final batch and both value heads. Real GPU
throughput and teacher-output qualification remain future work; the pinned
collector and active jobs are not updated by this change.
