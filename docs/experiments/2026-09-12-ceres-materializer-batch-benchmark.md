# Ceres materializer: one real shard at batch 128 and 512

The completed screen produced identical decoded arrays and metadata at both batch
sizes. Observed shard time fell from 10.483 to 6.745 seconds, but an unchanged
identity-check stage accounts for about 64% of that difference. This establishes
batch-512 compatibility on this shard, not a causal production speedup.

## Preregistered scope

Extract the existing per-shard loop into `ceres_target_mix.rewrite_shard`, with
full-corpus `rewrite` delegating to it and retaining manifest admission, recipe
validation and publication. The extraction preserves source/teacher identity,
feed alignment, copy, payload hashing, target readback and non-policy checks.
Direct callers must perform complete manifest admission and recipe validation,
use fresh outputs and retain final-publication obligations; a shard receipt is
not full-corpus qualification.

Run one pass per arm, baseline 128 first and 512 second, on the actual original
20M shard 0 (8,192 rows), using the full accepted 2,309-shard/18,910,484-row
manifest. Use CeresB50: equal BT4/Ceres policy weights, both temperatures 0.5.
The source arrays and BT4 policy use 512-row chunks. Compare all 17 decoded
arrays, dtype, shape, row order, complete attrs and numerical summary metrics.
Any mismatch fails compatibility. No training or strength decision is registered.

The frozen plan was independently reviewed before execution. Budget: 180 seconds
inclusive, CPUs 2–3, no GPU, two numeric and Blosc threads, 8 GiB address space,
2 GiB sampled RSS, 128 MiB fresh outputs and 150 GiB SSD reserve. STOP/deadline
checks remain frequent; heavier resource sampling uses a five-second cadence.
The running batch-128 full materializer was preserved. On failure, retain the
independent fragment outputs and receipts; do not admit them for training.

## Completed readout

The process exited 0 in 26.00 seconds, peak RSS 960,512 KiB. Internal elapsed time
was 25.129 seconds, including a 2.078-second decoded-output comparison.

| Seconds | Batch 128 | Batch 512 |
| --- | ---: | ---: |
| Full manifest admission | 1.674 | 1.510 |
| Complete shard operation | 10.483 | 6.745 |
| Source/BT4 identity checks | 4.002 | 1.607 |
| Ceres source/TPG alignment | 2.984 | 2.707 |
| Policy math | 0.736 | 0.877 |
| Unattributed shard work | 1.862 | 0.951 |
| Total arm, including admission | 12.262 | 8.329 |

All 17 arrays matched in decoded bytes, shape and dtype; complete attrs and
metrics matched. The policy-target SHA-256 is
`9e09fc9d10c310ac9cd4f4db74b50542b17d11526281c15d1151234c7aa19c4d`.
Both arms changed 8,138 rows, lost zero support entries, and retained
`legacy-root-position-v1` lineage with zero newly verified full-input-digest
shards. Compressed-file hashes/storage identities differ; decoded equivalence
does not imply identical compressed bytes.

Baseline-first cache warming, allocator state and concurrent-host scheduling
confound the timings. The unchanged source/BT4 identity work alone fell 2.395
seconds, approximately 64% of the 3.738-second shard difference. Policy math
became slightly slower. Lower residual time is compatible with fewer partial
chunk writes, but residual also includes reads, assembly, readback, guards and
other checks; it is not an isolated storage measurement. Instrumentation adds
some overhead. TPG alignment still uses batches of 128 in both arms.

Retain 512 as an output-equivalent candidate for future materializations. Do not
extrapolate a percentage saving to the active full run or 100M rows, change the
active run, or claim that TPG reconstruction was optimized. This single-shard
screen contains no inference, training or playing-strength evidence.

## Validation and retained evidence

Sixteen focused tests passed, including actual full-writer versus extracted
helper comparison across all arrays/attrs at different batch sizes. Scoped
explicit-interpreter type checking passed with zero errors/warnings; whole
Ruff and Vulture passed. Whole-project type checking was not repeated after
prior related broad-scope timeouts; no whole-type pass is claimed. Independent
source review and benchmark-plan review both passed.

The [compact evidence](evidence/ceres-materializer-batch-benchmark-20260912.json)
contains the complete result, per-array hashes, terminal details, source pins,
validation and review identities. Host-local plan, operator, logs and output
fragments remain under
`scratchpad/bt4_joint20/ceres_materializer_batch_benchmark_v1/`.
The run used base `7d60d9ecdb9f08b381d37073c7675ebb0d8374dd` plus the frozen
extraction diff, with producer SHA-256
`44de843884d37fc2709a8ebc95944299ec6412b8e97179db4eddeaf637a60e3d`.
The historical plan pins that uncommitted state; committing the same source
does not retroactively change the execution identity.
