# Exact batch gather CPU screen — September 22

Preallocation removes copying work in the isolated gather, but the absolute saving
is too small to prioritize a runtime change now. Preserve the prototype and
qualify the existing bounded host-batch overlap separately. No production patch,
training change, GPU benchmark or throughput claim follows from this screen.

| Real shard slices | Original median / batch | Prototype median / batch | Local ratio | Saving / batch |
| --- | ---: | ---: | ---: | ---: |
| 4 | 3.399 ms | 1.939 ms | 1.75x | 1.460 ms |
| 8 | 3.284 ms | 1.405 ms | 2.34x | 1.879 ms |

Both screens use 512-row batches from the first 256 rows of each selected cohort00
D overlay/base shard. They retain all persisted allowed fields and substitute the
real D policy/value overlay arrays. Fixed descending indices retain the original
chunk-grouped output ordering. Keys, dtypes and values match the original
`_slice_epoch_arrays` plus `_concat_sparse_batches` result. Each variant has three
100-call timings in fixed ABBAAB order. The original cProfile splits most measured
gather time between concatenation and row slicing; the full raw profile is banked.

The prototype constructs a zero-row concatenation to establish output schema,
then uses preallocated output slices with `np.take(..., out=..., mode="clip")`.
This is deliberately **not** a general implementation: only the fixed same-schema,
valid nonnegative indices are qualified. Mixed optional fields, metadata conflicts,
dtype promotion and invalid-index behavior would require separate compatibility
work. Whole-shard content requalification and actual epoch construction were not
performed. These timings omit refills/decoding, target preparation, pinning, transfer
and GPU computation. They cannot establish end-to-end training speedup.

The [training-path audit](evidence/exact-gather-screen-20260922/training-data-path-review.json)
pins the D summary and frozen code used for context. D's banked 1,290 training windows recorded 13,580.64s in
`batch_prefetch_wait_s` across 113,459 updates, approximately 120ms per update.
That phase includes sampling/preparation plus pinning and H2D issue; it is not pure
I/O wait. Comparing this scale with the 1.5–1.9ms local saving motivates the priority
decision; it is not an extrapolated throughput estimate. Existing opt-in host
batch overlap (#615) remains off in the frozen experiment and needs a separately
scheduled GPU qualification before adoption.

The CPU screens ran on cores 16/17 at nice 19 with two library threads, took 2.21s
and 1.93s respectively, and peaked below 178MiB RSS. The external tensor bank is
approximately 55MiB. This stays below the 10-minute/2-GiB-extra screen bounds.
Development source is main `4e8d463e1`; no frozen runtime was edited.

The [adjudication](evidence/exact-gather-screen-20260922/exact-gather-screen-adjudication-20260922.json)
links hashed scripts and timing/profile receipts. The [input manifest](evidence/exact-gather-screen-20260922/inputs.json)
pins the eight saved NumPy tensor archives and both fixed index schedules. Full
inputs, prototypes and receipts live at
`~/chess-artifacts/operations/exact-gather-screen-20260922/`. Source snapshots and
profiles are published with this record; tensor archives remain external.

Validation: exact fixed-input key/dtype/value parity passed for both screens;
all 11 documentation path guards passed. This is archived prototype evidence,
with no changes to supported runtime modules.
