# Opt-in upload-compaction buffering budget

The server can bound **retained replay ndarray payload** and the number of
resident source uploads across all trials. This does not bound process RSS,
concurrent decoding, writer scratch space, Python object/string overhead, or the
separate six-hour upload-deduplication records.

The feature defaults off. A launch-time opt-in matching the existing maximum
single-upload uncompressed-array allowance is:

```yaml
tune:
  distributed_upload_compact_buffer_bytes: 1073741824
  distributed_upload_compact_buffer_uploads: 2000
```

The corresponding standalone-server options are `--upload-compact-buffer-bytes`
and `--upload-compact-buffer-uploads`. A byte budget of zero disables the feature;
the source-upload default is 2,000. An enabled byte budget must cover the configured
per-upload uncompressed-byte limit, otherwise ordinary individually valid uploads
could never fit. These keys are driver-launch-fixed, not live reload settings.
No production configuration is changed by adding this feature.

The existing per-identity size and age targets still control ordinary batching.
Under aggregate pressure the server tries the oldest buffers first, including
other trials, using the existing durable compaction transaction. Successful flushes
retain receipt/commit-witness ordering. Failed flushes retain their accepted rows
and raw sources. A new admission that still cannot fit gets HTTP 503 with
`Retry-After: 1`; it is not a terminal `rejected` response, so the worker retains
its original upload. Only that unacknowledged request's staged copy is cleaned up.

Startup recovery applies the same budgets. Valid sources that do not fit remain
on disk and resume on subsequent manifest-triggered sweeps. Live recovery skips
sources already owned by an accumulator and never renames live in-flight groups.
Transient load, allocation, and directory-enumeration failures keep retry armed.
An oversized historical source is retained and reported, while smaller siblings
may proceed; it needs a larger configured budget or deliberate operator recovery.

Empty accepted sources cannot produce a compacted shard. They count against the
source-upload limit even when they share an identity, and are retained rather than
silently discarded. Enough empty sources can therefore exhaust this capacity and
cause backpressure until an operator addresses them. This is a bounded-memory
tradeoff, not a guarantee of uninterrupted admission under hostile input.

The default 2,000-source allowance permits a target-sized batch of one-row uploads.
The 1 GiB example follows the existing single-upload limit, not a measured safe
process-RSS allowance. In a bounded fixture, 2,048 minimal compact-policy rows
retained 45,883,392 ndarray bytes; optional fields and input widths change that
cost. Budget activation and production headroom assessment remain separate from
merging this opt-in implementation.
