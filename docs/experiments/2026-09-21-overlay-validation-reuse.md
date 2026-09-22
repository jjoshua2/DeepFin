# Reuse validated overlay identities during one exact epoch

The schema-2 overlay constructor repeated the same dense target validation nine
times per shard. The candidate retains the validated manifest, metadata and
composed content digest in the caller's `BaseSeal` context. Every reuse still
checks local membership/file identities, the sealed base tree, pinned receipts,
and qualified root membership/metadata. A changed dependency fails closed; it
never silently refreshes the cached identity. Returned metadata is copied.
A new epoch creates a new context; there is no global or persistent cache.

This change is stacked on `research/factorial58-teacher-mix` at
`2de1027470d32c8c970fbb38aaa3ef9b8aafd5c3`: main and the ordinary packed-Zarr
branch do not yet contain the schema-2 policy/value overlay implementation.
The active frozen factorial runtime remains unchanged.

## Bounded comparison

The control was untouched `/tmp/deepfin-factorial58-runtime` at `502cd02e0`.
Both arms read the same first eight qualified B shards (65,536 rows), with two
CPU cores, nice 19, and a 300-second cap per arm. The diagnostic invokes the
actual identity, scan, objective-census and game-aware planning stages. Batch
128 makes the small subset's game-diversity constraint valid; batch 512 was
correctly refused and that attempt is banked. No training setting changed.

| Measurement | Frozen control | Candidate |
| --- | ---: | ---: |
| Dense validations through planning | 72 | 8 |
| Selected identity checks | 1.657 s | 1.117 s |
| Shard scan | 6.479 s | 0.698 s |
| Objective census | 6.724 s | 0.811 s |
| Schedule planning | 0.150 s | 0.146 s |
| Constructor stages combined | 15.011 s | 2.772 s |

The **5.42×** reduction applies only to these eight-shard constructor stages.
The complete plans, objective counts, and ordered policy/value/game-ID byte
hashes match exactly. This single sequential warm-cache comparison excludes
full-corpus qualification and its global membership checks; it does not predict
500M-row startup or training throughput. Candidate accesses include the selected
base/overlay root and qualification-receipt checks. An earlier timing overlapping
lint is retained separately and excluded from the result.

## Validation and evidence

All 54 focused overlay tests pass, including real CPU-training parity. New tests
observe the actual nine-to-one validation reduction, equal complete schedules
and ordered tensors, fresh contexts, and rejection of target/base/receipt/inode,
root/local membership, metadata and mid-validation changes. Independent review
approved the implementation and independently checked the benchmark parity.

The focused static gate passes. Whole-repository Ruff passes; basedpyright
reports 23 diagnostics in seven unchanged test files. Baseline verification is
recorded in the receipt; these unrelated diagnostics are not suppressed here.

[Compact receipt](artifacts/2026-09-21-overlay-validation-reuse/receipt.json).
Full raw inputs, timings, rejected attempts, source patch, logs and reviewer
receipt are banked under
`/home/josh/chess-artifacts/operations/overlay-validation-reuse-20260921/`.
Publication does not adopt this code into the running factorial experiment.
