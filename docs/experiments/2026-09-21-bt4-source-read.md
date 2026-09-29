# BT4 source read-ahead CPU screen

## Question and fixed protocol

Can the derived-source BT4 value labeler avoid repeatedly decoding the same source
chunk without changing any inference batch or supervision? Its usual batch128
reads from x chunks containing512 rows, decoding each x chunk four times.
The candidate reads a bounded storage/batch-aligned block of the five source
columns and serves the same inference slices from memory. It changes no inference
batch size, WDL calculation, source validation, output format or provenance checks.

The [preregistered plan](evidence/2026-09-21-bt4-source-read/plan.json) fixes an
ABBA baseline/candidate order on8,192 real rows on NVMe and on an already banked
external ZIP shard. Success requires exact source/feed/batch-boundary hashes and
at least15% median source-read improvement on one location. One600-second cap,
two affinity CPUs14,15, nice19, two codec/BLAS threads,150GiB disk and40GiB RAM
floors; no GPU work or cache dropping. Any mismatch or resource breach stops it.

## Implementation and integrity

`source_batches` in `scripts/bt4_derived_wdl_sidecar.py` chooses the least common
multiple of x chunk rows and inference batch rows, up to2048 rows. Above that
limit it retains direct batch reads. A larger caller-selected inference batch
never gains additional read-ahead. It calls the existing resource/STOP guard
before each batch, including before loading a fresh block. The labeler retains
all source-domain validation, per-row LC0-feed hashes, immediate output readback,
source/summary mutation checks and partial-output preservation.

A2048-row x block is about44MiB. That is a block-size bound, not a process peak:
old yielded views and new blocks can coexist during rollover, and decoding/feed
preparation adds temporary arrays. No cross-shard or persistent data cache exists.
This helper also works with a manually opened ZipStore, as tested below; this PR
does not add ZIP source discovery/admission to the full BT4 producer.

## Readout

[Raw runs](evidence/2026-09-21-bt4-source-read/runs.jsonl) and
[completion](evidence/2026-09-21-bt4-source-read/complete.json) retain all8 runs.
All65,536 processed rows have the same source/feed/batch sequence digest across
arms and locations. Each traversal makes64 batches of128, including the same
float32 LC0 feed conversion and per-row feed hashing.

| Location | Median source read, baseline → candidate | Read speedup | CPU preparation including reads/digests |
| --- | --- | --- | --- |
| NVMe directory |0.384s →0.098s|3.91x|1.360s →1.038s (1.31x)|
| External packed ZIP |0.547s →0.134s|4.09x|1.522s →1.164s (1.31x)|

This passes the registered CPU-read screen. It is a small, cache-affected test
using already banked data, not cold-disk500M throughput or full teacher inference.
The external ZIP result exercises the source-reader helper; it does not establish
full-producer packed-source support. Inference, source discovery, full source
qualification and output writing are outside the timed loop. The opportunity is
less repeated decompression and metadata access with the same training targets.

Twenty-eight focused tests pass, including full tiny producer parity of inference
inputs, requests, output arrays and attributes against original unbuffered reads;
unaligned batch/chunk sizes, tails, oversized-chunk fallback, STOP before reads,
source mutation and malformed inputs. Publication includes independent review
and the final static-check status below. Active D training and the SF-free target
preparation use their existing frozen runtimes; neither was restarted or changed.


## Review and checks

[Independent review](evidence/2026-09-21-bt4-source-read/independent-review.json)
approves the implementation and claims; the reviewer independently ran all28tests
and recomputed the recorded timing/parity results. Focused basedpyright/vulture
and final whole-repository Ruff pass. Whole-repository basedpyright reports14
pre-existing diagnostics in six unchanged test files. Untouched main7939609c9
reproduces those14 diagnostics in two bounded groups:
[eleven](evidence/2026-09-21-bt4-source-read/baseline-typecheck.log) and
[three](evidence/2026-09-21-bt4-source-read/baseline-typecheck-other.log).
No diagnostics affect this change.
