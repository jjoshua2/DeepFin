# Packed Zarr exact-sampler qualification — 2026-09-20

## Question and decision

Can a single-file container remove external-drive small-file overhead while
preserving the existing exact-epoch sampler's ordered tensors and memory checks?
The matched pilot passes. This qualifies the opt-in ordinary-shard reader for
further measured use; no current training, target overlay, storage seal or queued
experiment was migrated.

The implementation was frozen at
`70faba7d11b7c4366be57f12dd551f3e75341414` before an independent root agent ran the
qualification. It reads ZIP_STORED archives with unchanged Zarr compressed chunks,
explicit lazy-store ownership, archive mutation checks, and existing array/codec
validation. Global directory replay discovery and training defaults are unchanged.
See [the API and limitations](../packed_zarr.md).

## Matched readout

Both representations supplied the same 16 shards / 131,072 rows, batch256,
seed121, 175 input planes, `lc0_root_legacy_meta`, and `history_rep_fix=True`.
Planning/loading used one worker each. The sampler's mirror memory contract was
enabled; full trainer augmentation/collation was not exercised.

| Representation | Planning | Consumer wall, including digest | Batch waits | Digest |
| --- | ---: | ---: | ---: | ---: |
| NVMe directory Zarr | 0.746s | 10.953s | 8.891s | 2.060s |
| External ZIP_STORED Zarr | 1.920s | 12.537s | 10.172s | 2.361s |

Every ordered batch tensor matched. Both sequence hashes are
`461ad75fe7d195d39478bc6be45a101d068c27ced5e79fcd957a3949e4b0255c`.
The representation/path-sensitive corpus and plan hashes intentionally differ;
matching row semantics does not permit replacing storage underneath an old plan.

The complete [raw receipt](artifacts/2026-09-20-packed-zarr/nvme-vs-external.json)
contains timing precision, plan parameters, paths and fingerprints. The host copy
is `/home/josh/chess-artifacts/operations/packed-exact-qualification-20260920/nvme-vs-external.json`.

A second independent qualification used 32 shards / 262,144 rows at the intended
batch512, with the same remaining sampler settings:

| Representation | Planning | Consumer wall, including digest | Batch waits | Digest |
| --- | ---: | ---: | ---: | ---: |
| NVMe directory Zarr | 1.493s | 24.430s | 20.025s | 4.403s |
| External ZIP_STORED Zarr | 4.081s | 25.878s | 21.771s | 4.104s |

All ordered tensors again matched, SHA256
`2bccddc7bef89b6b223af16bcdab8b31dad16698cab3badb8a26adc7e1fc6585`.
The [batch512 receipt](artifacts/2026-09-20-packed-zarr/batch512.json) records the
complete settings and measurements.

## Interpretation and limits

External packed storage approached the NVMe control in this small warm/cache-
affected trial. This is a fixed-order comparison, not independent replication or
cold-media bandwidth, and it does not establish billion-row scaling or GPU
training throughput. The original 16-shard sample cannot support batch512 under
the one-position-per-game constraint; the 32-shard follow-up qualifies batch512.
A larger working-set and actual training check remain separate adoption work.

Source-parent partitions must survive packing: flattening independently numbered
corpora would merge game namespaces. Packed overlays/base bindings are refused.
No writer or cache eviction policy is introduced by this change.

## Validation and independent review

The author ran 95 packed-reader, exact-epoch and existing overlay regressions,
including actual corrupt-CRC decode, truncated archives, FD closure on exceptions,
unsafe/duplicate members, unknown codecs, post-plan mutation, two-source ordered
batch parity, and refusal of an oversized working set before decoding wide inputs.
A further 57 legacy codec, NPZ, shard-validation and validation-cache tests pass
in the host environment (152 tests total). Scoped Ruff, Vulture and host
BasedPyright pass (zero type errors or warnings).

The independent root agent reviewed storage ownership, discovery, hash/decoder
integration and qualification code, found no actionable blockers, then ran the
real matched qualification above. That review is distinct from the author's tests.

## Training CLI follow-up

The opt-in reader now reaches the main-based offline launcher through
`--sampling-mode game_epoch --allow-packed-zarr`. Identity stamps and label
coverage include packed shards, staging preserves suffixes/source parents, and
all later epochs inherit the option. Replacement sampling and overlays refuse it;
a mixed CLI roster without the flag cannot silently omit its ZIP rows.

The follow-up passed 218 driver, converter, packed-reader and packed-CLI tests,
including a real two-epoch CPU training run and bad identity/partial-label
refusals. Root independently reviewed the CLI propagation and found no actionable
issues. No GPU training or frozen job adoption occurred. Main currently lacks the
frozen successor's recovery CLI; that integration remains separate.
