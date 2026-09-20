# Byte-preserving Zarr ZIP decoder pilot

This pilot measures the existing validated shard decoder with directory Zarr,
ZIP_STORED Zarr and NPZ on local NVMe and the external drive. It does not add ZIP
support to the exact-epoch sampler, qualify a migrated training corpus, or change
any production input paths.

Sixteen already verified local pilot shards supply 131,072 positions. Each ZIP
contains every original compressed chunk and metadata file unchanged. The packer
checks member names, ZIP_STORED mode, per-member SHA256 equality and source
stability. Eager decoded arrays must match the existing pilot receipt's complete
array digest, including keys, dtypes, shapes and values.

Zarr may implicitly create ZipStore instances when passed a ZIP filename. The
single-threaded decode call explicitly owns and closes all opened stores in a
finally block, including on decoding failure. This instrumentation is confined
to the benchmark; it changes no production loader behavior.

The existing storage sampler pilot completed before external access began. The
benchmark refuses to stat, resolve or access its external paths while predecessor
PID 1024483 exists. It ran on CPUs 4–5, with two numerical threads, no visible GPU,
16 GiB RSS ceiling, 32 GiB available-memory floor, 80 GiB disk reserve and a
30-minute deadline. Resource failure terminates even if its failure receipt
cannot be written. An independent reviewer approved the exact launch and reran
five focused tests; Ruff passed.

Caches are uncontrolled. This is one fixed-order traversal, local before external,
with directory Zarr, ZIP and NPZ in that order on each medium. ZIP copying and
verification precede decoding, so these results do not establish cold-drive
performance. Decoder timing includes validation and explicit Zarr store closure;
wall timing also includes full-array digest checks. The sampler's planning,
shuffle, prefetch and batch collation are not measured here.

## Results

| Medium | Layout | Validated decode (s) | Wall including digest (s) |
| --- | --- | ---: | ---: |
| nvme | zarr | 6.400 | 11.563 |
| nvme | zip | 6.499 | 12.045 |
| nvme | npz | 10.784 | 15.678 |
| external | zarr | 56.183 | 61.725 |
| external | zip | 6.856 | 11.787 |
| external | npz | 14.602 | 19.576 |

All six decoder cases completed and matched all 16 original full-array digests.
Local packing took 1.182s for 83,691,005 archive bytes; copying all 16 archives to the
external drive took 1.523s (separate from post-copy SHA verification).

Under this cache-affected decoder protocol, external ZIP was about 8.2 times faster
than external directory Zarr and close to local ZIP speed. This makes unchanged
compressed chunks in a single container a strong candidate for a future storage
adapter. It does **not** establish exact-sampler training throughput: discovery,
planning, content identity, store lifetimes and shuffled reads still need their
own qualification before any production migration.

[Raw decode result](artifacts/2026-09-20-zarr-zip-pilot/measure.json) and
[full member-level packing proof](artifacts/2026-09-20-zarr-zip-pilot/pack.json).
The runtime was frozen at 502cd02e072471c901255f3fdb580d6ea7b826d0; script and loader
SHA256s are recorded in the decode result. All outputs remain separate pilot
artifacts. No production sources or queues were changed by this measurement.
