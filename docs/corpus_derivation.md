# Corpus observations and row provenance

`scripts/derive_corpus_targets.py` derives replay targets from banked Stockfish
observations. A depth identifies a search rung; it does not uniquely identify the
observation when later narrowed phases also bank that depth.

## Policy and value observations

The defaults retain the existing behavior: `--policy-observation latest-phase`
reads each move at the latest phase carrying the scheme's requested depth, and
`--value-observation latest-phase` retains that value convention independently.
Unflagged derivations keep their existing arrays and metadata. `nodes-N` continues
to use its existing phase0 budget selection.

For uniform-depth schemes, `--policy-observation phase0` reads only the initial
complete all-legal block at that depth. It never overlays later narrowed searches.
A missing complete block uses the existing envelope budget; a legal-support
mismatch is fatal by default. This aligns a uniform-d9 policy with the phase0 d9
observations used by the rank sidecar; it does not qualify a whole corpus merely
because the flag was accepted.

Changing the policy observation does **not** change value selection. Select
`--value-observation phase0` separately if that intervention is intended.
`--value-depth D` remains an independent exact full-width value depth. Nondefault
observation selectors are currently rejected for top-K and node-budget schemes.
The summary's `scheme` and shard `derive_scheme_params` record nondefault
selectors; `derive_value_source` describes the actual value read.

For example, these arguments request phase0 d9 policy, the legacy latest-phase d9
value, and source row references:

```text
--scheme uniform-d9 --policy-observation phase0 \
--value-observation latest-phase --row-provenance
```

These are target semantics, not a registered compute plan. Choose the corpus
snapshot, temperature, limit, output and resource budget for the actual experiment.

## Bounded phase0 policy-support exclusions

`--max-policy-support-misses N` is a separate opt-in budget, defaulting to zero.
A positive budget requires a uniform-depth scheme with
`--policy-observation phase0`. It permits excluding a row only when its selected
complete phase0 policy block is missing legal moves or lists a legal move twice,
while preserving all legal-count rank slots and its full-width metadata.
Truncated/appended rank sequences and narrowed/incorrect width metadata are fatal.
Illegal extras, malformed selected-block ranks/scores, and source/history/input-key
failures remain fatal. An absent complete block still belongs to the separate
`--max-envelope-misses` budget; missing results keep their existing counter.

Before counting a support exclusion, the deriver reconstructs the full banked
history and verifies its original input key. This check does not increment the
emitted-row identity counters. It records source namespace, resolved source path,
config hash, shard, physical row, worker/game/ply, original and stored input keys,
selected depth, missing/duplicate move lists and reason in
`policy_support_misses.jsonl`. The completed summary includes the same records as
`realized.policy_support_exclusions`, the separate
`realized.rows_dropped_policy_support` count, the requested budget, and the evidence
filename (null when no row was excluded). Default-zero runs retain their existing
summary shape.

The budget applies to the entire source derivation, not independently to each
worker. Workers stop if their own count exceeds it; the coordinator checks their
combined count before repacking or publishing the completed summary. Failed runs
retain partial outputs and encountered exclusion evidence, including worker-local
files when a lane fails. Those files do not establish a completed corpus.

All target recipes intended for comparison must use the same source snapshot,
selector and exclusion bound so their surviving inputs match. The provenance-aware
rank sidecar verifies the exclusion ledger against the pinned summary, reconstructs
each excluded row's history and support defect, and refuses a derived reference to
an excluded row. Legacy rank traversal does not accept these drops without row
provenance. No raw row is changed, no shallower policy is substituted, and value
selection remains independent and unchanged. This narrow policy check does not
claim that every later-phase value observation has been structurally qualified.

## Optional compact row provenance

`--row-provenance` requires the generator's verified original history `input_key`.
It preserves references only for surviving rows, through game grouping, spill,
repack, shard cuts and the exact replay permutation. No replay array or trainer
input schema changes. The sidecar is committed inside each shard as
`row_provenance.npz`; shard attributes and summary shard entries pin its SHA-256.
The default writes no sidecar or additional provenance stamps.

The NPZ uses no pickle and no compression. It contains a UTF-8 JSON source table
as a uint8 array and a structured `records` array. Source directory/config namespace
and raw shard name occur once per table entry, rather than once per row.

| Record field | Width |
| --- | ---: |
| Source-table index | uint32, 4 bytes |
| Physical zero-based row within the raw shard | uint32, 4 bytes |
| Original game ID | int64, 8 bytes |
| Original ply and worker ID | int32 each, 8 bytes |
| Original full-history input key | 16 bytes |
| Float16-stored input key | 16 bytes |

Records require **56 bytes per surviving row**: 5.6 GB (5.22 GiB) at 100 million
rows, plus the small source tables and NPZ headers. Processing remains bounded by
the existing buffers/shard sizes. Game IDs stay unchanged in replay arrays; two
source namespaces distinguish otherwise colliding raw game/ply identifiers. Keep
companion corpora in distinct resolved source directories: the existing game-aware
loader namespaces those directories automatically, so they can share one loader.
Flattening or repacking both into a single physical parent still requires explicit
source-qualified game-ID remapping; the planner does not read this NPZ sidecar.

The original key hashes the generator's float32 full-history encoding. The stored
key hashes the exact float16 quantization converted back to float32 by the common
key function. Fractional input planes can change that key on storage, so neither
a root-position fingerprint nor a hash reconstructed from stored `x` can replace
the original key. The writer checks the stored key, game and ply against the row
being written before publishing provenance.

## Raw BT4 sidecar adapter

`scripts/adapt_raw_bt4_sidecars.py` joins previously computed raw BT4 labels to
these references without inference. Invoke its module with a pinned mapping and a
new output directory:

```text
python -m scripts.adapt_raw_bt4_sidecars --manifest MAPPING.json \
  --expected-manifest-sha256 SHA256 --out NEW_SIDECAR_DIRECTORY
```

The schema-1 mapping pins `derived_summary {path, sha256}`, the teacher's
`onnx {path, sha256}`, `policy_output`, `providers` and `remap`, and a `sources`
list. Each source specifies `source_dir`, `sidecar_dir`, its `manifest {path,
sha256}` and an immutable closed-shard `receipts {path, sha256}` snapshot.
Paths in that operational mapping are canonical absolute paths. A live progress
log is not an immutable receipt snapshot. See the adapter's CLI help and module
docstring for the complete interface.

The adapter verifies the raw source/sidecar receipts and reconstructed original
float32 history input, then joins physical rows and compares its exact float16
quantization with derived stored history. It retains the original teacher output;
it does not claim BT4 was evaluated again on quantized input. Grouped policy
gathers avoid repeated decompression per row. A private disk index uses 56 bytes
per raw row plus NPY headers, with an 8 GiB default cap and two cached shards in
memory; full raw verification and the extra history-key reconstruction pass happen
once per used raw shard, including when later output shards revisit it.

The output is the ordinary derived-source-bound BT4 sidecar: legal finite unit
mass, derived source fingerprints and SF policy hashes are validated, and the
completed summary retains transitive raw receipt and teacher/remap provenance.
The existing mixer admission checks remain in force. Output reuse and input
changes during preparation are refused; no live receipt adoption occurs.

## Phase0 d9 ranks

`scripts/sf_d9_rank_sidecar.py` always extracts the complete initial d9 observation;
it does not read the separately selected value target. On a provenance-bearing
source it verifies the pinned row-reference file and original/stored history keys,
then gathers ranks in actual derived row order. A single raw-prefix pass supports
cross-shard revisits and repacking, without replaying derivation filters or RNG.
Legacy sources retain their existing prefix/result-filter/permutation path.

The temporary rank/history cache uses `50 + 6 * top_k` bytes per raw row plus one
byte for duplicate-reference detection and NPY headers. At top3 this is about
6.9 GB per 100 million raw rows; `--max-provenance-cache-bytes` defaults to 8 GiB.
A single raw-shard cache is capped at 64 MiB. Input storage metadata and source
manifest/summary identities are rechecked before publication. Failure preserves
partial evidence; successful completion removes the private cache.

A real transfer still needs its preregistered bounded pilot, completed sidecars,
phase0 rank coverage and a matched game-aware schedule qualification. These tools
do not retrospectively relabel the existing 20M corpus or launch training.
