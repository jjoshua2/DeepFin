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
memory. Original and quantized history keys are collected during the same full
raw verification pass, once per used shard even when later output shards revisit
it. Cache entries are accepted only after verification and receipt checks succeed.

The output is the ordinary derived-source-bound BT4 sidecar: legal finite unit
mass, derived source fingerprints and SF policy hashes are validated, and the
completed summary retains transitive raw receipt and teacher/remap provenance.
The existing mixer admission checks remain in force. Output reuse and input
changes during preparation are refused; no live receipt adoption occurs.


## Optional raw BT4 value retention

`bt4_raw_corpus_sidecar.py --wdl-output NAME --wdl-output-kind probabilities`
can retain a previously qualified W/D/L side-to-move head while labeling future
closed shards. Use `logits` for a graph that emits logits. Both flags are required;
the writer never guesses a three-wide head, its activation, or a blend of heads.
The operator must establish the named head's W/D/L order and perspective from
that model's contract. Shape alone does not prove semantics.

The existing ORT call requests policy and value together from exactly the same
full-history input. `bt4_wdl_raw` stores three unmodified native float16/32/64
values per row; no softmax, normalization, clipping or dtype conversion is
applied. Finite shape/dtype checks always apply; probability heads also require
nonnegative bounded values and unit mass within native storage tolerance.

Optional `wdl` metadata binds the named output, declared kind/order/perspective,
native dtype, row count and array hash. Existing model/source hashes and
source/input/game/ply keys bind the same rows and teacher call. The progress
receipt retains this metadata; deep verification checks value bytes and shared
row identity without rerunning the teacher. At float32, raw triples add 12 bytes
per row (1.2 GB per 100 million rows before compression). Inference cost is
unmeasured; shared inputs and the teacher trunk are reused.

Policy-only defaults and existing compressed policy/key arrays remain unchanged.
Opting in at the same output root skips already completed policy-only shards and
labels only missing closed shards. It never backfills, overwrites or interprets
old policy completion as value coverage. Status and caught-up output report
value-bearing and policy-only rows separately. Existing value-bearing shards
must match the requested contract. `--verify-all` with both WDL flags explicitly
requires that value contract for every closed row and fails on missing coverage;
without those flags it validates any optional values that are present.

The current raw-to-derived adapter still transfers policy only. Retaining WDL is
banking evidence for a future explicitly qualified value join/blend; it does not
change training values, choose a mixture, or adopt a new runtime in a live driver.

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

## Incremental closed-shard selection

Both `derive_corpus_targets.py` and `sf_d9_rank_sidecar.py` accept optional
`--source-shards selection.json`. This selects original closed shards without
creating an alias corpus or rereading an earlier prefix. The JSON contract is:

```json
{
  "schema": 1,
  "source_dir": "<original resolved corpus directory>",
  "source_config_sha256": "<original configuration SHA-256>",
  "source_manifest_sha256": "<original manifest.json SHA-256>",
  "shards": [
    {"source_shard": "w00-00032.jsonl.zst", "rows": 8192,
     "source_sha256": "<closed raw shard SHA-256>"}
  ]
}
```

The row count is the actual closed-inventory claim, not a fixed shard size.
Duplicate names, path traversal, unknown or unclosed shards, wrong row counts,
source/config mismatches and wrong raw hashes are rejected. Selected payloads are
hashed once before processing, with bounded-memory reads. Final publication checks
that those files' device/inode/size/mtime/ctime and the selection/static manifest
remain unchanged. This is a local stable-file contract, not protection against a
privileged actor restoring file metadata. Growing progress logs and newly closed
unselected shards are not pinned, so continued generation is allowed.

Selection always follows the original corpus's canonical shard order, regardless
of JSON entry order. `--limit N` then caps raw physical rows in that selected
concatenation, before result/support/envelope filtering; it can stop inside its last
shard. The deriver's existing `--limit 0` means all selected rows. The rank tool
retains its positive, exact `--limit` contract: derive with the same explicit raw
count when producing a rank-joinable batch, and pass the same selection file to
both tools. A missing or different selection at rank generation is refused.

Both summaries record `source_selection`, including the original source binding,
canonical entries and selection-file path/hash. Original source namespaces, raw
shard paths and physical row offsets in `row_provenance.npz` are unchanged; output
shuffle and filtering still use the existing writer. No-flag commands retain their
original prefix behavior and emit no selection field. The mechanism selects data;
it does not qualify new data quality, a training schedule or playing strength.

## Raw-score policy rewrite on the legacy SF corpus

[`sf_policy_rewrite.py`](../scripts/sf_policy_rewrite.py) rewrites only
`policy_target` on an already derived, pinned schema3 corpus with the original
single-phase, all-legal d9 recipe. Its default `--score-space q --temperature
0.0005` is an identity control. The distinct Soft-SF control uses
`--score-space effective-cp --temperature 10`; temperature is then in centipawns,
not the saturated WDL-derived q scale. Scores come from the original float64
observation, including the generator's existing effective-cp mate-distance
encoding. No rank-gap rounding, alternate mate mapping, value change or inference
is involved. Softening ordinary centipawn gaps can coexist with sharpening
q-saturated mate positions.

```bash
PYTHONPATH=. python scripts/sf_policy_rewrite.py \
  --raw data/nnue_bootstrap/run03_s3 \
  --source data/nnue_derived/armB/qtemp_0.0005_hist_20m \
  --expected-source-summary-sha256 <pinned-original-summary-sha256> \
  --out <new-output-directory> --score-space effective-cp --temperature 10
```

The producer follows the original raw prefix, missing-result filter and seeded
within-shard permutation. It checks source-qualified raw configuration and ordered
worker/game/ply identities, complete legal d9 support, two compact-move mappings,
stored legal masks, and exact reconstruction of every original q/.0005 policy.
It requires the original zero-floor/search-value/history recipe and committed
shard attributes. Completed legacy raw corpora may supply only their mandatory
`summary.json`; an optional `manifest.json` is checked when present, and its
presence or absence must remain unchanged through publication.
Selected-source/provenance-repacked or already postprocessed
corpora are outside this narrow legacy path. Existing derivation defaults are
unchanged.

All 16 non-policy columns, including value labels and `x`, are copied with
compressed-file SHA-256 equality checks. Only game/ply, legal-mask and policy
arrays are decoded; raw observations are streamed and at most one output shard's
sparse legal scores is buffered. There is no dense full-corpus score intermediate.
The policies use float64 softmax, then float32 and float16 storage, with exact
readback and finite/legal-mass checks. Float16 can remove tiny tails; downstream
entropy calculations should normalize the actually stored targets.

The new summary records score space, temperature, source/code hashes, consumed raw
files, copied-file hashes and output policy hashes. The original derive summary
and shard provenance are preserved with explicit postprocessing metadata. Raw and
derived source storage identities are checked again before final publication.
History is **inherited** from the pinned original `input_key_verified` evidence
and unchanged `x`: this is not fresh history reconstruction, and it does not
retroactively qualify the original historical controls. The source input keys are
retained as an emitted-order digest rather than duplicated per-row text.

A failed run retains its `.writing` output and failure evidence and refuses reuse;
there is no resume or implicit partial-corpus success. The producer checks STOP
and a configurable free-disk reserve (150 GiB by default) between raw shards and
output commits. Operational runs still need an independently surviving time cap,
CPU/thread limits and a registered output allowance. This tool does not select a
training arm, launch training, or replace corpus/schedule qualification. The
[training-only sample](experiments/2026-09-08-soft-sf-qualified-training-sample.md) motivates
10 cp as a descriptive entropy match; it is not evidence of playing strength.

### WDL-only labels from the original derived SF corpus

`scripts/bt4_derived_wdl_sidecar.py` can bank a named BT4 WDL head beside a
pinned, completed original SF corpus. It reads only `x`, `game_id`, `ply_index`
and their presence flags in batches. It does no raw-history replay, policy
inference, legal-move mapping, deduplication or training-target blending. A bank
joined by original source directory, shard and physical row can subsequently be
shared by policy recipes that independently prove the same row/input lineage.
That downstream training join is not implemented here.

Supply the original `derive_targets_summary.json` SHA, model SHA and an explicit
W/D/L side-to-move output name and `logits` or `probabilities` kind. Head semantics
must come from the model contract; a three-wide shape alone does not establish
order, point of view or activation. Only that named output is requested from ORT.
Native float16/32/64 values are stored unchanged, with finite/shape validation and,
for probabilities, bounded rounding tolerance around unit mass.

```bash
python scripts/bt4_derived_wdl_sidecar.py \
  --source ORIGINAL_SF --expected-source-summary-sha256 SOURCE_SHA \
  --onnx TEACHER.onnx --expected-onnx-sha256 MODEL_SHA \
  --wdl-output EXACT_OUTPUT_NAME --wdl-output-kind probabilities \
  --out NEW_WDL_BANK --start-shard 0 --max-shards 1 \
  --batch-size 256 --threads 2 --gpu-mem-gb 0 \
  --max-seconds 900 --minimum-free-gib 150 --max-output-gib 1
```

The selection follows the summary's canonical contiguous shard order; a final
selection may end at the corpus boundary. For CUDA, use a positive memory budget
and an **explicit absolute shared `--gpu-lock`**. The disposable child owns ORT
and retains that lease until process exit. The parent bounds the invocation,
including lease wait and 30 seconds reserved for termination, and monitors STOP
and free space every two seconds. Output size is sampled every ten seconds and
at completion; the cap can miss transient overshoot and is not a RAM limit.
Use the usual low-priority affinity and numeric-thread environment for the host.
`OUT/STOP` and optional `--stop PATH` stop only this invocation. Original source
files and unrelated process owners are never modified.

Source admission requires committed/finalized original SF shard attributes,
verified history schema 3 and the `lc0_root_legacy_meta`/`v2_threats` regime.
During consumption, all 175 stored float16 planes must be finite; consumed
history, repetition, castling and side-to-move bits must be binary, metadata
planes constant, and the rule-50 plane one of the 101 stored `integer/100`
values. The shared `x_to_lc0_planes` converter recovers every clipped counter
exactly after float16 storage. Planes 110/111 are replaced by the converter;
extra planes are not teacher inputs. This preserves the actual LC0 feed on the
validated writer domain. It does **not** recover the original full float32
`input_key`, replay raw history, or retroactively qualify historical controls.

Each completed Zarr sidecar contains `bt4_wdl_raw`, `row_index`, `game_id`,
`ply_index` and a 32-byte `lc0_feed_sha256` per row, calculated after conversion
to the model's input dtype. Attributes bind the source summary, source storage
identity, hashes of the consumed source columns, teacher, output contract,
converter/producer code and stored-array hashes. All batches are read back before
atomic publication. Existing completed matching sidecars are content-verified and
skipped without a session; changed source storage, teacher, output kind, content,
or partial `.writing` output is refused. Partial output and per-invocation failure
receipts remain for inspection; no automatic overwrite or different-kind backfill.

At float32 WDL, payload overhead is 44 bytes per row for values plus feed digest,
plus an 8-byte row index and the source-native game/ply fields (normally 8+4 bytes),
before compression and small shard metadata. No dense policy or 112-plane feed
copy is stored. This producer is preparation tooling; it supplies no teacher
quality verdict or trained-value result.
