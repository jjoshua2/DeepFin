# Packed Zarr for exact epochs

`GameAwareEpochBuffer(..., allow_packed_zarr=True)` can read ordinary immutable
`shard_NNNNNN.zarr.zip` files alongside ordinary directory shards. This is an
explicit API option: rolling replay discovery and training defaults are unchanged.
It does not enable packed overlays or NPZ planning, move existing data, or update
any frozen experiment.

Use ZIP_STORED archives containing the original relative Zarr file names and bytes.
Zarr chunks remain compressed with their existing codec. The reader rejects
compressed ZIP entries, duplicate/unsafe names, symlinks and special entries,
missing root metadata, and non-Zarr members such as overlay/base-binding manifests.
Existing codec, dtype, shape and target validation remain active. Archives must be
complete and immutable before admission; a partial ZIP is not a resumable shard.

Packing must preserve source-directory partitions. Game identity is the resolved
shard parent plus source-local `game_id`. Do not flatten independently numbered
corpora into one directory. A staged directory can contain symlinks to separately
packed source directories, keeping the same partitions and shard order. Discovery
rejects duplicate staged shard indices, including a directory and ZIP for the same
index. Files from different source parents may still use the same original index
when staged under distinct indices.

Archive fingerprints cover the complete ZIP bytes with before/after descriptor and
path identity checks. Directory fingerprints retain the original tree hashing.
Representation changes therefore intentionally change corpus/epoch plan hashes,
even when every sampled tensor and row order matches. Build a new qualified epoch
plan; never replace storage beneath an existing plan or running epoch.

Planning opens lazy arrays inside an owned store context, reads declarations and
necessary game/eligibility columns, then closes the archive. It does not expand
`x` or policy targets merely to size them. Existing objective-mask callbacks may
read their required columns. Full decoding starts only after the existing working-
set qualification. Eager reads close on success and errors. Direct lazy callers
must use `with open_shard_arrays(path, lazy=True) as (arrays, meta):` and finish
all array accesses inside that context; unmanaged lazy ZIP reads are refused.

## Qualification without training adoption

Use separately prepared immutable directory and packed roots with the same shard
roster and source partitions. Resolved control targets must be directories and
resolved packed targets must be regular `.zarr.zip` files; aliases cannot make
both arms read the same representation. Matching-format symlinks are supported.
The bounded CPU command compares every ordered batch:

```bash
python scripts/qualify_packed_zarr_epoch.py \
  --directory /absolute/control-shards --packed /absolute/packed-shards \
  --result /absolute/artifacts/packed-qualification.json \
  --batch-size 512 --input-planes 175 \
  --input-history-encoding lc0_root_legacy_meta --history-rep-fix \
  --mirror-augmentation --working-set-gib 12 --seconds 1800
```

Supply the corpus's actual history/plane contract. The sampler still enforces
one position per game per batch: the original 16-shard pilot cannot support
batch512, so use a sufficiently large sample (32 pilot shards where feasible),
or explicitly measure batch256 and report that scope.

The report includes separate planning time, batch waits, consumer wall time,
digest overhead, plan identities and matching ordered tensor hashes. This is a
fixed-order, cache-affected exact-sampler measurement, not cold-disk bandwidth or
full training throughput; mirror/collation execution is not exercised. The CLI
is Linux-only and uses two allowed CPU cores, nice19 and no GPU. It samples
process RSS from `/proc/self/statm` and host `MemAvailable` from `/proc/meminfo`
every 0.5 seconds, stopping above 16GiB RSS or below 32GiB available host memory.
Unavailable or malformed measurements fail closed. These sampled limits are not
a hard allocation cap or a cgroup-aware memory guarantee. It observes `STOP`
beside the fresh result file. No corpus files are written, and a matching result
does not adopt the format in a live job.

## Offline training CLI

`lc0_control_train.py --sampling-mode game_epoch --allow-packed-zarr` admits
ordinary directory and `.zarr.zip` shards. Staging preserves each resolved source
parent and the archive suffix; source-local game namespaces stay separate.
Archives in a CLI input require this explicit flag, including mixed directories,
so omitting it cannot silently omit archived rows. Replacement sampling and
qualified target overlays reject the option.

All three corpus identity readers and both value-label coverage scans include
archives. They close archive stores after reading attrs or narrow label flags;
wide inputs/policies remain lazy until the sampler admits the working set.
The same flag reaches every later exact epoch, and
`realized_replay_after_guard.applied.allow_packed_zarr` records the realized mode.
The loss, optimizer, training tensor path and directory-only defaults are unchanged.
The launcher's rolling recovery checkpoints retain their existing behavior.
They save trainer state but do not persist the exact sampler cursor or prefetch
state. Packed admission does not qualify exact interrupted-epoch resume; a fresh
sampling pass still needs explicit corpus, storage and seed admission.
No existing frozen run is converted by this option.

Root-level `row_provenance.npz` from derivation may be retained as opaque provenance.
It is covered by the archive content hash and never interpreted as a training
array. Root-level `directory_producer_seal.json` is the same kind of opaque
member: packed admission does not interpret it and does not apply a dense
chunk-grid rule. The directory reader enforces that seal before fill-decoding
when the member or its per-array attribute is present. The seal binds each
array's raw `.zarray` bytes and the stored chunk-key inventory, and installs
those bytes onto the opened arrays before fill decoding. It does not
hash chunk payloads. A shard with neither keeps the legacy directory read,
including unsealed fill-elision, and is not newly validated. Other auxiliary
filenames and nested provenance or seal files remain rejected, as do overlays,
duplicate names, paths escaping the root and nonregular members.
