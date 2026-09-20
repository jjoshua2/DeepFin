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
roster and source partitions. The bounded CPU command compares every ordered batch:

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
uses two allowed CPU cores, nice19, no GPU, a 16GiB RSS cap and 32GiB host-memory
reserve. It observes `STOP` beside the fresh result file. No corpus files are
written, and a matching result does not adopt the format in a live job.

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
This main-based launcher does not yet expose the frozen runtime's recovery CLI;
future recovery integration must carry this option into its sampler reconstruction.
No existing frozen run is converted by this option.
