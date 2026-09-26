# BT4 stepper capacity and loader THP audit — September 23, 2026

This is a read-only source/capacity audit. No game, model inference, loader A/B,
training change, GPU run, or system-setting change was made.

## Stepper retention

At the current `v2_threats` 175-plane setting, each buffered ply in
`scripts/bt4_root_policy_stepper.py` owns an exact float32 input of
`175 × 8 × 8 × 4 = 44,800` bytes and a compact float32 teacher policy of
`1,858 × 4 = 7,432` bytes. Native WDL adds 6, 12, or 24 bytes for
float16/32/64. Thus 52,238–52,256 bytes per buffered ply is the array-data
floor, excluding ndarray/Python object headers, keys, metadata, active board
history, allocator fragmentation, and external evaluator buffers. Lazy search
logits do not add retained arrays until requested.

Using the 52,256-byte float64 case for planning:

| Concurrent games | Buffered plies per game | Raw retained arrays |
| ---: | ---: | ---: |
| 1 | 450 | 22.43 MiB |
| 128 | 100 | 0.623 GiB |
| 512 | 225 | 5.606 GiB |
| 1,024 | 450 | 22.426 GiB |

These are illustrative capacity scenarios, not measured game lengths or an
admitted batch size. The stepper has no live-slot or byte budget. A pending
root temporarily also holds a prepared input and a private expected-input
byte copy (89,600 bytes/slot), while `inference_inputs()` stacks another
44,800 bytes/slot and briefly owns per-root copies. Evaluation and all-batch
validation add native outputs and immutable record copies before old objects
can be released. A future scheduler needs an explicit live-slot and memory
budget that covers this peak plus model/ONNX buffers; the raw-array numbers
above are not an RSS limit. The existing generator's 450-ply default is a
source setting, not a proposed stepper batch size.

Source anchors: stepper `BT4PlayedPly`, `_immutable_array`, `prepare_roots`,
`PreparedBatch.inference_inputs`, and `apply_root_outputs`; plane count in
`chess_anti_engine/encoding/encode.py`; compact width in
`chess_anti_engine/moves/encode.py`; `DEFAULT_MAX_PLIES = 450` in
`scripts/gen_random_selfplay_shards.py`.

## Loader and transparent huge pages

The exact-epoch loader eagerly decodes shards via `load_shard_arrays(...,
lazy=False)`, which materializes Zarr proxies with `np.asarray`. It validates
arrays, retains decoded chunks, and uses `np.take` for owned compaction/gather
copies. The shuffle replay buffer also allocates `np.zeros` output arrays.
There is no explicit `madvise`/`MADV_HUGEPAGE` call in the searched repository
package/scripts paths.

Observed locally, without touching the trainer: NumPy 1.26.2's installed
`__init__.py` (lines 407–435) enables its allocator's huge-page advice by
default on Linux kernel >=4.6 when `NUMPY_MADVISE_HUGEPAGE` is unset. A fresh
system-Python process on kernel 6.18.33.2 reported
`np.core.multiarray._get_madvise_hugepage() == True`. The system's THP
`enabled` and `defrag` files both selected `[madvise]`. The parent separately
reported `NUMPY_MADVISE_HUGEPAGE` and `MALLOC_ARENA_MAX` unset for the host
trainer and an E window with train 30.470 s, wait 15.952 s, memory PSI avg60
about 9.69%, and a 10-second delta of compact_stall 2,542,
compact_fail 2,541, compact_success 1, kswapd scan 104,055, with zero direct
scan/allocstall/swap-in/out. Those host measurements were not repeated here.

Inference: eligible NumPy-owned large arrays may request THP under that
default; the code path and fresh-process setting do **not** establish that
the trainer's individual loader buffers received huge pages, that compaction
was caused by them, or that toggling advice would improve throughput. A
sandbox `/proc` lookup cannot determine whether a host PID exists or which
NumPy module that host process loaded. The parent verified the host E trainer
PID 2156939 remains alive (window 1105/1290, 97,240 updates); my earlier
"exited" inference from this sandbox's invisible `/proc` PID was incorrect.

For a future quiet slot, the smallest diagnostic is the same frozen eight-shard
exact-loader fixture in separate child processes with only
`NUMPY_MADVISE_HUGEPAGE=0` versus the default changed, set before NumPy
import. Use fixed source hashes/order/workers and an ABBA order with the same
cache-control policy; cap at 12 minutes on two low-priority CPU cores. Record
decoded-output hashes, wall/user/system time, peak RSS, memory PSI, and
before/after THP/compaction vmstat counters. Do not change sysctl, active
training, source shards, or model settings. This is a proposed test, not a
result or performance claim.
