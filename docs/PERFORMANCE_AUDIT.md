# Performance Audit Status

Last updated: 2026-07-15

This is the coverage index for the performance pass. Detailed hypotheses,
commands, thresholds, measurements, and rejected experiments remain in
`docs/experiment_ledger.md`; this file answers the narrower question: which
speed-critical runtime surfaces have been examined, and what is still waiting
on operational evidence?

## Coverage

| Runtime surface | Status | Main conclusions |
|---|---|---|
| Encoding, move generation, and native boards | Complete | Production uses the native 175-plane/BF16 path and caller-owned CBoard/MCTSTree APIs. Host-native plus LTO qualified; native-extension PGO regressed and was rejected. C/Python parity gates cover legal moves, history, mirroring, terminal roots, and board mutation. |
| Model inference and GPU broker | Complete, GPU re-profile queued | Shared pinned slots, compact legal-policy transport, BF16 input, device-cached policy maps, AOT/eager routing, model hot-swap, and broker metadata copies were reviewed and benchmarked. `inference_mode` missed its gate (1.7%) and was rejected. A fresh GPU trace is queued after WSL `/dev/dxg` is reset; this is validation, not an unreviewed code area. |
| Gumbel/PUCT search | Complete | Native tree selection/backprop, virtual loss, root setup, policy extraction, C/Python boundaries, and match/selfplay call shapes were profiled. Compact Gumbel candidate state removed 99.66% of root-state bytes and made the zero-budget initialization mechanism 186x faster. Remaining Python glue was only 2.7% in the production-shaped native-candidate profile, below the conversion gate. |
| Stockfish labels and opponent play | Complete, one quiet-host replication queued | Host-native Stockfish PGO qualified, request execution now uses engine-owning work stealing, and background starvation polling was reduced 77.5% without material completion latency. A 32 MiB hash measured 2.34% faster under a saturated host but missed the 3% gate; repeat only when the host is quiet. |
| Selfplay finalization and worker completion | Complete | Native SF finalization removed the dominant Python/GIL block (at least 1.69x); detached shard ownership removed materialization from completion locks; upload and Stockfish waits overlap independent work. Continuous scheduling and oversubscription have explicit telemetry/kill gates. |
| Replay storage, sampling, augmentation, and H2D | Complete | Zstd2 plus bit-shuffle, background shard ingest, disk-buffer refresh, sparse labels, sampling, target rebuild, and augmentation were profiled. Policy mirroring is 69.4% faster and input mirroring deletes a redundant copy. Compact H2D preparation now retains stored float16/uint8/int8/int16 sources while preserving exact consumer dtypes, cutting the major source arrays to 0.50x/0.25x bytes. |
| Training loop, losses, and optimizers | Complete, restart readout queued | Compile modes, Aurora update graphs, optimizer scopes, loss construction, scalar synchronizations, batch preparation, and retry lifecycle were examined. Loss metrics now use one device-to-host materialization; cross-step replay prefetch measured 20.9% faster in the registered overlap mechanism. GPU kernel/step attribution should be refreshed after `/dev/dxg` recovery. |
| UCI and match play | Complete | Time management, abort checks, root reuse, walker pools, evaluator routing, and persistent-board options were reviewed. Root-snapshot reuse (0.98%) and persistent dual Python/C boards (24% glue reduction but materially more state machinery) missed their gates and were rejected. Shared native Gumbel and inference improvements benefit match play directly. |
| Distributed server, uploads, and assets | Complete | Manifest publication, model/asset download, shard validation/compaction, leases, backpressure, pending recovery, and upload concurrency were reviewed. Expensive materialization/network work no longer serializes completed-game callbacks. |
| Compiler, Python runtime, and Python-to-C candidates | Complete | GCC 15, native flags, LTO, extension PGO, Stockfish PGO, and remaining Python-owned hot blocks were measured. The broad Python-to-native screen found no remaining qualifying search/replay conversion; the qualifying finalization block was converted. CPython 3.14 free-threaded is not currently viable because Ray/project extensions/Numcodecs do not form a compatible production toolchain. |

## Operational Readout Queue

These items do not justify more speculative code changes. They require the
next natural training restart or a quieter machine state:

1. Read whole-iteration games/hour and phase telemetry for the merged
   Stockfish work-stealing and background-polling changes.
2. Read `train_time_s`, steps/s, and peak GPU memory for cross-step prefetch,
   replay mirroring, scalar-sync coalescing, compact Gumbel roots, and compact
   replay H2D sources as one restart bundle; mechanism benchmarks must not be
   reported as additive end-to-end speedups.
3. After WSL `/dev/dxg` is reset, capture a fresh GPU training and inference
   trace before changing kernels, streams, CUDA graphs, or copy scheduling.
4. Repeat the 16 MiB versus 32 MiB Stockfish hash yardstick only on a quiet
   host. Keep 16 MiB unless the pre-registered 3% gate clears.

## Exit Rule

The static and CPU-profiled pass is complete when every row above is complete
or has a named external readout. Reopen a row only when new production
telemetry identifies a material phase, a dependency upgrade changes the
toolchain, or a new algorithm/configuration changes the hot-path shape. Every
new optimization still needs a ledger entry and a precommitted deciding
yardstick before implementation.
