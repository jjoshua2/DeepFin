# BT4 root worker CUDA qualification — September 23, 2026

Status: PREREGISTERED; no GPU job has run. This is a bounded qualification
of the opt-in [BT4 root-policy worker](2026-09-23-bt4-root-policy-worker.md),
not a training launch or a searched BT4 actor. Results will be added here
only after the active loader releases the GPU slot and the separate reviewer
accepts the provider change.

## Frozen inputs and semantics

The selected model is
`/home/josh/projects/chess/data/lc0/onnx/BT4-it332-vanilla-winner.onnx`,
SHA-256 `1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0`
(read from the file on September 23; the prior G10 raw sidecar records the
same SHA). ONNX graph inspection found `/input/planes` float32
`[batch,112,8,8]`, `/output/policy` float32 `[batch,1858]`, and
`/output/wdl` float32 `[batch,3]`. The WDL output's producing operator is
Softmax. The banked BT4 sidecar names `/output/wdl` probabilities in
win/draw/loss order from side to move. The pilot verifies the exact ONNX SHA,
which pins that inspected Softmax graph, then checks the realized session's
input/head names, types and widths. Runtime WDL probability validation checks
range and unit mass; it does not independently inspect the producing operator.
An unexpected hash, schema or probability tensor fails before game
publication. The legal policy mapping is the reviewed board-aware
Leela-to-compact conversion, retained as T1 probabilities
separately from the native WDL and from the game-result target. No search
policy or tablebase-informed move choice is claimed.

The canonical shared GPU lock is
`/home/josh/projects/chess/scratchpad/gpu0_experiment.lock` on device 0,
with `CUDA_VISIBLE_DEVICES=0`, a 2 GiB ORT arena cap, and two CPU threads.
The worker refuses absent CUDA, provider fallback, changed provider options,
and a first-root profile with no CUDA Conv/MatMul/Gemm-family model work. It
records both CUDA and CPU node counts; this does not establish that every
operator ran on CUDA. Profiling ends after the first inference, before the
remaining calls. The child takes the shared lock nonblockingly and holds its
descriptor through process exit. A busy lock fails before session creation.
The outer supervisor never takes a second lock: it owns the deadline and
process group, starts its timer before launching the child, and treats a busy
lock as failed admission rather than waiting on the GPU slot.

The strict six-man pair is
`/home/josh/projects/chess/data/syzygy_3-4-5:/home/josh/projects/chess/data/syzygy_6`.
The earlier real-file smoke opened 1,000 WDL and 1,000 DTZ material entries;
the pilot must recheck capacity and file-stat inventory itself. The fixed
seven-piece FEN is `4Q3/8/8/8/1BR4r/8/2p2k1K/8 w - - 0 1`.
Independent python-chess enumeration gives its sole legal move `c4h4`.
That capture resets the clock and reaches
`4Q3/8/8/8/1B5R/8/2p2k1K/8 b - - 0 1`, where the real local WDL/DTZ
helper returns `1-0`. This is a fixture qualification, not a general proof
of six-man coverage.

## Two finite pilots, after GPU admission

1. **Correctness:** two games, seed 121, temperature 0, `max_plies=1`,
   `parallel_games=2`, fixed FEN above, expected model SHA and named heads
   above. Require two completed games, two accepted root rows, zero discards,
   `c4h4` in both, six-man `1-0` before a second inference, and white-root
   `wdl_target=0`. Independently read both NPZ files, verify summary SHA
   receipts, replay FEN/move, recompute exact float32 history input and
   source/input keys, and confirm native float32 WDL and compact T1 policy
   are distinct from outcome targets. Require the CUDA profile proof and
   unchanged table inventory. Any discrepancy is a failed pilot.
2. **Accepted-row and writer-cost pilot:** only after correctness passes,
   32 games, seed 122, temperature 0, `max_plies=1`, `parallel_games=4`,
   same model/FEN/table contract. Require 32 complete games, 32 accepted
   rows, zero discards, 32 verified game-file SHA receipts, and no partial
   files. Record total process wall time from the outer supervisor, worker
   run wall, first-root qualification time, post-qualification wall,
   `writer_wall_seconds`, compressed output bytes, accepted rows per total
   process second, and peak GPU/disk use. The first profiled batch contributes
   accepted rows, so post-qualification time is reported only for timing
   decomposition and is never the denominator for all accepted rows.
   This deliberately measures short-game per-file writer cost; it is not a
   representative full-game throughput estimate or a training admission.

Both pilots require distinct new output directories, at least 150 GiB free
before launch, and the same exact code/model/table identities. The outer
supervisor has a 300-second correctness deadline and a 900-second throughput
deadline, with a 30-second termination grace. It must own the complete
process group, signal/kill and reap the group on timeout or failure, and
verify no owned process remains before recording PASS. A timeout, missing
completion summary, stale provider proof, changed table inventory, or any
unreaped process leaves the pilot incomplete. The child must hold the global
GPU lock continuously from before ORT session creation through process exit;
the supervisor must not acquire it separately.

The decision is limited to whether this root-policy writer correctly produces
the fixed tiny bank with observed CUDA model compute and measured accepted
writer cost. C-search leaf wiring, representative generator throughput,
replay loading, target-recipe changes and training remain separate gates.
