# BT4 run11 variable-batch scheduling opportunity

**Source-only readout, 2026-09-27.** The completed BT4 full-prefix pilot emitted 453,108 positions from 4,096 complete games in 15,387 physical model calls. Its effective batch sizes ranged from 1 to 64. This is call-count evidence, not a measured GPU wall improvement or admitted 500M throughput; source, target, corpus and owner credit remain zero.

## Why calls were small

The worker starts 64 consecutive games and finishes the entire wave before admitting the next 64. Each call evaluates one root from every unfinished game. Game lengths vary from 7 to 342 emitted rows, so late calls in each wave become small. The saved per-game lengths reproduce all 64 bins of the physical-call histogram: 5,329 calls had at most eight rows, carrying only 15,672 positions (3.46% of the total). The previous fixed-32 utilization estimate was invalid; 6,933 observed calls already exceeded 32 rows.

With a 64-root limit, 453,108 fixed evaluations require at least **7,080 calls** by capacity alone. An immediate-replacement simulation over the unchanged per-game lengths uses **7,193 calls**, versus 15,387 observed. This is conditional arithmetic: changing the schedule can change model batch shape, floating output, sampled moves, game length, Syzygy timing and file bytes. It cannot be counted as an 8,194-call saving or a wall-speedup until complete matched outputs pass.

## Next bounded comparison

The first two fixed waves, game IDs 0–127, contain 14,253 rows and used 396 calls. A virtual immediate-replacement schedule predicts 320 calls if their individual game traces remain unchanged. The existing full-prefix worker and strict replay verifier admit 4,096-game primary or 1,024-game top-up banks; they **cannot execute or verify this 128-game A/B as written**. A separately reviewed 128-ID runner and strict subset verifier are required first.

If that gate passes, run A then B sequentially as sole-GPU tasks on the same 128 IDs, model, seed, full histories and 64-root cap. Require each per-game NPZ SHA-256 to match the original sealed run11 subset, plus keyed feed, raw legal-logit and WDL bytes, complete all-attempt strict replay, each arm's own full physical-call ledger, and rule50-aware six-man Syzygy WDL/DTZ parity. The ledger stream hash will differ by design as call rosters change. Only after parity may wall, inference, writer and tablebase timings be compared. Any changed game or target makes this a different scientific comparison.

The [compact arithmetic](evidence/2026-09-27-bt4-batching-opportunity.json) has SHA-256 `18553a99f9b73dd37965713e2739f3ac4729f197beb1520124fa46e1624ddb8b` and comes from the run11 summary SHA-256 `34464fae162d391945dc09b8027dc9864236de59da7cb69a05cb9c3acbc7470f`, the source-only packet SHA-256 `fe31f153952b6271c2aa2852dc9d32f2cc8dd32e43bfc94ce7078cc14353eae3`, and independent arithmetic review SHA-256 `40d400417feab772f5592b95dd61f3fc4a09b605250c3ed81d306e770d418bfa`. The same pilot's source-plus-readback capacity appears in [the 500M capacity record](https://github.com/jjoshua2/DeepFin/pull/920). Bulk game files and receipts remain outside Git.
