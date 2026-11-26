# Search and evaluation paths: inventory and consolidation plan

Status: design note, 2026-11-21. No code changes. Line numbers are origin/main at
`15d515cce` (after #1440); rows and lines added in the review revision are at
`cce9b9412` and marked so. Where a claim was not checked in code it is marked
*unverified*.

The question this answers: speed work keeps landing in one search path at a time
(#1440's row compaction and duplicate fan-out reached only the multi-GPU PUCT pool).
Which paths exist, what they share, what can be one layer, and what is not worth
merging.

## 1. Inventory

### 1.1 Paths

| # | Path | Entry point | Callers | Tree / batching model | Tree persistence |
|---|---|---|---|---|---|
| A | **UCI classic Gumbel** (1-GPU default, `SearchWalkers=1`) | `SearchWorker._run_gumbel_chunk` → `gumbel_c.run_gumbel_root_many_c` (`uci/search.py:4661`) | UCI engine, `uci/selfcheck.py:623` | one tree, sequential halving; leaves of one chunk batched inside C (`_mcts_tree.c` Gumbel sim state) | carried across moves (reply book / tree reuse) |
| B | **UCI multi-GPU PUCT pool** (tournament path; `--devices`) | `MultiGpuPucvPool` (`uci/multi_gpu_pucv_pool.py:393`) via `_run_pucv_pool_chunk` (`uci/search.py:4718`) | UCI engine | one shared tree, one worker thread per device, `batch_descend_puct` up to `VLGather` leaves with virtual loss, 2-slot async pipeline per worker | carried |
| C | **UCI single-device PUCT** (`PucvChunker`) | `mcts/puct_vl.py:74`, built by `SearchWorker._build_pucv` (`uci/search.py:1463`) | UCI single-PUCV mode; root-parallel Gumbel (`uci/root_parallel_gumbel.py:1494`) | one tree, gather + 2-slot pipeline (same shape as B, one device) | carried |
| D | **Root-parallel Gumbel** (RPG) | `RootParallelGumbelPool` (`uci/root_parallel_gumbel.py:406`), `_run_rpg_chunk` (`uci/search.py:4744`) | UCI engine (multi-device) | Gumbel halving at the root, one `PucvChunker` per candidate group | carried |
| E | **Walker pool** (`SearchWalkers>1`) | `WalkerPool` (`uci/walker_pool.py:99`), `_run_walker_chunk` (`uci/search.py:4608`) | UCI engine | N walker threads, each descends single leaves (`walker_descend_puct`) and batches its own gather | carried |
| F | **UCI evaluator stack (routing chain)** | `ThreadSafeGPUDispatcher` / `CUDAOwnerDispatcher` / `BatchCoalescingDispatcher` / `MultiGPUDispatcher` (`inference_dispatcher.py:82,208,363,633`), `DegradingMultiGPUDispatcher` (`uci/device_guard.py:215`), `EncodedEvalCache`; wired in `uci/__main__.py:890-930` (@cce9b9412) | **every UCI path, including the single-GPU classic default (A).** The evaluator is always wrapped in `ThreadSafeGPUDispatcher` (`:904`); with compile on (the default) and one walker also in `CUDAOwnerDispatcher` (`:913`); `BatchCoalescingDispatcher` instead for walkers > 1 (path E) or eager with coalescing (`:915`); then `EncodedEvalCache` when `EvalCacheEntries > 0` (`:924`). With several devices and `--no-multi-gpu-pucv`, `DegradingMultiGPUDispatcher` (`:890`) routes calls across devices. | not a search; the wrapper layers every A/E evaluator call goes through | n/a |
| G | **Self-play collect, serial** | `selfplay/network_turn.py` → `run_gumbel_root_many_c` with the game's persistent tree (`network_turn.py:1211-1215`) | `selfplay.manager.play_batch`; `simple_rl/collect.py` (RL proof collector, evaluator at `collect.py:414`) | many boards (games) per call, one leaf buffer; **a supplied tree disables the 2-group pipeline** (`mcts/gumbel_c.py:1168-1190`) | per game, carried (`gumbel_tree_reuse=True`, `selfplay/config.py:107`) |
| H | **Self-play / match, pipelined** | `run_gumbel_root_many_c` without a tree, ≥64 boards, async evaluator → 2-group overlap (`gumbel_c.py:1168`) | `selfplay/match.py:184` (arena / match play), benches | many boards, two leaf groups alternating with the device | cold roots every ply |
| I | **Arena** | `scripts/arena_standard.py` (evaluator at `:1649`), plays through `selfplay/match.py` | ladders, `simple_rl/evaluate.py` | as H (cold tree by default; `tree_reuse` field `arena_standard.py:358`) | `cold` unless set |
| J | **Distributed worker, local evaluator** | `worker.py:1445-1463` (`ThreadedDispatcher` / `AOTEvaluator` / `DirectGPUEvaluator`) | distributed self-play | as G/H inside `play_batch` | as G |
| K | **Distributed worker → slot broker** | `SlotInferenceClient` / `MultiSlotInferenceClient` (`inference.py:3626,3970`; built `worker.py:1654-1662`) → `SlotBroker` / `SharedSlotBroker` (`inference.py:2509,4339`) | distributed self-play with a shared broker process | **across processes**: many workers' leaves into one broker batch | as G |
| L | **AOT broker / AOTEvaluator** | `AOTEvaluator` (`inference.py:1870`): one `.pt2` package per bucket | worker `--aot-dir` | evaluator only | n/a |
| M | **Python Gumbel** (@cce9b9412) | `mcts/gumbel.py:1707 run_gumbel_root_many` | `selfplay/match.py:218` and `selfplay/network_turn.py:1270`, taken when volatility search is on (`network_turn.py:1198-1200`) or the C extension is missing | Python sequential halving over many boards; no persistent C tree, no per-child root stats | cold |
| N | **`searchmoves` beside a pool** (@cce9b9412) | `SearchWorker._run_restricted_chunk_beside_pool` (`uci/search.py:3308`) | UCI `go … searchmoves` while a multi-GPU pool is live | a classic Gumbel chunk (the restriction is honoured only by the C path) on the primary through `GuardedPrimaryEvaluator`; if the primary fails, rerun on the pool's surviving devices via `PoolRowsEvaluator` | the restricted chunk's own tree |
| O | **Device-guard evaluators** (@cce9b9412) | `uci/device_guard.py`: `DeviceCallRunner` (`:82`), `DegradingMultiGPUDispatcher` (`:215`), `GuardedPrimaryEvaluator` (`:482`), `PoolRowsEvaluator` (`:524`) | F's multi-device chain; N | evaluator wrappers: stall/raise detection, device drop, rows served by a pool | n/a |

### 1.2 Evaluator classes and speed features

| Feature | Where it lives | A classic | B pool | C PucvChunker | E walkers | G/H self-play | I arena | K broker |
|---|---|---|---|---|---|---|---|---|
| Pinned host buffers, in-place submit (`get_input_buffer` + `evaluate_inplace_async`) | `DirectGPUEvaluator` (`inference.py:802`, buffers `:860-888`, `:987`, `:1182`) | yes | yes | yes | no (a plain numpy buffer, `walker_pool.py:267`, passed to `evaluate_encoded`) | yes when the evaluator is `DirectGPUEvaluator` | yes | slot shared memory (`SlotBroker`) |
| CUDA graphs: torch.compile cudagraph trees on padded buckets | `pad_to_compiled_buckets=True` + compile (`inference.py:825,1381`) | UCI compile (`uci/__main__.py:880-915`) | yes, per-worker capture | via UCI evaluator | via UCI evaluator | no | no | broker's own compile (*unverified*) |
| CUDA graphs: replay of the eager forward per exact batch size | `_EagerForwardGraphs` (`inference.py:612`), `cuda_graphs=True` | no | no | no | no | RL collector yes (`simple_rl/collect.py:414`, `pad_to_compiled_buckets=False`) | yes (`arena_standard.py:1649`) | no |
| Bucket padding | `_COMPILED_BATCH_BUCKETS` (`inference.py:1381`); `ThreadedDispatcher._next_bucket` (`inference_threaded.py:65`) | yes | yes | yes | yes | no (exact sizes) | no | own buckets |
| Terminal rows kept out of the forward | — | **yes**: C backprops terminals inline, never buffered (`_mcts_tree.c:2070`) | **yes since #1440** (`multi_gpu_pucv_pool.py:220,1495`) | **no**: zeroed terminal rows forwarded unless the eval cache is on (`puct_vl.py:266` `gpu_full`; cache path skips them `:340`) | yes: terminals backprop inline and are not queued (`walker_pool.py:356`) | yes (same C as A) | yes | depends on the search |
| Duplicate leaves in one batch | — | **guard**: C17 flush-and-resume (`_mcts_tree.c:2051`) | **fan-out since #1440** (`_scatter_outputs`, `multi_gpu_pucv_pool.py:268`) | **no**: each duplicate forwarded | no handling (per-walker vloss only) | guard (as A) | guard | depends |
| Eval cache | `PucvEvalCache` (`inference_cache.py:598`); `EncodedEvalCache` (`:144`) | `EncodedEvalCache` wrapper when `EvalCacheEntries>0` (`uci/__main__.py:924`) | `PucvEvalCache` (`multi_gpu_pucv_pool.py:514`) | `PucvEvalCache` (`puct_vl.py:121`) | via wrapper | no | no | no |
| Pipelined slots (overlap search with the device) | 2-slot handles | within C only | 2 slots per worker | 2 slots | no (synchronous per walker; concurrency from N walkers) | **H only**; G is serial | H | broker overlaps across clients |
| Failure / drain / quarantine | `async_safety.py`, pool `_drain_failed` | process-level | yes (device drop, degrade) | partial (`_release_outstanding`) | vloss release (`walker_pool.py` B7 guard) | process-level | process-level | `broker_hang.py` |

**Forward chunking (PR #1511).** With graphs on, `DirectGPUEvaluator` forwards a call larger than
`cuda_graph_max_rows` (512 by default) as consecutive graph-replayed chunks of at most that many rows. Before,
such a call ran as one eager forward. This changes forward shapes and forward counters for the consumers that run
with graphs on:
- the arena (`scripts/arena_standard.py`);
- the RL collector, at `--eval-chunk-rows`, default 512;
- `scripts/core_finite_collect.py` (protocol `serving` with `cuda_graphs`): `serving['forwards']` no longer counts
  above-cap calls, whose chunks are graph replays (`cuda_graph_stats.forward_chunks`);
- `scripts/bench_arena_identity.py`: its `forwards` counts evaluator calls, which are no longer network forwards
  above the cap.

### 1.3 Duplicated logic (diverged or able to)

- **The submit/finish pipeline.** `PucvChunker._prepare_submit` / finish (`puct_vl.py:~325-400`) and `MultiGpuPucvPool._prepare_submit` / `_finish_pending` (`multi_gpu_pucv_pool.py:1495-`) are the same design, copied. They have already diverged: #1440's compaction, fan-out and vectorised scatter exist only in the pool, and the pool's per-buffer output arrays and counters are not in `PucvChunker`.
- **Terminal handling: four implementations.**
  - the Gumbel C sim state (`_mcts_tree.c:2070`);
  - `batch_descend_puct` (inline backprop plus `is_term` flags, `_mcts_tree.c:~3133`);
  - pool compaction;
  - the walker inline path.

  All four must agree on what "terminal" means (twofold repetition, 50-move, TB-proven). Today they share `cboard_search_terminal` in C. The *consumers* differ.
- **Duplicate handling: three behaviours.** The Gumbel guard (changes the search: smaller batches), pool fan-out (does not), and none (PucvChunker, walkers).
- **Virtual loss: three implementations.** Gumbel `vloss_w` in C; PUCT `vloss_mode` (legacy / virtual-mean) for B and C; walker per-leaf apply/remove in Python plus C. A release bug was fixed separately in the walker (B7) and in the pool (`_release_outstanding`).
- **Tablebase probing: five places.**
  - UCI root (`uci/tb_root.py`);
  - UCI leaf (`uci/tb_leaf.py`, applied in PUCT through `mcts/puct_tb.py:68 override_leaf_values` and `:97 mark_batch_solved`);
  - Gumbel C path (`tb_probe` argument, `gumbel_c.py:908`);
  - self-play witness (`selfplay/strict_tb.py`);
  - `chess_anti_engine/tablebase.py`.

  Only the UCI paths are covered by the production-Syzygy rules in `CLAUDE.md`.
- **Stop / time checks: four clocks.**
  - `SearchWorker._should_stop_search` plus `_time_capped_chunk` (`uci/search.py:3729,3826`);
  - the walker reserve (`:4565`; its decay race is fixed in #1449);
  - the pool's own stall threshold;
  - Gumbel's per-chunk budget.

  The walker race shows how two clocks drift apart.
- **Leaf encoding.** It is done in C during descent for every path, through the shared C encoder. This is already one layer, and is not a divergence risk.

## 2. The structural difference

| | Match (UCI) | Self-play / arena |
|---|---|---|
| Trees | one big tree | many small trees (one per game) |
| Batch source | many leaves of **one** tree (virtual loss, gather) | a few leaves from **each** of many trees |
| Bound by | latency and the clock (each `go` must finish on time; first-move behaviour matters) | throughput (positions per GPU-hour) |
| Batch size | up to `VLGather` (512); compacted batches vary | concurrency × topk; fixed shape per ply |
| Duplicates | frequent (15.4% of pool rows; 88% on a fresh root) | rare per tree; the C17 guard handles them |
| Terminals | frequent in endgames (16.8% of pool rows overall) | handled in C |

The **schedulers** differ for good reasons and should stay separate: the Gumbel halving sim state, PUCT gather with virtual loss, and walkers. What does not depend on the scheduler is everything after "here are N leaf rows". That part can be **one batch-assembly / evaluation layer**:

1. Take rows plus their leaf ids and `is_term` flags.
2. Drop terminals and fold duplicates (`_distinct_leaf_rows`).
3. Consult the eval cache.
4. Pick the bucket (compiled buckets or exact-size eager graphs, chosen by the evaluator, not the caller).
5. Fill pinned buffers and submit into one of 2+ slots.
6. Await, scatter back into reusable per-slot outputs (`_scatter_outputs`), and fan out duplicates.
7. On failure, drain or quarantine with a single receipt protocol.

The pool already implements steps 1–7. `PucvChunker` implements 3, 5, 6 (without fan-out) and 7 (partly). Gumbel C does 1–2 inside C and hands Python an already compact buffer.

## 3. Staged consolidation plan

Numbers come from banked measurements:
- **pool profile** (`~/chess-artifacts/operations/rl-claude-20261119/engine-speed-profile/`): 16.8% terminal, 15.4% duplicate, 2.55% TB rows; 67.8% of forwarded rows were unique evaluations;
- **#1440 idle timing** (`idle/pvp_*`): geomean ×1.13 at movetime 5000, ×1.16 at 100k nodes, ×1.30 at 20k nodes; endgames ×1.7–1.9;
- **collector profiles** (`collect-throughput/logs/prof_*.txt`, 14 py-spy profiles over sims 16–64 per move, K = 1–8 collector processes, concurrency 64–256): the GPU-wait share depends heavily on the configuration and on how it is counted. Samples in a CUDA sync frame (`torch/cuda/streams.py`, `synchronize`, `.item`) over all samples give 9–51% (lowest `c256_s16`, highest `mp3prof_c128`); the engine reviewer's count gives 18–79%. The 51–56% quoted in the first version of this note fits one configuration, not the collector in general.

Order (revised after review): measure first, then build one batch layer, not two.

| Step | What | Expected gain | Risk | Verification |
|---|---|---|---|---|
| **0** | **Cheap measurement that gates step 4**: GPU utilisation and collector rows/s at the production K on an idle GPU, with the configuration recorded. | none; decides whether step 4 is worth anything | none | `nvidia-smi` utilisation trace plus collect-throughput rows/s, banked with the configuration |
| **1+3** | **One batch layer, done together.** Move `_distinct_leaf_rows` / `_scatter_outputs` / `PoolSearchCounts`-style counters and the submit/finish/drain pipeline (slots, cache, receipts) into one module, with the pool's implementation as the single one. `PucvChunker` becomes a thin user of it, or is retired in favour of a one-worker pool (the one-fast-path rule). C and RPG then get compaction and fan-out because they run the same code, not because it is copied in. | the pool's ×1.13–1.30 on 1-device PUCT and RPG (same waste mechanism; not measured on C); stops the next speed fix landing in one copy | medium (failure paths, device drop, receipts) | the #1440 tree-identity harness on the row-pure evaluator (identity + fuzz + mutants); full UCI/pool suites; receipt and startup-failure tests; mutants on drain; idle fixed-nodes timing parent vs PR |
| ~~2~~ | ~~Port compaction and fan-out into `PucvChunker`.~~ **Dropped**: it would create a second copy of the batch layer, the divergence this note is about. Superseded by 1+3. | | | |
| **4** | Only if step 0 shows idle GPU at production K. Self-play G: allow overlap with persistent trees. The 2-group pipeline builds ephemeral sub-trees (`gumbel_c.py:1168-1190`); instead, split the boards into two groups that each keep their own persistent trees, and alternate them against the device. | up to the idle share step 0 measures; ~0 if K processes already saturate the GPU | high: C sim-state changes | **target identity needs the row-pure evaluator**: real-net outputs differ with batch size, and the buffer-full flush makes Gumbel depend on batch composition, so "bitwise identical targets per game" with the real net is not a valid check. Use the row-pure evaluator for identity, and real-net throughput (rows/s at production K) separately |
| **5** | Make the bucket choice an evaluator property (compiled buckets vs exact-size eager graphs) rather than a caller flag. | small; fewer misconfigurations (e.g. `pad_to_compiled_buckets` vs `cuda_graphs` mutual exclusion, `inference.py:579`) | low | evaluator tests (`tests/test_eager_cuda_graphs.py`) |

**Not worth doing:**

- **Porting compaction or fan-out to classic Gumbel (A/G/H/I).** That path already never forwards terminals, and folds duplicates by the C17 guard. There is nothing to compact.
- **A duplicate *guard* on the pool instead of fan-out.** It changes the search, so it must earn its place by a fixed-time strength match. It stays a candidate behind a match, not a refactor.
- **Merging the schedulers** (one "search engine" for match and self-play). Halving vs PUCT vs walkers serve different budgets; a merge would make every change a cross-path risk for no speed gain.
- **Walkers (E).** A non-default path that already keeps terminals out. (The evaluator stack F is *not* in this list: it sits on the single-GPU default, so its per-call cost — lock, owner-thread hop, cache lookup — is on the default path and belongs in any profile of A.)
- **Unifying the TB code paths** is worth an audit, but not a merge: root, leaf and witness semantics differ (DTZ at the root, WDL at leaves, witness for self-play strictness). The risk is in rule-50 / history handling, which is better pinned by tests than by sharing code.

## 4. The single-GPU default for matches

The tournament path is the pool (B); the single-GPU default is classic Gumbel (A). The evidence today:
- The pool on one GPU (`--devices cuda:0,cuda:0`, i.e. two workers, one with CUDA graphs) runs about 17.5k visits/s in middlegames on an idle GPU (#1440 parent), and up to about 31–35k in TB-adjacent endgames with #1440.
- Classic Gumbel on one GPU has **no idle timing**: `engine-speed-profile/scripts/run_classic.sh` is staged, not run.
- Nps does not compare across the two searches anyway: a Gumbel visit and a PUCT visit buy different things.

So the default should **not** be switched on speed. A switch needs:
1. idle-GPU timing of A (`run_classic.sh`), for reference only;
2. a fixed-time strength match, A vs B-on-one-GPU, at a realistic time control, with a pre-registered threshold. This is the deciding yardstick, per `docs/eval_protocol.md`.

Until then, classic Gumbel stays the validated default. Step 1+3 above makes B and C cheaper either way.

### 4.1 Measured on main (75d063c97, idle GPU, 2026-11-21)

**First search of a process: a median 60–130 ms (spread 20–300 ms) on the first one or two searches, not CUDA-graph capture.**
- Design: 5 fresh processes per arm, two passes of `go nodes 20000` over four positions, `ucinewgame` between gos. Banked under `engine-speed-profile/idle/fs_*`; the script is `scripts/run_firstsearch.sh`.
- Pass-1 extra over pass 2 (median):
  - pool with compile: +62 ms on the first search, +52 ms on the second;
  - pool without compile: +131 ms;
  - classic with compile: +98 ms, first search only;
  - classic eager: not measurable in its noise.
- It is present without compile, so it is not graph capture. It is not front-loaded at `go` either: in classic, the first info line at about 12k nodes is on time and the extra comes later in the search. That fits growth costs (tree arena, buffers, cache memory first reaching a size) rather than a one-off init. The cause is not isolated further.
- **The earlier −20.8% (#1440, r1, mt5000) cell is withdrawn: not reproduced under the first-search design (`go nodes 20000`, `ucinewgame` between gos — not a rerun of the original 5 s clocked conditions); most likely noise.** A cost of this size would be about 1–6% of a 5 s search, not 20%.
- An `isready` warm-up (one throwaway search of about one real move's size) is **parked**.
  - Launch condition: a time control where move 1 is ≤ 2 s, or evidence of a first-move loss.
  - Acceptance: `scripts/run_firstsearch.sh` shows the pass-1 deficit gone.

**TB-adjacent / 7-piece endgames forward about 35% fewer rows/s than middlegames (`end_rook` is middlegame-like).**
- Per-search counters (#1453), pool `--devices cuda:0,cuda:0`, compile on, all standard positions at mt5000 and n100k. Banked: `engine-speed-profile/idle/evals_pool_main.{jsonl,summary.txt}`.
- Middlegames and openings: about 17.3k evals/s (16.4–17.7k) at 87–98% unique leaves. That is the forward-bound rate of two workers on one GPU.
- TB-adjacent / 7-piece endgames (`end_minor7`, `tb_adjacent7`): 32–35k leaves/s but only 9.8k–12.1k evals/s, because only 28–38% of leaves are unique (terminal 45–65%, duplicate up to 27%). `end_rook` runs 17.4–17.5k evals/s at 98% unique, like a middlegame.
- **So the pool's next speed-up is per-leaf CPU/tree work in TB-adjacent / 7-piece endgames, not the device.**
