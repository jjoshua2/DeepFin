# Transposition reuse and exploration accounting: direction and diagnostic plan

**2026-11-22 · Diagnostic implemented; first SL-base reading banked (see [First reading](#first-reading-on-the-sl-base)) · baseline `e182d821ca3575e2c27a224c9dc0560a3c08a0fa`**

## Decision and scope

Start by measuring the identities DeepFin actually encounters. Preserve the present search while determining whether the opportunity is repeated network evaluation, overly restrictive donor lookup, repeated exploration through equivalent states, or useful generalization between different positions. These require different interventions.

This change supplies a bounded, opt-in **post-search occupancy census**, a reproducible diagnostic driver, and this research direction. It does not adopt learned counts, change search allocation, enable caches, or claim a strength improvement. A later experiment needs its own realized configuration and result record. No live changes, collection, arena, GPU experiment, or training are authorized by this plan.

### Paper connection

[MCTS Is Counting the Wrong Thing: Learning Exploration Counts for LLM Reasoning](https://arxiv.org/abs/2611.22301) motivates learned pseudo-counts for exploration. Its count head is frozen at inference: it reflects training coverage, not a live counter of redundant simulations in the current search. Its reported LLM results do not establish a chess gain. Chess supplies exact structural and rule-state identities, while DeepFin additionally has history-sensitive model inputs. The transferable hypothesis is that exploration should respond to useful new evidence. Exact reuse and honest accounting are the first controls; learning a novelty measure is a later candidate when those controls leave a measurable problem.

## Four identities that must remain separate

| Identity | Includes / purpose | Sharing boundary |
|---|---|---|
| Structural board | Pieces, side to move, castling, pseudo-capturable en passant under the current native key | Legal moves and path-independent board computation; equality must be verified after hashing. |
| Complete rule state | Structural board plus halfmove clock and sufficient repetition history for draw semantics | Needed before treating continuations and draw outcomes as equivalent. A rare draw difference can determine the result. |
| Model input | Actual encoded history/features, dtype and representation; model/encoding namespace; any auxiliary inputs | Identical deterministic input can support raw evaluation reuse. Equal boards need not produce equal input. |
| Opponent-history belief | Information about an opponent's tendencies inferred from earlier choices | A separate modeling question. It does not alter legal moves; it can alter a history-conditioned prediction. |

The native structural key is conservative relative to legal-position equivalence: a pinned pawn can make an en-passant capture pseudo-legal but illegal, and that EP state remains distinct here. The current strict Gumbel context comparison is an implementation identity, including the ordered active history stack and encoder history. It may distinguish contexts beyond the minimum needed for rule equivalence. Conversely, matching reconstructed planes does not establish matching draw semantics. In ideal play, the complete rule state determines the game; using history to predict a particular opponent is an additional objective.

The census observes existing draw behavior; it does not certify that behavior as the rules of chess. The [twofold-as-proven-draw audit](2026-11-19-twofold-repetition-as-proven-draw.md) remains a separate search-policy concern. [#1421](https://github.com/jjoshua2/DeepFin/pull/1421) fixed stale-context root reuse; it did not remove the first-repeat terminal shortcut. A future graph experiment must separate that shortcut from legitimate repetition claims and must not share a path-dependent solved flag across incompatible histories.

## Existing work and what its evidence establishes

These are reported historical results, not reruns on the present checkpoint. Merged source is not evidence of a currently loaded binary or configuration.

| Work | Finding and practical consequence |
|---|---|
| [C17 / #278](https://github.com/jjoshua2/DeepFin/pull/278), July 27 | Virtual loss eliminated pending duplicates while retaining batching. On 2,000 frozen audit positions, target expected regret improved **54.2→46.8 cp**; a node-matched larger-budget control reached **49.6 cp**. The residual **2.8 cp** supports allocation effects beyond extra tree nodes. This was **target quality**, on history-free audit inputs, not Elo or a current-model result. |
| [C17 audit](https://github.com/jjoshua2/DeepFin/blob/e182d821ca3575e2c27a224c9dc0560a3c08a0fa/docs/rl_loop_audit.md#L474) | The proposed root-count explanation was refuted: root maximum/total visits stayed identical. Effects came through completed values and allocation. A per-prepare-call observer read **0%** while the accumulated evaluator batch contained **77.5%** duplicates. That is the 8-board run (128 leaves per board); the same audit measured **38.7%** at 256 boards (4 leaves per board), so the rate is a function of leaves per board, not a single figure ([audit table](https://github.com/jjoshua2/DeepFin/blob/e182d821ca3575e2c27a224c9dc0560a3c08a0fa/docs/rl_loop_audit.md#L755-L760)). Measure the actual flush boundary. |
| [#280](https://github.com/jjoshua2/DeepFin/pull/280), July 28 | Padding was substantial, but saved rows did not translate proportionally into forward time. Keep real requests, padding, and wall time separate. |
| [#470](https://github.com/jjoshua2/DeepFin/pull/470), [#472](https://github.com/jjoshua2/DeepFin/pull/472), [#473](https://github.com/jjoshua2/DeepFin/pull/473), August 26 | The native NNUE canonical graph and identical-qsearch control have real correctness/parity tests. The [current tactical-bank table](https://github.com/jjoshua2/DeepFin/blob/e182d821ca3575e2c27a224c9dc0560a3c08a0fa/docs/fastq_search.md) reports **431.84→342.67 evaluations/call**, but shared-machine median time **199.7→302.2 μs**. Reuse was not a demonstrated speedup. FastQ's much larger reduction also changes search policy. |
| [#474](https://github.com/jjoshua2/DeepFin/pull/474), [#478](https://github.com/jjoshua2/DeepFin/pull/478) | The [NNUE/Gumbel readout](https://github.com/jjoshua2/DeepFin/blob/e182d821ca3575e2c27a224c9dc0560a3c08a0fa/docs/nnue_gumbel_readout.md) separates reuse from policy. Lessons: delta cumulative work counters; preserve full histories; hash actual artifacts; compare complete policy/value outputs, since identical played moves can conceal changed targets. Banks generated by different search arms are not a paired quality sample. |
| [#477](https://github.com/jjoshua2/DeepFin/pull/477), August | Global-margin lazy evaluation failed its utility gate: **0.488%** predicted served probes versus **20%** required. A previously quoted wall-time ceiling was withdrawn because its denominators did not match. Do not infer speed from reuse counts alone. |
| [#485](https://github.com/jjoshua2/DeepFin/pull/485), [#487](https://github.com/jjoshua2/DeepFin/pull/487), August 29 | Gumbel donor reuse now requires legal-action and history-context agreement. Tests distinguish exact-context reuse from history mismatch; `tt_stats()` exposes hits, reuse, action rejects, and context rejects. This is safe donor expansion copying, **not a shared-visit neural search graph**. The historical ledger entry was deploy-gated; current realized reuse remains to be measured. |
| [#363](https://github.com/jjoshua2/DeepFin/pull/363), [#1198](https://github.com/jjoshua2/DeepFin/pull/1198), [#1210](https://github.com/jjoshua2/DeepFin/pull/1210) | Exact-input cache work found compact/single-walker bypasses and a BF16 transport defect. Healthy-looking zero counters did not mean the cache ran. Cold-cache overhead and actual evaluator requests must be measured; a TT miss need not imply a GPU evaluation when a lower cache is active. |
| [#1440](https://github.com/jjoshua2/DeepFin/pull/1440), [#1453](https://github.com/jjoshua2/DeepFin/pull/1453), [#1469](https://github.com/jjoshua2/DeepFin/pull/1469), November 21 | Multi-GPU PUCT evaluates distinct nonterminal pending leaves once and fans results back to unchanged visits. A shared-GPU profile found **16.8% terminal / 15.4% duplicate / 67.8% unique**; no idle-GPU speed claim. Follow-ups fixed last-chunk counters misreported as per-search and added a nonzero duplicate fixture. This does not change classic Gumbel or make duplicate visits explore elsewhere. |

## Stage 1 — implemented occupancy census

`scripts/profile_transposition_reuse.py` drives the real Gumbel runner from frozen full-history roots, with an optional actual checkpoint. A native opt-in census scans cached `CBoard`s after search, subject to an explicit bound, and groups them by:

1. Exact structural identity.
2. Exact current search-context identity.
3. Reconstructed **float32 encoded-input identity** under the selected encoding.

Publish cached-board count, unique groups and duplicate-node counts for each identity, positions with multiple search contexts, and maximum contexts per position. Hash matches are checked with exact equality. The census scans **all** cached boards or refuses: exceeding `max_cached_boards` never returns a prefix as a full result. A pending Gumbel evaluator batch is also refused; callers must stop all other native walkers before inspecting a tree. That refusal covers Gumbel only: PUCT (`select_leaves`/integrate, the multi-GPU pool, `walker_descend_puct`/`walker_integrate_leaf`) releases the GIL inside each call and keeps no tree-level in-flight flag, so a census beside a running PUCT walker is not detected. This is a documented limit beside the native function, not a refusal, because enforcing it would add an atomic counter to every GIL-released hot-path entry. Encoded-input equality and structural equality are separate partitions, not assumed to be nested.

**This is occupancy, not an evaluator request log.** Cached roots and recipients populated through reuse are included. Boards no longer retained are absent. Multiple cached boards with equal reconstructed planes neither prove repeated NN calls nor quantify saved GPU work. Float32 reconstruction also does not establish equality of actual BF16/FP16 transport or auxiliary-input bytes. The report must not name these counts “NN savings,” “cache hit rate,” or independent evidence samples.

The initial CPU synthetic smoke demonstrates reachability and grouping mechanism only. Validate known exact twins, different histories on the same structural board, distinct boards, empty populations and exceeded bounds, and unchanged full search outputs with census disabled/enabled. No current-checkpoint performance conclusion follows from a synthetic evaluator.

**Gate:** proceed when grouping controls are discriminating, bounds are enforced, and full search-output parity holds. The driver starts a separate tree for each root/budget and records one census per tree; it does not aggregate transpositions across those searches. The cached population includes roots and reused recipients, so low-frequency or zero-overlap findings only describe this scope. If the bound is exceeded, the run fails without publishing a partial report; explicitly choose a larger bounded canary before repeating. Occupancy alone cannot approve an optimization.


### Reproduce the first diagnostic

After the repository's locked CPU/dev setup and native rebuild, run from the checkout:

```bash
uv run --no-sync python -m scripts.profile_transposition_reuse \
  --synthetic-smoke --simulations 64 256 \
  --output /absolute/experiment-output/transposition-smoke.json
```

This uses a deterministic synthetic evaluator on the initial board. It establishes that the command runs, not a production reuse frequency. The report preserves action, complete policy, legal mask and root value, plus an output digest, separate search/census times and isolated TT counter deltas.

For a real checkpoint, supply its actual `CollectSettings` JSON and an ordered JSONL root corpus. Each record has exactly `id` and `moves`, replayed legally from the standard initial position:

```json
{"id":"game-17-ply-4","moves":["g1f3","g8f6","b1c3","b8c6"]}
```

The driver passes `allow_terminal_root_shortcuts=True`, the literal at production's selfplay call site, and records it in each row's `realized_shape`; a test compares the two literals. FEN-only/truncated histories, null moves, terminal/claimable-draw roots, relation-enabled checkpoints and tablebase-backed collection profiles are refused in v1. Relation inputs must be supported explicitly before admitting those checkpoints; silently omitting a trained input would change the search. Preserve game identifiers when sampling multiple roots; those roots are not independent games for confidence intervals. Prefer banked game histories and predeclare sampling/strata before examining overlap. The first `max_roots` records are selected in file order and the complete corpus is hashed.

```bash
uv run --no-sync python -m scripts.profile_transposition_reuse \
  --checkpoint /absolute/checkpoint/trainer.pt \
  --collect-settings /absolute/experiment-input/collect-settings.json \
  --roots /absolute/experiment-input/full-histories.jsonl \
  --simulations 64 256 --max-roots 32 --max-cached-boards 8192 \
  --noise-scale 1 --seed 20261122 \
  --output /absolute/experiment-output/transposition-checkpoint.json
```

Run the process alone. The driver uses the shared `build_configs` and `build_selfplay_search_shape` helpers, and records the expanded requested settings and every realized shape. It deliberately runs fresh single-root CPU searches with two Torch threads, no tablebases/evaluation cache, and a constant supplied root-noise scale. It does **not** replay the collection's GPU batching, warm-tree carry, noise schedule, or playout-cap sampling. Checkpoint/roots/native binary digests, encoder/repetition regime, source commit/dirty flag and runtime versions are recorded. Use a clean pinned checkout for evidence. Existing output files are refused. No actual production checkpoint or frozen corpus was available in this development environment; that readout remains pending.

### Development validation, not production evidence

The initial-board synthetic smoke above, seed `20261122`, reported:

| Requested simulations | Cached boards | Unique structures | Unique search contexts | Unique reconstructed inputs |
|---:|---:|---:|---:|---:|
| 64 | 65 | 65 | 65 | 65 |
| 256 | 257 | 256 | 257 | 257 |

The larger fixture distinguishes one repeated structural board from two different contexts/inputs. This verifies why the identities need separate columns; it does not estimate their frequency under a trained net or a batched collection. The focused development suite passed **132 tests**, including a real saved tiny-model replay, native grouping controls, pending/limit refusals, complete-output parity and warm-root continuation parity. The saved tiny model validates the checkpoint path; it is not the production model.

## First reading on the SL base

Bank: `~/chess-artifacts/operations/rl-claude-20261119/transposition-census-slbase/` — `~/chess-artifacts/operations/rl-claude-20261119/transposition-census-slbase/README.md` [`ff80151e`], `~/chess-artifacts/operations/rl-claude-20261119/transposition-census-slbase/output/summary_64_256_800.txt` [`ab3f5b9d`], `~/chess-artifacts/operations/rl-claude-20261119/transposition-census-slbase/output/summary_3200.txt` [`8327fa76`]. Driver at the #1499 head, SL base, the pilot's collect settings with the Syzygy pair dropped, CPU, fresh single-root trees. 32 roots from 8 pilot games at plies 12/30/60/100. The 3,200-sim row is 6 ply-12 roots only; the seventh root hit the CPU budget.

| Simulations | Repeated boards (structural duplicate share) | Same search context / model input | TT probe hits → context-rejected |
|---:|---:|---:|---:|
| 64 | 1.2% | 0% | 7 → 7 (100%) |
| 256 | 5.0% | 0% | 217 → 217 (100%) |
| 800 | 8.9% | 0.06% | 1,337 → 1,330 (99.5%) |
| 3,200 | 19.0% | 0.76% | 2,786 → 2,726 (97.8%) |

Repeated boards grow with budget, but almost none share the model input: history planes make the same board a different NN input, and the strict context check rejects 98–100% of TT hits.

**Limits.** These are cached-board **occupancy** counts, not evaluator calls saved. The roots are not independent (four per game), so no intervals are given. Tablebases were dropped, because the driver refuses a tablebase profile. Root noise was a constant 1.0, so roots at plies ≥ 30 ran noisier than the pilot's decaying schedule. The bank ran before this driver passed production's root-shortcut flag; its README checked all 32 roots and none had an immediate mate or terminal-draw root action, so that setting changed nothing there.

**Consequence.** Stage 3's first two candidates are an exact-input cache and a context-aware donor index. At these budgets both have under 1% of cached boards to find: same-context and same-input boards are at most 0.76%, even at 3,200 simulations. That figure is occupancy **within one fresh single-root search**. An exact-input cache could also hit across consecutive moves of a game, or inside a carried (reused) tree, and this census observes neither. So the evidence makes no case for either candidate within a single search, and **cross-move reuse is unmeasured**. The next measurement is Stage 2's actual request accounting, which counts evaluator calls rather than occupancy and should count cross-move and carried-tree repeats explicitly. Beside it sits a separate question that this census cannot answer: whether an **approximate** share across histories is acceptable, i.e. how far the net's value and policy move between histories that reach the same board ([below](#history-sensitivity-would-an-approximate-share-be-nearly-free)).

## History sensitivity: would an approximate share be nearly free?

The census says exact sharing has almost nothing to find because history planes make each transposition a different input. The next question is how far the net's outputs actually move between two real move orders that reach the same board. If they barely move, an evaluation could be shared across histories approximately. Two pre-registered CPU scopings on the SL base answer it. Neither is a candidate, and neither measures a search outcome.

### Replay: same board, different move orders

Bank `~/chess-artifacts/operations/rl-claude-20261119/history-sensitivity-scoping/`: `~/chess-artifacts/operations/rl-claude-20261119/history-sensitivity-scoping/PREREG.md` [`271b8527`], sealed before any output; `~/chess-artifacts/operations/rl-claude-20261119/history-sensitivity-scoping/README.md` [`468e86a0`]; `~/chess-artifacts/operations/rl-claude-20261119/history-sensitivity-scoping/output/summary.json` [`7840451e`]; addendum `~/chess-artifacts/operations/rl-claude-20261119/history-sensitivity-scoping/ADDENDUM_1.md` [`32eaf0af`].

From the 32 census roots, the net's top-4 prior moves were expanded to depth 4 and every node forwarded. That gave 1,041 groups of distinct paths to the same board. The input carries 8 frames (`LC0_HISTORY_FRAMES`, `chess_anti_engine/encoding/lc0.py`). Verdict on the pre-committed screening bands: **AMBIGUOUS**.

| criterion | value | NEARLY FREE band |
|---|---:|---|
| median group \|ΔQ\| spread | 0.0051 | ≤ 0.01: met |
| p90 group \|ΔQ\| spread | **0.0352** | ≤ 0.03: **failed** |
| median group sym-KL | 0.00047 nats | ≤ 0.02: met |
| median spread / sibling gap | 0.033 | ≤ 0.10: met |

No CHANGES THE SEARCH band was reached. The built-in control passed: 1,020 pairs extended so that they differ only outside the 8 frames had bitwise-identical inputs and exactly equal outputs. The other 21 differ only in clock or repetition state.

A "first one wins" cache would inject a |ΔQ| error of 0.0048 median and 0.033 p90. It would change the top-1 move for 4.8% of members and flip 1.66% of sibling orderings at the parent. Depth-4 transpositions are the easy case; the census's 3,200-simulation transpositions are deeper.

**Disclosed deviations.** (a) Before the seal, a development smoke of the byte-identical script ran on the first root only (into scratch, discarded) and printed that root's verdict line; the PREREG discloses it, and the bands were fixed before it. (b) The script groups only `depth > 0` nodes (`history_sensitivity_replay.py`, the grouping loop), an exclusion the PREREG did not state: it drops 9 groups where a 4-ply return path reaches the root's own board. With the roots included there are 1,050 groups, spread median 0.00518 and p90 0.03660, so the verdict is still AMBIGUOUS (bank `ADDENDUM_1.md`).

### Rule50: is the clock or the history doing it?

Bank `~/chess-artifacts/operations/rl-claude-20261119/rule50-bucketing-scoping/`: `~/chess-artifacts/operations/rl-claude-20261119/rule50-bucketing-scoping/PREREG.md` [`2648143d`], sealed before any output; `~/chess-artifacts/operations/rl-claude-20261119/rule50-bucketing-scoping/README.md` [`d5143e72`]; `~/chess-artifacts/operations/rl-claude-20261119/rule50-bucketing-scoping/output/part_a_summary.json` [`7d1ed2ca`]; `~/chess-artifacts/operations/rl-claude-20261119/rule50-bucketing-scoping/output/part_b_summary.json` [`2f925d4e`]; addendum `~/chess-artifacts/operations/rl-claude-20261119/rule50-bucketing-scoping/ADDENDUM_1.md` [`bc95432d`].

The owner proposed bucketing the 50-move clock in the key instead of matching it exactly. Clock figures here are the **half-move clock** (the owner counts the rule in moves: half-move clock 2c = move c). The clock enters the input as one scalar plane, 109 = min(half-move clock, 100)/100. Re-encoding 512 boards at 5 clocks changed no other plane.

- **The replay's tail is history, not the clock value.** This is a re-analysis of the banked observations, post hoc.
  - Pairs that differ only in the clock (window otherwise identical) move Q by a median of 0.0001 (max 0.0033).
  - In-tree pairs whose clocks differ also differ in *when* a pawn moved or a capture happened. Those move Q by 0.0071 median / 0.039 p90, against 0.0034 / 0.027 for history-only pairs.
  - Post hoc: keying on board + repetition + **exact** clock re-reads the replay's bands as NEARLY FREE, but only by 0.001 on p90 (0.02897 vs 0.03), on a split chosen after the verdict and on the replay's proxy repetition flags. Any bucket of width ≥ 4 merges every real clock-differing pair and reads AMBIGUOUS again.
- **Controlled sweep.** Board and history frames were fixed, and only the half-move clock was swept over 0..99. That covered 512 primary boards stratified by piece count and |Q|. A declared addition, descriptive only, added 120 real high-clock positions from pilot shards; **the verdict reads the 512 primary boards only**.
  - **Verdict on the pre-committed band** (within-bucket |ΔQ| median ≤ 0.002 and p90 ≤ 0.01, sym-KL median ≤ 0.005): **NONE**. No scheme coarser than exact passes: none, w16, the geometric shape, w8, w4, w2. Even w2 fails on p90, 0.063.
  - **Shape of the curve.** The net is nearly insensitive at low clocks: adjacent-step |ΔQ| p90 ≤ 0.001 below half-move clock 30 (≤ 0.002 on the real high-clock positions). It is steep near 100: adjacent-step p90 0.058 at half-move clock 90 on real high-clock positions, and |Q(99) − Q(0)| up to 0.95. The worst w8 bucket starts at half-move clock ≥ 88 on 415 of 512 boards, mostly endgames. In the owner's units the steep zone starts at about **moves 40–44**, which agrees with his "1 vs 10 moves is identical, 45 vs 48 is very different".
  - **Post-hoc hybrid** (descriptive, needs fresh confirmation): width 4 below half-move clock 48 and exact above passes on both populations (64 keys). It was chosen from 12 post-hoc shapes tried on the same data. It is consistent with the owner's idea: coarse at low clocks, exact near the draw.
- **What output spread cannot show.**
  - The 50-move draw and Syzygy DTZ verdicts depend on the exact clock. The TT context (`cboard_context_match`, `_mcts_tree.c`) and the terminal test (`_lc0_ext.c`) must keep it exact.
  - A bucketed clock belongs only in the **evaluation** cache key. Those caches fingerprint the encoded row, so quantising plane 109 in the key copy would leave the TT and terminal paths untouched. This is a code reading; nothing was built.
  - While the history frames stay in the key, bucketing the clock adds almost no hits: 18 of 1,041 extended control pairs, and no in-tree pair. The real gain needs history out of the key too, which is the AMBIGUOUS question above.

### How lc0 handles it (cited from source)

Read from the lc0 source at commit `d8ce482` (v0.33.0-dev, a local clone at `~/local_engines/lc0_033`, outside this repository):

- **The NN cache keys on the current board plus its repetition count; it ignores the history frames and the clock.** `~/local_engines/lc0_033/src/neural/memcache.cc` [`3bae948c`] (lines 38–43): `ComputeEvalPositionHash` returns `pos.pos.back().Hash()`, and `Position::Hash()` is `HashCat({us_board_.Hash(), repetitions_})` (`~/local_engines/lc0_033/src/chess/position.cc` [`0387843a`], lines 59–61). Its TODO says it ignores repetitions and history; it ignores the earlier repetitions and the history frames, not the current repetition count.
- **The DAG search's TT key is short by default.**
  - The key is `history.HashLast(params_.GetCacheHistoryLength() + 1)` (`~/local_engines/lc0_033/src/search/dag_classic/search.cc` [`dbe9a497`], line 2095). `HashLast` also mixes in the rule50 ply (`~/local_engines/lc0_033/src/chess/position.cc` [`0387843a`], lines 141–149).
  - `CacheHistoryLength` defaults to 0 (`~/local_engines/lc0_033/src/search/classic/params.cc` [`5c28b7b3`], line 573), so the key is the current position (board plus repetition count) plus the rule50 ply.
  - The option's help text says that when it is shorter than the net's history, "the search will use eval of the same position with different history taken from cache".
- **So lc0 shares evaluations across histories approximately, by design and by default.** Its NN cache also ignores the clock; its DAG TT key does not.

KataGo was not checked: no source was available.

### Evaluation-key grouping: log clock buckets, history depth, repetition

Bank `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/`: `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/PREREG.md` [`b4e81cf5`], sealed with its scripts before any output; `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/README.md` [`e5a38477`]; `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/output/part_a_summary.json` [`402ba79a`]; `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/output/part_b_summary.json` [`64c3a097`]; `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/output/part_c_summary.json` [`e9d16cbc`]; `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/pair-cost/README.md` [`2908f45d`]. SL base, CPU, fp32. All clocks are **half-move clocks**; the owner counts moves, so half-move clock 2m is move m.

The owner asked for clock buckets that are "log based on moves until draw, … exact after 45 [moves] and 0 to 10 grouped", and whether repetition or history could be grouped the same way. These three readings say what an **evaluation** cache key may group. The TT context, terminal, threefold, 50-move and Syzygy logic stay exact in every case.

**(A) Log-shaped clock buckets — NOT CONFIRMED.**
- **Schemes and post-hoc scores.** The owner's shape as literally stated is H1: [0–20] as one bucket, octaves of the half-moves left, exact from half-move clock 90 (17 buckets). Its geometric variant is H2 (26 buckets). Both fail the band, on the wide **low-clock** buckets as much as near the draw: the worst buckets are 41–60, 0–20 and 0–19 as well as 86–89 and 88–89. H1's single [0–20] bucket alone costs per-bucket p90 0.018 on the primary sample, and H2's [0–19] 0.017. H3 (exact from half-move clock 80, 43 buckets) passed post hoc.
- **Fresh confirmation, pre-registered:** "NOT CONFIRMED: H3 fails the pre-committed band on the fresh sample". Over 331 fresh boards it reads p90 0.0147 against a 0.01 limit. Its median 0.00228 also misses the 0.002 limit. Its failures sit in its width-8 low-clock buckets. H3 had also failed post hoc on the supplementary sample (p90 0.0228).
  - **Sampling shortfall:** 331 of 504 registered boards. No boards at half-move clock ≥ 80, and 79 of 126 at 60–79, because a global per-game cap starved the high-clock strata. Disclosed, not re-run. The verdict therefore rests on clocks below 80.
- **The finer hybrid passes everywhere tested.** Width 4 below half-move clock 48 and exact above (64 buckets) passes on the primary, supplementary and fresh samples (fresh 0.0010 / 0.0060). It was chosen post hoc, from 12 shapes.
- **Top band: "unread, insufficient evidence".** The fresh sample had no boards with a real half-move clock ≥ 80. The whole pilot holds only 26 such games (319 rows), and all were already used. Syzygy adjudication at ≤ 6 men makes long shuffles rare, so high clocks matter mainly for match play against engines, not for collection-time cache hits. A top-band confirmation of the hybrid is **registered and waiting** on RL run 1 generation 2's collection (`~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/topband/PREREG.md` [`d3eab569`]: verdict on the ±10-of-real-clock subset, at most 5 rows per game per band, at least 150 boards and 30 games per band or the reading is again "unread").
- **Top band, read 2026-11-26 (run Nov 22 19:51–20:30, the sealed script unchanged): "HY CONFIRMED on real high-clock boards", by the pre-committed rule, but the test had no power.**
  - Input: gen-2's collection, 42 shards and 796,341 rows, checked against its manifest. 301 earlier games were excluded. T (80–99) gave 216 boards from 75 games and U (60–79) gave 300 boards from 124 games, so the minimum n was met.
  - HY within ±10 of the real clock, pooled: 516 boards, |ΔQ| median, p90 and max all exactly 0. H3 (secondary, not a verdict) has median 0.00021, but its p90 of 0.0108 misses 0.01.
  - **Why HY could not fail.** HY is exact from clock 48, and the ±10 window around real clocks of 60–99 never goes below 50, so every HY bucket in the verdict statistic holds one clock. The PREREG fixed the verdict statistic and the bands before anyone noticed this, so the verdict stands as registered. It says nothing about HY beyond "exact is exact".
  - **What the reading does show** is the descriptive pair cost: on real high-clock boards the net is clock-sensitive. For a clock difference of 1, T gives |ΔQ| median 0.0059 / p90 0.047, and U gives 0.0002 / 0.0061. At a difference of 4, T gives 0.030 / 0.188. Any bucket wider than one clock above 60 would therefore be costly, and HY's exact region above 48 is load-bearing, not slack.
  - Bank: `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/topband/output/topband_summary.json` [`62714e40`], per-position observations `.../sweep_topband.jsonl` [`dc853167`], log `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/topband/topband_run.log` [`d1f705b9`], sums `.../SHA256SUMS.output` [`e3f3e1b8`].
- **What any bucket buys is small.** Every scheme merges 100% of the real clock-differing transposition pairs seen, because their half-move clocks are ≤ 3 at root plies 12 to 100. On the clock differences real pairs have (1–4 half-moves), the cost is median ≤ 0.0013 and p90 ≤ 0.0104 below half-move clock 50 in every population. A few d = 4 cells sit just over 0.01 (0.0103 at 40–49 on the supplementary sample). The cost is large above 80 (`pair-cost`, post hoc, descriptive). Pure clock-only differences move Q by ≤ 0.0033. So a clock bucket is cheap but buys only the 18-of-1,041 class of transpositions whose history frames already match.

**(B) History depth — the actionable reading.** The verdict, fixed wording: "smallest d = 2: a key of board + rule state + the last 1 plies". Here **rule state** means the half-move clock plus all 8 window repetition flags.
- **Limits:** one net (the SL base), CPU fp32. Leaves come from 32 roots in 8 games, so they are not independent. "NEARLY FREE" is a band on |ΔQ| / KL / ratio, not a playing-strength claim.

| first differing ply back (d) | controlled pairs | median \|ΔQ\| | p90 \|ΔQ\| | band | real depth-6 pairs, band |
|---:|---:|---:|---:|---|---|
| 1 | 228 | 0.0031 | 0.0360 | AMBIGUOUS | AMBIGUOUS (p90 0.038) |
| 2 | 256 | 0.0016 | 0.0167 | NEARLY FREE | NEARLY FREE (p90 0.023) |
| 3–7 | 183–272 each | 0.0002–0.0011 | 0.0058–0.0083 | NEARLY FREE | NEARLY FREE at d = 3, 4 |
| 8 (outside the input) | 278 | 0 | 0 | identical inputs 278/278 (control PASS) | — |

- **How the pairs were built.** Controlled pairs are commuting swaps that keep the board, the half-move clock and the window repetition flags identical; every row was forwarded at batch 1. Real pairs come from a top-3, depth-6 expansion of 16 census roots, and they agree with the controlled pairs, including the rule-state-equal subset.
- **What each key would group.** The numbers below are **fractions of same-board PAIRS, not of evaluator requests or network calls.** Structural duplicates are only 5–19% of nodes at 256–3,200 sims (census above), and these fractions apply within that share. The verdict's key includes rule state, so it groups only **rule-state-equal** pairs. **It groups 25.7% (depth 4) and 30.3% (depth 6) of same-board pairs.** The "ignoring rule state" column describes a key that drops the clock and the repetition flags; the verdict does not cover that key. Recomputed in `~/chess-artifacts/operations/rl-claude-20261119/eval-key-grouping-scoping/ADDENDUM_1.md` [`6779c2f9`], which corrects the bank README's mislabelled table.

  | key keeps | replay depth-4 pairs (1,409): rule-state-equal | replay depth-4: ignoring rule state | real depth-6 pairs (3,780): rule-state-equal | real depth-6: ignoring rule state |
  |---|---:|---:|---:|---:|
  | board (+ rule state) only | 58.2% (reads AMBIGUOUS) | 100% | 47.5% (reads AMBIGUOUS) | 100% |
  | **+ last 1 ply (the verdict's key)** | **25.7%** | 36.5% | **30.3%** | 49.8% |
  | + last 2 plies | 0% | 0% | 13.9% | 20.7% |
  | exact input | 0% | 0% | 0% | 0% |
- **Limit:** commuting swaps are the mildest history difference, and real paths reach only d ≤ 4.

**(C) Repetition.**
- **Encoding.** Each frame carries one binary repetition plane (no count, so "twice" encodes as "once").
- **The current position's flag is load-bearing.** Flipping the frame-0 flag moves Q by a median of 0.078 (p90 0.216), so it stays in any key. Older frames' flags are smaller but not free: frames 1 and 2 have medians 0.020 and 0.015 (p90 0.080 and 0.073), and frames 3–7 have medians 0.005–0.007. Any key keeps every frame's flag.
- **Caveat:** these are synthetic flips. Only 4 of 512 boards carry a real flag.

### Consequence for Stage 3

- **Exact candidates:** the exact-input cache and context-aware donor index keep their census reading. They have under 1% to find within one search, and cross-move reuse is unmeasured.
- **An approximate share is neither ruled in nor out.** The typical error is small, but one group in ten moves value by more than 0.035, and almost all of that comes from history, not the clock.
- **Hypothesis, not a reading:** a key of board + repetition state + exact clock (dropping the move-order frames, keeping the rule state). Post hoc it passes the replay's p90 band by 0.001 (0.02897 vs 0.03), on a statistic chosen after the verdict and on the replay's proxy repetition flags. The pre-registered history-depth reading above refines it: keep the **last ply** as well (d ≥ 2 reads NEARLY FREE; d = 1 does not). The candidate key is therefore board + rule state + last ply.
- **A bucketed clock adds little on top**, because real transpositions rarely differ in the clock alone. If it is tried, the post-hoc hybrid shape (w4 below half-move clock 48, exact above) is the one to confirm on fresh data, and only in the evaluation key.
- **Stage 2's request accounting comes first.** Only it shows whether repeated boards cost real evaluator calls at match budgets.

## Next experiment (design only; for decision)

**(a) Stage 2 request accounting — how many real network calls each key would save.**
- **Where.** The match engine (UCI), at 800, 3,200 and 10,000+ simulations per move, playing CONSECUTIVE moves with tree reuse. That is the cross-move case the census cannot see.
- **What to count.** For every evaluator request, the row's key three ways:
  1. exact input (today's fingerprint);
  2. board + rule state + last ply: the truncated key defined in (b), which is exactly the key (B) tested;
  3. board + rule state.

  In each case, "rule state" is the half-move clock plus every frame's repetition flag.

  Count distinct keys per search and across a game's searches, so the output is evaluator requests a cache would have answered, not occupancy.
- **Where to hook it.** At the evaluator entry, where `PucvEvalCache` and `EncodedEvalCache` already see each row.
  - #1507's `CAE_BATCH_TRACE` records forward sizes and timing per forward, not row identities. It can carry a per-forward count of new keys if a keyed counter is added beside it.
  - The census path cannot help: it is post-search occupancy.
- **Rule:** off by default; outputs (action, policy, value) must be identical with it on and off.

**(b) In-search arm (after (a) shows real savings).**
- **What it is.** An evaluation cache keyed on board + rule state + last ply, default off. Matched simulations against the exact-key control on the same positions and seeds.
- **What it must not touch.** TT context (`cboard_context_match`), terminal, tablebase and repetition logic stay exact. Only the evaluation lookup is approximate.
- **Where the key is computed — a small change.**
  - Today the key fingerprints the whole encoded row (`encoded_row_fingerprint128` in `_mcts_tree.c`; `_encoded_row_key` / `_encoded_row_digest` in `inference_cache.py`).
  - The truncated key hashes:
    - planes 0–25: frames 0 and 1, each with its repetition flag;
    - **every frame's repetition plane, 13·i + 12 for i = 2…7: planes 38, 51, 64, 77, 90, 103;**
    - planes 104–174: castling, side to move, half-move clock, en passant, the bias plane and the v2 feature planes.
  - **This is exactly the key (B) tested: board + half-move clock + all 8 window repetition flags + the last ply.** The B1 constructions held all 8 flags identical. A key that dropped the frame 2–7 repetition planes would be coarser than the verdict covers, and Part C shows those flags are not free (frame 2 flip: median 0.0148, p90 0.073).
  - The feature planes are assumed board-only; verify that before relying on it.
  - It needs a list of byte ranges instead of one (planes 0–25, six single planes, 104–174), in both the C fingerprint and the Python key.
  - **The full-row digest that `PucvEvalCache.get` checks on a hit must use the same truncation. Otherwise every cross-history hit is silently rejected as a miss** — the repo's signature defect, so the take-effect proof is a hit counter that moves.
- **Cost.** CPU hashing over fewer bytes than today, and no GPU cost. The saving is the forwards answered.
- **Screen, then yardstick** (per `docs/eval_protocol.md`):
  - **Screen:** policy-target and value-target quality against the frozen deep-SF / BT4 references at matched sims, plus nps. The screen decides whether the arm earns an arena; it is not a strength claim.
  - **Strength claim:** only from the paired arena (`scripts/arena_standard.py`). Use `matched_time` for the nps gain, and `matched_sims` to show no quality loss.
  - Sample size: the replay's effect sizes (|ΔQ| median 0.002–0.003, p90 ≈ 0.017 at d = 2) need a paired design on a few thousand positions to resolve a target-quality change of the size of C17's 2.8 cp residual. Fix it in the record from (a)'s hit rate.
- **Kill rule:**
  - At the screen, stop the arm if a target-quality regression has a 95% interval that excludes zero against the exact-key control, or if the hit rate is below 5% of requests at match budgets.
  - At the arena, stop if the `matched_sims` paired arena's 95% interval lies wholly below zero, or if the `matched_time` paired arena's interval does not lie above zero.
- **Success rule:** adoption needs both arenas: a `matched_time` gain whose interval excludes zero, and a `matched_sims` result whose interval includes zero or is above it. The exact numbers go in the record before launch.

## Stage 2 — actual request accounting

Extend the existing [C17 observer](https://github.com/jjoshua2/DeepFin/blob/e182d821ca3575e2c27a224c9dc0560a3c08a0fa/chess_anti_engine/mcts/gumbel_c.py#L89-L188) at actual evaluator flushes, preserving its disabled fast path:

- Count real/padded rows, within-flush duplicates, cross-flush repeats, and distinct exact inputs over a declared search lifetime. Preserve raw dtype/bytes, auxiliary relations, compact legal ordering, and model/encoding provenance.
- Distinguish requested rows, cache hits, and rows actually forwarded to the model. Observe both sides of an enabled cache; do not enable one for the baseline.
- Use isolated before/after `tt_stats()` snapshots. These counters are process-cumulative; concurrency can contaminate deltas. Do not publish a hit rate without a genuine probe-attempt denominator.
- Require identical action, legal mask, complete policy, and returned value with observation on/off. Record coverage and overhead; unobserved means unavailable, not zero.

Existing source reports costly exact hashing in a CPU-only regime. Sample entire batches if needed, never sparse row elements; disclose sampling and avoid interpreting partial observations as a full census.

**Proposed diagnostic budget, not queued work:** at most 32 frozen histories, 64 simulations each, two repeat passes, including cold/carried-tree and controlled transposition/repetition cases. Cap preliminary work at 10 CPU-minutes and any separately authorized checkpoint/GPU pilot at 15 GPU-minutes. Time one small canary first; stop and retain partial evidence at the cap. Freeze roots/checkpoint/settings before reading treatment results. This is descriptive development data, not a powered strength test.

## Stage 3 — choose the smallest supported intervention

| Event-level finding | Candidate / falsifier |
|---|---|
| Repeated exact model requests consume material forward time | Test the existing exact-input cache or bounded deduplication first. Reject if hit savings are absorbed by hashing, copies, batching loss, or memory cost. |
| Context-compatible donors exist but structural indexing repeatedly selects another context | Test a bounded context-aware donor index. Preserve legal/context guards. Reject if most rejects are genuine input/rule differences or measured reuse remains negligible. |
| Different routes repeatedly reproduce downstream work after evaluation reuse is controlled | Prototype a true graph with explicit cycle handling, shared safe state payloads, and parent-local edge statistics. Keep path-dependent draw decisions separate. Reject incorrect backup accounting or gains explained solely by evaluation caching. |
| Material allocation inefficiency remains between different positions | Only then investigate learned coverage/pseudo-counts, with exact reuse and ordinary visit counts as controls. Reject if novelty scoring suppresses tactical refutations or its overhead removes equal-time gains. Training requires separate authorization. |

Sharing a board's static computation does not justify sharing every searched Q, visit, or exploration denominator. A repeat traversal can reveal new descendants. The current [NNUE graph design](https://github.com/jjoshua2/DeepFin/blob/e182d821ca3575e2c27a224c9dc0560a3c08a0fa/docs/nnue_position_dag.md) is useful infrastructure and precedent, not proof that Gumbel's sequential-halving accounting transfers unchanged to a graph.

## Stage 4 — controlled readouts and adoption

For a selected candidate, preregister a separate experiment before running it:

- **Fixed simulations:** same roots, histories, checkpoint, seeds and realized settings. Report requests, real model evaluations, unique inputs, allocation/coverage, target quality, and memory. Include a work-matched control when a change buys more distinct work.
- **Fixed time:** include hashing/copying/synchronization and actual batching costs. Use paired openings, seat balance, interleaved arm order, and the same cache/carry policy. A cheaper simulation is valuable only if it improves useful search within the budget.
- **Validity:** observation-only controls must be identical. Reuse-only controls require full-output parity or a declared numerical tolerance when NN batch shape changes. Search-changing arms need outcome/quality readouts; teacher agreement and NPS alone cannot establish strength.
- **Decision:** publish failures and adverse slices, especially repetition/draw and tactical positions. Advance only if the mechanism takes effect, correctness holds, and the chosen deciding measure improves with uncertainty reported. Otherwise stop or revise the mechanism; do not add a learned module to rescue an unexplained null.

The remaining question is empirical: how much reusable computation and genuinely new exploration the current search produces per unit time. This sequence makes that measurable before changing what a visit means.
