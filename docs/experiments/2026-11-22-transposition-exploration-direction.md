# Transposition reuse and exploration accounting: direction and diagnostic plan

**2026-11-22 · Diagnostic implemented; production measurement pending · baseline `e182d821ca3575e2c27a224c9dc0560a3c08a0fa`**

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
| [C17 audit](https://github.com/jjoshua2/DeepFin/blob/e182d821ca3575e2c27a224c9dc0560a3c08a0fa/docs/rl_loop_audit.md#L474) | The proposed root-count explanation was refuted: root maximum/total visits stayed identical. Effects came through completed values and allocation. A per-prepare-call observer read **0%** while the accumulated evaluator batch contained **77.5%** duplicates. Measure the actual flush boundary. |
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

Publish cached-board count, unique groups and duplicate-node counts for each identity, positions with multiple search contexts, and maximum contexts per position. Hash matches are checked with exact equality. The census scans **all** cached boards or refuses: exceeding `max_cached_boards` never returns a prefix as a full result. A pending Gumbel evaluator batch is also refused; callers must stop all other native walkers before inspecting a tree. Encoded-input equality and structural equality are separate partitions, not assumed to be nested.

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

FEN-only/truncated histories, null moves, terminal/claimable-draw roots, relation-enabled checkpoints and tablebase-backed collection profiles are refused in v1. Relation inputs must be supported explicitly before admitting those checkpoints; silently omitting a trained input would change the search. Preserve game identifiers when sampling multiple roots; those roots are not independent games for confidence intervals. Prefer banked game histories and predeclare sampling/strata before examining overlap. The first `max_roots` records are selected in file order and the complete corpus is hashed.

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
