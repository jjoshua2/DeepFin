# Search and encode paths: map and consolidation plan

**Mapped at commit `863f12aa9` on `main` (2026-11-13); the selfplay, encode and
gate statements were re-verified at `9a5afddcb` and updated for the
refuse-instead-of-fall-back change (section 8).** `file:line` references are as
of those commits; later commits will move them. Paths are relative to
`chess_anti_engine/` unless they start with `scripts/`, `docs/` or `tests/`.

Purpose: the reference for collapsing the codebase to ONE search path and ONE
encode path shared by selfplay and the match engine, so a fix or speedup lands
once. Sections 1-7 are the map; the last section is the ordered plan.

Since `863f12aa9` the UCI default has changed to classic Gumbel (`--walkers 1`,
`uci/__main__.py:899`) and `Threads` no longer selects the walker pool, on the
grounds that classic Gumbel is the only strength-validated search path
(section 1). Rows below that still describe the walker default are marked.

---

## 1. Search implementations

### One C tree, four descent primitives

`mcts/_mcts_tree.c` (6762 lines) is shared by every C path: node/child arena,
`tree_expand` (:776), backprop (:1180), virtual loss (:1198/:1209), the
transposition table, solved-status propagation (:1025), the value-provider hook,
and terminal detection `cboard_search_terminal` (`mcts/_search_terminal.h:59`:
true game over, plus a 2-fold repetition scored as a draw, lc0 style).

| Primitive | C entry | Inner selection |
|---|---|---|
| A. Gumbel state machine | `start_gumbel_sims` / `continue_gumbel_sims` (`gss_*` :1536-2107, :4220, :4682) | `tree_gumbel_collect_leaf` (:3560): forced root action, then `tree_gumbel_select_child` (:3365); PUCT `tree_select_child` (:3627) only if `full_tree=False` |
| B. Python-driven PUCT | `select_leaves` (:2428) | `tree_select_leaf` (:971), PUCT |
| C. Walker PUCT | `walker_descend_puct[_tb]` / `walker_integrate_leaf` (:2560, :2776) | `tree_select_leaf`, PUCT + virtual loss |
| D. Batched PUCT | `batch_descend_puct[_tb]` / `batch_integrate_leaves` (:2882, :3172) | `tree_select_leaf`, PUCT + virtual loss, batched |

### Drivers

| # | Driver | Language | Primitive | Callers |
|---|---|---|---|---|
| 1 | `mcts/gumbel_c.py:812 run_gumbel_root_many_c` | Python over C | A | **Production selfplay** (`selfplay/network_turn.py:1210`); arena matched_sims (`selfplay/match.py:186`, used by `scripts/arena_standard.py`, `scripts/match_vs_handicapped_sf.py`, `arena.py`, `scripts/match_checkpoints.py`, the worker arena task); UCI classic mode (`uci/search.py:2440`); `eval/puzzles.py:440`; `scripts/audit_targets.py`; `scripts/search_gain_probe.py`; `scripts/diagnose_gumbel_roots.py`; benches; 35 test files |
| 2 | `mcts/gumbel.py:1726 run_gumbel_root_many` | Pure Python, python-chess boards, own `Node` tree | — | Selfplay (`network_turn.py:1258`) and arena (`match.py:220`) only when volatility search is configured (refused for selfplay at yaml load, `utils/config_yaml.py`) or inside the explicit `ext_check.python_reference_paths` opt-in; a stale build that cannot import the C runner is **refused** (section 8). Research: `scripts/audit_targets.py`, `scripts/lc0_adapter_probe.py`, `scripts/gumbel_scale_probe.py`, `scripts/profile_selfplay.py`, `bench/play_batch_timing.py`; 17 test files incl. the parity test `tests/test_mcts_uci_parity_gates.py` |
| 3 | `mcts/puct_c.py:78 run_mcts_many_c` | Python over C | B | Selfplay PUCT branch (`network_turn.py:1296`), taken when `mcts_type != "gumbel"` and sims > max(64, fast_simulations); `SearchConfig.mcts_type` **defaults to `"puct"`** (`selfplay/config.py:42`), so only a yaml with `mcts: gumbel` (production) avoids it; `match.py:232` when `mcts_type="puct"`, which is the **default** of the worker arena task (`worker.py:4143`, `arena_cfg.get("mcts", "puct")`) and of the `arena.py` CLI (:149) |
| 4 | `mcts/puct.py:337 run_mcts_many` | Pure Python, own tree | — | Stood in for #3 when `puct_c` failed to import (now refused, section 8); `eval/puzzles.py` PUCT branch, the **default** when no `gumbel_cfg` is passed, which `tune/trainable_phases.py:243` does when `puzzle_interval > 0` (production: 0); benches; 7 test files |
| 5 | `uci/walker_pool.py WalkerPool` | Python threads over C | C | UCI `--walkers > 1` / `SearchWalkers` only (default now 1, `uci/__main__.py:899`); it was the default launch at `863f12aa9` |
| 6 | `mcts/puct_vl.py:74 PucvChunker` | Python 2-slot async pipeline over C | D | UCI `UseVL=true` at 1 thread (`uci/search.py:879`); also the per-candidate engine inside #8 |
| 7 | `uci/multi_gpu_pucv_pool.py MultiGpuPucvPool` | One Python thread per GPU over C | D | UCI with `--devices` listing >= 2 devices; switched on automatically (`__main__.py:866`, `search.py:610`) |
| 8 | `uci/root_parallel_gumbel.py RootParallelGumbelPool` | Python orchestration + Python halving | D | UCI `SearchParallel=gumbel` with >= 2 devices (`search.py:671`) |
| 9 | `mcts/one_ply.py one_ply_value_backups` | 1-ply backup, not a search | — | `scripts/avi_*` only |

What each driver re-implements:

- **#2 and #4** decide terminality with python-chess `board.is_game_over()`
  (`gumbel.py:1259`, `puct.py:402`): no claimable draws, no 2-fold rule, so
  their terminal semantics differ from every C path.
- **#3** decides leaf terminality in Python too (`puct_c.py:228`).
- **#8** re-implements completed-Q scoring, the halving schedule and final
  selection in Python (sharing `halving_keep_count` and
  `_select_top_m_with_gumbel` from `gumbel.py`). Its inner descent is PUCT,
  not Gumbel (its own docstring, :65). No cross-move tree reuse
  (`advance_root` returns False, `search.py:1421`).
- **#6 and #7** each carry their own copy of the 2-slot pipeline:
  `PucvChunker._run_pipelined / _prepare_cached_submit / _finish_pending /
  _materialize_cached_outputs` against `MultiGpuPucvPool._pipeline_until_done /
  _prepare_submit / _finish_pending / _materialize_cached_outputs`.

### Which path runs where (at `863f12aa9`)

- **Production selfplay: #1.** The production yaml sets `mcts: gumbel`,
  `gumbel_vloss_weight: 1`, `gumbel_c_scale: 0.1`, `syzygy_in_search: true`
  (read the live yaml for the current sims/topk/policy_temp — do not copy them
  from here). The PUCT branch is unreachable there: `use_gumbel` is true
  whenever the type is gumbel or sims <= max(64, fast_simulations)
  (`network_turn.py:1180`). A config without `mcts:` gets the `SearchConfig`
  default `"puct"` and does take it above that sim count.
- **Default UCI launch: #1** (classic Gumbel, `--walkers 1`). At `863f12aa9` it
  was #5, the PUCT walker pool with 2 walkers.

### Recorded strength validation

- **#1 classic Gumbel** (UCI `--walkers 1`, or the arena): c_scale 0.025
  measured +301 Elo at 8k sims (ledger :3759; provenance and the
  sim-count caveat at :8331); the Stockfish node-anchor table run with
  `--walkers 1` (ledger :3935); the c_scale ranking ladder (ledger
  "2026-08-10 RANKING LADDER", :42836/:42916); the C17 virtual-loss fix
  (:7903); every arena matched_sims Elo in the ledger.
- **#5 walker pool: none found.** Ledger :4118-4121: "the shipped UCI default
  is `--walkers 2` … Nothing here measures the walker pool." The only
  deployment (`docs/experiments/2026-11-12-rl-week-one.md:168`, two PUCT walkers
  vs Rofchade) lost both games on time; that is not a strength reading.
- **#6, #7, #8: none found.** `docs/design_multi_gpu_search.md` §6-7
  pre-registers a strength ladder that was never run (throughput only).
- **#2, #3, #4: none found** beyond parity tests.
- The virtual-loss change on the UCI Gumbel path (PR #286, ledger :3683) is
  still recorded as pre-registered and unread.

## 2. Config surface

Config objects: `GumbelConfig` (`mcts/gumbel.py:323`); `MCTSConfig`
(`mcts/puct.py:43`); selfplay `SearchConfig` (`selfplay/config.py`);
`SelfplaySearchShape` (`selfplay/network_turn.py`, which exists because
`vloss_weight` and `target_batch` are runner arguments, not `GumbelConfig`
fields); `WalkerPoolConfig`, `MultiGpuPucvConfig`, `RootParallelGumbelConfig`;
`EngineOptions` (`uci/engine.py:206`) plus the `SEARCH_OPTIONS` registry
(`mcts/search_options.py:143`); the arena's `SideSearch`; and the constants
`PLAY_SEARCH_DEFAULTS` / `PLAY_PUCT_DEFAULTS` / `PLAY_SEARCH_VLOSS_WEIGHT=3`
(`gumbel.py:87`, `:297`, `:318`).

Existing guards: `PY_ONLY_GUMBEL_KNOBS = {volatility_q_scale, volatility_fpu}`
refused on the C path (`gumbel.py:135`, `:246`); `C_ONLY_GUMBEL_KNOBS =
{full_tree}` refused on the Python path (:172, :266); `INERT_GUMBEL_KNOBS =
{c_puct, cpuct_factor, cpuct_base, fpu_reduction}` (:119). For the UCI modes,
`search_options.inert_reason` already encodes the field x path table below.

Columns: SC = selfplay C (#1), Ar = arena C (#1), UG = UCI Gumbel (#1),
Py = Python Gumbel (#2), RPG = #8, W = walker (#5), V = pucv (#6),
Pool = #7, PC = `run_mcts_many_c` (#3). Y = honoured, – = dropped/inert,
R = refused loudly.

| Field | SC | Ar | UG | Py | RPG | W | V | Pool | PC |
|---|---|---|---|---|---|---|---|---|---|
| simulations | Y | Y | Y (per chunk) | Y | Y (per chunk) | Y | Y | Y | Y |
| topk | Y | Y | Y | Y | Y | – | – | – | – |
| c_scale / c_visit (descent) | Y | Y | Y | Y | root fallback only | – | – | – | – |
| c_scale_root / c_visit_root / q_visit_exp_root | Y | Y | Y | Y | Y | – | – | – | – |
| halving_div | Y | Y | Y | Y | Y | – | – | – | – |
| policy_temp | Y (disables bf16 compact leaves) | Y | Y | Y | Y (`set_policy_temp`, rpg:1150) | – | – (method exists on `PucvChunker`; only RPG calls it) | – | – (no field) |
| add_noise / gumbel_scale | Y per game | Y | Y (`root_noise_scale`) | Y | Y | – | – | – | Dirichlet only |
| temperature | 0, then per-game resample | Y | forced 0 | Y | 0 | – | – | – | Y (visits) |
| full_tree | Y | Y | Y | R | – | – | – | – | – |
| target_max_visit_cap / target_untempered_prior / target_q_rescale | Y (stored target only) | n/a | n/a | Y | n/a | – | – | – | – |
| volatility_* | R (yaml loader) | R / reroutes to Py | – | Y | – | – | – | – | – |
| vloss_weight | Y (yaml) | Y (explicit arg) | Y (3) | – | Y | Y | Y | Y | – |
| target_batch / MinibatchSize | Y | Y | Y | – | – | – | – | – | – |
| c_puct | – (inert) | – | – | – | Y (descent) | Y | Y | Y | Y |
| cpuct_factor / cpuct_base | – | – | – | – | Y | Y | Y | Y | – (no field) |
| fpu_reduction | – | – | – | – | Y | Y | Y | Y | Y |
| fpu_at_root | – (yaml key inert) | – | – | – | read, pinned 0.0 | read, pinned 0.0 | read, pinned 0.0 | read, pinned 0.0 | Y |
| tb_probe in leaves | Y | Y | Y | R | Y | Y | Y | Y | – |
| compute_relations | Y | Y | Y | Y | Y | Y | Y | Y | R (`puct_c.py:94`) |
| input_history_encoding / extra features | Y | Y | Y | Y | Y | Y | Y | Y | Y |

**`fpu_at_root` on the PUCT engines (settled by reading).** Every UCI PUCT
engine carries the value into its C descent: `WalkerPoolConfig.fpu_at_root`
→ `walker_pool.py:217` → `walker_descend_puct[_tb]` (:301/:308);
`PucvChunker(fpu_at_root=)` (`puct_vl.py:88`, :114) → `descend_batch`
(:250); `MultiGpuPucvConfig.fpu_at_root` (`multi_gpu_pucv_pool.py:55`) →
:439/:489; `RootParallelGumbelConfig.fpu_at_root`
(`root_parallel_gumbel.py:212`) → per-candidate `PucvChunker` (:1050). In C
it is the FPU reduction used for the first `tree_select_child` call at the
descent's root, after which `fpu_reduction` applies (`_mcts_tree.c:979-989`);
for RPG that "root" is the candidate node. **But all four construction sites
hard-code 0.0** (`uci/search.py:501` walker, `:600` pool, `:726` RPG, `:891`
pucv), and no UCI option or CLI flag sets it. So it reaches the search but is
pinned at 0.0, not tunable. The selfplay yaml `fpu_at_root` and
`fpu_reduction` keys are accepted and do nothing (the yaml's own comment says
so). The UCI play default is `policy_temp 1.0`; switching to any PUCT mode
except RPG drops policy_temp entirely.

## 3. Encode paths

- **E1 — C encoders compiled into `_mcts_tree`.**
  `batch_encode_146[_lc0_root[_legacy_meta]][_bf16]` (`_mcts_tree.c:6610-6626`)
  and the leaf encoders inside the Gumbel state machine and the walker/batch
  descents (`cboard_encode_planes_into*`, :4130-4200). Used by the selfplay root
  batch (`network_turn.py:505-529`), `gumbel_c` leaves, `puct_c`, and every
  UCI descent.
- **E2 — C encoders compiled into `_lc0_ext`.** `CBoard.encode_full` /
  `encode_*` and `encode_full_batch` (`_lc0_ext.c:979-997`), reached through
  `encoding/cboard_encode.py encode_cboard` / `encode_cboard_batch`. Used for
  the UCI root eval (`uci/search.py _ensure_root_eval_cached`), the production
  selfplay `record_lc0_root_input` side row (`network_turn.py:747`,
  `encode_cboard_batch`), `stockfish_turn`, `resume`, RPG, `model_inputs` and
  the worker's broker warmup probe.
- **E1 and E2 are the same headers compiled twice** (`_cboard_impl.h`,
  `_features_impl.h`), each with its own global state — which is why
  `encoding/rep_fix.py apply` must set the flag in both modules.
- **E3 — Python encoder.** `encoding/encode.py encode_position /
  encode_position_into / encode_positions_batch / _encode_fused_c`:
  python-chess board → bitboards → `_lc0_ext.encode_piece_planes` +
  `_features_ext.compute_extra_features`. Repetition planes are computed in
  Python with full look-back (`encoding/lc0.py:583 _check_repetitions`), so
  **E3 always encodes the `history_rep_fix=on` regime regardless of the global
  flag.** Used by the Python searches (#2, #4), `puct_c`'s no-CBoard branch,
  `eval/puzzles.py`, the `deepfin --selfcheck` oracle
  (`uci/selfcheck.py _python_reference_planes`), research scripts, and the
  side row of the Python per-ply reference recorder (`network_turn.py:937`;
  never the production recorder, which uses E2). `build_model` no longer
  encodes a position to count planes (it uses `input_plane_count`), and the
  worker broker warmup probe uses E2; both were shown bit-identical
  (`tests/test_c_core_required.py`).
- **E4 — pure-Python features** (`encoding/features.py
  extra_feature_planes_fast`): a test oracle. When `_features_ext` lacks a
  symbol, E3 (`encode.py _extra_feature_planes`) and the replay v1→v2 upgrade
  (`plane_decode.recompute_extra_planes`) now **refuse** instead of using it,
  unless inside `ext_check.python_reference_paths`. The old all-zero
  feature-plane branch of `encode_cboard` is deleted: a board without
  `encode_full` is refused, and under the opt-in a missing features extension
  is still an error.
- **History handling.** `CBoard.from_board` reads python-chess's private
  `board._stack` (`_lc0_ext.c:415`). `cboard_from_board_fast`
  (`cboard_encode.py:29`) builds a board with no history; it is `puct_c`'s
  fallback. The trainer repairs stored planes (`train/trainer.py`,
  `lc0.py x_to_lc0_planes` / `fill_lc0_history_repeat`). Selfplay also has the
  C ply recorder `batch_process_ply` (`_mcts_tree.c:5511`).
- **`rep_fix.apply` is all-or-nothing:** it resolves the setter in both
  `_lc0_ext` and `_mcts_tree` first and raises `RepFixUnavailableError`
  (setting neither) if either module cannot be imported or lacks
  `set_history_rep_fix`; it used to skip that module with a warning, leaving
  the two encoders in different regimes.
- **rep_fix application points:** `model/__init__.py:518` (`build_model`, also
  the UCI route via `uci/model_loader.py`); `selfplay/manager.py:421`;
  `selfplay/match.py:157` (re-applied every move, per model);
  `scripts/net_source.py:582` (ONNX); `scripts/shared_teacher_owner.py:282`,
  `scripts/generate_bootstrap.py:140/228`, `scripts/gen_sf_rooted_corpus.py:1807`,
  `scripts/gen_random_selfplay_shards.py:2343`,
  `scripts/bt4_root_policy_worker.py:1192`, `scripts/nnue_*_readout.py`,
  `scripts/fuzz_*`.
- **The buffer plane-count guard** `check_encode_buffer_planes` is repeated at
  each evaluator-sourced encode site; `puct_c.py:265` calls itself the eighth.

## 4. Board representation (python-chess usage)

Rough count of `chess.*` vs `CBoard` references: `uci/` 62 vs 26,
`mcts/` 43 vs 21, `encoding/` 362 vs 22, `tablebase.py` 38 vs 4,
`moves/` 105 vs 2.

python-chess APIs the engine runtime relies on:

- **`uci/engine.py`:** `chess.Board()` / `Board(fen)` holds game state;
  `chess.Move.from_uci`; `mv in board.legal_moves`; `push` / `pop`;
  `copy(stack=)`; `turn`; `chess.popcount(board.occupied)` for the time manager.
- **`uci/search.py`:** `SearchWorker.run` takes a `chess.Board`;
  `board.fen()` is the tree identity; `any(board.legal_moves)` and
  `board.is_checkmate()` at the root; `_declined_root_reason` uses
  `is_repetition`, `is_insufficient_material`, `halfmove_clock`;
  `move_to_index` / `index_to_move*` (in `moves/`) go through python-chess;
  PV and `_board_after` push python-chess boards; 5 `CBoard.from_board` calls.
- **`mcts/gumbel_c.py`:** the entry point takes `list[chess.Board]`, each
  converted with `CBoard.from_board`.
- **`mcts/root_tactics.py:23`:** mate-in-1 detection via `is_game_over`,
  `legal_moves`, `gives_check`, `copy`.
- **`tablebase.py`:** every probe uses `chess.syzygy.Tablebase`, and
  **`SyzygyProbe.apply` builds `chess.Board(cb.fen())` for every
  tablebase-eligible leaf** (:624) — python-chess in the leaf hot path of every
  search that uses tablebases.
- **`_lc0_ext.c:415`:** depends on the private `_stack` attribute.

## 5. Evaluator / inference paths

- **Production selfplay:** `MultiSlotInferenceClient` / `SlotInferenceClient`
  (`worker.py:1561`) to a per-trial `SlotBroker` process
  (`inference.py:1941`), when `distributed_inference_broker_enabled` is set
  (it is in the production yaml; shared broker off, AOT off, compiled
  reduce-overhead). Without the broker the worker builds `ThreadedDispatcher`,
  `AOTEvaluator` (`--aot-dir`) or `DirectGPUEvaluator` (`worker.py:1344`).
  `SharedSlotBroker` (:3771) and AOT are optional.
- **UCI:** `DirectGPUEvaluator` compiled with max-autotune
  (`uci/__main__.py:322`), wrapped by mode: `MultiGPUDispatcher` for multiple
  `--devices`; `ThreadSafeGPUDispatcher`; `CUDAOwnerDispatcher` at 1 walker or
  `BatchCoalescingDispatcher` above 1 (:338-340); `EncodedEvalCache` (:349,
  only when `EvalCacheEntries > 0`, default 0); per-device factories for #7/#8
  (:373); `PucvEvalCache` inside #6/#7.
- **Arena matched_sims:** a hoisted `DirectGPUEvaluator`
  (`scripts/arena_standard.py:1563`), or a throwaway `LocalModelEvaluator`
  per call when none is passed (the stream-pool OOM hazard documented at
  `selfplay/match.py:119`).
- **ONNX:** `onnx/load.py` wraps `onnxruntime` as an `nn.Module` (4672-wide
  policy, Ceres input), used via `scripts/net_source.py` and the lc0 adapter.
- **Python searches** fall back to `LocalModelEvaluator` (`gumbel.py`,
  `puct.py`, `puct_c.py:107`, `selfplay/state.py:870`).

## 6. Tablebase probing per path

Leaf probing: one class, `SyzygyProbe` (`tablebase.py:484`), but **three
wirings with two eligibility rules**:

| Wiring | Used by | Eligibility | Writes into |
|---|---|---|---|
| `gumbel_c._tb_override` (:590; call sites :1524-1850) | selfplay, arena, UCI classic Gumbel | **any halfmove clock** (C filter `get_pending_tb_leaves`, `_mcts_tree.c:4918`, checks castling and piece count only) | the evaluator's returned `wdl`, in place |
| inline in `uci/walker_pool.py:340` | walker pool | **halfmove_clock == 0 only** | a copy |
| `mcts/puct_tb.py` (`descend_batch` / `override_leaf_values` / `mark_batch_solved`) | pucv, pool, RPG | **halfmove_clock == 0 only** | a copy |

**Can the in-place write in `gumbel_c` leak into `EncodedEvalCache` on the
UCI classic path? No (settled by reading).** `EncodedEvalCache` declares
`supports_inplace_api = False` (`inference_cache.py:210`) and refuses to
forward `evaluate_encoded_async` (:196-207), so `gumbel_c`'s `_inplace` and
`_has_async` arms (`gumbel_c.py:978`, :1024) are both off and the leaf eval
goes through `evaluate_encoded` or `evaluate_legal_bf16`. Both store
**separate copies** in the cache (:405; :534-535) and return a **freshly
allocated** `wdl_out` filled row by row from those entries (:417-425;
:545-555). `_tb_override` (`gumbel_c.py:1850`) therefore mutates only the
fresh output, never a cached entry. With the cache off (the default,
`EvalCacheEntries=0`), the write lands in the inner evaluator's output
buffer (for the in-place arm, a view of the pinned slot tensor,
`gumbel_c.py:1824-1825`), which the next batch overwrites; there is no
cache to poison. The PUCT routes copy before overriding regardless.

Other probing:

- **Root (UCI):** `_try_tb_shortcut` → `_try_tb_root_bestmove` picks the DTZ
  move (`uci/search.py:1442`, :2783). The root wdl is probed in
  `_ensure_root_eval_cached`, and `pre_wdl_logits_tb_probed=True` is passed on.
- **Root (selfplay):** `allow_terminal_root_shortcuts`, plus solved-root
  witness recovery (`gumbel_c._recover_root_witness`), which adjudicates
  (`network_turn.py:1220`).
- **Adjudication / rescoring (selfplay):** `selfplay/finalize.py`,
  `selfplay/manager.py`, `selfplay/strict_tb.py` via `tb_adjudicate_result` /
  `rescore_game_samples` / `rule50_match_*`. In `rule50_match_v1` mode strict
  selfplay runs only on the C Gumbel path (`network_turn.py:1076`).
- **Unsupported:** the Python searches refuse tablebases (`match.py:141/181`);
  `run_mcts_many_c` has none.

## 7. Duplicated-logic hot spots

1. **Leaf tablebase wiring.** PR #1246 touched 13 files (C, walker pool,
   search, engine, tablebase, `match_vs_uci`); #1247 touched 9 (C, `puct_tb`,
   `puct_vl`, pool, RPG, search). Three copies, two eligibility rules.
2. **Root pre-expansion.** Three near-identical methods —
   `_ensure_pucv_pool_root_expanded` / `_ensure_pucv_root_expanded` /
   `_ensure_walker_root_expanded` (`uci/search.py:2567/2604/2628`) — plus RPG's
   `prepare_root` and `gumbel_c`'s own root preparation.
3. **The 2-slot pipeline.** Two copies: `mcts/puct_vl.py:205-435` and
   `uci/multi_gpu_pucv_pool.py:427-683`.
4. **vloss / target_batch plumbing.** The C17 fix had to land separately in
   selfplay (yaml), the arena (`match.py:91`, PR #286), UCI `_run_gumbel_chunk`
   (`uci/search.py:2475`), and four other UCI sites.
5. **Sequential halving and completed-Q scoring.** Three copies: C
   (`gss_score_and_halve`), the Python reference (`gumbel.py:1561`) and RPG's
   Python. The hard-coded divide-by-2 defect in the Python copy is noted at
   `gumbel.py:1520`.
6. **Terminal detection.** C `cboard_search_terminal`, python-chess
   `is_game_over()` (#2, #3, #4), and UCI root checks in Python
   (`uci/search.py:2185`, :2825).
7. **Encoder logic.** Compiled twice (E1/E2) plus Python E3, whose rep_fix
   regime is fixed; rep_fix must be applied at 15+ entry points.
8. **Search config.** Spread across about nine objects (section 2).
   `PLAY_*` vs `SELFPLAY_GUMBEL_C_SCALE` are deliberately split and pinned by a
   test — that split is intended, not duplication.
9. **Deleted descent knobs** are still passed to C as literals
   (`gumbel.py:351`).

## 8. C-core completeness gates (refuse, never fall back)

A missing `_mcts_tree` was always a hard `ImportError` (`network_turn.py`
imports it unguarded). The silent degradations came from a **stale or partial
build**: a symbol missing from an otherwise importable extension selected the
Python search, the Python per-ply recorder (which drops seed history), the
Python slot classifier, fp32 instead of bf16 leaf transport, Python feature
planes or legal-move generation, the Python terminal scan, all-zero feature
planes, or a rep_fix regime set in only one encoder.

One definition of "complete" now lives in `chess_anti_engine/ext_check.py`:
`REQUIRED_EXTENSIONS`, `REQUIRED_SYMBOLS` (per module), the required `CBoard`
and `MCTSTree` methods, and the `_mcts_tree` ABI (`gumbel_c._REQUIRED_MCTS_ABI`).
`tests/test_c_core_required.py` scans the package for `try: from <ext> import`
and `hasattr`/`getattr` probes of extension symbols and fails when one is
neither required nor in `GUARDED_OPTIONAL_ALLOWLIST`.

| Entry point | Check | On failure |
|---|---|---|
| UCI engine | `uci/selfcheck.enforce_startup_gate` (shared `extension_problems` + content freshness + non-zero feature planes) | exit 2, "refusing to start" (bypass `DEEPFIN_ALLOW_UNVERIFIED_EXTENSIONS=1`, development only) |
| `run.py` train/tune | `ext_check.require_c_core` before `run_tune` | `SystemExit` naming each missing symbol |
| `worker.py` | `ext_check.require_c_core` right after argument parsing | `SystemExit` naming each missing symbol |
| `scripts/train.sh` | `check_c_extensions_fresh.py --check-symbols` (presence, mtime, gcc, LTO recipe, plus symbols + ABI in a child interpreter); `scripts/graceful_restart.py` preflight passes the same flags | refuses to start (bypass `TRAIN_SKIP_C_EXT_CHECK=1`) |
| `play_batch` (any caller: worker, `tune/trainable_phases.py` eval games, scripts) | `_require_compiled_search` (gumbel_c / puct_c imported) and `SelfplayState.create` → `_probe_c_extensions` (tree, per-ply, encoders incl. bf16, `classify_games`, `temperature_resample`) | `CCoreIncompleteError` |
| point of use | network turn (Python Gumbel / PUCT search, Python per-ply recorder), arena move selection (`match.py`), `run_gumbel_root_many_c` (bf16 encoders), `encode.py`, `plane_decode.py`, `cboard_encode.py`, `moves/encode.py`, `root_tactics.py` | `CCoreIncompleteError` via `ext_check.refuse_fallback` |
| `rep_fix.apply` | both setters present before either is called | `RepFixUnavailableError` |

The Python reference implementations stay importable as parity oracles.
Running one **in place of** a missing C symbol requires
`ext_check.python_reference_paths(reason)`, a context manager no production
module enters (tests use the `python_reference_paths` fixture in
`tests/conftest.py`). Config-selected Python paths are unchanged: volatility
search still selects the Python Gumbel search, and the Python slot classifier
remains the normal path for SF-refute plies (`state.py`).

---

## Target architecture

**Core: the C `MCTSTree` plus the Gumbel state machine (primitive A), driven
by one Python driver** — `gumbel_c.run_gumbel_root_many_c`, refactored into a
steppable `GumbelSearch` object. Why this one:

- it produces the training targets today;
- every recorded strength result is on it;
- it already supports multi-board batching, virtual loss, tree reuse
  (`root_node_ids` + `root_context_matches`), restricted root moves
  (`allowed_root_indices_batch`), tablebase leaves, solved-root witness
  recovery, relations, and bf16 / in-place transport;
- `tree_gumbel_collect_leaf` already descends from a forced root action, which
  is exactly the per-candidate step RPG needs.

**Selfplay adds:** per-game sim budgets (playout cap), per-ply noise schedules,
temperature resampling, stored-target extraction (cap / untempered prior /
rescale / diagnostics), strict tablebase adjudication, and the game loop
(`batch_process_ply`).

**The UCI engine adds:** time management and abort, chunk interruption, ponder,
MultiPV and info lines, `advance_root` reuse, the root DTZ shortcut, a
python-chess-free protocol layer, and multi-GPU as RPG running its
per-candidate work on primitive A rather than `PucvChunker`.

**Keep only as test references:** `gumbel.py run_gumbel_root_many` (the parity
oracle and the only home of volatility search) and `puct.py`.
**Delete:** `puct_c.run_mcts_many_c` and the selfplay PUCT branch.
**Retire or mark experimental:** the single-GPU walker pool and single-GPU
pucv.

**Not to be retired without a match: the multi-GPU modes.** The target
tournament hardware (TCEC) is 4 GPUs, so a multi-GPU search is the tournament
path, not an optional mode. The owner recalls a 2-GPU test in which the PUCT
pool had the highest throughput and classic Gumbel was fastest on one GPU; no
record of it was found in the ledger or `docs/experiments`, and throughput is
not playing strength. Which multi-GPU mode (PUCT pool or root-parallel Gumbel)
wins games at 4 GPUs is unmeasured and is decided by the
`docs/design_multi_gpu_search.md` §7 ladder on rented hardware — scheduled
only after the net's strength gap has closed (owner's sequencing, 2026-11-13).
The consolidation target is therefore two paths: classic Gumbel on one GPU,
and the multi-GPU mode that wins that ladder.

### Ordered steps

Sizes: S/M/L. "Risk" says what the step can change.

1. **(S, measurement only)** Experiment record on main, then run the match the
   walker default never had: walker pool vs `--walkers 1` classic Gumbel, same
   checkpoint, paired openings, fixed clock, pause window. For example
   `PYTHONPATH=. python3 scripts/match_vs_uci.py --engine-a "python -m
   chess_anti_engine.uci --checkpoint C --walkers 1" --engine-b "python -m
   chess_anti_engine.uci --checkpoint C --walkers 2" --clock-base-ms 30000
   --clock-inc-ms 100 --openings <book> --games 200 --games-out ...`.
   Still worth running to know what the walker pool is worth, but it **no
   longer gates the default flip** (step 7).
2. **(S, no behaviour change)** Fold the three `_ensure_*_root_expanded`
   methods into one helper. Proof: UCI tests plus a byte-identical fixed-node
   move log before/after.
3. **(S, no behaviour change) — done (section 8).** Selfplay, arena, worker,
   training and the encoders refuse a stale or partial build instead of
   falling back to a Python path. Proof: a fixed-seed CPU `play_batch` digest
   over every recorded array, identical before and after, plus the parity
   suites and `tests/test_e2e_smoke.py`.
4. **(S, measurement instruments only)** Flip the worker arena task default
   (`worker.py:4137`) and the `arena.py --mcts` default from puct to gumbel,
   then delete `run_mcts_many_c` and the selfplay PUCT branch; keep `puct.py`
   as a test reference. Changes those arena rulers, not training data; note
   the ruler break in the record.
5. **(M, no behaviour change)** Extract one pipelined PUCT runner shared by
   `PucvChunker` and `MultiGpuPucvPool`. Proof:
   `tests/test_async_pipeline_safety.py`, `tests/test_pucv_leaf_tablebase.py`,
   `tests/test_vloss_release_on_eval_error.py`, plus a fixed-seed 1-GPU pool
   vs the chunker on the same tree.
6. **(M, changes selfplay targets if the rule moves on the Gumbel path —
   experiment record required)** One leaf-tablebase eligibility rule and one
   implementation (`puct_tb`-style, writing into a copy). Moving the Gumbel
   path to clock == 0 changes selfplay targets because `syzygy_in_search` is
   true; moving the PUCT paths to any-clock changes only the engine. Order:
   first move the walker pool onto `puct_tb` with its rule unchanged (no
   behaviour change); then score the rule question on the frozen deep-SF audit
   set before any training compute (CLAUDE.md experiment rules 1 and 5).
7. **(M, changes engine strength)** Make classic Gumbel the UCI default
   (`--walkers 1`) and stop `Threads` from selecting the walker pool; mark
   single-GPU Threads>1 / UseVL experimental (the multi-GPU PUCT pool keeps
   its status until the ladder is run). **Done since `863f12aa9`**,
   justified by classic Gumbel being the only strength-validated path; step 1
   then measures what the retired default was worth.
8. **(L, changes engine strength, multi-GPU only)** Rebase RPG's
   per-candidate work onto primitive A (forced root action + Gumbel descent),
   deleting its Python halving copy and the PUCT descent and adding tree
   reuse. Proof: with one group, RPG's root decision bit-identical to classic
   at fixed sims; strength via the `docs/design_multi_gpu_search.md` §7
   ladder. `MultiGpuPucvPool` is retired only if it loses that ladder.
9. **(M, no search change)** Make the classic Gumbel chunk interruptible:
   `continue_gumbel_sims` stops at a batch boundary on a stop event / deadline,
   closing the opaque-chunk clock risk from the 2026-11-12 deployment
   (`docs/experiments/2026-11-12-rl-week-one.md`). Proof: existing deadline
   tests plus a slow-evaluator test.
10. **(M, must be bit-identical)** One encoder source: build the encoder only
    in `_lc0_ext` and have `_mcts_tree` call it through a capsule, so there is
    one rep_fix flag; make E3 regime-aware or test-only (the zero-feature
    fallback in `cboard_encode` is already deleted, section 8). Proof:
    `scripts/fuzz_batch_encode_diff.py` / `scripts/fuzz_cboard_diff.py` under
    both rep_fix values, and bit-identity of production-shape encodes. Not
    bit-identical = a training-target change needing a record.
11. **(L)** Remove python-chess from the engine: `SearchWorker.run` and the
    engine take a `CBoard` plus a move list; UCI move parsing and legality via
    `CBoard.legal_move_indices`; `CBoard.from_board` stops reading `_stack`;
    `SyzygyProbe` probes from `CBoard` fields (a C prober, or at least no
    `chess.Board` per leaf). Safe if a fixed-node UCI move log stays
    byte-identical. A prober change also touches selfplay leaves, so it needs
    bit-identical override rows on a frozen leaf set.

**Steps that would change selfplay training targets** — each needs an
experiment record and the deep-SF audit first: step 6 if the rule moves on
the Gumbel path; any change to Gumbel-path terminal semantics; step 10 if not
bit-identical; anything that puts `policy_temp` or virtual loss into a
different arm. Steps 2, 3, 5 and 9 cannot change selfplay or engine
behaviour by construction; step 4 changes only arena rulers; steps 7 and 8
change engine strength only.
