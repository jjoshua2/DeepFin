from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.mcts import GumbelConfig, MCTSConfig
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishUCI

try:
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c as _run_mcts_many_c
    _HAS_C_TREE = True
except ImportError:
    _HAS_C_TREE = False

try:
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,
    )
    _HAS_GUMBEL_C = True
except ImportError:
    _HAS_GUMBEL_C = False

if TYPE_CHECKING:
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,  # noqa: F401,F811
    )
    from chess_anti_engine.mcts.puct_c import (
        run_mcts_many_c as _run_mcts_many_c,  # noqa: F401,F811
    )
import chess

from chess_anti_engine.moves import (
    POLICY_SIZE,
    index_to_move,
    legal_move_mask,
)
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.finalize import finalize_game
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import (
    SOFT_RESIGN_CONSECUTIVE,
    SOFT_RESIGN_THRESHOLD,
    BatchStats,
    CompletedGameBatch,
    SelfplayState,
    _NetRecord,
)
from chess_anti_engine.selfplay.temperature import (
    apply_policy_temperature,
    temperature_for_ply,
)
from chess_anti_engine.tablebase import tb_adjudicate_result


# Backward-compatible alias: tests and bench scripts import this name directly.
# The implementation lives in ``selfplay.temperature`` as ``apply_policy_temperature``.
_apply_temperature = apply_policy_temperature


def _choose_curriculum_opponent_move(
    *,
    rng: np.random.Generator,
    legal_indices: np.ndarray,
    cand_indices: list[int],
    cand_scores: list[float],
    regret_limit: float,
) -> int:
    """Choose the curriculum opponent move index from Stockfish candidates.

    Returns a policy index (int).  No python-chess objects needed.
    """
    if not cand_indices:
        return int(legal_indices[int(rng.integers(len(legal_indices)))])

    if not math.isfinite(float(regret_limit)):
        # No regret filter = full-strength SF. MultiPV lists PVs in rank order
        # so cand_indices[0] is SF's best move. Used by eval / gate matches.
        return cand_indices[0]

    best_score = max(float(s) for s in cand_scores)
    acceptable = [
        idx
        for idx, score in zip(cand_indices, cand_scores, strict=False)
        if (best_score - float(score)) <= float(regret_limit) + 1e-12
    ]
    if not acceptable:
        acceptable = [cand_indices[0]]
    return acceptable[int(rng.integers(len(acceptable)))]


def play_batch(
    model: torch.nn.Module | None,
    *,
    device: str,
    rng: np.random.Generator,
    stockfish: StockfishUCI | StockfishPool,
    evaluator: BatchEvaluator | None = None,
    games: int,
    target_games: int = 0,
    on_game_complete: Callable[[CompletedGameBatch], None] | None = None,
    on_step: Callable[[], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    # Config groups (frozen dataclasses with sensible defaults).
    opponent: OpponentConfig = OpponentConfig(),
    temp: TemperatureConfig = TemperatureConfig(),
    search: SearchConfig = SearchConfig(),
    opening: OpeningConfig = OpeningConfig(),
    diff_focus: DiffFocusConfig = DiffFocusConfig(),
    game: GameConfig = GameConfig(),
) -> tuple[list[ReplaySample], BatchStats]:
    """Play a batch of games.

    Design goals:
    - keep GPU busy via batched inference
    - keep SF queries minimal (one per SF ply)
    - compute volatility targets from a consistent network-side WDL series without per-ply overhead

    Continuous mode (stop_fn provided, target_games=0): runs forever with all
    finished slots recycled immediately, until stop_fn() returns True.  Samples
    are delivered incrementally via on_game_complete; the returned sample list
    is empty to avoid unbounded memory growth.

    Finite mode (target_games > 0 or stop_fn is None): plays exactly
    target_games (or `games` if target_games=0) then returns all samples.
    """

    requested_batch = int(games)
    continuous = stop_fn is not None and int(target_games) <= 0
    target = int(target_games) if int(target_games) > 0 else requested_batch
    batch_size = min(requested_batch, target)
    if batch_size <= 0:
        raise ValueError("play_batch requires at least one game")

    state = SelfplayState.create(
        model=model,
        device=device,
        rng=rng,
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=batch_size,
        continuous=continuous,
        target=target,
        opponent=opponent,
        temp=temp,
        search=search,
        opening=opening,
        diff_focus=diff_focus,
        game=game,
    )
    eval_impl = state.evaluator

    all_samples: list[ReplaySample] = []


    # ── Network turn ──────────────────────────────────────────────────────────

    def _run_network_turn(net_idxs: list[int]) -> None:
        if not net_idxs:
            return

        _bsz = len(net_idxs)
        # Bucket size for torch.compile shape stability
        _ROOT_BUCKETS = (32, 64, 128, 256, 512)
        _padded_bsz = _bsz
        for _b in _ROOT_BUCKETS:
            if _b >= _bsz:
                _padded_bsz = _b
                break

        _cb_encode_list = [state.cboards[_idx] for _idx in net_idxs]

        # Fast path: encode directly into evaluator's pinned buffer (zero-copy H2D)
        _use_inplace = state.batch_enc_146 is not None and hasattr(eval_impl, "get_input_buffer")
        if _use_inplace:
            # get_input_buffer + evaluate_inplace exist only on DirectGPU-style evaluators;
            # _use_inplace is the runtime guard.
            _buf = eval_impl.get_input_buffer(_padded_bsz)
            # batch_encode_146 memsets + encodes into first _bsz slots;
            # remaining slots are zero-padded by get_input_buffer's pinned memory.
            assert state.batch_enc_146 is not None
            state.batch_enc_146(_cb_encode_list, _buf)
            pol_logits_padded, wdl_logits_raw_padded = eval_impl.evaluate_inplace(
                _padded_bsz, copy_out=True,
            )
        else:
            xs_batch = np.empty((_bsz, 146, 8, 8), dtype=np.float32)
            if state.batch_enc_146 is not None:
                state.batch_enc_146(_cb_encode_list, xs_batch)
            else:
                for _j, _idx in enumerate(net_idxs):
                    xs_batch[_j] = state.cboards[_idx].encode_146()
            if _padded_bsz > _bsz:
                _pad = np.zeros((_padded_bsz - _bsz, *xs_batch.shape[1:]), dtype=xs_batch.dtype)
                xs_padded = np.concatenate([xs_batch, _pad], axis=0)
            else:
                xs_padded = xs_batch
            pol_logits_padded, wdl_logits_raw_padded = eval_impl.evaluate_encoded(xs_padded)

        pol_logits = pol_logits_padded[:_bsz]
        wdl_logits_raw = wdl_logits_raw_padded[:_bsz]
        # Pure numpy softmax (avoids torch tensor creation roundtrip for small arrays)
        wdl_f = wdl_logits_raw.astype(np.float64, copy=True)
        wdl_f -= wdl_f.max(axis=-1, keepdims=True)
        np.exp(wdl_f, out=wdl_f)
        wdl_f /= wdl_f.sum(axis=-1, keepdims=True)
        wdl_est = wdl_f.astype(np.float32, copy=False)

        is_full = rng.random(size=len(net_idxs)) < float(search.playout_cap_fraction)

        eff_full_sims = [int(search.simulations)] * len(net_idxs)
        eff_fast_sims = [int(search.fast_simulations)] * len(net_idxs)
        sample_weights = [1.0] * len(net_idxs)
        for j, idx in enumerate(net_idxs):
            win_p = float(wdl_est[j][0])
            if win_p < SOFT_RESIGN_THRESHOLD:
                state.consecutive_low_winrate[idx] += 1
            else:
                state.consecutive_low_winrate[idx] = 0

            if state.consecutive_low_winrate[idx] >= SOFT_RESIGN_CONSECUTIVE:
                ratio = win_p / SOFT_RESIGN_THRESHOLD
                sample_weights[j] = 0.1 + 0.9 * ratio

            state.sf_resign_scale[idx] = 1.0

        probs_list: list[np.ndarray | None] = [None] * len(net_idxs)
        actions: list[int | None] = [None] * len(net_idxs)
        values_list: list[float | None] = [None] * len(net_idxs)
        masks_list: list[np.ndarray | None] = [None] * len(net_idxs)

        # Per-game temperature based on each game's own ply count.
        temps = [
            temperature_for_ply(
                ply=state.cboards[i].ply // 2 + 1,
                temperature=float(temp.temperature),
                drop_plies=int(temp.drop_plies),
                after=float(temp.after),
                decay_start_move=int(temp.decay_start_move),
                decay_moves=int(temp.decay_moves),
                endgame=float(temp.endgame),
            )
            for i in net_idxs
        ]

        def _run_mcts_group(idxs: list[int], sims_per: list[int], *, per_game_noise: list[bool] | None = None) -> None:
            if not idxs:
                return

            # All games share one MCTS call with per-game sim budgets and
            # temperature applied after search.  Maximizes GPU batch size.
            group = idxs
            sub_cboards = [state.cboards[net_idxs[j]] for j in group]
            sub_temps = [temps[j] for j in group]
            sub_sims = [int(sims_per[j]) for j in group]
            sim_count = max(sub_sims)

            gumbel_low_sims = max(64, int(search.fast_simulations))
            use_gumbel = (str(search.mcts_type) == "gumbel") or (int(sim_count) <= int(gumbel_low_sims))
            sub_pol = pol_logits[group, :]
            sub_wdl = wdl_logits_raw[group, :]

            if use_gumbel:
                _gumbel_fn = _run_gumbel_root_many_c if _HAS_GUMBEL_C else run_gumbel_root_many
                # C gumbel only uses cboards; Python fallback needs python-chess boards
                sub_boards = sub_cboards if _HAS_GUMBEL_C else [state.boards[net_idxs[j]] for j in group]
                sub_noise = [per_game_noise[j] for j in group] if per_game_noise is not None else None
                # Map group indices to game-level root IDs for tree reuse
                sub_root_ids = [state.root_ids[net_idxs[j]] for j in group] if state.mcts_tree is not None else None
                _gumbel_result = _gumbel_fn(
                    state.model,
                    sub_boards,  # type: ignore[arg-type] # CBoard or Board; dispatched by _HAS_GUMBEL_C branch
                    device=device,
                    rng=rng,
                    cfg=GumbelConfig(simulations=int(sim_count), temperature=1.0, add_noise=True),
                    evaluator=eval_impl,
                    pre_pol_logits=sub_pol,
                    pre_wdl_logits=sub_wdl,
                    per_game_simulations=sub_sims,
                    per_game_add_noise=sub_noise,
                    cboards=sub_cboards,
                    nn_cache=state.nn_cache,
                    tree=state.mcts_tree,
                    root_node_ids=sub_root_ids,
                    tb_probe=state.tb_probe if game.syzygy_in_search else None,
                )
                # C version returns 6-tuple (with tree, root_ids), Python returns 4-tuple
                p_sub, a_sub, v_sub, m_sub = _gumbel_result[:4]
                # Store returned root IDs for tree reuse
                if state.mcts_tree is not None and len(_gumbel_result) >= 6:
                    _ret_root_ids = _gumbel_result[5]
                    for gi, jj in enumerate(group):
                        state.root_ids[net_idxs[jj]] = _ret_root_ids[gi]
            else:
                # PUCT needs python-chess boards.  When the C fast path is
                # active, boards[idx] stays at the starting position — rebuild
                # current state from move_idx_history (same as _finalize_game).
                if state.has_c_ply:
                    sub_boards = []
                    for j in group:
                        idx = net_idxs[j]
                        b = state.boards[idx].copy(stack=False)
                        for _mi in state.move_idx_history[idx]:
                            b.push(index_to_move(_mi, b))
                        sub_boards.append(b)
                else:
                    sub_boards = [state.boards[net_idxs[j]] for j in group]
                _puct_fn = _run_mcts_many_c if _HAS_C_TREE else run_mcts_many
                p_sub, a_sub, v_sub, m_sub = _puct_fn(
                    state.model,
                    sub_boards,
                    device=device,
                    rng=rng,
                    cfg=MCTSConfig(
                        simulations=int(sim_count),
                        temperature=1.0,
                        fpu_reduction=float(search.fpu_reduction),
                        fpu_at_root=float(search.fpu_at_root),
                    ),
                    evaluator=eval_impl,
                    pre_pol_logits=sub_pol,
                    pre_wdl_logits=sub_wdl,
                    cboards=sub_cboards,
                )

            # Re-select actions with per-game temperature from the
            # improved policy (probs are temperature-independent).
            _temps_arr = np.array(sub_temps, dtype=np.float64)
            _need_resample = _temps_arr != 1.0
            if _need_resample.any():
                _p_stack = np.stack(p_sub)  # (G, 4672)
                if state.has_classify_c:
                    # C path: GIL released during pow/sample
                    _a_arr = np.array(a_sub, dtype=np.int32)
                    _rand_arr = rng.random(len(sub_temps))
                    assert state.c_temp_resample is not None
                    state.c_temp_resample(_p_stack, _temps_arr, _a_arr, _rand_arr)
                    for gi in range(len(a_sub)):
                        a_sub[gi] = int(_a_arr[gi])
                else:
                    _nonzero = _need_resample.nonzero()[0]
                    for gi in _nonzero:
                        p = _p_stack[gi]
                        t = _temps_arr[gi]
                        legal = np.flatnonzero(p > 0)
                        if len(legal) == 0:
                            continue
                        if t <= 0:
                            a_sub[gi] = int(legal[np.argmax(p[legal])])
                        else:
                            pw = np.power(p[legal], 1.0 / t)
                            ps = float(pw.sum())
                            if ps > 0:
                                pw /= ps
                                a_sub[gi] = int(rng.choice(legal, p=pw))

            for jj, p, a, v, m in zip(group, p_sub, a_sub, v_sub, m_sub, strict=True):
                probs_list[jj] = p
                actions[jj] = a
                values_list[jj] = float(v)
                masks_list[jj] = m
                # Advance tree root to chosen move's child for next-ply reuse
                if state.mcts_tree is not None:
                    game_idx = net_idxs[jj]
                    rid = state.root_ids[game_idx]
                    if rid >= 0:
                        child = state.mcts_tree.find_child(rid, int(a))
                        state.root_ids[game_idx] = child  # -1 if not found

        # Run all games in one MCTS call with per-game sim budgets and noise.
        # Full-sim games get Gumbel noise for exploration; fast games don't
        # (KataGo playout cap convention).
        all_idxs = list(range(len(net_idxs)))
        combined_sims = [
            int(eff_full_sims[j]) if bool(is_full[j]) else int(eff_fast_sims[j])
            for j in all_idxs
        ]
        noise_flags = [bool(is_full[j]) for j in all_idxs]
        _run_mcts_group(all_idxs, combined_sims, per_game_noise=noise_flags)

        # Pre-allocate reusable buffers for per-sample computation
        _lg_buf = np.empty(POLICY_SIZE, dtype=np.float64)
        _swdl_buf = np.empty(3, dtype=np.float32)
        _df_enabled = bool(diff_focus.enabled)
        _df_q_w = float(diff_focus.q_weight)
        _df_p_s = float(diff_focus.pol_scale)
        _df_slope = float(diff_focus.slope)
        _df_min = float(diff_focus.min_keep)

        # ── C-accelerated per-ply processing (GIL released) ──────────
        if state.has_c_ply and len(net_idxs) > 0:
            _n = len(net_idxs)
            _cb_list = [state.cboards[net_idxs[j]] for j in range(_n)]
            _actions_arr = np.array(actions, dtype=np.int32)
            _values_arr = np.array(values_list, dtype=np.float64)
            # All slots filled by _run_mcts_group above; cast for np.stack's strict ArrayLike protocol.
            _probs_arr = np.stack(cast("list[np.ndarray]", probs_list)).astype(np.float32, copy=False)

            assert state.c_process_ply is not None
            (c_x, c_probs, c_wdl_net, c_wdl_search, c_priority,
             c_keep, c_mask, c_ply, c_pov, c_over) = state.c_process_ply(
                _cb_list, pol_logits[:_n], wdl_logits_raw[:_n],
                _actions_arr, _values_arr, _probs_arr,
                int(_df_enabled), float(_df_q_w), float(_df_p_s), float(_df_min), float(_df_slope),
            )

            # Pre-extract Python scalars from numpy arrays (batch conversion
            # is cheaper than per-element conversion in the loop).
            _c_ply_list = c_ply.tolist()
            _c_pov_list = c_pov.tolist()
            _c_priority_list = c_priority.tolist()
            _c_keep_list = c_keep.tolist()
            _c_over_list = c_over.tolist()
            _is_full_list = is_full.tolist()
            _sw_list = sample_weights
            _act_list = _actions_arr.tolist()

            for j in range(_n):
                idx = net_idxs[j]
                state.move_idx_history[idx].append(_act_list[j])
                state.last_net_full[idx] = _is_full_list[j]

                state.samples_per_game[idx].append(
                    _NetRecord(
                        c_x[j], c_probs[j], c_wdl_net[j], c_wdl_search[j],
                        chess.WHITE if _c_pov_list[j] else chess.BLACK,
                        _c_ply_list[j], _is_full_list[j],
                        _c_priority_list[j], _sw_list[j], _c_keep_list[j],
                        c_mask[j],
                    ),
                )

                if _c_over_list[j]:
                    state.done_arr[idx] = 1

        else:
            # Python fallback (original per-ply loop)
            for j, (idx, probs, a, v) in enumerate(zip(net_idxs, probs_list, actions, values_list, strict=True)):
                assert probs is not None and a is not None and v is not None

                board_before = state.boards[idx]
                ply_index = int(len(board_before.move_stack))
                pov_color = board_before.turn

                mask = masks_list[j]
                if mask is None:
                    mask = legal_move_mask(board_before)

                np.copyto(_lg_buf, pol_logits[j])
                _lg_buf[~mask] = -1e9
                _lg_buf -= float(np.max(_lg_buf))
                np.exp(_lg_buf, out=_lg_buf)
                _lg_buf[~mask] = 0.0
                s = float(_lg_buf.sum())
                if s > 0:
                    raw = (_lg_buf / s).astype(np.float32, copy=False)
                else:
                    raw = mask.astype(np.float32) / float(mask.sum())

                imp = np.maximum(probs.astype(np.float32, copy=False), 1e-12)
                raw_c = np.maximum(raw, 1e-12)
                kl = float(np.sum(raw_c * (np.log(raw_c) - np.log(imp))))

                orig_q = float(wdl_est[j][0] - wdl_est[j][2])
                best_q = float(v)
                q_surprise = abs(best_q - orig_q)

                difficulty = q_surprise * _df_q_w + kl * _df_p_s
                if not math.isfinite(difficulty):
                    difficulty = 1.0
                if not _df_enabled:
                    keep_prob = 1.0
                else:
                    keep_prob = max(_df_min, min(1.0, difficulty * _df_slope))

                move = index_to_move(int(a), board_before)
                board_before.push(move)
                state.cboards[idx].push_index(int(a))
                state.move_idx_history[idx].append(int(a))

                d_raw = float(wdl_est[j][1])
                rem = max(0.0, 1.0 - d_raw)
                q = float(max(-rem, min(rem, best_q)))
                w_search = 0.5 * (rem + q)
                _swdl_buf[0] = w_search
                _swdl_buf[1] = d_raw
                _swdl_buf[2] = rem - w_search
                search_wdl_est = _swdl_buf.copy()
                if not np.all(np.isfinite(search_wdl_est)):
                    search_wdl_est = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                state.last_net_full[idx] = bool(is_full[j])

                # ``not state.has_c_ply`` ⇒ ``state.batch_enc_146 is None``
                # (coupled import in SelfplayState.create) ⇒ ``_use_inplace is False``
                # ⇒ ``xs_batch`` was set in the else-branch.
                state.samples_per_game[idx].append(
                    _NetRecord(
                        x=xs_batch[j],
                        policy_probs=probs,
                        net_wdl_est=wdl_est[j] if np.all(np.isfinite(wdl_est[j])) else np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        search_wdl_est=search_wdl_est,
                        pov_color=pov_color,
                        ply_index=ply_index,
                        has_policy=bool(is_full[j]),
                        priority=float(difficulty),
                        sample_weight=float(sample_weights[j]),
                        keep_prob=float(keep_prob),
                        legal_mask=mask.view(np.uint8),
                    ),
                )

                if state.cboards[idx].is_game_over():
                    state.done_arr[idx] = 1

    # ── Stockfish annotation + opponent moves ─────────────────────────────────

    def _eff_nodes(idx: int) -> int | None:
        if state.base_nodes <= 0:
            return None
        fast_scale = 1.0 if bool(state.last_net_full[idx]) else 0.25
        return max(1, int(round(float(state.base_nodes) * float(fast_scale))))

    def _submit_sf_queries(idxs: list[int]) -> dict[int, Any]:
        """Submit SF queries to pool without blocking. Returns futures dict.

        Only valid when ``stockfish`` is a ``StockfishPool`` — callers guard with isinstance.
        """
        assert isinstance(state.stockfish, StockfishPool)
        return {idx: state.stockfish.submit(state.cboards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}

    def _finish_sf_annotation_and_moves(
        idxs: list[int], *, play_curriculum_moves: bool,
        futures: dict[int, Any] | None = None,
    ) -> None:
        """Collect SF results (from futures or synchronous) and process."""
        if not idxs:
            return
        if futures is not None:
            results = {idx: futures[idx].result() for idx in idxs if idx in futures}
        elif isinstance(state.stockfish, StockfishPool):
            futs = {idx: state.stockfish.submit(state.cboards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}
            results = {idx: fut.result() for idx, fut in futs.items()}
        else:
            results = {idx: state.stockfish.search(state.cboards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}
        _process_sf_results(idxs, results=results, play_curriculum_moves=play_curriculum_moves)

    def _process_sf_results(
        idxs: list[int], *, results: dict, play_curriculum_moves: bool,
    ) -> None:
        if not idxs:
            return

        sf_policy_temp_local = float(game.sf_policy_temp)
        sf_policy_label_smooth_local = float(game.sf_policy_label_smooth)

        def _softmax_np(x: np.ndarray) -> np.ndarray:
            z = x.astype(np.float64, copy=False)
            z = z - float(np.max(z))
            e = np.exp(z)
            s = float(e.sum())
            if s <= 0:
                return np.full_like(z, 1.0 / float(z.size))
            return e / s

        def _flip_wdl_pov(wdl: np.ndarray) -> np.ndarray:
            wdl = np.asarray(wdl, dtype=np.float32)
            if wdl.shape != (3,):
                return wdl.astype(np.float32, copy=False)
            return np.array([float(wdl[2]), float(wdl[1]), float(wdl[0])], dtype=np.float32)

        regret_limit = (
            float(opponent.wdl_regret_limit)
            if opponent.wdl_regret_limit is not None
            else float("inf")
        )

        for idx in idxs:
            res = results[idx]
            legal_indices = state.cboards[idx].legal_move_indices()
            if legal_indices.size == 0:
                state.done_arr[idx] = 1
                continue

            _turn = bool(state.cboards[idx].turn)
            legal_set = {int(x) for x in legal_indices}

            a_idx = uci_to_policy_index(res.bestmove_uci, _turn)
            if a_idx < 0 or a_idx not in legal_set:
                a_idx = int(legal_indices[0])

            cand_idxs: list[int] = []
            cand_scores: list[float] = []
            if getattr(res, "pvs", None):
                for pv in res.pvs:
                    if pv.wdl is None:
                        continue
                    a = uci_to_policy_index(pv.move_uci, _turn)
                    if a < 0 or a not in legal_set:
                        continue
                    w_sf, d_sf = float(pv.wdl[0]), float(pv.wdl[1])
                    cand_idxs.append(a)
                    cand_scores.append(w_sf + 0.5 * d_sf)

            if not cand_idxs:
                cand_idxs = [a_idx]
                cand_scores = [0.0]

            scores = np.array(cand_scores, dtype=np.float64) / max(1e-6, sf_policy_temp_local)
            p_top = _softmax_np(scores).astype(np.float32, copy=False)

            p_sf = np.zeros((POLICY_SIZE,), dtype=np.float32)
            for a, p in zip(cand_idxs, p_top, strict=False):
                p_sf[int(a)] += float(p)

            if sf_policy_label_smooth_local > 0.0:
                n_legal = legal_indices.size
                if n_legal > 0:
                    p_sf *= (1.0 - sf_policy_label_smooth_local)
                    p_sf[legal_indices] += sf_policy_label_smooth_local / float(n_legal)

            ps = float(p_sf.sum())
            if ps > 0:
                p_sf /= ps

            if state.samples_per_game[idx]:
                rec = state.samples_per_game[idx][-1]
                if rec.sf_policy_target is None and rec.sf_move_index is None:
                    rec.sf_policy_target = p_sf
                    rec.sf_move_index = a_idx
                    if res.wdl is not None:
                        rec.sf_wdl = _flip_wdl_pov(res.wdl)
                    _sf_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
                    _sf_mask[legal_indices] = 1
                    rec.sf_legal_mask = _sf_mask

            if not play_curriculum_moves or state.selfplay_arr[idx]:
                continue

            _opp_move_idx = _choose_curriculum_opponent_move(
                rng=rng,
                legal_indices=legal_indices,
                cand_indices=cand_idxs,
                cand_scores=cand_scores,
                regret_limit=regret_limit,
            )

            state.cboards[idx].push_index(_opp_move_idx)
            state.move_idx_history[idx].append(_opp_move_idx)
            # Advance tree root through opponent's move
            if state.mcts_tree is not None and state.root_ids[idx] >= 0:
                state.root_ids[idx] = state.mcts_tree.find_child(state.root_ids[idx], _opp_move_idx)
            if state.cboards[idx].is_game_over():
                state.done_arr[idx] = 1

    # ── Main game loop (rolling batch) ────────────────────────────────────────
    if continuous:
        max_steps = 2**62  # effectively infinite; stop_fn controls exit
    else:
        max_steps = int(target) * (int(game.max_plies) // 2 + 2)  # safety bound

    def _tb_adjudicate_active_games() -> int:
        """Scan active games; if any current position is TB-eligible and
        probable AND this game's per-game roll said "adjudicate", mark the
        game done and stash the TB-proven result. Runs once per step;
        per-active-game cost is ~10µs after the popcount prefilter rejects
        non-endgame positions.

        Games whose roll landed on "play through" (controlled by
        ``syzygy_adjudicate_fraction``) are skipped here and finish
        naturally, so the NN continues training on endgame positions with
        its own labels rather than losing endgame skill entirely.
        """
        assert state.tb_probe is not None and game.syzygy_path is not None
        max_p = state.tb_probe.max_pieces
        adjudicated = 0
        for i in range(batch_size):
            if state.done_arr[i] or state.finalized_arr[i] or state.tb_result_arr[i] is not None:
                continue
            if not state.tb_adj_roll_arr[i]:
                continue
            cb = state.cboards[i]
            occ = int(cb.occ_white) | int(cb.occ_black)
            if occ.bit_count() > max_p or int(cb.castling) != 0:
                continue
            board = chess.Board(cb.fen())
            result = tb_adjudicate_result(board, game.syzygy_path)
            if result is not None:
                state.tb_result_arr[i] = result
                state.done_arr[i] = 1
                adjudicated += 1
        return adjudicated
    _t_net = 0.0
    _t_sf = 0.0

    for _step in range(max_steps):  # skylos: ignore (_step loop var unused by convention)
        # Allow caller to update model/evaluator between moves.
        if on_step is not None:
            on_step()
        if stop_fn is not None and stop_fn():
            break

        # Tablebase adjudication before the classify pass. Any game that's
        # now TB-eligible gets marked done; the classify will then skip it
        # and the finalize path uses the stashed TB result. Runs at most
        # once per game — the ``state.tb_result_arr`` stash is the idempotency key.
        if state.tb_probe is not None and game.syzygy_adjudicate:
            _tb_adjudicate_active_games()

        net_idxs, selfplay_opp_idxs, curriculum_opp_idxs, all_done = state.classify_active_slots()
        if all_done:
            break

        # Submit SF queries for curriculum games FIRST — curriculum boards
        # are disjoint from net/selfplay boards, so we can overlap SF I/O
        # with the full combined network turn below.
        _sf_futures: dict[int, Any] | None = None
        if curriculum_opp_idxs and isinstance(state.stockfish, StockfishPool):
            _t0 = time.time()
            _sf_futures = _submit_sf_queries(curriculum_opp_idxs)
            _t_sf += time.time() - _t0

        # Combined network turn: merge net_idxs + selfplay_opp_idxs into a
        # single MCTS call.  This doubles the GPU batch size (256 vs 128),
        # giving much better GPU utilization and pipeline overlap.  Both
        # sets are disjoint and independent — same MCTS treatment, same
        # per-ply processing.
        _combined_net_idxs = net_idxs + selfplay_opp_idxs
        if _combined_net_idxs:
            _t0 = time.time()
            _run_network_turn(_combined_net_idxs)
            _t_net += time.time() - _t0

        # Submit selfplay SF queries immediately after network turn
        # (boards now have the move pushed).  These run in the SF pool
        # while we collect curriculum results below.
        _sf_sp_futures: dict[int, Any] | None = None
        if selfplay_opp_idxs and isinstance(state.stockfish, StockfishPool):
            _t0 = time.time()
            _sf_sp_futures = _submit_sf_queries(selfplay_opp_idxs)
            _t_sf += time.time() - _t0

        # Collect curriculum SF results (overlapped with combined net turn)
        if curriculum_opp_idxs:
            _t0 = time.time()
            _finish_sf_annotation_and_moves(curriculum_opp_idxs, play_curriculum_moves=True, futures=_sf_futures)
            _t_sf += time.time() - _t0

        # Collect selfplay SF move results (submitted above, overlapped with curriculum)
        if selfplay_opp_idxs:
            _t0 = time.time()
            _finish_sf_annotation_and_moves(selfplay_opp_idxs, play_curriculum_moves=True, futures=_sf_sp_futures)
            _t_sf += time.time() - _t0
            # Submit label queries immediately; collect after (overlaps with
            # finalization below for negligible extra latency).
            selfplay_label_idxs = [i for i in selfplay_opp_idxs if not state.done_arr[i]]
            _sf_label_futures: dict[int, Any] | None = None
            if selfplay_label_idxs and isinstance(state.stockfish, StockfishPool):
                _t0 = time.time()
                _sf_label_futures = _submit_sf_queries(selfplay_label_idxs)
                _t_sf += time.time() - _t0
            if selfplay_label_idxs:
                _t0 = time.time()
                _finish_sf_annotation_and_moves(selfplay_label_idxs, play_curriculum_moves=False, futures=_sf_label_futures)
                _t_sf += time.time() - _t0

        # Finalize completed games and optionally recycle slots
        for i in range(batch_size):
            if state.done_arr[i] and not state.finalized_arr[i]:
                finalize_game(state, i, all_samples, on_game_complete)
                state.finalized_arr[i] = 1
                state.games_completed += 1
                if continuous or state.games_started < target:
                    state.recycle_slot(i)

        # Reset tree when it gets too large and no roots reference old nodes
        if state.mcts_tree is not None and state.mcts_tree.node_count() > 500_000:
            if all(rid < 0 for rid in state.root_ids):
                state.mcts_tree.reset()

    # ── Timing summary ─────────────────────────────────────────────────────────
    logging.getLogger("chess_anti_engine.worker").info(
        "play_batch timing: net=%.1fs sf=%.1fs (net %.0f%%, sf %.0f%%)",
        _t_net, _t_sf,
        _t_net / max(0.001, _t_net + _t_sf) * 100,
        _t_sf / max(0.001, _t_net + _t_sf) * 100,
    )
    if state.nn_cache is not None:
        _nc_stats = state.nn_cache.stats()
        _nc_total = _nc_stats["hits"] + _nc_stats["misses"]
        _nc_hit_rate = _nc_stats["hits"] / _nc_total if _nc_total > 0 else 0.0
        logging.getLogger("chess_anti_engine.worker").info(
            "nncache: hits=%d misses=%d (hit_rate=%.1f%%)  "
            "inserts=%d collisions=%d  count=%d/%d (%.1f%% full)",
            _nc_stats["hits"], _nc_stats["misses"], 100.0 * _nc_hit_rate,
            _nc_stats["inserts"], _nc_stats["insert_collisions"],
            _nc_stats["count"], _nc_stats["cap"],
            100.0 * _nc_stats["count"] / max(1, _nc_stats["cap"]),
        )

    # ── Return results ────────────────────────────────────────────────────────
    sf_nodes = int(getattr(stockfish, "nodes", 0) or 0)
    return all_samples, state.stats.to_batch_stats(
        games=state.games_completed,
        positions=len(all_samples),
        sf_nodes=sf_nodes if sf_nodes > 0 else None,
    )
