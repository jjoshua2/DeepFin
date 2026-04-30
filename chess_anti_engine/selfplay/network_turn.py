"""Run one network turn for a batch of network-assigned slots.

The C fast paths (``batch_encode_146``, ``batch_process_ply``,
``temperature_resample``) consume plain Python lists of ``CBoard``
objects + numpy arrays by reference; the zero-copy inplace path writes
into the evaluator's pinned buffer. Torch-compile shape-stability
buckets are kept as a module constant so the evaluator sees a stable
set of batch shapes.

``_run_mcts_group`` is nested inside ``run_network_turn`` so it can
capture the large local pol/wdl logit matrices without a parameter
explosion.
"""

from __future__ import annotations

import math
from typing import cast

import chess
import numpy as np

from chess_anti_engine.mcts import GumbelConfig, MCTSConfig
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.mcts.puct import run_mcts_many

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

from chess_anti_engine.moves import POLICY_SIZE, index_to_move, legal_move_mask
from chess_anti_engine.selfplay.state import (
    SOFT_RESIGN_CONSECUTIVE,
    SOFT_RESIGN_THRESHOLD,
    SelfplayState,
    _NetRecord,
)
from chess_anti_engine.selfplay.temperature import temperature_for_ply


# torch.compile shape-stability buckets.
_ROOT_BUCKETS: tuple[int, ...] = (32, 64, 128, 256, 512)


def _padded_batch_size(bsz: int) -> int:
    """Round ``bsz`` up to the next torch.compile-friendly bucket."""
    for b in _ROOT_BUCKETS:
        if b >= bsz:
            return b
    return bsz


def _apply_forced_moves(state: SelfplayState, net_idxs: list[int]) -> list[int]:
    """Push the only-legal-move for each forced game and return remaining indices.

    Forced-move (only one legal move) games skip both NN eval and MCTS — the
    policy target would collapse to one-hot regardless and the value head
    can't compare alternatives. ``state.boards`` is only kept in sync with
    ``state.cboards`` on the Python fallback path; the C-ply fast path
    replays history from ``move_idx_history`` instead.
    """
    forced_idxs: list[int] = []
    forced_actions: list[int] = []
    play_idxs: list[int] = []
    for idx in net_idxs:
        legal = state.cboards[idx].legal_move_indices()
        if legal.size == 1:
            forced_idxs.append(idx)
            forced_actions.append(int(legal[0]))
        else:
            play_idxs.append(idx)
    if not forced_idxs:
        return play_idxs
    for idx, act in zip(forced_idxs, forced_actions, strict=True):
        state.cboards[idx].push_index(act)
        if not state.has_c_ply:
            state.boards[idx].push(index_to_move(act, state.boards[idx]))
        state.move_idx_history[idx].append(act)
        state.last_net_full[idx] = True
        if state.cboards[idx].is_game_over():
            state.done_arr[idx] = 1
  # No tree carryover — without an MCTS root expansion there's nothing to reuse.
        if state.mcts_tree is not None:
            state.root_ids[idx] = -1
    return play_idxs


def _evaluate_root_batch(
    state: SelfplayState, net_idxs: list[int],
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Encode + run one batched root NN eval.

    Returns ``(xs_batch_or_None, pol_logits, wdl_logits_raw, wdl_softmax)``.
    The fast path encodes into the evaluator's pinned input buffer (zero-copy
    H2D), in which case ``xs_batch`` is None — callers that need per-game x
    arrays must use the fallback path.
    """
    eval_impl = state.evaluator
    bsz = len(net_idxs)
    padded_bsz = _padded_batch_size(bsz)
    cb_encode_list = [state.cboards[idx] for idx in net_idxs]

    use_inplace = state.batch_enc_146 is not None and hasattr(eval_impl, "get_input_buffer")
    if use_inplace:
  # evaluate_inplace + get_input_buffer exist on DirectGPU-style evaluators only.
        buf = eval_impl.get_input_buffer(padded_bsz)  # pyright: ignore[reportAttributeAccessIssue]
        assert state.batch_enc_146 is not None
        state.batch_enc_146(cb_encode_list, buf)
        pol_padded, wdl_padded = eval_impl.evaluate_inplace(  # pyright: ignore[reportAttributeAccessIssue]
            padded_bsz, copy_out=True,
        )
        xs_batch: np.ndarray | None = None
    else:
        xs_batch = np.empty((bsz, 146, 8, 8), dtype=np.float32)
        if state.batch_enc_146 is not None:
            state.batch_enc_146(cb_encode_list, xs_batch)
        else:
            for j, idx in enumerate(net_idxs):
                xs_batch[j] = state.cboards[idx].encode_146()
        if padded_bsz > bsz:
            pad = np.zeros((padded_bsz - bsz, *xs_batch.shape[1:]), dtype=xs_batch.dtype)
            xs_padded = np.concatenate([xs_batch, pad], axis=0)
        else:
            xs_padded = xs_batch
        pol_padded, wdl_padded = eval_impl.evaluate_encoded(xs_padded)

    pol_logits = pol_padded[:bsz]
    wdl_logits_raw = wdl_padded[:bsz]
  # Pure numpy softmax avoids torch tensor roundtrip for small arrays.
    wdl_f = wdl_logits_raw.astype(np.float64, copy=True)
    wdl_f -= wdl_f.max(axis=-1, keepdims=True)
    np.exp(wdl_f, out=wdl_f)
    wdl_f /= wdl_f.sum(axis=-1, keepdims=True)
    return xs_batch, pol_logits, wdl_logits_raw, wdl_f.astype(np.float32, copy=False)


def _compute_resign_weights(
    state: SelfplayState, wdl_est: np.ndarray, net_idxs: list[int],
) -> list[float]:
    """Soft-resign sample weights based on consecutive low-winrate plies."""
    sample_weights = [1.0] * len(net_idxs)
    for j, idx in enumerate(net_idxs):
        win_p = float(wdl_est[j][0])
        if win_p < SOFT_RESIGN_THRESHOLD:
            state.consecutive_low_winrate[idx] += 1
        else:
            state.consecutive_low_winrate[idx] = 0
        if state.consecutive_low_winrate[idx] >= SOFT_RESIGN_CONSECUTIVE:
            sample_weights[j] = 0.1 + 0.9 * (win_p / SOFT_RESIGN_THRESHOLD)
    return sample_weights


def run_network_turn(state: SelfplayState, net_idxs: list[int]) -> None:
    """Run one network turn for the games in ``net_idxs``.

    Side effects:
    * Advances ``state.cboards[idx]`` (and ``state.boards[idx]`` in the
      Python fallback) with the chosen move.
    * Appends an ``_NetRecord`` to ``state.samples_per_game[idx]``.
    * Appends the chosen policy index to ``state.move_idx_history[idx]``.
    * Updates ``state.consecutive_low_winrate``, ``state.last_net_full``,
      ``state.root_ids``, and ``state.done_arr``.
    """
    if not net_idxs:
        return

    net_idxs = _apply_forced_moves(state, net_idxs)
    if not net_idxs:
        return

    eval_impl = state.evaluator
    rng = state.rng
    search = state.search
    temp = state.temp
    diff_focus = state.diff_focus

    xs_batch, pol_logits, wdl_logits_raw, wdl_est = _evaluate_root_batch(state, net_idxs)
    _cb_encode_list = [state.cboards[_idx] for _idx in net_idxs]

    is_full = rng.random(size=len(net_idxs)) < float(search.playout_cap_fraction)
    full_sims = int(search.simulations)
    fast_sims = int(search.fast_simulations)
    sample_weights = _compute_resign_weights(state, wdl_est, net_idxs)

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

    def _run_mcts_group(
        idxs: list[int], sims_per: list[int], *, per_game_noise: list[bool] | None = None,
    ) -> None:
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
                device=state.device,
                rng=rng,
                cfg=GumbelConfig(simulations=int(sim_count), temperature=1.0, add_noise=True),
                evaluator=eval_impl,
                pre_pol_logits=sub_pol,
                pre_wdl_logits=sub_wdl,
                per_game_simulations=sub_sims,
                per_game_add_noise=sub_noise,
                cboards=sub_cboards,
                tree=state.mcts_tree,
                root_node_ids=sub_root_ids,
                tb_probe=state.tb_probe if state.game.syzygy_in_search else None,
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
            # current state from move_idx_history (handled inside replay_board).
            sub_boards = [state.replay_board(net_idxs[j]) for j in group]
            _puct_fn = _run_mcts_many_c if _HAS_C_TREE else run_mcts_many
            p_sub, a_sub, v_sub, m_sub = _puct_fn(
                state.model,
                sub_boards,
                device=state.device,
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
    is_full_py = is_full.tolist()
    combined_sims = [full_sims if is_full_py[j] else fast_sims for j in all_idxs]
    _run_mcts_group(all_idxs, combined_sims, per_game_noise=is_full_py)

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
        _actions_arr = np.array(actions, dtype=np.int32)
        _values_arr = np.array(values_list, dtype=np.float64)
        # All slots filled by _run_mcts_group above; cast for np.stack's strict ArrayLike protocol.
        _probs_arr = np.stack(cast("list[np.ndarray]", probs_list)).astype(np.float32, copy=False)

        assert state.c_process_ply is not None
        (c_x, c_probs, c_wdl_net, c_wdl_search, c_priority,
         c_keep, c_mask, c_ply, c_pov, c_over) = state.c_process_ply(
            _cb_encode_list, pol_logits[:_n], wdl_logits_raw[:_n],
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
        _is_full_list = is_full_py
        _sw_list = sample_weights
        _act_list = _actions_arr.tolist()

        for j in range(_n):
            idx = net_idxs[j]
            state.move_idx_history[idx].append(_act_list[j])
            state.last_net_full[idx] = _is_full_list[j]

            # Skip training on positions with a single legal move: the policy
            # target collapses to one-hot regardless of search, so there's no
            # learning signal. Same shape as low-sim filtering (has_policy=False).
            _has_policy = bool(_is_full_list[j]) and int(c_mask[j].sum()) > 1
            state.samples_per_game[idx].append(
                _NetRecord(
                    c_x[j], c_probs[j], c_wdl_net[j], c_wdl_search[j],
                    chess.WHITE if _c_pov_list[j] else chess.BLACK,
                    _c_ply_list[j], _has_policy,
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
            # ⇒ ``xs_batch`` was set in the else-branch above.
            assert xs_batch is not None
            state.samples_per_game[idx].append(
                _NetRecord(
                    x=xs_batch[j],
                    policy_probs=probs,
                    net_wdl_est=wdl_est[j] if np.all(np.isfinite(wdl_est[j])) else np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    search_wdl_est=search_wdl_est,
                    pov_color=pov_color,
                    ply_index=ply_index,
                    has_policy=bool(is_full[j]) and int(mask.sum()) > 1,
                    priority=float(difficulty),
                    sample_weight=float(sample_weights[j]),
                    keep_prob=float(keep_prob),
                    legal_mask=mask.view(np.uint8),
                ),
            )

            if state.cboards[idx].is_game_over():
                state.done_arr[idx] = 1


__all__ = ["run_network_turn"]
