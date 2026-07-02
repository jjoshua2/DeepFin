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
from typing import Any, cast

import chess
import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.mcts._mcts_tree import batch_compute_relations
from chess_anti_engine.encoding.features import extra_feature_plane_count
from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_ROOT,
    LC0_HISTORY_ROOT_LEGACY_META,
    c_input_history_mode,
    normalize_lc0_history_encoding,
    uses_lc0_root_history,
)
from chess_anti_engine.mcts import GumbelConfig, MCTSConfig
from chess_anti_engine.mcts.gumbel import (
    run_gumbel_root_many,
    volatility_search_enabled,
    warn_volatility_python_path,
)
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.mcts.sampling import sample_action_with_temperature

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
    _run_gumbel_root_many_c = None
    _HAS_GUMBEL_C = False

from chess_anti_engine.moves import (
    POLICY_SIZE,
    index_to_move,
    legal_move_mask,
    policy_batch_to_full_if_needed,
)
from chess_anti_engine.selfplay.state import (
    SOFT_RESIGN_CONSECUTIVE,
    SOFT_RESIGN_THRESHOLD,
    SelfplayState,
    _NetRecord,
)
from chess_anti_engine.selfplay.temperature import temperature_for_ply


# torch.compile shape-stability buckets.
_ROOT_BUCKETS: tuple[int, ...] = (32, 64, 128, 256, 512)


def _batch_encoder_pair(state: SelfplayState):
    hist_enc = normalize_lc0_history_encoding(state.game.input_history_encoding)
    if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
        return state.batch_enc_146_lc0_root_legacy_meta, state.batch_enc_146_lc0_root_legacy_meta_bf16
    if uses_lc0_root_history(hist_enc):
        return state.batch_enc_146_lc0_root, state.batch_enc_146_lc0_root_bf16
    return state.batch_enc_146, state.batch_enc_146_bf16


def _padded_batch_size(bsz: int) -> int:
    """Round ``bsz`` up to the next torch.compile-friendly bucket."""
    for b in _ROOT_BUCKETS:
        if b >= bsz:
            return b
    return bsz


def _resample_actions_with_temperature(
    probs_list: list[np.ndarray],
    actions: list[int],
    temps: list[float],
    rng: np.random.Generator,
    *,
    c_temp_resample: Any | None,
    use_c_resample: bool,
    preserve_zero_temperature_actions: bool,
) -> None:
    """Apply final move temperature to search policies in-place.

    Gumbel's temperature-0 action is the sequential-halving survivor, not the
    argmax of the returned improved policy. ``preserve_zero_temperature_actions``
    keeps that survivor and only resamples when the requested temperature is
    positive.
    """
    temps_arr = np.array(temps, dtype=np.float64)
    if preserve_zero_temperature_actions:
        need_resample = temps_arr > 0.0
    else:
        need_resample = temps_arr != 1.0
    if not need_resample.any():
        return

    p_stack = np.stack(probs_list)  # (G, 4672)
    if use_c_resample and not preserve_zero_temperature_actions:
        # The C helper treats t == 1 as a no-op, which is correct for PUCT
        # because its search call already sampled at temperature 1. Gumbel
        # uses the Python path so t == 1 can sample from the improved policy
        # after starting from the halving survivor.
        actions_arr = np.array(actions, dtype=np.int32)
        rand_arr = rng.random(len(temps))
        assert c_temp_resample is not None
        c_temp_resample(p_stack, temps_arr, actions_arr, rand_arr)
        for gi in range(len(actions)):
            actions[gi] = int(actions_arr[gi])
        return

    for gi in need_resample.nonzero()[0]:
        p = p_stack[gi]
        t = float(temps_arr[gi])
        legal = np.flatnonzero(p > 0)
        if len(legal) == 0:
            continue
        if preserve_zero_temperature_actions:
            matches = np.flatnonzero(legal == int(actions[gi]))
            fallback_idx = int(matches[0]) if matches.size else 0
        else:
            fallback_idx = int(np.argmax(p[legal]))
        actions[gi] = sample_action_with_temperature(
            rng,
            legal,
            p[legal],
            t,
            argmax_idx=fallback_idx,
        )


def _refresh_gumbel_action_diag(
    diag: dict[str, float] | None,
    probs: np.ndarray,
    action: int,
) -> dict[str, float] | None:
    """Update action-dependent diagnostics after final temperature resampling."""
    if diag is None:
        return None
    p = np.asarray(probs, dtype=np.float64)
    legal = np.flatnonzero(p > 0.0)
    if legal.size == 0:
        return diag
    legal_p = np.maximum(p[legal], 0.0)
    total = float(legal_p.sum(dtype=np.float64))
    if total <= 0.0 or not math.isfinite(total):
        return diag
    legal_p = legal_p / total
    top_action = int(legal[int(np.argmax(legal_p))])
    out = dict(diag)
    out["action_prob"] = max(0.0, float(p[int(action)])) if 0 <= int(action) < p.shape[0] else 0.0
    out["argmax_is_action"] = 1.0 if top_action == int(action) else 0.0
    return out


def _scheduled_scale(
    *,
    start: float,
    after: float,
    decay_start: int,
    decay_moves: int,
    move_number: int,
) -> float:
    if decay_start <= 0 or decay_moves <= 0 or move_number < decay_start:
        return start
    if move_number >= decay_start + decay_moves:
        return after
    frac = float(move_number - decay_start) / float(decay_moves)
    return start + (after - start) * frac


def _scheduled_gumbel_scale(search, *, move_number: int, curriculum: bool = False) -> float:
    if curriculum:
        return _scheduled_scale(
            start=float(search.curriculum_gumbel_scale),
            after=float(search.curriculum_gumbel_scale_after),
            decay_start=int(search.curriculum_gumbel_scale_decay_start_move),
            decay_moves=int(search.curriculum_gumbel_scale_decay_moves),
            move_number=move_number,
        )
    return _scheduled_scale(
        start=float(search.gumbel_scale),
        after=float(search.gumbel_scale_after),
        decay_start=int(search.gumbel_scale_decay_start_move),
        decay_moves=int(search.gumbel_scale_decay_moves),
        move_number=move_number,
    )


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
    hist_enc = normalize_lc0_history_encoding(state.game.input_history_encoding)
    batch_enc, batch_enc_bf16 = _batch_encoder_pair(state)

    use_inplace = batch_enc is not None and hasattr(eval_impl, "get_input_buffer")
    use_bf16_input = (
        state.has_c_ply
        and batch_enc_bf16 is not None
        and bool(getattr(eval_impl, "supports_input_bf16_bits", False))
    )
    if use_inplace:
  # evaluate_inplace + get_input_buffer exist on DirectGPU-style evaluators only.
        if use_bf16_input and hasattr(eval_impl, "get_input_buffer_bf16_bits"):
            buf = eval_impl.get_input_buffer_bf16_bits(padded_bsz)  # pyright: ignore[reportAttributeAccessIssue]
            assert batch_enc_bf16 is not None
            batch_enc_bf16(cb_encode_list, buf)
        else:
            buf = eval_impl.get_input_buffer(padded_bsz)  # pyright: ignore[reportAttributeAccessIssue]
            assert batch_enc is not None
            batch_enc(cb_encode_list, buf)
        if state.game.record_relations:
            _rel = np.zeros((padded_bsz, 5, 64, 64), dtype=np.uint8)
            batch_compute_relations(cb_encode_list, _rel)
            pol_padded, wdl_padded = eval_impl.evaluate_inplace(  # pyright: ignore[reportAttributeAccessIssue]
                padded_bsz, copy_out=True, relations=_rel,
            )
        else:
            pol_padded, wdl_padded = eval_impl.evaluate_inplace(  # pyright: ignore[reportAttributeAccessIssue]
                padded_bsz, copy_out=True,
            )
        xs_batch: np.ndarray | None = None
    else:
        dtype = np.uint16 if use_bf16_input else np.float32
        xs_batch = np.empty(
            (bsz, input_plane_count(state.game.input_extra_features), 8, 8), dtype=dtype,
        )
        if use_bf16_input:
            assert batch_enc_bf16 is not None
            batch_enc_bf16(cb_encode_list, xs_batch)
        elif batch_enc is not None:
            batch_enc(cb_encode_list, xs_batch)
        else:
            for j, idx in enumerate(net_idxs):
                xs_batch[j] = encode_cboard(
                    state.cboards[idx],
                    input_history_encoding=hist_enc,
                    input_extra_features=state.game.input_extra_features,
                )
        if padded_bsz > bsz:
            pad = np.zeros((padded_bsz - bsz, *xs_batch.shape[1:]), dtype=xs_batch.dtype)
            xs_padded = np.concatenate([xs_batch, pad], axis=0)
        else:
            xs_padded = xs_batch
        if state.game.record_relations:
            _rel = np.zeros((padded_bsz, 5, 64, 64), dtype=np.uint8)
            batch_compute_relations(cb_encode_list, _rel)
            pol_padded, wdl_padded = eval_impl.evaluate_encoded(xs_padded, relations=_rel)
        else:
            pol_padded, wdl_padded = eval_impl.evaluate_encoded(xs_padded)
        if use_bf16_input:
            xs_batch = None

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


def _expand_policy_logits_for_ply(
    pol_logits: np.ndarray,
    *,
    policy_encoding: str,
) -> np.ndarray:
    pol = np.asarray(pol_logits, dtype=np.float32)
    return policy_batch_to_full_if_needed(pol, policy_encoding=policy_encoding, fill_value=-1e9)


def _append_records_via_c(
    state: SelfplayState, net_idxs: list[int],
    *, cb_encode_list, pol_logits: np.ndarray, wdl_logits_raw: np.ndarray,
    actions: list[int | None], values_list: list[float | None],
    probs_list: list[np.ndarray | None],
    gumbel_diags: list[dict[str, float] | None],
    is_full_py: list[bool], sample_weights: list[float],
    diff_focus,
) -> None:
    """C-fast-path per-ply append (GIL released inside ``c_process_ply``)."""
    n = len(net_idxs)
    actions_arr = np.array(actions, dtype=np.int32)
    values_arr = np.array(values_list, dtype=np.float64)
    # All slots filled by _run_mcts_group above; cast for np.stack's strict ArrayLike protocol.
    probs_arr = np.stack(cast("list[np.ndarray]", probs_list)).astype(np.float32, copy=False)
    pol_logits_full = _expand_policy_logits_for_ply(
        pol_logits,
        policy_encoding=state.game.policy_encoding,
    )

    assert state.c_process_ply is not None
    alt_lc0_root_xs = None
    if (
        bool(state.game.record_lc0_root_input)
        and not uses_lc0_root_history(state.game.input_history_encoding)
    ):
        _n_extra = extra_feature_plane_count(state.game.input_extra_features)
        alt_lc0_root_xs = [cb.encode_full(1, _n_extra) for cb in cb_encode_list]
    _want_rel = bool(state.game.record_relations)
    c_result = state.c_process_ply(
        cb_encode_list, pol_logits_full[:n], wdl_logits_raw[:n],
        actions_arr, values_arr, probs_arr,
        int(diff_focus.enabled), float(diff_focus.q_weight),
        float(diff_focus.pol_scale), float(diff_focus.min_keep), float(diff_focus.slope),
        c_input_history_mode(state.game.input_history_encoding),
        extra_feature_plane_count(state.game.input_extra_features),
        int(_want_rel),
    )
    c_rel = c_result[12] if _want_rel else None
    (c_x, c_probs, c_wdl_net, c_wdl_search, c_priority,
     c_priority_policy_kl, c_priority_q_delta,
     c_keep, c_mask, c_ply, c_pov, c_over) = c_result[:12]

    # Pre-extract Python scalars from numpy arrays (batch conversion is cheaper
    # than per-element conversion in the loop).
    c_ply_list = c_ply.tolist()
    c_pov_list = c_pov.tolist()
    c_priority_list = c_priority.tolist()
    c_priority_policy_kl_list = (
        c_priority_policy_kl.tolist()
        if isinstance(c_priority_policy_kl, np.ndarray)
        else list(c_priority_policy_kl)
    )
    c_priority_q_delta_list = (
        c_priority_q_delta.tolist()
        if isinstance(c_priority_q_delta, np.ndarray)
        else list(c_priority_q_delta)
    )
    c_keep_list = c_keep.tolist()
    c_over_list = c_over.tolist()
    act_list = actions_arr.tolist()

    for j in range(n):
        idx = net_idxs[j]
        state.move_idx_history[idx].append(act_list[j])
        state.last_net_full[idx] = is_full_py[j]

        # Skip training on positions with a single legal move: the policy
        # target collapses to one-hot regardless of search, so there's no
        # learning signal. Same shape as low-sim filtering (has_policy=False).
        has_policy = bool(is_full_py[j]) and int(c_mask[j].sum()) > 1
        state.samples_per_game[idx].append(
            _NetRecord(
                c_x[j], c_probs[j], c_wdl_net[j], c_wdl_search[j],
                chess.WHITE if c_pov_list[j] else chess.BLACK,
                c_ply_list[j], has_policy,
                c_priority_list[j], sample_weights[j], c_keep_list[j],
                c_mask[j],
                x_lc0_root=(None if alt_lc0_root_xs is None else alt_lc0_root_xs[j]),
                relations=(None if c_rel is None else c_rel[j]),
                priority_policy_kl=c_priority_policy_kl_list[j],
                priority_q_delta=c_priority_q_delta_list[j],
                gumbel_policy_diag=gumbel_diags[j],
            ),
        )

        if c_over_list[j]:
            state.done_arr[idx] = 1


def _append_records_via_python(
    state: SelfplayState, net_idxs: list[int],
    *, xs_batch: np.ndarray | None,
    pol_logits: np.ndarray, wdl_est: np.ndarray,
    probs_list: list[np.ndarray | None], actions: list[int | None],
    values_list: list[float | None],
    gumbel_diags: list[dict[str, float] | None],
    masks_list: list[np.ndarray | None],
    is_full: np.ndarray, sample_weights: list[float],
    diff_focus,
) -> None:
    """Python fallback per-ply loop (only runs when C ply path is unavailable)."""
    lg_buf = np.empty(POLICY_SIZE, dtype=np.float64)
    swdl_buf = np.empty(3, dtype=np.float32)
    df_enabled = bool(diff_focus.enabled)
    df_q_w = float(diff_focus.q_weight)
    df_p_s = float(diff_focus.pol_scale)
    df_slope = float(diff_focus.slope)
    df_min = float(diff_focus.min_keep)
    pol_logits_full = _expand_policy_logits_for_ply(
        pol_logits,
        policy_encoding=state.game.policy_encoding,
    )

    # ``not state.has_c_ply`` ⇒ ``state.batch_enc_146 is None``
    # (coupled import in SelfplayState.create) ⇒ ``_use_inplace is False``
    # ⇒ ``xs_batch`` was set in the else-branch of _evaluate_root_batch.
    assert xs_batch is not None

    for j, (idx, probs, a, v) in enumerate(
        zip(net_idxs, probs_list, actions, values_list, strict=True),
    ):
        assert probs is not None
        assert a is not None
        assert v is not None

        board_before = state.boards[idx]
        ply_index = len(board_before.move_stack)
        pov_color = board_before.turn
        x_lc0_root = (
            encode_position(
                board_before, add_features=True,
                input_history_encoding=LC0_HISTORY_ROOT,
                input_extra_features=state.game.input_extra_features,
            )
            if (
                bool(state.game.record_lc0_root_input)
                and not uses_lc0_root_history(state.game.input_history_encoding)
            )
            else None
        )

        mask = masks_list[j]
        if mask is None:
            mask = legal_move_mask(board_before)

        np.copyto(lg_buf, pol_logits_full[j])
        lg_buf[~mask] = -1e9
        lg_buf -= float(np.max(lg_buf))
        np.exp(lg_buf, out=lg_buf)
        lg_buf[~mask] = 0.0
        s = float(lg_buf.sum())
        if s > 0:
            raw = (lg_buf / s).astype(np.float32, copy=False)
        else:
            raw = mask.astype(np.float32) / float(mask.sum())

        imp = np.maximum(probs.astype(np.float32, copy=False), 1e-12)
        raw_c = np.maximum(raw, 1e-12)
        kl = float(np.sum(raw_c * (np.log(raw_c) - np.log(imp))))

        orig_q = float(wdl_est[j][0] - wdl_est[j][2])
        best_q = float(v)
        q_delta = best_q - orig_q
        q_surprise = abs(q_delta)

        difficulty = q_surprise * df_q_w + kl * df_p_s
        if not math.isfinite(difficulty):
            difficulty = 1.0
        keep_prob = (
            max(df_min, min(1.0, difficulty * df_slope)) if df_enabled else 1.0
        )

        move = index_to_move(int(a), board_before)
        board_before.push(move)
        state.cboards[idx].push_index(int(a))
        state.move_idx_history[idx].append(int(a))

        d_raw = float(wdl_est[j][1])
        rem = max(0.0, 1.0 - d_raw)
        q = float(max(-rem, min(rem, best_q)))
        w_search = 0.5 * (rem + q)
        swdl_buf[0] = w_search
        swdl_buf[1] = d_raw
        swdl_buf[2] = rem - w_search
        search_wdl_est = swdl_buf.copy()
        if not np.all(np.isfinite(search_wdl_est)):
            search_wdl_est = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        state.last_net_full[idx] = bool(is_full[j])
        state.samples_per_game[idx].append(
            _NetRecord(
                x=xs_batch[j],
                policy_probs=probs,
                net_wdl_est=(
                    wdl_est[j] if np.all(np.isfinite(wdl_est[j]))
                    else np.array([0.0, 1.0, 0.0], dtype=np.float32)
                ),
                search_wdl_est=search_wdl_est,
                pov_color=pov_color,
                ply_index=ply_index,
                has_policy=bool(is_full[j]) and int(mask.sum()) > 1,
                priority=float(difficulty),
                sample_weight=float(sample_weights[j]),
                keep_prob=float(keep_prob),
                legal_mask=mask.view(np.uint8),
                x_lc0_root=x_lc0_root,
                priority_policy_kl=float(kl),
                priority_q_delta=float(q_delta),
                gumbel_policy_diag=gumbel_diags[j],
            ),
        )

        if state.cboards[idx].is_game_over():
            state.done_arr[idx] = 1


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
    gumbel_diags: list[dict[str, float] | None] = [None] * len(net_idxs)

    # Per-game action temperature based on each game's own ply count.  Pure
    # selfplay can keep exploration while SF-curriculum games stay sharp.
    temps = [
        temperature_for_ply(
            ply=state.cboards[i].ply // 2 + 1,
            temperature=(
                float(temp.selfplay_temperature)
                if bool(state.selfplay_arr[i]) and temp.selfplay_temperature is not None
                else float(temp.temperature)
            ),
            drop_plies=int(temp.drop_plies),
            after=float(temp.after),
            decay_start_move=(
                int(temp.selfplay_decay_start_move)
                if bool(state.selfplay_arr[i]) and temp.selfplay_decay_start_move is not None
                else int(temp.decay_start_move)
            ),
            decay_moves=(
                int(temp.selfplay_decay_moves)
                if bool(state.selfplay_arr[i]) and temp.selfplay_decay_moves is not None
                else int(temp.decay_moves)
            ),
            endgame=(
                float(temp.selfplay_endgame)
                if bool(state.selfplay_arr[i]) and temp.selfplay_endgame is not None
                else float(temp.endgame)
            ),
        )
        for i in net_idxs
    ]
    gumbel_scales = [
        _scheduled_gumbel_scale(
            search,
            move_number=state.cboards[i].ply // 2 + 1,
            curriculum=not bool(state.selfplay_arr[i]),
        )
        for i in net_idxs
    ]

    def _run_mcts_group(
        idxs: list[int],
        sims_per: list[int],
        *,
        per_game_noise: list[bool] | None = None,
        per_game_gumbel_scale: list[float] | None = None,
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
        diag_sub: list[dict[str, float] | None]

        if use_gumbel:
            sub_noise = [per_game_noise[j] for j in group] if per_game_noise is not None else None
            sub_scale = [
                float(per_game_gumbel_scale[j])
                for j in group
            ] if per_game_gumbel_scale is not None else [float(search.gumbel_scale) for _ in group]
            gumbel_cfg = GumbelConfig(
                simulations=int(sim_count),
                topk=int(search.gumbel_topk),
                temperature=0.0,
                c_scale=float(search.gumbel_c_scale),
                add_noise=True,
                gumbel_scale=1.0,
                input_history_encoding=state.game.input_history_encoding,
                input_extra_features=state.game.input_extra_features,
                policy_encoding=state.game.policy_encoding,
                compute_relations=bool(state.game.record_relations),
                volatility_q_scale=float(search.volatility_q_scale),
                volatility_fpu=float(search.volatility_fpu),
                volatility_anchor=float(search.volatility_anchor),
            )
            if volatility_search_enabled(gumbel_cfg):
                warn_volatility_python_path()
            used_gumbel_c = _HAS_GUMBEL_C and not volatility_search_enabled(gumbel_cfg)
            if used_gumbel_c:
                # Map group indices to game-level root IDs for tree reuse.
                sub_root_ids = [state.root_ids[net_idxs[j]] for j in group] if state.mcts_tree is not None else None
                gumbel_c_fn = cast(Any, _run_gumbel_root_many_c)
                _gumbel_result = gumbel_c_fn(
                    state.model,
                    sub_cboards,
                    device=state.device,
                    rng=rng,
                    cfg=gumbel_cfg,
                    evaluator=eval_impl,
                    pre_pol_logits=sub_pol,
                    pre_wdl_logits=sub_wdl,
                    per_game_simulations=sub_sims,
                    per_game_add_noise=sub_noise,
                    per_game_gumbel_scale=sub_scale,
                    cboards=sub_cboards,
                    tree=state.mcts_tree,
                    root_node_ids=sub_root_ids,
                    tb_probe=state.tb_probe if state.game.syzygy_in_search else None,
                    allow_terminal_root_shortcuts=True,
                    return_diagnostics=True,
                )
            else:
                sub_boards = [state.replay_board(net_idxs[j]) for j in group]
                _gumbel_result = run_gumbel_root_many(
                    state.model,
                    sub_boards,
                    device=state.device,
                    rng=rng,
                    cfg=gumbel_cfg,
                    evaluator=eval_impl,
                    pre_pol_logits=sub_pol,
                    pre_wdl_logits=sub_wdl,
                    per_game_simulations=sub_sims,
                    per_game_add_noise=sub_noise,
                    per_game_gumbel_scale=sub_scale,
                    return_diagnostics=True,
                )
            # C diagnostics return tree/root ids before diagnostics; Python diagnostics do not.
            p_sub, a_sub, v_sub, m_sub = _gumbel_result[:4]
            if used_gumbel_c and len(_gumbel_result) >= 7:
                diag_sub = cast(list[dict[str, float] | None], _gumbel_result[6])
            elif (not used_gumbel_c) and len(_gumbel_result) >= 5:
                diag_sub = cast(list[dict[str, float] | None], _gumbel_result[4])
            else:
                diag_sub = [None] * len(group)
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
                    input_history_encoding=state.game.input_history_encoding,
                    input_extra_features=state.game.input_extra_features,
                    policy_encoding=state.game.policy_encoding,
                    compute_relations=bool(state.game.record_relations),
                ),
                evaluator=eval_impl,
                pre_pol_logits=sub_pol,
                pre_wdl_logits=sub_wdl,
                cboards=sub_cboards,
            )
            diag_sub = [None] * len(group)

        # Re-select actions with per-game temperature from the returned policy.
        # For Gumbel, temperature <= 0 must preserve the sequential-halving
        # survivor returned by the search instead of taking policy argmax.
        _resample_actions_with_temperature(
            p_sub,
            a_sub,
            sub_temps,
            rng,
            c_temp_resample=state.c_temp_resample,
            use_c_resample=bool(state.has_classify_c),
            preserve_zero_temperature_actions=bool(use_gumbel),
        )

        for local_j, (jj, p, a, v, m) in enumerate(zip(group, p_sub, a_sub, v_sub, m_sub, strict=True)):
            probs_list[jj] = p
            actions[jj] = a
            values_list[jj] = float(v)
            masks_list[jj] = m
            gumbel_diags[jj] = (
                _refresh_gumbel_action_diag(diag_sub[local_j], p, int(a))
                if use_gumbel else diag_sub[local_j]
            )
            # Advance tree root to chosen move's child for next-ply reuse
            if state.mcts_tree is not None:
                game_idx = net_idxs[jj]
                rid = state.root_ids[game_idx]
                if rid >= 0:
                    child = state.mcts_tree.find_child(rid, int(a))
                    state.root_ids[game_idx] = child  # -1 if not found

    # Run all games in one MCTS call with per-game sim budgets and noise.
    # Full-sim games get configured Gumbel noise for exploration. Fast games
    # remain deterministic (KataGo playout cap convention).
    all_idxs = list(range(len(net_idxs)))
    is_full_py = is_full.tolist()
    combined_sims = [full_sims if is_full_py[j] else fast_sims for j in all_idxs]
    scale_py = [
        float(gumbel_scales[j]) if bool(is_full_py[j]) else 0.0
        for j in all_idxs
    ]
    noise_py = [scale > 0.0 for scale in scale_py]
    _run_mcts_group(
        all_idxs,
        combined_sims,
        per_game_noise=noise_py,
        per_game_gumbel_scale=scale_py,
    )

    if state.has_c_ply and len(net_idxs) > 0:
        _append_records_via_c(
            state, net_idxs,
            cb_encode_list=_cb_encode_list,
            pol_logits=pol_logits, wdl_logits_raw=wdl_logits_raw,
            actions=actions, values_list=values_list, probs_list=probs_list,
            gumbel_diags=gumbel_diags,
            is_full_py=is_full_py, sample_weights=sample_weights,
            diff_focus=diff_focus,
        )
    else:
        _append_records_via_python(
            state, net_idxs,
            xs_batch=xs_batch,
            pol_logits=pol_logits, wdl_est=wdl_est,
            probs_list=probs_list, actions=actions, values_list=values_list,
            gumbel_diags=gumbel_diags,
            masks_list=masks_list,
            is_full=is_full, sample_weights=sample_weights,
            diff_focus=diff_focus,
        )


__all__ = ["run_network_turn"]
