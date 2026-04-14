"""Gumbel MCTS with C-accelerated tree + CBoard operations.

Uses MCTSTree (array-based C tree) for select/expand/backprop and CBoard
for board state management.  The entire tree traversal loop runs in C,
eliminating Python interpreter overhead that was the dominant CPU bottleneck.

Architecture:
  - MCTSTree: array-based C tree holding N/W/prior/children per node
  - CBoard: C chess board for encoding, legal moves, terminal detection
  - Tree traversal: C gumbel_collect_leaves (improved-policy selection)
  - Expand: C expand_from_logits (softmax + tree insert)
  - Backprop: C backprop_many (batched value propagation)
  - Sequential halving scoring: C gumbel_score_candidates
"""
from __future__ import annotations

import math

import numpy as np
import chess
import torch

from chess_anti_engine.inference import BatchEvaluator, LocalModelEvaluator
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _gumbel,
    _softmax,
    _wdl_to_q,
    _sigma_scale,
)
from chess_anti_engine.moves import POLICY_SIZE

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.mcts._mcts_tree import MCTSTree, NNCache


@torch.no_grad()
def run_gumbel_root_many_c(
    model: torch.nn.Module | None,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
    evaluator: BatchEvaluator | None = None,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
    per_game_simulations: list[int] | None = None,
    per_game_add_noise: list[bool] | None = None,
    cboards: list | None = None,
    nn_cache: "NNCache | None" = None,
    tree: "MCTSTree | None" = None,
    root_node_ids: list[int] | None = None,
) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray], "MCTSTree", list[int]]:
    """Gumbel root search with MCTSTree C tree + CBoard.

    Same API as ``run_gumbel_root_many`` -- drop-in replacement.
    """
    import time as _time
    _t_init = 0.0; _t_prepare = 0.0; _t_gpu = 0.0; _t_finish = 0.0
    _t_score = 0.0; _t_policy = 0.0; _t_python_glue = 0.0
    _n_gpu_calls = 0; _n_gpu_positions = 0
    _t_func_start = _time.perf_counter()

    n_boards = len(boards)
    if n_boards == 0:
        return [], [], [], [], None, []

    sim_budget = max(1, int(cfg.simulations))

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_gumbel_root_many_c requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)

    # -- 1. Batch root evaluation ------------------------------------------
    root_cboards = cboards if cboards is not None else [CBoard.from_board(b) for b in boards]

    _has_async = hasattr(eval_impl, 'evaluate_encoded_async')

    if pre_pol_logits is not None and pre_wdl_logits is not None:
        pol_logits_batch = np.asarray(pre_pol_logits, dtype=np.float32)
        wdl_logits_batch = np.asarray(pre_wdl_logits, dtype=np.float32)
    else:
        xs = np.empty((n_boards, 146, 8, 8), dtype=np.float32)
        for _i, _cb in enumerate(root_cboards):
            xs[_i] = _cb.encode_146()
        if _has_async:
            pol_t, wdl_t, event = eval_impl.evaluate_encoded_async(xs)
            if event is not None:
                event.synchronize()
            pol_logits_batch = pol_t.numpy()
            wdl_logits_batch = wdl_t.numpy()
        else:
            pol_logits_batch, wdl_logits_batch = eval_impl.evaluate_encoded(xs)

    root_qs = [_wdl_to_q(wdl_logits_batch[i]) for i in range(n_boards)]

    # -- 2. Init C tree + roots --------------------------------------------
    _own_tree = tree is None
    if _own_tree:
        tree = MCTSTree()

    probs_out: list[np.ndarray | None] = [None] * n_boards
    actions_out: list[int | None] = [None] * n_boards
    values_out: list[float] = list(root_qs)

    root_ids: list[int] = [-1] * n_boards  # node IDs in C tree
    root_legal: list[np.ndarray | None] = [None] * n_boards
    root_pri: list[np.ndarray | None] = [None] * n_boards
    remaining_per_board: list[list[int] | None] = [None] * n_boards
    if per_game_simulations is not None:
        budget_remaining: list[int] = [max(1, int(s)) for s in per_game_simulations]
    else:
        budget_remaining: list[int] = [sim_budget] * n_boards
    gumbels_per_board: list[np.ndarray | None] = [None] * n_boards

    _full_tree = bool(cfg.full_tree)
    _c_puct = float(cfg.c_puct)
    _fpu_reduction = float(cfg.fpu_reduction)
    _c_visit = float(cfg.c_visit)
    _c_scale = float(cfg.c_scale)

    _t0 = _time.perf_counter()
    for i in range(n_boards):
        root_cb = root_cboards[i]
        legal_idx = root_cb.legal_move_indices()

        if root_cb.is_game_over() or legal_idx.size == 0:
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            root_pri[i] = np.zeros(POLICY_SIZE, dtype=np.float64)
            continue

        root_legal[i] = legal_idx

        # Softmax priors
        ll = pol_logits_batch[i][legal_idx].astype(np.float64)
        ll -= ll.max()
        e = np.exp(ll)
        s = float(e.sum())
        priors = (e / s) if s > 0 else np.full_like(e, 1.0 / e.size)

        pri = np.zeros(POLICY_SIZE, dtype=np.float64)
        pri[legal_idx] = priors
        root_pri[i] = pri

        # Reuse existing root from persistent tree, or create new one
        _reused = False
        if root_node_ids is not None and root_node_ids[i] >= 0:
            rid = root_node_ids[i]
            if tree.is_expanded(rid):
                root_ids[i] = rid
                _reused = True

        if not _reused:
            rid = tree.add_root(1, float(root_qs[i]))
            root_ids[i] = rid
            tree.expand(rid, legal_idx.astype(np.int32), priors)

        if legal_idx.size == 1:
            a0 = int(legal_idx[0])
            p = np.zeros((POLICY_SIZE,), dtype=np.float32)
            p[a0] = 1.0
            probs_out[i] = p
            actions_out[i] = a0
            continue

        # Gumbel noise -> select top-m
        log_pri = np.log(np.maximum(pri[legal_idx], 1e-12))
        _noise_this = per_game_add_noise[i] if per_game_add_noise is not None else cfg.add_noise
        g = _gumbel(rng, legal_idx.size) if _noise_this else np.zeros(legal_idx.size, dtype=np.float64)
        score = g + log_pri

        _game_budget = budget_remaining[i]
        if _game_budget <= 1:
            m = 1
        else:
            m_cap = max(2, (_game_budget + 1) // 2)
            m = int(min(int(cfg.topk), int(legal_idx.size), int(m_cap)))
            m = max(2, m)

        kth = min(m - 1, int(score.size) - 1)
        top_idx = np.argpartition(-score, kth)[:m]
        cands = legal_idx[top_idx].astype(int).tolist()

        remaining_per_board[i] = list(cands)
        # Store gumbel values indexed by legal_idx for scoring
        g_full = np.zeros(POLICY_SIZE, dtype=np.float64)
        g_full[legal_idx] = g
        gumbels_per_board[i] = g_full

    _t_init = _time.perf_counter() - _t0
    # -- 3. Sequential halving with C tree ---------------------------------

    _max_leaves_per_rep = n_boards * max(2, int(cfg.topk))
    _BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 4096)
    _enc_buf = np.empty((_max_leaves_per_rep * 2, 146, 8, 8), dtype=np.float32)

    # ---- Fused C simulation loop -----------------------------------------
    # Uses start_gumbel_sims/continue_gumbel_sims to run the entire
    # sequential halving loop in C, only returning to Python for GPU eval.
    _root_ids_arr = np.array(root_ids, dtype=np.int32)
    _budget_arr = np.array(budget_remaining, dtype=np.int32)
    _root_qs_arr = np.array(root_qs, dtype=np.float64)

    _tp0 = _time.perf_counter()
    n_leaves = tree.start_gumbel_sims(
        root_cboards, _root_ids_arr, remaining_per_board,
        gumbels_per_board, root_pri, _budget_arr, _root_qs_arr,
        _c_scale, _c_visit, _c_puct, _fpu_reduction, _full_tree,
        _enc_buf, nn_cache,
    )
    _t_prepare += _time.perf_counter() - _tp0

    while n_leaves is not None:
        n_leaves = int(n_leaves)
        padded = n_leaves
        for _b in _BUCKETS:
            if _b >= n_leaves:
                padded = min(_b, len(_enc_buf))
                break
        enc_slice = _enc_buf[:padded]
        _tg0 = _time.perf_counter()
        if _has_async:
            pol_t, wdl_t, event = eval_impl.evaluate_encoded_async(enc_slice)
            if event is not None:
                event.synchronize()
            pol_all = pol_t[:n_leaves].numpy()
            wdl_all = wdl_t[:n_leaves].numpy()
        else:
            pol_all, wdl_all = eval_impl.evaluate_encoded(enc_slice)
            pol_all = pol_all[:n_leaves]
            wdl_all = wdl_all[:n_leaves]
        _t_gpu += _time.perf_counter() - _tg0
        _n_gpu_calls += 1
        _n_gpu_positions += n_leaves

        _tp0 = _time.perf_counter()
        n_leaves = tree.continue_gumbel_sims(pol_all, wdl_all)
        _t_prepare += _time.perf_counter() - _tp0

    # Retrieve final remaining candidates from C state
    remaining_per_board = tree.get_gumbel_remaining()

    # -- 4. Build improved policies from C tree ----------------------------
    _tp0 = _time.perf_counter()
    for i in range(n_boards):
        if probs_out[i] is not None:
            continue

        pri = root_pri[i]
        remaining = remaining_per_board[i]
        rid = root_ids[i]
        if pri is None or remaining is None or rid < 0:
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            continue

        legal = np.nonzero(pri > 0)[0].astype(int)
        root_q_i = float(root_qs[i])

        # Get children stats from C tree (completed_q already negated)
        child_actions, child_visits, child_q = tree.get_children_q(rid, root_q_i)
        action_to_slot = {}
        for j in range(child_actions.size):
            action_to_slot[int(child_actions[j])] = j

        max_visit = int(child_visits.max()) if child_visits.size > 0 else 0

        completed_q = np.empty(legal.size, dtype=np.float64)
        for j, a in enumerate(legal):
            slot = action_to_slot.get(int(a))
            if slot is not None:
                completed_q[j] = float(child_q[slot])
            else:
                completed_q[j] = root_q_i

        logits_imp = np.log(np.maximum(pri[legal], 1e-12)) + _sigma_scale(
            max_visit=int(max_visit), cfg=cfg,
        ) * completed_q
        imp_all = _softmax(logits_imp)
        probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
        probs[legal] = imp_all.astype(np.float32)

        best_a = int(remaining[0])
        if cfg.temperature <= 0:
            action = best_a
        else:
            p = imp_all.astype(np.float64, copy=True)
            if cfg.temperature != 1.0:
                p = np.power(np.maximum(p, 0.0), 1.0 / float(cfg.temperature))
            ps = float(p.sum())
            if ps > 0:
                p /= ps
                action = int(rng.choice(legal, p=p))
            else:
                action = best_a

        probs_out[i] = probs
        actions_out[i] = action

        # Value from child
        slot = action_to_slot.get(best_a)
        if slot is not None and int(child_visits[slot]) > 0:
            # completed_q for best_a
            j_best = np.searchsorted(legal, best_a)
            if j_best < legal.size and int(legal[j_best]) == best_a:
                values_out[i] = float(completed_q[j_best])
            else:
                values_out[i] = root_q_i
        else:
            values_out[i] = root_q_i

    # Build legal masks
    legal_masks_out: list[np.ndarray] = []
    for i in range(n_boards):
        mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
        rl = root_legal[i]
        if rl is not None:
            mask[rl] = True
        legal_masks_out.append(mask)

    _t_policy = _time.perf_counter() - _tp0
    _t_total = _time.perf_counter() - _t_func_start
    _t_python_glue = _t_total - _t_init - _t_prepare - _t_gpu - _t_finish - _t_score - _t_policy
    if n_boards >= 64:  # only log for realistic batches
        import sys
        print(
            f"gumbel profile: total={_t_total:.3f}s init={_t_init:.3f} prep={_t_prepare:.3f} "
            f"gpu={_t_gpu:.3f}({_n_gpu_calls}calls,{_n_gpu_positions}pos) "
            f"finish={_t_finish:.3f} score={_t_score:.3f} policy={_t_policy:.3f} glue={_t_python_glue:.3f}",
            file=sys.stderr, flush=True,
        )
    return probs_out, actions_out, values_out, legal_masks_out, tree, root_ids


@torch.no_grad()
def run_gumbel_root_c(
    model: torch.nn.Module,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, int, float]:
    probs, acts, vals, _masks, _tree, _rids = run_gumbel_root_many_c(
        model, [board], device=device, rng=rng, cfg=cfg,
    )
    return probs[0], acts[0], float(vals[0])
