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
from chess_anti_engine.mcts._mcts_tree import MCTSTree


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
) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray]]:
    """Gumbel root search with MCTSTree C tree + CBoard.

    Same API as ``run_gumbel_root_many`` -- drop-in replacement.
    """
    n_boards = len(boards)
    if n_boards == 0:
        return [], [], [], []

    sim_budget = max(1, int(cfg.simulations))

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_gumbel_root_many_c requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)

    # -- 1. Batch root evaluation ------------------------------------------
    root_cboards = [cboard_from_board_fast(b) for b in boards]

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
    tree = MCTSTree()

    probs_out: list[np.ndarray | None] = [None] * n_boards
    actions_out: list[int | None] = [None] * n_boards
    values_out: list[float] = list(root_qs)

    root_ids: list[int] = [-1] * n_boards  # node IDs in C tree
    root_legal: list[np.ndarray | None] = [None] * n_boards
    root_pri: list[np.ndarray | None] = [None] * n_boards
    remaining_per_board: list[list[int] | None] = [None] * n_boards
    budget_remaining: list[int] = [sim_budget] * n_boards
    gumbels_per_board: list[np.ndarray | None] = [None] * n_boards

    _full_tree = bool(cfg.full_tree)
    _c_puct = float(cfg.c_puct)
    _fpu_reduction = float(cfg.fpu_reduction)
    _c_visit = float(cfg.c_visit)
    _c_scale = float(cfg.c_scale)

    for i in range(n_boards):
        root_cb = root_cboards[i]
        legal_idx = root_cb.legal_move_indices()

        if boards[i].is_game_over() or legal_idx.size == 0:
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

        # Add root to C tree
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
        g = _gumbel(rng, legal_idx.size) if cfg.add_noise else np.zeros(legal_idx.size, dtype=np.float64)
        score = g + log_pri

        if sim_budget <= 1:
            m = 1
        else:
            m_cap = max(2, (sim_budget + 1) // 2)
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

    # -- 3. Sequential halving with C tree ---------------------------------

    _max_leaves = n_boards * max(2, int(cfg.topk))
    _enc_buf = np.empty((_max_leaves, 146, 8, 8), dtype=np.float32)
    # Second encoding buffer for pipelined GPU/CPU overlap: while the GPU
    # reads from one buffer, the next rep's leaves are encoded into the other.
    _enc_buf2 = np.empty((_max_leaves, 146, 8, 8), dtype=np.float32) if _has_async else None

    # Cache CBoard by node_id to avoid replaying full action paths from root.
    # Root CBoards are seeded; children are built from parent copy+push.
    _cb_cache: dict[int, CBoard] = {}
    for i in range(n_boards):
        rid = root_ids[i]
        if rid >= 0:
            _cb_cache[rid] = root_cboards[i]

    while True:
        active = [
            i for i in range(n_boards)
            if (
                probs_out[i] is None
                and remaining_per_board[i] is not None
                and len(remaining_per_board[i]) >= 1
                and budget_remaining[i] > 0
            )
        ]
        if not active:
            break

        visits_per_action: dict[int, int] = {}
        for bi in active:
            rem = remaining_per_board[bi]
            assert rem is not None
            if len(rem) <= 1:
                visits_per_action[bi] = int(budget_remaining[bi])
                continue
            rounds_left = max(1, int(math.ceil(math.log2(len(rem)))))
            vpa = int(budget_remaining[bi] // (len(rem) * rounds_left))
            visits_per_action[bi] = max(1, vpa)

        max_reps = max(visits_per_action.values(), default=0)

        # ---- Helper: collect leaves + build CBoards for one rep ----
        def _prepare_rep(rep: int, enc_buf: np.ndarray):
            """Tree traversal + CBoard building + encoding for *rep*.

            Returns None if no queries, otherwise a tuple of
            (all_bi, leaf_ids, node_paths, need_eval, leaf_cbs,
             leaf_legal, n_leaves, leaf_enc_slice).
            Terminal nodes are backpropped immediately.
            """
            p_root_ids = []
            p_forced = []
            p_bi = []
            for bi in active:
                rem = remaining_per_board[bi]
                if rem is None or rep >= visits_per_action[bi]:
                    continue
                rid = root_ids[bi]
                if rid < 0:
                    continue
                for action in rem:
                    p_root_ids.append(rid)
                    p_forced.append(int(action))
                    p_bi.append(bi)

            if not p_root_ids:
                return None

            n_queries = len(p_root_ids)
            root_ids_arr = np.array(p_root_ids, dtype=np.int32)
            forced_arr = np.array(p_forced, dtype=np.int32)

            p_leaf_ids, p_node_paths, p_action_paths = tree.gumbel_collect_leaves(
                root_ids_arr, forced_arr,
                _c_scale, _c_visit, _c_puct, _fpu_reduction, _full_tree,
            )

            p_leaf_cbs: list[CBoard | None] = [None] * n_queries
            p_terminal_vals: list[float | None] = [None] * n_queries
            p_need_eval: list[int] = []

            for qi in range(n_queries):
                bi = p_bi[qi]
                np_arr = p_node_paths[qi]

                if np_arr.size <= 1:
                    p_terminal_vals[qi] = float(root_qs[bi])
                    continue

                cached_cb = None
                replay_start = 0
                for di in range(np_arr.size - 2, -1, -1):
                    anc = int(np_arr[di])
                    if anc in _cb_cache:
                        cached_cb = _cb_cache[anc]
                        replay_start = di + 1
                        break

                if cached_cb is None:
                    ap = p_action_paths[qi]
                    cb = root_cboards[bi].copy()
                    for a in ap:
                        cb.push_index(int(a))
                else:
                    cb = cached_cb.copy()
                    ap = p_action_paths[qi]
                    for ai in range(replay_start - 1, ap.size):
                        cb.push_index(int(ap[ai]))

                if cb.is_game_over():
                    p_terminal_vals[qi] = float(cb.terminal_value())
                else:
                    p_leaf_cbs[qi] = cb
                    p_need_eval.append(qi)

            # Backprop terminals immediately
            term_paths = []
            term_values = []
            p_need_eval_set = set(p_need_eval)
            for qi in range(n_queries):
                if p_terminal_vals[qi] is not None and qi not in p_need_eval_set:
                    term_paths.append(p_node_paths[qi])
                    term_values.append(float(p_terminal_vals[qi]))
            if term_paths:
                tree.backprop_many(term_paths, term_values)

            if not p_need_eval:
                return None

            n_leaves = len(p_need_eval)
            p_leaf_legal: list[np.ndarray | None] = [None] * n_leaves
            for li, qi in enumerate(p_need_eval):
                _cb = p_leaf_cbs[qi]
                enc_buf[li] = _cb.encode_146()
                p_leaf_legal[li] = _cb.legal_move_indices()

            return (
                p_bi, p_leaf_ids, p_node_paths, p_need_eval,
                p_leaf_cbs, p_leaf_legal, n_leaves, enc_buf[:n_leaves],
            )

        # ---- Helper: sync GPU results and expand+backprop ----------
        def _finish_rep(prep, pol_logits_leaf, wdl_logits_leaf):
            """Expand nodes and backprop values for a completed rep."""
            (
                f_bi, f_leaf_ids, f_node_paths, f_need_eval,
                f_leaf_cbs, f_leaf_legal, f_n_leaves, _enc_slice,
            ) = prep
            q_vals = np.array(
                tree.batch_wdl_to_q(wdl_logits_leaf), dtype=np.float64,
            )
            expand_paths = []
            expand_values = []
            for li in range(f_n_leaves):
                qi = f_need_eval[li]
                nid = int(f_leaf_ids[qi])
                legal = f_leaf_legal[li]
                if legal.size > 0 and not tree.is_expanded(nid):
                    tree.expand_from_logits(nid, legal, pol_logits_leaf[li])
                _cb_cache[nid] = f_leaf_cbs[qi]
                expand_paths.append(f_node_paths[qi])
                expand_values.append(float(q_vals[li]))
            if expand_paths:
                tree.backprop_many(expand_paths, expand_values)

        # ---- Pipelined simulation loop --------------------------------
        # When async GPU eval is available, overlap GPU inference for rep N
        # with CPU work (tree traversal + CBoard + encoding) for rep N+1.
        # The two encoding buffers alternate so the GPU can read from one
        # while the CPU writes to the other.
        if _has_async and _enc_buf2 is not None:
            # Pipelined path: overlap GPU and CPU work
            _bufs = [_enc_buf, _enc_buf2]
            cur_buf_idx = 0

            # Prepare first rep
            cur_prep = _prepare_rep(0, _bufs[cur_buf_idx])
            for rep in range(max_reps):
                if cur_prep is None:
                    # Nothing to eval this rep; prepare next
                    cur_buf_idx ^= 1
                    if rep + 1 < max_reps:
                        cur_prep = _prepare_rep(rep + 1, _bufs[cur_buf_idx])
                    else:
                        cur_prep = None
                    continue

                # Launch async GPU eval for current rep
                pol_t, wdl_t, event = eval_impl.evaluate_encoded_async(
                    cur_prep[7],  # leaf_enc_slice
                )

                # While GPU works, prepare NEXT rep's CPU work
                next_buf_idx = cur_buf_idx ^ 1
                if rep + 1 < max_reps:
                    next_prep = _prepare_rep(rep + 1, _bufs[next_buf_idx])
                else:
                    next_prep = None

                # Sync GPU and finish current rep
                if event is not None:
                    event.synchronize()
                pol_logits_leaf = pol_t.numpy()
                wdl_logits_leaf = wdl_t.numpy()
                _finish_rep(cur_prep, pol_logits_leaf, wdl_logits_leaf)

                # Advance pipeline
                cur_prep = next_prep
                cur_buf_idx = next_buf_idx
        else:
            # Synchronous fallback (no async support)
            for rep in range(max_reps):
                prep = _prepare_rep(rep, _enc_buf)
                if prep is None:
                    continue
                pol_logits_leaf, wdl_logits_leaf = eval_impl.evaluate_encoded(
                    prep[7],  # leaf_enc_slice
                )
                _finish_rep(prep, pol_logits_leaf, wdl_logits_leaf)

        # Sequential halving: score and prune candidates using C tree
        for bi in active:
            rem = remaining_per_board[bi]
            pri = root_pri[bi]
            g_full = gumbels_per_board[bi]
            rid = root_ids[bi]
            if rem is None or pri is None or g_full is None or rid < 0:
                continue

            budget_remaining[bi] = max(0, int(budget_remaining[bi] - visits_per_action[bi] * len(rem)))
            if len(rem) <= 1:
                continue

            cands_arr = np.array(rem, dtype=np.int32)
            g_arr = np.array([float(g_full[a]) for a in rem], dtype=np.float64)
            scores = tree.gumbel_score_candidates(
                rid, cands_arr, g_arr, pri, _c_scale, _c_visit,
            )
            # Sort by score descending, keep top half
            order = np.argsort(-scores)
            keep = max(1, (len(rem) + 1) // 2)
            remaining_per_board[bi] = [rem[int(j)] for j in order[:keep]]

    # -- 4. Build improved policies from C tree ----------------------------
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

    return probs_out, actions_out, values_out, legal_masks_out


@torch.no_grad()
def run_gumbel_root_c(
    model: torch.nn.Module,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, int, float]:
    probs, acts, vals, _masks = run_gumbel_root_many_c(
        model, [board], device=device, rng=rng, cfg=cfg,
    )
    return probs[0], acts[0], float(vals[0])
