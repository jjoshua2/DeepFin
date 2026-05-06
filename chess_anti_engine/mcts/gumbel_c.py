"""Gumbel MCTS with C-accelerated tree + CBoard operations.

Uses MCTSTree (array-based C tree) for select/expand/backprop and CBoard
for board state management.  The entire tree traversal loop runs in C,
eliminating Python interpreter overhead that was the dominant CPU bottleneck.

Architecture:
  - MCTSTree: array-based C tree holding N/W/prior/children per node
  - CBoard: C chess board for encoding, legal moves, terminal detection
  - Gumbel simulation: C start_gumbel_sims / continue_gumbel_sims state machine
    (tree traversal + sequential-halving scoring all in C).
  - Expand: C expand_from_logits (softmax + tree insert)
  - Backprop: C backprop_many (batched value propagation)
"""
from __future__ import annotations

import logging as _logging
from typing import cast

import chess
import numpy as np
import torch

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import (  # noqa: F401  # skylos: ignore (AsyncBatchEvaluator used via stringified cast)
    AsyncBatchEvaluator,
    BatchEvaluator,
    LocalModelEvaluator,
    _COMPILED_BATCH_BUCKETS,
)
from chess_anti_engine.mcts._mcts_tree import MCTSTree, batch_encode_146
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _gumbel,
    _sigma_scale,
    _softmax,
    _wdl_to_q,
)
from chess_anti_engine.moves import POLICY_SIZE

_log = _logging.getLogger(__name__)


def _tb_override(tree: MCTSTree | None, probe, wdl: np.ndarray) -> None:
    if probe is None or tree is None:
        return
    indices, leaves = tree.get_pending_tb_leaves(probe.max_pieces)
    if not leaves:
        return
    # solved_out feeds mark_tb_solved so subtrees with proven WDL short-circuit
    # MCTS selection (and propagate up). 0 = no TB hit / skip.
    solved_out = np.zeros(len(leaves), dtype=np.int8)
    probe.apply(leaves, wdl, indices=indices, solved_out=solved_out)
    if (solved_out != 0).any():
        tree.mark_tb_solved(indices.astype(np.int32, copy=False), solved_out)


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
    tree: MCTSTree | None = None,
    root_node_ids: list[int] | None = None,
    tb_probe=None,
    pre_wdl_logits_tb_probed: bool = False,
    target_batch: int = 0,
) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray], MCTSTree, list[int]]:
    """Gumbel root search with MCTSTree C tree + CBoard.

    Same API as ``run_gumbel_root_many`` -- drop-in replacement.
    """
    import time as _time
    _t_init = 0.0
    _t_prepare = 0.0
    _t_gpu = 0.0
    _t_finish = 0.0
    _t_score = 0.0
    _t_policy = 0.0
    _t_python_glue = 0.0
    _n_gpu_calls = 0
    _n_gpu_positions = 0
    _t_func_start = _time.perf_counter()

    n_boards = len(boards)
    if n_boards == 0:
        return [], [], [], [], (tree if tree is not None else MCTSTree()), []

    sim_budget = max(1, int(cfg.simulations))

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_gumbel_root_many_c requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)

  # -- 1. Batch root evaluation ------------------------------------------
    root_cboards = cboards if cboards is not None else [CBoard.from_board(b) for b in boards]

    _has_async = hasattr(eval_impl, 'evaluate_encoded_async')
  # All async-capable evaluators conform to the protocol; _has_async is the runtime check.
    _async_eval = cast("AsyncBatchEvaluator", eval_impl)
    _use_pipeline = _has_async and n_boards >= 64

  # Zero-copy path: when the evaluator exposes get_input_buffer + evaluate_inplace_async
  # (DirectGPUEvaluator with n_slots>=needed), we route the C tree walks to write
  # encodes directly into pinned host memory. This eliminates one numpy memcpy per
  # rep AND lets H2D DMA start immediately when the C walk returns. Pipelined mode
  # also requires 2 slots so submit(g=0)+submit(g=1) don't share output buffers
  # (otherwise the next async submit overwrites pol/wdl before C reads them, forcing
  # a defensive .numpy().copy()).
    _slots_needed = 2 if _use_pipeline else 1
    _inplace = (
        hasattr(eval_impl, "get_input_buffer")
        and hasattr(eval_impl, "evaluate_inplace_async")
        and getattr(eval_impl, "n_slots", 1) >= _slots_needed
    )

    if pre_pol_logits is not None and pre_wdl_logits is not None:
        pol_logits_batch = np.asarray(pre_pol_logits, dtype=np.float32)
        wdl_logits_batch = np.asarray(pre_wdl_logits, dtype=np.float32)
    elif _inplace:
        root_buf = eval_impl.get_input_buffer(n_boards, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
        batch_encode_146(root_cboards, root_buf)
        pol_t, wdl_t, event = eval_impl.evaluate_inplace_async(n_boards, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
        if event is not None:
            event.synchronize()
        pol_logits_batch = pol_t.numpy()
        wdl_logits_batch = wdl_t.numpy()
    else:
        xs = np.empty((n_boards, 146, 8, 8), dtype=np.float32)
        batch_encode_146(root_cboards, xs)
        if _has_async:
            pol_t, wdl_t, event = _async_eval.evaluate_encoded_async(xs)
            if event is not None:
                event.synchronize()
            pol_logits_batch = pol_t.numpy()
            wdl_logits_batch = wdl_t.numpy()
        else:
            pol_logits_batch, wdl_logits_batch = eval_impl.evaluate_encoded(xs)

  # Override root wdl_logits before root_qs is derived (root_qs seeds FPU
  # and the values_out initial pass). UCI may pass cached logits that already
  # include this override; selfplay passes raw batched model logits.
    if tb_probe is not None and not pre_wdl_logits_tb_probed:
        tb_probe.apply(root_cboards, wdl_logits_batch)

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
    budget_remaining: list[int]
    if per_game_simulations is not None:
        budget_remaining = [max(1, int(s)) for s in per_game_simulations]
    else:
        budget_remaining = [sim_budget] * n_boards
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

  # Reuse existing root from persistent tree, or create new one.
  # Skip when pipelining — pipeline creates its own sub-trees.
        if not _use_pipeline:
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
        score: np.ndarray = g + log_pri

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

  # Floor at 256 so single-game UCI (n_boards=1) gets a usefully-sized GPU
  # batch. _enc_buf is _max_leaves_per_rep*2, so this gives a 512-slot buffer
  # minimum (~19 MB). Without the floor, 1 board × topk=32 caps at 64 slots
  # and gss_step flushes the halving round across 4-5 tiny GPU calls.
    _max_leaves_per_rep = max(256, n_boards * max(2, int(cfg.topk)))
    _BUCKETS = _COMPILED_BATCH_BUCKETS

  # ---- Pipelined simulation: split games into 2 groups ----------------
  # While GPU evaluates group A's leaves, C does tree walks for group B,
  # and vice versa.  CPU (C tree walks) and GPU overlap on separate hardware.

    def _pad_for_bucket(nl, buf_len):
        for _b in _BUCKETS:
            if _b >= nl:
                return min(_b, buf_len)
        return min(nl, buf_len)

    if _use_pipeline:
        mid = n_boards // 2
        _grp = [list(range(mid)), list(range(mid, n_boards))]
        _trees = [MCTSTree(), MCTSTree()]
        _max_grp = max(mid, n_boards - mid)  # ceil half for odd splits
        _leaf_cap = max(512, _max_grp * max(2, int(cfg.topk)) * 2)
        if _inplace:
  # Pinned-host views: C writes encodes directly here, eval reads from the
  # same memory (no memcpy on submit). Two slots so g=0 / g=1 outputs don't
  # collide.
            _max_batch = getattr(eval_impl, "_max_batch", _leaf_cap)
            _leaf_cap = min(_leaf_cap, _max_batch)
            _enc_bufs = [
                eval_impl.get_input_buffer(_leaf_cap, slot=g)  # pyright: ignore[reportAttributeAccessIssue]
                for g in range(2)
            ]
        else:
            _enc_bufs = [
                np.empty((_leaf_cap, 146, 8, 8), dtype=np.float32)
                for _ in range(2)
            ]

  # Create fresh root nodes in each sub-tree and build local root_ids
        _sub_root_ids: list[list[int]] = [[], []]
        for g in range(2):
            for i in _grp[g]:
                _pri_i = root_pri[i]
                _legal_i = root_legal[i]
                if _pri_i is None or _legal_i is None:
                    _sub_root_ids[g].append(-1)
                    continue
                priors = _pri_i[_legal_i].astype(np.float64)
                rid = _trees[g].add_root(1, float(root_qs[i]))
                _trees[g].expand(rid, _legal_i.astype(np.int32), priors)
                _sub_root_ids[g].append(rid)

  # Start both groups
        _n_leaves: list[int | None] = [None, None]
        _tp0 = _time.perf_counter()
        for g in range(2):
            idx = _grp[g]
            ng = len(idx)
            if ng == 0:
                continue
            _cb_g = [root_cboards[i] for i in idx]
            _rid_g = np.array(_sub_root_ids[g], dtype=np.int32)
  # idx slots were populated in the init loop above, so items are non-None.
            _rem_g = cast("list[list[int]]", [remaining_per_board[i] for i in idx])
            _gum_g = cast("list[np.ndarray]", [gumbels_per_board[i] for i in idx])
            _pri_g = cast("list[np.ndarray]", [root_pri[i] for i in idx])
            _bud_g = np.array([budget_remaining[i] for i in idx], dtype=np.int32)
            _rqs_g = np.array([root_qs[i] for i in idx], dtype=np.float64)

            _n_leaves[g] = _trees[g].start_gumbel_sims(
                _cb_g, _rid_g, _rem_g, _gum_g, _pri_g, _bud_g, _rqs_g,
                _c_scale, _c_visit, _c_puct, _fpu_reduction, _full_tree,
                _enc_bufs[g], 0, int(target_batch),
            )
        _t_prepare += _time.perf_counter() - _tp0

  # Pipeline loop --------------------------------------------------
  # Each group independently cycles: GPU eval → C tree walks → GPU eval → ...
  # We overlap GPU(A) with C(B) by launching async GPU for one group,
  # then doing continue_gumbel_sims for the other.

        def _drain_sequential(g):
            """Drain remaining simulation for group g without pipelining."""
            nonlocal _t_gpu, _t_prepare, _n_gpu_calls, _n_gpu_positions
            while _n_leaves[g] is not None:
                nl = int(_n_leaves[g])
                padded = _pad_for_bucket(nl, len(_enc_bufs[g]))
                _tg0 = _time.perf_counter()
                if _inplace:
                    pol_t, wdl_t, ev = eval_impl.evaluate_inplace_async(padded, slot=g)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    pol_t, wdl_t, ev = _async_eval.evaluate_encoded_async(_enc_bufs[g][:padded])
                if ev is not None:
                    ev.synchronize()
                _t_gpu += _time.perf_counter() - _tg0
                _n_gpu_calls += 1
                _n_gpu_positions += nl
                _tp0 = _time.perf_counter()
                _wdl_slice = wdl_t[:nl].numpy()
                _tb_override(_trees[g], tb_probe, _wdl_slice)
                _n_leaves[g] = _trees[g].continue_gumbel_sims(
                    pol_t[:nl].numpy(), _wdl_slice)
                _t_prepare += _time.perf_counter() - _tp0

  # Main pipelined loop: GPU(g) overlaps with C tree walks(other).
  # The first iteration runs asymmetrically (no pending group 1 results
  # yet), then settles into a steady-state pattern:
  #   1. Submit GPU(0) async
  #   2. C tree walks for group 1 using previous GPU(1) results
  #   3. Sync GPU(0), copy results out of pinned buffers
  #   4. Submit GPU(1) async
  #   5. C tree walks for group 0 using copied GPU(0) results
  #   6. Sync GPU(1), copy results → _pending_g1 for next iteration
  #
  # We copy results from pinned buffers immediately after sync because
  # the next evaluate_encoded_async reuses the same _pinned_pol /
  # _pinned_wdl buffers, invalidating any views.
        _pending_g1 = None  # (pol_np, wdl_np) — synced + copied numpy

        _max_iters = n_boards * max(max(budget_remaining), 1) + 100
        for _ in range(_max_iters):
            if _n_leaves[0] is None and _n_leaves[1] is None:
                break

  # Only one group active → flush pending, then drain
            if _n_leaves[0] is None:
                if _pending_g1 is not None:
                    _tp0 = _time.perf_counter()
                    _tb_override(_trees[1], tb_probe, _pending_g1[1])
                    _n_leaves[1] = _trees[1].continue_gumbel_sims(*_pending_g1)
                    _t_prepare += _time.perf_counter() - _tp0
                    _pending_g1 = None
                _drain_sequential(1)
                break
            if _n_leaves[1] is None and _pending_g1 is None:
                _drain_sequential(0)
                break

  # Both active — one pipelined iteration:

  # 1) Submit GPU for group 0 (async — GPU starts working)
            nl0 = int(_n_leaves[0])
            padded0 = _pad_for_bucket(nl0, len(_enc_bufs[0]))
            _tg0 = _time.perf_counter()
            if _inplace:
                pol_t0, wdl_t0, ev0 = eval_impl.evaluate_inplace_async(padded0, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                pol_t0, wdl_t0, ev0 = _async_eval.evaluate_encoded_async(_enc_bufs[0][:padded0])

  # 2) While GPU processes group 0, do C tree walks for group 1
  #    (continue_gumbel_sims releases GIL; CPU and GPU run in parallel)
            if _pending_g1 is not None:
                _tp0 = _time.perf_counter()
                _tb_override(_trees[1], tb_probe, _pending_g1[1])
                _n_leaves[1] = _trees[1].continue_gumbel_sims(*_pending_g1)
                _t_prepare += _time.perf_counter() - _tp0
                _pending_g1 = None

                if _n_leaves[1] is None:
  # Group 1 finished — wait for GPU(0) and drain group 0
                    if ev0 is not None:
                        ev0.synchronize()
                    _t_gpu += _time.perf_counter() - _tg0
                    _n_gpu_calls += 1
                    _n_gpu_positions += nl0
                    _tp0 = _time.perf_counter()
                    _wdl0 = wdl_t0[:nl0].numpy()
                    _tb_override(_trees[0], tb_probe, _wdl0)
                    _n_leaves[0] = _trees[0].continue_gumbel_sims(
                        pol_t0[:nl0].numpy(), _wdl0)
                    _t_prepare += _time.perf_counter() - _tp0
                    _drain_sequential(0)
                    break

  # 3) Wait for GPU(0). Slot-aware path: pinned slot 0 outputs stay live
  # until the next submit(slot=0), which happens at the *top* of next loop
  # iteration — well after step (5) reads them. Legacy path must copy
  # because both submits share one output buffer.
            if ev0 is not None:
                ev0.synchronize()
            _t_gpu += _time.perf_counter() - _tg0
            _n_gpu_calls += 1
            _n_gpu_positions += nl0
            if _inplace:
                pol_np0 = pol_t0[:nl0].numpy()
                wdl_np0 = wdl_t0[:nl0].numpy()
            else:
                pol_np0 = pol_t0[:nl0].numpy().copy()
                wdl_np0 = wdl_t0[:nl0].numpy().copy()

  # 4) Submit GPU for group 1 (async — safe: group 0 results consumed below)
            if _n_leaves[1] is not None:
                nl1 = int(_n_leaves[1])
                padded1 = _pad_for_bucket(nl1, len(_enc_bufs[1]))
                _tg1 = _time.perf_counter()
                if _inplace:
                    pol_t1, wdl_t1, ev1 = eval_impl.evaluate_inplace_async(padded1, slot=1)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    pol_t1, wdl_t1, ev1 = _async_eval.evaluate_encoded_async(_enc_bufs[1][:padded1])

  # 5) While GPU processes group 1, do C tree walks for group 0
  #    (uses copied numpy arrays — safe from pinned buffer reuse)
            _tp0 = _time.perf_counter()
            _tb_override(_trees[0], tb_probe, wdl_np0)
            _n_leaves[0] = _trees[0].continue_gumbel_sims(pol_np0, wdl_np0)
            _t_prepare += _time.perf_counter() - _tp0

  # 6) Sync GPU(1), copy results for next iteration's step 2.
  # Same-condition re-check — ev1/_tg1/nl1/pol_t1/wdl_t1 are all bound
  # from step (4). Pyright can't narrow across the blocks.
            if _n_leaves[1] is not None:
                _ev1 = ev1  # pyright: ignore[reportPossiblyUnboundVariable]
                _tg1_l = _tg1  # pyright: ignore[reportPossiblyUnboundVariable]
                _nl1 = nl1  # pyright: ignore[reportPossiblyUnboundVariable]
                _pol1 = pol_t1  # pyright: ignore[reportPossiblyUnboundVariable]
                _wdl1 = wdl_t1  # pyright: ignore[reportPossiblyUnboundVariable]
                if _ev1 is not None:
                    _ev1.synchronize()
                _t_gpu += _time.perf_counter() - _tg1_l
                _n_gpu_calls += 1
                _n_gpu_positions += _nl1
  # Inplace path: slot 1's pinned outputs persist until next submit(slot=1)
  # in step (4) of next iter — safe to alias. Legacy path shares one output
  # buffer across both submits, so we must clone before submit(slot=0).
                if _inplace:
                    _pending_g1 = (_pol1[:_nl1].numpy(), _wdl1[:_nl1].numpy())
                else:
                    _pending_g1 = (
                        _pol1[:_nl1].numpy().copy(),
                        _wdl1[:_nl1].numpy().copy(),
                    )
        else:
            raise RuntimeError(f"pipeline loop did not converge in {_max_iters} iterations")

  # Retrieve remaining candidates from both trees, merge back
        _rem_a = _trees[0].get_gumbel_remaining()
        _rem_b = _trees[1].get_gumbel_remaining()
        remaining_per_board = [None] * n_boards
        for gi, i in enumerate(_grp[0]):
            remaining_per_board[i] = _rem_a[gi] if gi < len(_rem_a) else None
        for gi, i in enumerate(_grp[1]):
            remaining_per_board[i] = _rem_b[gi] if gi < len(_rem_b) else None

  # Store tree refs + root IDs for policy extraction. `None` on the outer
  # type is reserved for the non-pipelined else-branch below (signals
  # "use single tree"); pipelined path always populates every slot.
        _tree_for_board: list[MCTSTree | None] | None = list[MCTSTree | None]([None] * n_boards)
        _rid_for_board: list[int] | None = [0] * n_boards
        for g in range(2):
            for gi, i in enumerate(_grp[g]):
                _tree_for_board[i] = _trees[g]
                _rid_for_board[i] = _sub_root_ids[g][gi]

    else:
  # Non-pipelined fallback (small batches or no async)
        if _inplace:
            _max_batch = getattr(eval_impl, "_max_batch", _max_leaves_per_rep * 2)
            _cap = min(_max_leaves_per_rep * 2, _max_batch)
            _enc_buf = eval_impl.get_input_buffer(_cap, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            _enc_buf = np.empty((_max_leaves_per_rep * 2, 146, 8, 8), dtype=np.float32)
        _root_ids_arr = np.array(root_ids, dtype=np.int32)
        _budget_arr = np.array(budget_remaining, dtype=np.int32)
        _root_qs_arr = np.array(root_qs, dtype=np.float64)

        _tp0 = _time.perf_counter()
        n_leaves = tree.start_gumbel_sims(
            root_cboards, _root_ids_arr,
            cast("list[list[int]]", remaining_per_board),
            cast("list[np.ndarray]", gumbels_per_board),
            cast("list[np.ndarray]", root_pri),
            _budget_arr, _root_qs_arr,
            _c_scale, _c_visit, _c_puct, _fpu_reduction, _full_tree,
            _enc_buf, 0, int(target_batch),
        )
        _t_prepare += _time.perf_counter() - _tp0

        while n_leaves is not None:
            n_leaves = int(n_leaves)
            padded = _pad_for_bucket(n_leaves, len(_enc_buf))
            _tg0 = _time.perf_counter()
            if _inplace:
                pol_t, wdl_t, event = eval_impl.evaluate_inplace_async(padded, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
                if event is not None:
                    event.synchronize()
                pol_all = pol_t[:n_leaves].numpy()
                wdl_all = wdl_t[:n_leaves].numpy()
            elif _has_async:
                pol_t, wdl_t, event = _async_eval.evaluate_encoded_async(_enc_buf[:padded])
                if event is not None:
                    event.synchronize()
                pol_all = pol_t[:n_leaves].numpy()
                wdl_all = wdl_t[:n_leaves].numpy()
            else:
                pol_all, wdl_all = eval_impl.evaluate_encoded(_enc_buf[:padded])
                pol_all = pol_all[:n_leaves]
                wdl_all = wdl_all[:n_leaves]
            _t_gpu += _time.perf_counter() - _tg0
            _n_gpu_calls += 1
            _n_gpu_positions += n_leaves

            _tp0 = _time.perf_counter()
            _tb_override(tree, tb_probe, wdl_all)
            n_leaves = tree.continue_gumbel_sims(pol_all, wdl_all)
            _t_prepare += _time.perf_counter() - _tp0

        remaining_per_board = cast("list[list[int] | None]", tree.get_gumbel_remaining())
        _tree_for_board = None  # signal to use single tree
        _rid_for_board = None

  # -- 4. Build improved policies from C tree ----------------------------
    _tp0 = _time.perf_counter()
    for i in range(n_boards):
        if probs_out[i] is not None:
            continue

        pri = root_pri[i]
        remaining = remaining_per_board[i]
        if _tree_for_board is not None and _rid_for_board is not None:
            _qtree = _tree_for_board[i]
            rid = _rid_for_board[i]
        else:
            assert tree is not None
            _qtree = tree
            rid = root_ids[i]
        if pri is None or remaining is None or rid < 0:
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            continue

        legal = np.nonzero(pri > 0)[0].astype(int)
        root_q_i = float(root_qs[i])

  # Get children stats from C tree (completed_q already negated)
        assert _qtree is not None
        child_actions, child_visits, child_q = _qtree.get_children_q(rid, root_q_i)
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
    if _log.isEnabledFor(_logging.DEBUG):
        _avg_batch = _n_gpu_positions / max(1, _n_gpu_calls)
        _log.debug(
            "gumbel profile (n_boards=%d): total=%.3fs init=%.3f prep=%.3f "
            "gpu=%.3f(%dcalls,%dpos,avg=%.1f) "
            "finish=%.3f score=%.3f policy=%.3f glue=%.3f%s",
            n_boards, _t_total, _t_init, _t_prepare,
            _t_gpu, _n_gpu_calls, _n_gpu_positions, _avg_batch,
            _t_finish, _t_score, _t_policy, _t_python_glue,
            " PIPELINE" if _use_pipeline else "",
        )
  # When pipelining, sub-trees are ephemeral — invalidate root IDs so
  # the caller doesn't try to reuse nodes that don't exist in the main tree.
    _ret_root_ids = root_ids if not _use_pipeline else [-1] * n_boards
    assert tree is not None
    return (
        cast("list[np.ndarray]", probs_out),
        cast("list[int]", actions_out),
        values_out,
        legal_masks_out,
        tree,
        _ret_root_ids,
    )

