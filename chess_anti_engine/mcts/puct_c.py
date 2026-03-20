"""PUCT MCTS using the C-accelerated tree.

Drop-in replacement for `run_mcts_many` / `run_mcts` that stores the tree
in C arrays instead of Python Node objects.  Selection, expansion, and
backprop all happen in C; Python only handles board replay at leaves,
position encoding, and GPU evaluation.
"""
from __future__ import annotations

import numpy as np
import chess
import torch

from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.inference import BatchEvaluator, LocalModelEvaluator
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct import MCTSConfig, _value_scalar_from_wdl_logits
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import index_to_move_fast, legal_move_indices

try:
    from chess_anti_engine.encoding._lc0_ext import CBoard
    _HAS_CBOARD = True
except ImportError:
    _HAS_CBOARD = False


def _softmax_legal(logits: np.ndarray, legal_idx: np.ndarray) -> np.ndarray:
    """Softmax over legal moves only. Returns priors for each legal index."""
    ll = logits[legal_idx].astype(np.float64)
    ll -= np.max(ll)
    e = np.exp(ll)
    s = float(e.sum())
    return (e / s) if s > 0 else np.full_like(e, 1.0 / e.size)


def _replay_board(root_board: chess.Board, action_path: np.ndarray) -> chess.Board:
    """Replay action path on root board to get leaf board."""
    board = root_board.copy(stack=True)
    for a in action_path:
        move = index_to_move_fast(int(a), board)
        board.push(move)
    return board


def _replay_board_cached(
    board_cache: dict[int, chess.Board],
    node_path: np.ndarray,
    action_path: np.ndarray,
) -> chess.Board:
    """Build leaf board from closest cached ancestor in the path."""
    # Walk backwards through node_path to find cached ancestor
    for depth in range(len(node_path) - 2, -1, -1):
        ancestor_id = int(node_path[depth])
        ancestor_board = board_cache.get(ancestor_id)
        if ancestor_board is not None:
            board = ancestor_board.copy(stack=True)
            for a in action_path[depth:]:
                move = index_to_move_fast(int(a), board)
                board.push(move)
            return board
    # Should not reach here — root is always cached
    raise RuntimeError("No cached ancestor found")


def _terminal_value(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1/2-1/2":
        return 0.0
    if res == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    return 1.0 if board.turn == chess.BLACK else -1.0


@torch.no_grad()
def run_mcts_many_c(
    model: torch.nn.Module | None,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: MCTSConfig,
    evaluator: BatchEvaluator | None = None,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray]]:
    """Run PUCT MCTS using C-accelerated tree.

    Same API as `run_mcts_many` — drop-in replacement.
    """
    n_boards = len(boards)
    if n_boards == 0:
        return [], [], [], []

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_mcts_many_c requires model or evaluator")
        eval_impl = LocalModelEvaluator(
            model,
            device=device,
            use_amp=bool(cfg.use_amp),
            amp_dtype=str(cfg.amp_dtype),
        )

    # ── 1. Root evaluation ───────────────────────────────────────────────
    use_cboard = _HAS_CBOARD
    root_cboards: list[CBoard] | None = [CBoard.from_board(b) for b in boards] if use_cboard else None
    if pre_pol_logits is not None and pre_wdl_logits is not None:
        pol_logits_all = np.asarray(pre_pol_logits, dtype=np.float32)
        wdl_logits_all = np.asarray(pre_wdl_logits, dtype=np.float32)
    else:
        xs = encode_positions_batch(boards, add_features=True)
        pol_logits_all, wdl_logits_all = eval_impl.evaluate_encoded(xs)

    # ── 2. Build C tree + init roots ─────────────────────────────────────
    tree = MCTSTree()
    root_ids = np.empty(n_boards, dtype=np.int32)
    board_cache: dict[int, chess.Board] = {}

    if use_cboard:
        cb_cache: dict[int, CBoard] = {}

    for i, b in enumerate(boards):
        root_board = b.copy(stack=True)
        root_value = _terminal_value(root_board) if root_board.is_game_over() else _value_scalar_from_wdl_logits(wdl_logits_all[i].reshape(-1))
        root_id = tree.add_root(1, root_value)
        root_ids[i] = root_id
        board_cache[root_id] = root_board

        if use_cboard:
            root_cb = root_cboards[i] if root_cboards is not None else CBoard.from_board(b)
            cb_cache[root_id] = root_cb
            legal_idx = root_cb.legal_move_indices()
        else:
            legal_idx = legal_move_indices(root_board)

        if not root_board.is_game_over() and legal_idx.size > 0:
            priors = _softmax_legal(pol_logits_all[i], legal_idx)

            # Add Dirichlet noise at root
            if cfg.dirichlet_eps > 0:
                noise = rng.dirichlet([cfg.dirichlet_alpha] * int(legal_idx.size)).astype(np.float64)
                priors = (1 - cfg.dirichlet_eps) * priors + cfg.dirichlet_eps * noise

            tree.expand(root_id, legal_idx.astype(np.int32), priors)

    # ── 3. Simulations ───────────────────────────────────────────────────
    c_puct = float(cfg.c_puct)
    fpu_root = float(cfg.fpu_at_root)
    fpu_tree = float(cfg.fpu_reduction)

    # Limit cache size: keep root entries + up to 4x simulations of non-root entries.
    _cache_max = n_boards + 4 * int(cfg.simulations)
    _root_id_set = set(int(root_ids[i]) for i in range(n_boards))

    for _ in range(int(cfg.simulations)):
        # Select one leaf per root (all in C)
        leaves = tree.select_leaves(root_ids, c_puct, fpu_root, fpu_tree)

        leaf_data: list[tuple[int, np.ndarray, chess.Board, CBoard | None]] = []
        terminal_backprops: list[tuple[np.ndarray, float]] = []

        for leaf_id, action_path, node_path, is_exp in leaves:
            if is_exp:
                continue

            if len(node_path) >= 2:
                parent_id = int(node_path[-2])
                parent_board = board_cache.get(parent_id)
                if parent_board is not None:
                    board = parent_board.copy(stack=True)
                    board.push(index_to_move_fast(int(action_path[-1]), parent_board))
                else:
                    board = _replay_board_cached(board_cache, node_path, action_path)
            else:
                board = board_cache.get(int(node_path[0]), chess.Board()).copy(stack=True)

            if board.is_game_over():
                terminal_backprops.append((node_path, _terminal_value(board)))
                continue

            if use_cboard:
                # CBoard path: copy + push_index (34x faster)
                if len(node_path) >= 2:
                    parent_id = int(node_path[-2])
                    parent_cb = cb_cache.get(parent_id)
                    if parent_cb is not None:
                        cb = parent_cb.copy()
                        cb.push_index(int(action_path[-1]))
                    else:
                        # Fallback: replay from root
                        root_cb = cb_cache[int(node_path[0])]
                        cb = root_cb.copy()
                        for a in action_path:
                            cb.push_index(int(a))
                else:
                    cb = cb_cache.get(int(node_path[0]))
                    if cb is not None:
                        cb = cb.copy()
                    else:
                        cb = CBoard.from_board(board)
            else:
                cb = None
            leaf_data.append((leaf_id, node_path, board, cb))

        # Backprop terminals
        if terminal_backprops:
            t_paths = [tb[0] for tb in terminal_backprops]
            t_values = [tb[1] for tb in terminal_backprops]
            tree.backprop_many(t_paths, t_values)

        if not leaf_data:
            continue

        leaf_xs = encode_positions_batch([ld[2] for ld in leaf_data], add_features=True)

        pol_batch, wdl_batch = eval_impl.evaluate_encoded(leaf_xs)
        q_values = tree.batch_wdl_to_q(wdl_batch.reshape(-1, 3))

        node_paths = []
        values = []
        for j, (leaf_id, node_path, board, cb) in enumerate(leaf_data):
            legal_idx = (cb.legal_move_indices() if cb is not None
                         else legal_move_indices(board))
            if legal_idx.size > 0:
                tree.expand_from_logits(leaf_id, legal_idx.astype(np.int32), pol_batch[j])
            else:
                tree.expand(leaf_id, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))
            board_cache[leaf_id] = board
            if use_cboard and cb is not None:
                cb_cache[leaf_id] = cb
            node_paths.append(node_path)
            values.append(float(q_values[j]))
        tree.backprop_many(node_paths, values)

        if use_cboard and len(cb_cache) > _cache_max:
            evict = [k for k in cb_cache if k not in _root_id_set]
            for k in evict[:len(evict) // 2]:
                del cb_cache[k]
        if len(board_cache) > _cache_max:
            evict = [k for k in board_cache if k not in _root_id_set]
            for k in evict[:len(evict) // 2]:
                del board_cache[k]

    # ── 4. Extract results ───────────────────────────────────────────────
    probs_list: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    legal_masks: list[np.ndarray] = []

    for i in range(n_boards):
        root_id = int(root_ids[i])
        child_actions, child_visits = tree.get_children_visits(root_id)

        probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
        mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)

        if child_actions.size > 0:
            visits_f = child_visits.astype(np.float32)
            s = float(visits_f.sum())
            if s > 0:
                probs[child_actions] = visits_f / s
            mask[child_actions] = True

        # Action selection (sample over children only, not full POLICY_SIZE)
        if child_actions.size == 0:
            action = 0
        elif cfg.temperature <= 0:
            action = int(child_actions[np.argmax(child_visits)])
        else:
            cp = visits_f.astype(np.float64)
            if cfg.temperature != 1.0:
                cp = np.power(np.maximum(cp, 0.0), 1.0 / float(cfg.temperature))
            cps = float(cp.sum())
            if not np.isfinite(cps) or cps <= 0:
                action = int(child_actions[np.argmax(child_visits)])
            else:
                cp /= cps
                action = int(child_actions[rng.choice(child_actions.size, p=cp)])

        probs_list.append(probs)
        actions.append(action)
        values.append(float(tree.node_q(root_id)))
        legal_masks.append(mask)

    return probs_list, actions, values, legal_masks


@torch.no_grad()
def run_mcts_c(
    model: torch.nn.Module | None,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: MCTSConfig,
    evaluator: BatchEvaluator | None = None,
) -> tuple[np.ndarray, int, float]:
    probs_list, actions, values, _masks = run_mcts_many_c(
        model, [board], device=device, rng=rng, cfg=cfg, evaluator=evaluator,
    )
    return probs_list[0], actions[0], float(values[0])
