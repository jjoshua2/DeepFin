"""Detailed profiling of the C-tree MCTS simulation loop.

Breaks down per-simulation costs: C selection, board creation,
terminal checks, encoding, GPU eval, expansion, backprop.

Run:
    PYTHONPATH=. python3 scripts/profile_mcts_detail.py [--compile]
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any, cast

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct import MCTSConfig, _value_scalar_from_wdl_logits
from chess_anti_engine.mcts.puct_c import _softmax_legal, _terminal_value
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves.encode import index_to_move_fast, legal_move_indices

try:
    from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
    from chess_anti_engine.encoding.cboard_encode import encode_cboard as _encode_cboard
except ImportError:
    _CBoard = None
    _encode_cboard = None

_HAS_CBOARD = _CBoard is not None and _encode_cboard is not None


def make_boards(n: int, rng: np.random.Generator) -> list[chess.Board]:
    boards = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(int(rng.integers(0, 40))):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                b = chess.Board()
                break
            b.push(moves[int(rng.integers(0, len(moves)))])
        boards.append(b)
    return boards


def profile_mcts(
    model: torch.nn.Module,
    boards: list[chess.Board],
    device: str,
    cfg: MCTSConfig,
) -> dict[str, float]:
    rng = np.random.default_rng(42)
    eval_impl = LocalModelEvaluator(model, device=device)
    n_boards = len(boards)

    timers: dict[str, float] = defaultdict(float)

    # Root eval
    t0 = time.perf_counter()
    xs = [encode_position(b, add_features=True) for b in boards]
    pol_all, wdl_all = eval_impl.evaluate_encoded(np.stack(xs, axis=0))
    timers["root_eval"] = time.perf_counter() - t0

    # Build tree
    t0 = time.perf_counter()
    tree = MCTSTree()
    root_ids = np.empty(n_boards, dtype=np.int32)
    cboard_cls = _CBoard
    encode_cboard_fn = _encode_cboard
    use_cboard = cboard_cls is not None and encode_cboard_fn is not None

    cb_cache: dict[int, Any] = {}
    board_cache: dict[int, chess.Board] = {}

    for i, b in enumerate(boards):
        root_q = _value_scalar_from_wdl_logits(wdl_all[i].reshape(-1))
        root_id = tree.add_root(1, root_q)
        root_ids[i] = root_id

        if use_cboard:
            assert cboard_cls is not None
            root_cb = cboard_cls.from_board(b)
            cb_cache[root_id] = root_cb
            legal_idx = root_cb.legal_move_indices()
        else:
            root_board = b.copy(stack=False)
            board_cache[root_id] = root_board
            legal_idx = legal_move_indices(root_board)

        if legal_idx.size > 0:
            priors = _softmax_legal(pol_all[i], legal_idx)
            if cfg.dirichlet_eps > 0:
                noise = rng.dirichlet([cfg.dirichlet_alpha] * int(legal_idx.size)).astype(np.float64)
                priors = (1 - cfg.dirichlet_eps) * priors + cfg.dirichlet_eps * noise
            tree.expand(root_id, legal_idx.astype(np.int32), priors)
    timers["tree_init"] = time.perf_counter() - t0

    c_puct = float(cfg.c_puct)
    fpu_root = float(cfg.fpu_at_root)
    fpu_tree = float(cfg.fpu_reduction)
    total_leaves = 0
    total_terminals = 0

    for _ in range(int(cfg.simulations)):
        # C selection
        t0 = time.perf_counter()
        leaves = tree.select_leaves(root_ids, c_puct, fpu_root, fpu_tree)
        timers["c_select"] += time.perf_counter() - t0

        # Board construction + terminal check
        t0 = time.perf_counter()
        leaf_data = []
        terminal_paths = []
        terminal_values = []

        for i, (leaf_id, action_path, node_path, is_exp) in enumerate(leaves):
            if is_exp:
                continue

            if use_cboard:
                assert cboard_cls is not None
                if len(node_path) >= 2:
                    parent_id = int(node_path[-2])
                    parent_cb = cb_cache.get(parent_id)
                    if parent_cb is not None:
                        cb = parent_cb.copy()
                        cb.push_index(int(action_path[-1]))
                    else:
                        root_cb = cb_cache[int(node_path[0])]
                        cb = root_cb.copy()
                        for a in action_path:
                            cb.push_index(int(a))
                else:
                    cb = cb_cache.get(int(node_path[0]))
                    if cb is not None:
                        cb = cb.copy()
                    else:
                        cb = cboard_cls.from_board(chess.Board())

                if cb.is_game_over():
                    terminal_paths.append(node_path)
                    terminal_values.append(cb.terminal_value())
                    total_terminals += 1
                    continue
                leaf_data.append((leaf_id, node_path, cb))
                total_leaves += 1
            else:
                if len(node_path) >= 2:
                    parent_id = int(node_path[-2])
                    parent_board = board_cache.get(parent_id)
                    if parent_board is not None:
                        board = parent_board.copy(stack=False)
                        board.push(index_to_move_fast(int(action_path[-1]), parent_board))
                    else:
                        board = board_cache[int(root_ids[i])].copy(stack=False)
                        for a in action_path:
                            board.push(index_to_move_fast(int(a), board))
                else:
                    board = board_cache.get(int(node_path[0]), chess.Board())

                if board.is_game_over():
                    terminal_paths.append(node_path)
                    terminal_values.append(_terminal_value(board))
                    total_terminals += 1
                    continue
                leaf_data.append((leaf_id, node_path, board))
                total_leaves += 1
        timers["board_build"] += time.perf_counter() - t0

        # Terminal backprop
        t0 = time.perf_counter()
        if terminal_paths:
            tree.backprop_many(terminal_paths, terminal_values)
        timers["terminal_backprop"] += time.perf_counter() - t0

        if not leaf_data:
            continue

        # Encoding
        t0 = time.perf_counter()
        if use_cboard:
            assert encode_cboard_fn is not None
            leaf_xs = [encode_cboard_fn(ld[2]) for ld in leaf_data]
        else:
            leaf_xs = [encode_position(ld[2], add_features=True) for ld in leaf_data]
        xb = np.stack(leaf_xs, axis=0)
        timers["encode"] += time.perf_counter() - t0

        # GPU eval
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pol_batch, wdl_batch = eval_impl.evaluate_encoded(xb)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timers["gpu_eval"] += time.perf_counter() - t0

        # Vectorized WDL → Q
        t0 = time.perf_counter()
        q_values = tree.batch_wdl_to_q(wdl_batch.reshape(-1, 3))
        timers["wdl_to_q"] += time.perf_counter() - t0

        # Expand + cache boards
        t0 = time.perf_counter()
        node_paths = []
        values = []
        if use_cboard:
            for j, (leaf_id, node_path, cb) in enumerate(leaf_data):
                legal_idx = cb.legal_move_indices()
                if legal_idx.size > 0:
                    tree.expand_from_logits(leaf_id, legal_idx.astype(np.int32), pol_batch[j])
                else:
                    tree.expand(leaf_id, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))
                cb_cache[leaf_id] = cb
                node_paths.append(node_path)
                values.append(float(q_values[j]))
        else:
            for j, (leaf_id, node_path, board) in enumerate(leaf_data):
                legal_idx = legal_move_indices(board)
                if legal_idx.size > 0:
                    tree.expand_from_logits(leaf_id, legal_idx.astype(np.int32), pol_batch[j])
                else:
                    tree.expand(leaf_id, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))
                board_cache[leaf_id] = board
                node_paths.append(node_path)
                values.append(float(q_values[j]))
        timers["expand"] += time.perf_counter() - t0

        # Backprop
        t0 = time.perf_counter()
        tree.backprop_many(node_paths, values)
        timers["backprop"] += time.perf_counter() - t0

    timers["total_leaves"] = total_leaves
    timers["total_terminals"] = total_terminals
    return dict(timers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--boards", type=int, default=32)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.boards <= 0:
        raise SystemExit("--boards must be > 0")
    if args.simulations <= 0:
        raise SystemExit("--simulations must be > 0")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")
    rng = np.random.default_rng(0)

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=384, num_layers=8,
        num_heads=12, ffn_mult=2, use_smolgen=True,
    )).to(device)
    model.eval()
    if args.compile and device.startswith("cuda"):
        model = cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))
        for _ in range(5):
            model(torch.randn(4, 146, 8, 8, device=device))
        torch.cuda.synchronize()

    boards = make_boards(args.boards, rng)
    cfg = MCTSConfig(simulations=args.simulations)

    # Warmup
    profile_mcts(model, boards[:2], device, MCTSConfig(simulations=4))

    all_timers = []
    for _ in range(args.repeats):
        t = profile_mcts(model, boards, device, cfg)
        all_timers.append(t)

    # Average
    keys = [k for k in all_timers[0] if k not in ("total_leaves", "total_terminals")]
    avg = {k: np.mean([t[k] for t in all_timers]) for k in keys}
    total = sum(avg.values())

    print(f"Device: {device}, Boards: {args.boards}, Sims: {args.simulations}, Compile: {args.compile}, CBoard: {_HAS_CBOARD}")
    print(f"Total leaves: {int(all_timers[0]['total_leaves'])}, terminals: {int(all_timers[0]['total_terminals'])}")
    print()
    print(f"{'Phase':<20s} {'Time (ms)':>10s} {'%':>6s} {'Per-leaf (µs)':>14s}")
    print("-" * 55)

    leaves = max(1, int(all_timers[0]["total_leaves"]))
    for k in ["root_eval", "tree_init", "c_select", "board_build", "terminal_backprop",
              "encode", "gpu_eval", "wdl_to_q", "expand", "backprop"]:
        v = avg.get(k, 0)
        pct = v / total * 100 if total > 0 else 0
        per_leaf = v / leaves * 1e6
        print(f"  {k:<18s} {v*1000:10.1f} {pct:6.1f}% {per_leaf:14.1f}")

    print("-" * 55)
    print(f"  {'TOTAL':<18s} {total*1000:10.1f}")


if __name__ == "__main__":
    main()
