from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.inference import BatchEvaluator, LocalModelEvaluator
from chess_anti_engine.mcts.puct import (
    Node,
    _backprop,
    _expand_sparse,
    _select_child,
    _terminal_value,
)
from chess_anti_engine.mcts.puct import (
    _value_scalar_from_wdl_logits as _wdl_to_q,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import legal_move_indices


def _gumbel(rng: np.random.Generator, size: int) -> np.ndarray:
    u = rng.random(size=size)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


from chess_anti_engine.mcts.sampling import sample_action_with_temperature  # noqa: E402
from chess_anti_engine.utils.numpy_helpers import softmax_1d as _softmax  # noqa: E402


@dataclass
class GumbelConfig:
    simulations: int = 50
    topk: int = 16
    temperature: float = 1.0
    c_visit: float = 50.0
    c_scale: float = 1.0
    c_puct: float = 2.5
    fpu_reduction: float = 1.2
    full_tree: bool = True
    add_noise: bool = True  # Gumbel noise at root; disable for max-strength (non-training) search


def _masked_priors(pol_logits: np.ndarray, board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (full_priors, mask, legal_indices)."""
    legal_idx = legal_move_indices(board)
    if legal_idx.size == 0:
        return (np.zeros(POLICY_SIZE, dtype=np.float64),
                np.zeros(POLICY_SIZE, dtype=np.bool_),
                legal_idx)
  # Compute softmax only over legal moves
    legal_logits = pol_logits[legal_idx].astype(np.float64)
    legal_logits -= legal_logits.max()
    e = np.exp(legal_logits)
    s = float(e.sum())
    legal_priors = (e / s) if s > 0 else np.full_like(e, 1.0 / e.size)
  # Scatter into full-size arrays
    pri = np.zeros(POLICY_SIZE, dtype=np.float64)
    pri[legal_idx] = legal_priors
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    mask[legal_idx] = True
    return pri, mask, legal_idx


def _sigma_scale(*, max_visit: int, cfg: GumbelConfig) -> float:
    return float(cfg.c_scale) * (float(cfg.c_visit) + float(max_visit))


def _completed_q(*, root_q: float, root: Node, action: int) -> float:
    child = root.children.get(int(action))
    if child is None or child.N <= 0:
        return float(root_q)
    return float(-child.Q)


def _root_score(
    *,
    log_prior: float,
    gumbel: float,
    q_hat: float,
    max_visit: int,
    cfg: GumbelConfig,
) -> float:
    return float(gumbel + log_prior + _sigma_scale(max_visit=max_visit, cfg=cfg) * q_hat)


def _improved_policy_probs(
    *,
    node: Node,
    cfg: GumbelConfig,
) -> tuple[list[int], np.ndarray]:
    children = node.children
    actions = list(children.keys())
    if not actions:
        return [], np.zeros((0,), dtype=np.float64)

    n_act = len(actions)
    logits = np.empty(n_act, dtype=np.float64)
    completed_q = np.empty(n_act, dtype=np.float64)
    max_visit = 0
    v_pi = node.Q

    for i, a in enumerate(actions):
        ch = children[a]
        n = ch.N
        if n > max_visit:
            max_visit = n
        logits[i] = math.log(max(ch.prior, 1e-12))
        completed_q[i] = (-ch.W / n) if n > 0 else v_pi

    probs = _softmax(logits + _sigma_scale(max_visit=max_visit, cfg=cfg) * completed_q)
    return actions, probs


def _select_full_gumbel_child(node: Node, *, cfg: GumbelConfig) -> tuple[int, Node]:
    children = node.children
    actions, probs = _improved_policy_probs(node=node, cfg=cfg)
    if not actions:
        raise ValueError("Cannot select from an unexpanded node with no children")

    total_visits = 0
    for a in actions:
        total_visits += children[a].N
    inv_total = 1.0 / float(1 + total_visits)

    best_idx = 0
    best_score = -1e30
    for i, a in enumerate(actions):
        score = float(probs[i]) - float(children[a].N) * inv_total
        if score > best_score:
            best_score = score
            best_idx = i
    a = int(actions[best_idx])
    return a, children[a]


def _init_root_from_logits(
    board: chess.Board,
    *,
    pol_logits: np.ndarray,
    root_q: float,
) -> tuple[Node, np.ndarray, np.ndarray]:
    root = Node(board.copy(stack=True), parent=None, prior=1.0)
    if root.board.is_game_over():
        root.N = 1
        root.W = _terminal_value(root.board)
        zeros = np.zeros((POLICY_SIZE,), dtype=np.float64)
        return root, zeros, zeros.astype(np.bool_)
    pri, mask, legal_idx = _masked_priors(pol_logits, root.board)
    if legal_idx.size > 0:
        _expand_sparse(root, legal_idx, pri[legal_idx])
    root.N = 1
    root.W = float(root_q)
    return root, pri, mask


def _collect_forced_leaf(
    *,
    root: Node,
    forced_action: int,
    cfg: GumbelConfig,
) -> tuple[Node | None, list[Node], float | None]:
    child = root.children.get(int(forced_action))
    if child is None:
        return None, [root], float(root.Q)

    node = child
    path = [root, child]
    while node.expanded and node.children:
        if cfg.full_tree:
            _, node = _select_full_gumbel_child(node, cfg=cfg)
        else:
            _, node = _select_child(node, c_puct=float(cfg.c_puct), fpu_reduction=float(cfg.fpu_reduction))
        path.append(node)
  # Expanded nodes with children are never terminal — skip is_game_over()
  # here. Terminal detection happens after the loop exits.

    if node.board.is_game_over():
        return None, path, _terminal_value(node.board)
    return node, path, None


@torch.no_grad()
def run_gumbel_root_many(
    model: torch.nn.Module | None,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
    evaluator: BatchEvaluator | None = None,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
    per_game_simulations: list[int] | None = None,  # skylos: ignore  # pylint: disable=unused-argument  # API parity with run_gumbel_root_many_async
    per_game_add_noise: list[bool] | None = None,  # skylos: ignore  # pylint: disable=unused-argument  # API parity with run_gumbel_root_many_async
) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray]]:
    """Root Gumbel search with sequential halving.

    This follows the paper's root-search structure much more closely than the
    previous shallow approximation:
      1. Evaluate the root once for priors + root value.
      2. Sample top-m root actions using gumbel(log π(a)).
      3. Allocate actual subtree simulations to those candidates via
         sequential halving, forcing each simulation through a chosen root
         action and then using ordinary tree search below the root.
      4. Build the returned policy from completed-Q policy improvement on the
         searched root.

    Returns (probs_list, actions, root_values), where root_values are the best
    searched child values from the root perspective.
    """
    n_boards = len(boards)
    if n_boards == 0:
        return [], [], [], []

    sim_budget = max(1, int(cfg.simulations))

  # ── 1. Batch root evaluation ─────────────────────────────────────────────
    if pre_pol_logits is not None and pre_wdl_logits is not None:
  # Reuse logits computed by the caller (saves one forward pass per ply).
        pol_logits_batch = np.asarray(pre_pol_logits, dtype=np.float32)  # (B, POLICY_SIZE)
        wdl_logits_batch = np.asarray(pre_wdl_logits, dtype=np.float32)  # (B, 3)
    else:
        xs = encode_positions_batch(boards, add_features=True)
        eval_impl = evaluator
        if eval_impl is None:
            if model is None:
                raise ValueError("run_gumbel_root_many requires model or evaluator")
            eval_impl = LocalModelEvaluator(model, device=device)
        pol_logits_batch, wdl_logits_batch = eval_impl.evaluate_encoded(xs)

    root_qs = [_wdl_to_q(wdl_logits_batch[i]) for i in range(n_boards)]

  # ── 2. Per-board root init + Gumbel candidate selection ──────────────────
    probs_out: list[np.ndarray | None] = [None] * n_boards
    actions_out: list[int | None] = [None] * n_boards
    values_out: list[float] = list(root_qs)

    roots: list[Node | None] = [None] * n_boards
    priors: list[np.ndarray | None] = [None] * n_boards
    candidates_per_board: list[list[int] | None] = [None] * n_boards
    remaining_per_board: list[list[int] | None] = [None] * n_boards
    budget_remaining: list[int] = [sim_budget] * n_boards
    gumbels_per_board: list[dict[int, float] | None] = [None] * n_boards

    for i, b in enumerate(boards):
        root, pri, mask = _init_root_from_logits(
            b,
            pol_logits=pol_logits_batch[i],
            root_q=float(root_qs[i]),
        )
        roots[i] = root
        priors[i] = pri

        if root.board.is_game_over():
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            values_out[i] = float(root.Q)
            continue

        legal = np.nonzero(mask)[0]

        if legal.size == 0:
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            continue

        if legal.size == 1:
            a0 = int(legal[0])
            p = np.zeros((POLICY_SIZE,), dtype=np.float32)
            p[a0] = 1.0
            probs_out[i] = p
            actions_out[i] = a0
            continue

  # Gumbel noise → select top-m. Keep m small enough that sequential
  # halving can still allocate at least one visit per action each phase.
        log_pri = np.log(np.maximum(pri[legal], 1e-12))
        g = _gumbel(rng, legal.size) if cfg.add_noise else np.zeros(legal.size, dtype=np.float64)
        score: np.ndarray = g + log_pri

        if sim_budget <= 1:
            m = 1
        else:
            m_cap = max(2, (sim_budget + 1) // 2)
            m = int(min(int(cfg.topk), int(legal.size), int(m_cap)))
            m = max(2, m)

        kth = min(m - 1, int(score.size) - 1)
        top_idx = np.argpartition(-score, kth)[:m]
        cands = legal[top_idx].astype(int).tolist()

        candidates_per_board[i] = cands
        remaining_per_board[i] = list(cands)
        gumbels_per_board[i] = {int(a): float(gg) for a, gg in zip(legal.tolist(), g.tolist(), strict=True)}

  # ── 3. Sequential halving with real subtree simulations ──────────────────
  # Resolve evaluator once for all leaf evaluations below.
    leaf_eval: BatchEvaluator | None = evaluator
    if leaf_eval is None and model is not None:
        leaf_eval = LocalModelEvaluator(model, device=device)

    while True:
        active = []
        for i in range(n_boards):
            rem_i = remaining_per_board[i]
            if (
                probs_out[i] is None
                and rem_i is not None
                and len(rem_i) >= 1
                and budget_remaining[i] > 0
            ):
                active.append(i)
        if not active:
            break

        visits_per_action: dict[int, int] = {}
        for bi in active:
            rem = remaining_per_board[bi]
            assert rem is not None
            if len(rem) <= 1:
                visits_per_action[bi] = int(budget_remaining[bi])
                continue
            rounds_left = int(np.ceil(np.log2(len(rem))))
            vpa = int(budget_remaining[bi] // max(1, len(rem) * rounds_left))
            visits_per_action[bi] = max(1, vpa)

        max_reps = max(visits_per_action.values(), default=0)
        for rep in range(max_reps):
            leaf_nodes: list[Node] = []
            leaf_paths: list[list[Node]] = []
            leaf_entries: list[tuple[int, int]] = []

            for bi in active:
                rem = remaining_per_board[bi]
                root = roots[bi]
                if rem is None or root is None or rep >= visits_per_action[bi]:
                    continue
                for action in rem:
                    leaf, path, terminal_value = _collect_forced_leaf(
                        root=root,
                        forced_action=int(action),
                        cfg=cfg,
                    )
                    if terminal_value is not None:
                        _backprop(path, float(terminal_value))
                    elif leaf is not None:
                        leaf_nodes.append(leaf)
                        leaf_paths.append(path)
                        leaf_entries.append((bi, int(action)))

            if not leaf_nodes:
                continue

            leaf_xs = encode_positions_batch([node.board for node in leaf_nodes], add_features=True)
            if leaf_eval is None:
                raise ValueError("run_gumbel_root_many requires model or evaluator")
            pol_logits_leaf, wdl_logits_leaf = leaf_eval.evaluate_encoded(leaf_xs)

            for node, path, pol_logits, wdl_logits in zip(
                leaf_nodes, leaf_paths, pol_logits_leaf, wdl_logits_leaf, strict=True
            ):
                pri, _, legal_idx = _masked_priors(pol_logits, node.board)
                if legal_idx.size > 0:
                    _expand_sparse(node, legal_idx, pri[legal_idx])
                _backprop(path, _wdl_to_q(wdl_logits.reshape(-1)))

        for bi in active:
            rem = remaining_per_board[bi]
            root = roots[bi]
            pri = priors[bi]
            gmap = gumbels_per_board[bi]
            if rem is None or root is None or pri is None or gmap is None:
                continue
  # Re-bind as non-Optional locals so pyright narrows inside the lambda below.
            pri_nn = pri
            gmap_nn = gmap
            root_nn = root

            budget_remaining[bi] = max(0, int(budget_remaining[bi] - visits_per_action[bi] * len(rem)))
            if len(rem) <= 1:
                continue

            max_visit = max((root_nn.children[int(a)].N for a in rem if int(a) in root_nn.children), default=0)
            rem.sort(
                key=lambda a: _root_score(
                    log_prior=float(np.log(max(float(pri_nn[int(a)]), 1e-12))),
                    gumbel=float(gmap_nn.get(int(a), 0.0)),
                    q_hat=_completed_q(root_q=float(root_qs[bi]), root=root_nn, action=int(a)),
                    max_visit=int(max_visit),
                    cfg=cfg,
                ),
                reverse=True,
            )
            remaining_per_board[bi] = rem[: max(1, (len(rem) + 1) // 2)]

  # ── 4. Build improved policies ────────────────────────────────────────────
    for i in range(n_boards):
        if probs_out[i] is not None:
            continue  # handled (empty / single-legal cases)

        root = roots[i]
        cands = candidates_per_board[i]
        pri = priors[i]
        remaining = remaining_per_board[i]
        if root is None or cands is None or pri is None or remaining is None:
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            continue

        legal = np.nonzero(pri > 0)[0].astype(int)
        max_visit = max((root.children[int(a)].N for a in legal if int(a) in root.children), default=0)
        completed_q = np.array(
            [_completed_q(root_q=float(root_qs[i]), root=root, action=int(a)) for a in legal],
            dtype=np.float64,
        )
        logits_imp = np.log(np.maximum(pri[legal], 1e-12)) + _sigma_scale(
            max_visit=int(max_visit),
            cfg=cfg,
        ) * completed_q
        imp_all = _softmax(logits_imp)
        probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
        probs[legal] = imp_all.astype(np.float32)

        best_a = int(remaining[0])
  # Gumbel sequential halving leaves the survivor at remaining[0]; map
  # that back to its position in the full ``legal`` array (= imp_all).
        argmax_idx = int(np.searchsorted(legal, best_a)) if legal.size > 0 else 0
        action = sample_action_with_temperature(
            rng, legal, imp_all, float(cfg.temperature),
            argmax_idx=argmax_idx,
        )

        probs_out[i] = probs
        actions_out[i] = action

        values_out[i] = _completed_q(root_q=float(root_qs[i]), root=root, action=best_a)

  # Build legal masks from root children
    legal_masks_out: list[np.ndarray] = []
    for i in range(n_boards):
        root = roots[i]
        mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
        if root is not None:
            for a in root.children:
                mask[a] = True
        legal_masks_out.append(mask)

  # Every slot of probs_out/actions_out is set above (terminal/fallback/main paths).
    return (
        cast(list[np.ndarray], probs_out),
        cast(list[int], actions_out),
        values_out,
        legal_masks_out,
    )


@torch.no_grad()
def run_gumbel_root(
    model: torch.nn.Module,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, int, float]:
    probs, acts, vals, _masks = run_gumbel_root_many(model, [board], device=device, rng=rng, cfg=cfg)
    return probs[0], acts[0], float(vals[0])
