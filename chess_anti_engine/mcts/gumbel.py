from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask, index_to_move
from chess_anti_engine.mcts.puct import Node, _backprop, _expand, _select_child, _terminal_value
from chess_anti_engine.utils.amp import inference_autocast


def _gumbel(rng: np.random.Generator, size: int) -> np.ndarray:
    # Sample Gumbel(0,1): -log(-log(U))
    u = rng.random(size=size)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = float(e.sum())
    return e / s if s > 0 else np.full_like(x, 1.0 / x.size)


def _wdl_to_q(wdl_logits: np.ndarray) -> float:
    """WDL logits → scalar Q ∈ [-1, 1] from side-to-move perspective (W-L)."""
    p = _softmax(wdl_logits.astype(np.float64))
    return float(p[0] - p[2])


@dataclass
class GumbelConfig:
    simulations: int = 50
    topk: int = 16
    child_sims: int = 8  # kept for API compat, no longer used
    temperature: float = 1.0
    c_visit: float = 50.0
    c_scale: float = 1.0
    c_puct: float = 2.5
    fpu_reduction: float = 1.2
    full_tree: bool = True


def _masked_priors(pol_logits: np.ndarray, board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    mask = legal_move_mask(board)
    pl = pol_logits.astype(np.float64, copy=True)
    pl[~mask] = -1e9
    pri = _softmax(pl)
    pri[~mask] = 0.0
    return pri, mask


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
    actions = list(node.children.keys())
    if not actions:
        return [], np.zeros((0,), dtype=np.float64)

    max_visit = max((node.children[a].N for a in actions), default=0)
    logits = np.array(
        [np.log(max(float(node.children[a].prior), 1e-12)) for a in actions],
        dtype=np.float64,
    )
    v_pi = float(node.Q)
    completed_q = np.array(
        [
            float(-node.children[a].Q) if node.children[a].N > 0 else v_pi
            for a in actions
        ],
        dtype=np.float64,
    )
    probs = _softmax(logits + _sigma_scale(max_visit=max_visit, cfg=cfg) * completed_q)
    return actions, probs


def _select_full_gumbel_child(node: Node, *, cfg: GumbelConfig) -> tuple[int, Node]:
    actions, probs = _improved_policy_probs(node=node, cfg=cfg)
    if not actions:
        raise ValueError("Cannot select from an unexpanded node with no children")

    total_visits = sum(int(node.children[a].N) for a in actions)
    scores = np.array(
        [
            float(probs[i]) - (float(node.children[a].N) / float(1 + total_visits))
            for i, a in enumerate(actions)
        ],
        dtype=np.float64,
    )
    best_idx = int(np.argmax(scores))
    a = int(actions[best_idx])
    return a, node.children[a]


def _init_root_from_logits(
    board: chess.Board,
    *,
    pol_logits: np.ndarray,
    root_q: float,
) -> tuple[Node, np.ndarray, np.ndarray]:
    root = Node(board.copy(stack=False), parent=None, prior=1.0)
    pri, mask = _masked_priors(pol_logits, root.board)
    _expand(root, pri)
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
        if bool(cfg.full_tree):
            _, node = _select_full_gumbel_child(node, cfg=cfg)
        else:
            _, node = _select_child(node, c_puct=float(cfg.c_puct), fpu_reduction=float(cfg.fpu_reduction))
        path.append(node)
        if node.board.is_game_over():
            return None, path, _terminal_value(node.board)

    if node.board.is_game_over():
        return None, path, _terminal_value(node.board)
    return node, path, None


@torch.no_grad()
def run_gumbel_root_many(
    model: torch.nn.Module,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int], list[float]]:
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
        return [], [], []

    sim_budget = max(1, int(cfg.simulations))

    # ── 1. Batch root evaluation ─────────────────────────────────────────────
    if pre_pol_logits is not None and pre_wdl_logits is not None:
        # Reuse logits computed by the caller (saves one forward pass per ply).
        pol_logits_batch = np.asarray(pre_pol_logits, dtype=np.float32)   # (B, POLICY_SIZE)
        wdl_logits_batch = np.asarray(pre_wdl_logits, dtype=np.float32)   # (B, 3)
    else:
        xs = [encode_position(b, add_features=True) for b in boards]
        xt = torch.from_numpy(np.stack(xs, axis=0)).to(device)
        with inference_autocast(device=device, enabled=True, dtype="auto"):
            root_out = model(xt)
        policy_out = root_out["policy"] if "policy" in root_out else root_out["policy_own"]
        pol_logits_batch = policy_out.detach().float().cpu().numpy()  # (B, POLICY_SIZE)
        wdl_logits_batch = root_out["wdl"].detach().float().cpu().numpy()  # (B, 3)

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
        g = _gumbel(rng, legal.size)
        score = g + log_pri

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

            leaf_xs = [encode_position(node.board, add_features=True) for node in leaf_nodes]
            leaf_xt = torch.from_numpy(np.stack(leaf_xs, axis=0)).to(device)
            with inference_autocast(device=device, enabled=True, dtype="auto"):
                leaf_out = model(leaf_xt)
            policy_out = leaf_out["policy"] if "policy" in leaf_out else leaf_out["policy_own"]
            pol_logits_leaf = policy_out.detach().float().cpu().numpy()
            wdl_logits_leaf = leaf_out["wdl"].detach().float().cpu().numpy()

            for node, path, pol_logits, wdl_logits in zip(
                leaf_nodes, leaf_paths, pol_logits_leaf, wdl_logits_leaf, strict=True
            ):
                pri, _ = _masked_priors(pol_logits, node.board)
                _expand(node, pri)
                _backprop(path, _wdl_to_q(wdl_logits.reshape(-1)))

        for bi in active:
            rem = remaining_per_board[bi]
            root = roots[bi]
            pri = priors[bi]
            gmap = gumbels_per_board[bi]
            if rem is None or root is None or pri is None or gmap is None:
                continue

            budget_remaining[bi] = max(0, int(budget_remaining[bi] - visits_per_action[bi] * len(rem)))
            if len(rem) <= 1:
                continue

            max_visit = max((root.children[int(a)].N for a in rem if int(a) in root.children), default=0)
            rem.sort(
                key=lambda a: _root_score(
                    log_prior=float(np.log(max(float(pri[int(a)]), 1e-12))),
                    gumbel=float(gmap.get(int(a), 0.0)),
                    q_hat=_completed_q(root_q=float(root_qs[bi]), root=root, action=int(a)),
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

        values_out[i] = _completed_q(root_q=float(root_qs[i]), root=root, action=best_a)

    return probs_out, actions_out, values_out


@torch.no_grad()
def run_gumbel_root(
    model: torch.nn.Module,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, int, float]:
    probs, acts, vals = run_gumbel_root_many(model, [board], device=device, rng=rng, cfg=cfg)
    return probs[0], acts[0], float(vals[0])
