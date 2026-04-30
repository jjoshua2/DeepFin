from __future__ import annotations

import math
from dataclasses import dataclass

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


def _resolve_root_logits(
    boards: list[chess.Board],
    *,
    model: torch.nn.Module | None,
    evaluator: BatchEvaluator | None,
    device: str,
    pre_pol_logits: np.ndarray | None,
    pre_wdl_logits: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, BatchEvaluator | None]:
    """Phase 1: get root pol/wdl logits + a leaf evaluator for later phases.

    Reuses caller-provided ``pre_*_logits`` (one forward pass saved per ply
    when called from selfplay). Always resolves a ``leaf_eval`` for use by
    sequential halving — even when pre-logits short-circuit phase 1.
    """
    if pre_pol_logits is not None and pre_wdl_logits is not None:
        pol = np.asarray(pre_pol_logits, dtype=np.float32)
        wdl = np.asarray(pre_wdl_logits, dtype=np.float32)
        leaf_eval = evaluator if evaluator is not None else (
            LocalModelEvaluator(model, device=device) if model is not None else None
        )
        return pol, wdl, leaf_eval

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_gumbel_root_many requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)
    xs = encode_positions_batch(boards, add_features=True)
    pol, wdl = eval_impl.evaluate_encoded(xs)
    return pol, wdl, eval_impl


def _evaluate_and_backprop_leaves(
    leaf_nodes: list[Node],
    leaf_paths: list[list[Node]],
    leaf_eval: BatchEvaluator | None,
) -> None:
    """Batched leaf NN eval + expand + backprop. No-op when ``leaf_nodes`` is empty."""
    if not leaf_nodes:
        return
    if leaf_eval is None:
        raise ValueError("run_gumbel_root_many requires model or evaluator")
    leaf_xs = encode_positions_batch([node.board for node in leaf_nodes], add_features=True)
    pol_logits_leaf, wdl_logits_leaf = leaf_eval.evaluate_encoded(leaf_xs)
    for node, path, pol_logits, wdl_logits in zip(
        leaf_nodes, leaf_paths, pol_logits_leaf, wdl_logits_leaf, strict=True,
    ):
        pri, _, legal_idx = _masked_priors(pol_logits, node.board)
        if legal_idx.size > 0:
            _expand_sparse(node, legal_idx, pri[legal_idx])
        _backprop(path, _wdl_to_q(wdl_logits.reshape(-1)))


def _select_top_m_with_gumbel(
    *,
    legal: np.ndarray,
    pri: np.ndarray,
    sim_budget: int,
    topk: int,
    add_noise: bool,
    rng: np.random.Generator,
) -> tuple[list[int], dict[int, float]]:
    """Sample top-m root actions via Gumbel(logit + noise). Caller filters trivial cases.

    Returns (cands, gumbels_for_all_legal). ``m`` is bounded so sequential halving
    can still allocate ≥1 visit per action per round.
    """
    log_pri = np.log(np.maximum(pri[legal], 1e-12))
    g = _gumbel(rng, legal.size) if add_noise else np.zeros(legal.size, dtype=np.float64)
    score: np.ndarray = g + log_pri

    if sim_budget <= 1:
        m = 1
    else:
        m_cap = max(2, (sim_budget + 1) // 2)
        m = max(2, int(min(int(topk), int(legal.size), int(m_cap))))

    kth = min(m - 1, int(score.size) - 1)
    top_idx = np.argpartition(-score, kth)[:m]
    cands = legal[top_idx].astype(int).tolist()
    gumbels = {int(a): float(gg) for a, gg in zip(legal.tolist(), g.tolist(), strict=True)}
    return cands, gumbels


@dataclass
class _BoardSearchState:
    """Per-board state for sequential halving. ``finished`` short-circuits halving."""
    root: Node
    priors: np.ndarray
    candidates: list[int] | None
    remaining: list[int] | None
    gumbels: dict[int, float] | None
    finished_probs: np.ndarray | None
    finished_action: int | None
    finished_value: float | None


def _init_board_search_state(
    board: chess.Board,
    *,
    pol_logits: np.ndarray,
    root_q: float,
    sim_budget: int,
    cfg: GumbelConfig,
    rng: np.random.Generator,
) -> _BoardSearchState:
    """Phase 2 per-board: init root, early-exit trivial cases, else select top-m."""
    root, pri, mask = _init_root_from_logits(board, pol_logits=pol_logits, root_q=root_q)

    def _finish(probs: np.ndarray, action: int, value: float) -> _BoardSearchState:
        return _BoardSearchState(
            root=root, priors=pri, candidates=None, remaining=None, gumbels=None,
            finished_probs=probs, finished_action=action, finished_value=value,
        )

    if root.board.is_game_over():
        return _finish(np.zeros((POLICY_SIZE,), dtype=np.float32), 0, float(root.Q))

    legal = np.nonzero(mask)[0]
    if legal.size == 0:
        return _finish(np.zeros((POLICY_SIZE,), dtype=np.float32), 0, root_q)

    if legal.size == 1:
        a0 = int(legal[0])
        p = np.zeros((POLICY_SIZE,), dtype=np.float32)
        p[a0] = 1.0
        return _finish(p, a0, root_q)

    cands, gumbels = _select_top_m_with_gumbel(
        legal=legal, pri=pri, sim_budget=sim_budget,
        topk=int(cfg.topk), add_noise=cfg.add_noise, rng=rng,
    )
    return _BoardSearchState(
        root=root, priors=pri,
        candidates=cands, remaining=list(cands), gumbels=gumbels,
        finished_probs=None, finished_action=None, finished_value=None,
    )


def _collect_forced_leaves_round(
    *,
    active: list[int],
    states: list[_BoardSearchState],
    visits_per_action: dict[int, int],
    rep: int,
    cfg: GumbelConfig,
) -> tuple[list[Node], list[list[Node]]]:
    """One sequential-halving round: walk forced lines from root, collect non-terminal leaves.

    Backprops terminal-value paths immediately; returns leaves that need NN eval.
    """
    leaf_nodes: list[Node] = []
    leaf_paths: list[list[Node]] = []
    for bi in active:
        st = states[bi]
        rem = st.remaining
        if rem is None or rep >= visits_per_action[bi]:
            continue
        for action in rem:
            leaf, path, terminal_value = _collect_forced_leaf(
                root=st.root, forced_action=int(action), cfg=cfg,
            )
            if terminal_value is not None:
                _backprop(path, float(terminal_value))
            elif leaf is not None:
                leaf_nodes.append(leaf)
                leaf_paths.append(path)
    return leaf_nodes, leaf_paths


def _halve_remaining_for_board(
    st: _BoardSearchState,
    *,
    root_q: float,
    cfg: GumbelConfig,
) -> None:
    """Re-rank ``st.remaining`` by completed-Q and halve it. No-op when ≤1 candidate left."""
    rem = st.remaining
    if rem is None or st.gumbels is None or len(rem) <= 1:
        return
    pri = st.priors
    gmap = st.gumbels
    root = st.root
    max_visit = max(
        (root.children[int(a)].N for a in rem if int(a) in root.children),
        default=0,
    )
    rem.sort(
        key=lambda a: _root_score(
            log_prior=float(np.log(max(float(pri[int(a)]), 1e-12))),
            gumbel=float(gmap.get(int(a), 0.0)),
            q_hat=_completed_q(root_q=root_q, root=root, action=int(a)),
            max_visit=int(max_visit),
            cfg=cfg,
        ),
        reverse=True,
    )
    st.remaining = rem[: max(1, (len(rem) + 1) // 2)]


def _build_improved_policy_for_board(
    st: _BoardSearchState,
    *,
    root_q: float,
    cfg: GumbelConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float]:
    """Phase 4: completed-Q policy improvement at the searched root + temperature sample."""
    root = st.root
    pri = st.priors
    remaining = st.remaining
    if remaining is None or st.candidates is None:
        return np.zeros((POLICY_SIZE,), dtype=np.float32), 0, root_q

    legal = np.nonzero(pri > 0)[0].astype(int)
    max_visit = max(
        (root.children[int(a)].N for a in legal if int(a) in root.children),
        default=0,
    )
    completed_q = np.array(
        [_completed_q(root_q=root_q, root=root, action=int(a)) for a in legal],
        dtype=np.float64,
    )
    logits_imp = np.log(np.maximum(pri[legal], 1e-12)) + _sigma_scale(
        max_visit=int(max_visit), cfg=cfg,
    ) * completed_q
    imp_all = _softmax(logits_imp)
    probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
    probs[legal] = imp_all.astype(np.float32)

    best_a = int(remaining[0])
  # Gumbel sequential halving leaves the survivor at remaining[0]; map
  # that back to its position in the full ``legal`` array (= imp_all).
    argmax_idx = int(np.searchsorted(legal, best_a)) if legal.size > 0 else 0
    action = sample_action_with_temperature(
        rng, legal, imp_all, float(cfg.temperature), argmax_idx=argmax_idx,
    )
    value = _completed_q(root_q=root_q, root=root, action=best_a)
    return probs, action, value


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

  # ── 1. Batch root evaluation + resolve leaf evaluator for phase 3 ────────
    pol_logits_batch, wdl_logits_batch, leaf_eval = _resolve_root_logits(
        boards,
        model=model, evaluator=evaluator, device=device,
        pre_pol_logits=pre_pol_logits, pre_wdl_logits=pre_wdl_logits,
    )
    root_qs = [_wdl_to_q(wdl_logits_batch[i]) for i in range(n_boards)]

  # ── 2. Per-board root init + Gumbel candidate selection ──────────────────
    states: list[_BoardSearchState] = [
        _init_board_search_state(
            b,
            pol_logits=pol_logits_batch[i],
            root_q=float(root_qs[i]),
            sim_budget=sim_budget,
            cfg=cfg,
            rng=rng,
        )
        for i, b in enumerate(boards)
    ]
    budget_remaining: list[int] = [sim_budget] * n_boards

  # ── 3. Sequential halving with real subtree simulations ──────────────────
    while True:
        active = [
            i for i, st in enumerate(states)
            if st.finished_probs is None
            and st.remaining is not None
            and len(st.remaining) >= 1
            and budget_remaining[i] > 0
        ]
        if not active:
            break

        visits_per_action: dict[int, int] = {}
        for bi in active:
            rem = states[bi].remaining
            assert rem is not None
            if len(rem) <= 1:
                visits_per_action[bi] = int(budget_remaining[bi])
                continue
            rounds_left = int(np.ceil(np.log2(len(rem))))
            vpa = int(budget_remaining[bi] // max(1, len(rem) * rounds_left))
            visits_per_action[bi] = max(1, vpa)

        max_reps = max(visits_per_action.values(), default=0)
        for rep in range(max_reps):
            leaf_nodes, leaf_paths = _collect_forced_leaves_round(
                active=active, states=states,
                visits_per_action=visits_per_action, rep=rep, cfg=cfg,
            )
            _evaluate_and_backprop_leaves(leaf_nodes, leaf_paths, leaf_eval)

        for bi in active:
            st = states[bi]
            rem = st.remaining
            if rem is None:
                continue
            budget_remaining[bi] = max(
                0, int(budget_remaining[bi] - visits_per_action[bi] * len(rem)),
            )
            _halve_remaining_for_board(st, root_q=float(root_qs[bi]), cfg=cfg)

  # ── 4. Build improved policies + legal masks ─────────────────────────────
    probs_out: list[np.ndarray] = []
    actions_out: list[int] = []
    values_out: list[float] = []
    legal_masks_out: list[np.ndarray] = []
    for i, st in enumerate(states):
        if st.finished_probs is not None:
            probs_out.append(st.finished_probs)
            actions_out.append(int(st.finished_action or 0))
            values_out.append(float(st.finished_value if st.finished_value is not None else root_qs[i]))
        else:
            probs, action, value = _build_improved_policy_for_board(
                st, root_q=float(root_qs[i]), cfg=cfg, rng=rng,
            )
            probs_out.append(probs)
            actions_out.append(action)
            values_out.append(value)

        mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
        for a in st.root.children:
            mask[a] = True
        legal_masks_out.append(mask)

    return probs_out, actions_out, values_out, legal_masks_out


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
