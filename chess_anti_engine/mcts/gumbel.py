from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, overload

import chess
import numpy as np
import torch

from chess_anti_engine.encoding.lc0 import LC0_HISTORY_LEGACY
from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.encoding.features import EXTRA_FEATURES_V1, relation_matrices
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
from chess_anti_engine.mcts.root_tactics import (
    immediate_mate_root_policy,
    immediate_terminal_draw_indices,
)
from chess_anti_engine.moves import (
    POLICY_ENCODING_AZ_4672,
    POLICY_SIZE,
    policy_batch_to_full_if_needed,
)
from chess_anti_engine.moves.encode import legal_move_indices

GumbelManyResult = tuple[list[np.ndarray], list[int], list[float], list[np.ndarray]]
GumbelManyDiagnosticsResult = tuple[
    list[np.ndarray], list[int], list[float], list[np.ndarray], list[dict[str, float] | None]
]


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
    c_scale: float = 0.1
    c_puct: float = 2.5
    fpu_reduction: float = 1.2
    full_tree: bool = True
    add_noise: bool = True  # Backward-compatible gate; use gumbel_scale for partial noise.
    gumbel_scale: float = 1.0
    input_history_encoding: str = LC0_HISTORY_LEGACY
    input_extra_features: str = EXTRA_FEATURES_V1
    policy_encoding: str = POLICY_ENCODING_AZ_4672
  # Compute dynamic board-relation matrices per eval and pass them to the
  # evaluator as attention-bias input (model.use_dynamic_relations).
    compute_relations: bool = False
  # ── Volatility-aware search (Python path only; both default OFF) ────────
  # volatility_q_scale: exponent scaling the sigma(q) value-transform
  # constant per node by predicted volatility. The effective scale is
  #   c_scale * (volatility_anchor / vol)^volatility_q_scale
  # clipped to [1/volatility_factor_clip, volatility_factor_clip] x c_scale,
  # so at vol == anchor the behavior is IDENTICAL to today. High predicted
  # volatility -> smaller sigma -> flatter value transform -> candidates
  # survive halving longer; low volatility -> trust the value sooner.
    volatility_q_scale: float = 0.0
  # volatility_fpu: pessimistic first-play urgency — unvisited children's
  # completed value becomes mixed_value - volatility_fpu * vol (Q units).
    volatility_fpu: float = 0.0
  # Dataset-mean anchor for the volatility head's scalar summary (mean of
  # the 3 head components). Derive from a recent shard window before an
  # experiment (see configs/exp_volatility_search.yaml) and pin it here —
  # the normalization must stay frozen within an arena sweep.
    volatility_anchor: float = 0.05
    volatility_factor_clip: float = 4.0


def _policy_logits_to_full(pol_logits: np.ndarray, *, cfg: GumbelConfig) -> np.ndarray:
    return policy_batch_to_full_if_needed(
        np.asarray(pol_logits, dtype=np.float32),
        policy_encoding=cfg.policy_encoding,
        fill_value=-1e9,
    )


def gumbel_policy_diagnostics(
    *,
    probs: np.ndarray,
    action: int,
    legal: np.ndarray,
    candidates: list[int] | np.ndarray | None,
) -> dict[str, float]:
    """Cheap diagnostics for the final Gumbel training policy."""
    legal = np.asarray(legal, dtype=np.int64)
    if legal.size == 0:
        return {}

    p = np.asarray(probs, dtype=np.float64)
    legal_p = np.maximum(p[legal], 0.0)
    total = float(legal_p.sum(dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        return {}
    legal_p = legal_p / total

    top_local = int(np.argmax(legal_p))
    top_action = int(legal[top_local])
    top_prob = float(legal_p[top_local])
    action_prob = float(p[int(action)]) if 0 <= int(action) < p.shape[0] else 0.0
    positive = legal_p[legal_p > 0.0]
    entropy = float(-(positive * np.log(positive)).sum(dtype=np.float64))

    cand_mask = np.zeros_like(legal, dtype=np.bool_)
    cand_count = 0
    if candidates is not None:
        cand_set = {int(a) for a in candidates}
        cand_count = len(cand_set)
        if cand_count > 0:
            cand_mask = np.array([int(a) in cand_set for a in legal], dtype=np.bool_)
    cand_mass = float(legal_p[cand_mask].sum(dtype=np.float64)) if cand_count > 0 else 0.0
    non_cand_top = (
        float(legal_p[~cand_mask].max(initial=0.0)) if cand_count > 0 else 0.0
    )

    return {
        "top_prob": top_prob,
        "action_prob": max(0.0, float(action_prob)),
        "entropy": entropy,
        "eff_moves": float(np.exp(entropy)),
        "candidate_mass": cand_mass,
        "non_candidate_top_prob": non_cand_top,
        "argmax_is_candidate": 1.0 if (cand_count > 0 and bool(cand_mask[top_local])) else 0.0,
        "argmax_is_action": 1.0 if top_action == int(action) else 0.0,
        "legal_count": float(legal.size),
        "candidate_count": float(cand_count),
    }


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


def volatility_search_enabled(cfg: GumbelConfig) -> bool:
    """True when any volatility-aware search mechanism is switched on."""
    return float(cfg.volatility_q_scale) != 0.0 or float(cfg.volatility_fpu) != 0.0


_volatility_python_path_warned = False


def warn_volatility_python_path() -> None:
    """Warn once per process: volatility flags force the Python search path.

    The C fast path (mcts/gumbel_c.py) does not implement
    volatility_q_scale/volatility_fpu; callers that would normally take it
    must drop to run_gumbel_root_many and say so in the log. Porting to C is
    a follow-up gated on the arena clearing the Elo bar.
    """
    global _volatility_python_path_warned
    if not _volatility_python_path_warned:
        _volatility_python_path_warned = True
        import logging

        logging.getLogger("chess_anti_engine.mcts").warning(
            "volatility-aware Gumbel search enabled: forcing the (slower) "
            "Python search path - the C fast path does not implement "
            "volatility_q_scale/volatility_fpu. Use matched_sims, not "
            "matched_time, when comparing against the C path."
        )


def _volatility_sigma_factor(vol: float, cfg: GumbelConfig) -> float:
    """Multiplier on the sigma(q) scale for a node with predicted ``vol``.

    1.0 when the mechanism is off or vol equals the anchor; <1 (flatter)
    above the anchor, >1 (sharper) below it. Clipped so a wild head output
    cannot collapse or explode the transform.
    """
    k = float(cfg.volatility_q_scale)
    if k == 0.0:
        return 1.0
    anchor = max(1e-9, float(cfg.volatility_anchor))
    ratio = anchor / max(1e-9, float(vol))
    clip = max(1.0, float(cfg.volatility_factor_clip))
    return float(np.clip(ratio ** k, 1.0 / clip, clip))


def _volatility_fpu_penalty(vol: float, cfg: GumbelConfig) -> float:
    """Pessimistic unvisited-child value offset (Q units, subtracted)."""
    k = float(cfg.volatility_fpu)
    return k * float(vol) if k != 0.0 else 0.0


def _completed_q_transform(
    *,
    actions: list[int] | np.ndarray,
    priors: np.ndarray,
    visits: np.ndarray,
    qvalues: np.ndarray,
    raw_value: float,
    cfg: GumbelConfig,
    epsilon: float = 1e-8,
    sigma_factor: float = 1.0,
    fpu_penalty: float = 0.0,
) -> np.ndarray:
    """DeepMind mctx completed-by-mix-value Q transform for Gumbel scores.

    ``sigma_factor`` scales the sigma(q) constant (volatility_q_scale) and
    ``fpu_penalty`` is subtracted from the unvisited-children mix value
    (volatility_fpu). Both default to the exact legacy behavior.
    """
    actions_arr = np.asarray(actions, dtype=np.int64)
    visits_f = np.asarray(visits, dtype=np.float64)
    q = np.asarray(qvalues, dtype=np.float64)
    prior = np.maximum(np.asarray(priors, dtype=np.float64), np.finfo(np.float64).tiny)

    if actions_arr.size == 0:
        return np.zeros((0,), dtype=np.float64)

    visited = visits_f > 0.0
    sum_visits = float(visits_f.sum(dtype=np.float64))
    sum_probs = float(prior[visited].sum(dtype=np.float64)) if visited.any() else 0.0
    if sum_probs > 0.0 and np.isfinite(sum_probs):
        weighted_q = float((prior[visited] * q[visited] / sum_probs).sum(dtype=np.float64))
    else:
        weighted_q = float(raw_value)
    mixed_value = (float(raw_value) + sum_visits * weighted_q) / (sum_visits + 1.0)

    completed = np.where(visited, q, mixed_value - float(fpu_penalty))
    min_q = float(completed.min())
    max_q = float(completed.max())
    completed = (completed - min_q) / max(max_q - min_q, float(epsilon))
    max_visit = int(visits_f.max(initial=0.0))
    return float(sigma_factor) * _sigma_scale(max_visit=max_visit, cfg=cfg) * completed


def _completed_q(*, root_q: float, root: Node, action: int) -> float:
    child = root.children.get(int(action))
    if child is None or child.N <= 0:
        return float(root_q)
    return float(-child.Q)


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
    visits = np.empty(n_act, dtype=np.float64)
    qvalues = np.empty(n_act, dtype=np.float64)
    priors = np.empty(n_act, dtype=np.float64)
    v_pi = node.Q

    for i, a in enumerate(actions):
        ch = children[a]
        n = ch.N
        priors[i] = max(ch.prior, 1e-12)
        logits[i] = math.log(priors[i])
        visits[i] = float(n)
        qvalues[i] = (-ch.W / n) if n > 0 else v_pi

    q_logits = _completed_q_transform(
        actions=actions,
        priors=priors,
        visits=visits,
        qvalues=qvalues,
        raw_value=float(v_pi),
        cfg=cfg,
        sigma_factor=_volatility_sigma_factor(node.vol, cfg),
        fpu_penalty=_volatility_fpu_penalty(node.vol, cfg),
    )
    probs = _softmax(logits + q_logits)
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


def _eval_with_optional_volatility(
    eval_impl: BatchEvaluator,
    xs: np.ndarray,
    *,
    relations: np.ndarray | None,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate a batch; include the volatility head when the search needs it.

    Volatility-aware search is Python-path-only and requires an evaluator
    exposing ``evaluate_encoded_with_volatility`` (LocalModelEvaluator) —
    fail loud rather than silently searching with vol=0 everywhere.
    """
    if not volatility_search_enabled(cfg):
        if relations is not None:
            pol, wdl = eval_impl.evaluate_encoded(xs, relations=relations)
        else:
            pol, wdl = eval_impl.evaluate_encoded(xs)
        return pol, wdl, None
    fn = getattr(eval_impl, "evaluate_encoded_with_volatility", None)
    if fn is None:
        raise ValueError(
            "volatility-aware Gumbel search needs an evaluator with "
            "evaluate_encoded_with_volatility (LocalModelEvaluator); got "
            f"{type(eval_impl).__name__}"
        )
    if relations is not None:
        return fn(xs, relations=relations)
    return fn(xs)


def _resolve_root_logits(
    boards: list[chess.Board],
    *,
    model: torch.nn.Module | None,
    evaluator: BatchEvaluator | None,
    device: str,
    cfg: GumbelConfig,
    pre_pol_logits: np.ndarray | None,
    pre_wdl_logits: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, BatchEvaluator | None, np.ndarray | None]:
    """Phase 1: root pol/wdl logits (+ volatility) + a leaf evaluator.

    Reuses caller-provided ``pre_*_logits`` (one forward pass saved per ply
    when called from selfplay). Always resolves a ``leaf_eval`` for use by
    sequential halving — even when pre-logits short-circuit phase 1.
    With volatility-aware search enabled the pre-logits shortcut is skipped
    (they don't carry the volatility head) and the roots are re-evaluated.
    """
    vol_on = volatility_search_enabled(cfg)
    if pre_pol_logits is not None and pre_wdl_logits is not None and not vol_on:
        pol = _policy_logits_to_full(pre_pol_logits, cfg=cfg)
        wdl = np.asarray(pre_wdl_logits, dtype=np.float32)
        leaf_eval = evaluator if evaluator is not None else (
            LocalModelEvaluator(model, device=device) if model is not None else None
        )
        return pol, wdl, leaf_eval, None

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_gumbel_root_many requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)
    xs = encode_positions_batch(
        boards,
        add_features=True,
        input_history_encoding=cfg.input_history_encoding,
        input_extra_features=cfg.input_extra_features,
    )
    rel = (
        np.stack([relation_matrices(b) for b in boards], axis=0)
        if cfg.compute_relations else None
    )
    pol, wdl, vol = _eval_with_optional_volatility(eval_impl, xs, relations=rel, cfg=cfg)
    pol = _policy_logits_to_full(pol, cfg=cfg)
    return pol, wdl, eval_impl, vol


def _evaluate_and_backprop_leaves(
    leaf_nodes: list[Node],
    leaf_paths: list[list[Node]],
    leaf_eval: BatchEvaluator | None,
    cfg: GumbelConfig,
) -> None:
    """Batched leaf NN eval + expand + backprop. No-op when ``leaf_nodes`` is empty."""
    if not leaf_nodes:
        return
    if leaf_eval is None:
        raise ValueError("run_gumbel_root_many requires model or evaluator")
    leaf_xs = encode_positions_batch(
        [node.board for node in leaf_nodes],
        add_features=True,
        input_history_encoding=cfg.input_history_encoding,
        input_extra_features=cfg.input_extra_features,
    )
    leaf_rel = (
        np.stack([relation_matrices(node.board) for node in leaf_nodes], axis=0)
        if cfg.compute_relations else None
    )
    pol_logits_leaf, wdl_logits_leaf, vol_leaf = _eval_with_optional_volatility(
        leaf_eval, leaf_xs, relations=leaf_rel, cfg=cfg,
    )
    pol_logits_leaf = _policy_logits_to_full(pol_logits_leaf, cfg=cfg)
    for li, (node, path, pol_logits, wdl_logits) in enumerate(zip(
        leaf_nodes, leaf_paths, pol_logits_leaf, wdl_logits_leaf, strict=True,
    )):
        if vol_leaf is not None:
            node.vol = float(vol_leaf[li])
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
    gumbel_scale: float,
    rng: np.random.Generator,
) -> tuple[list[int], dict[int, float]]:
    """Sample top-m root actions via Gumbel(logit + noise). Caller filters trivial cases.

    Returns (cands, gumbels_for_all_legal). ``m`` is bounded so sequential halving
    can still allocate ≥1 visit per action per round.
    """
    log_pri = np.log(np.maximum(pri[legal], 1e-12))
    scale = float(gumbel_scale) if add_noise else 0.0
    g = scale * _gumbel(rng, legal.size) if scale > 0.0 else np.zeros(legal.size, dtype=np.float64)
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

    mate = immediate_mate_root_policy(board)
    if mate is not None:
        probs, action, value = mate
        return _finish(probs, action, value)

    if root_q > 0.0 and legal.size > 1:
        terminal_draws = immediate_terminal_draw_indices(board)
        if terminal_draws:
            keep = np.array([int(a) not in terminal_draws for a in legal], dtype=np.bool_)
            if keep.any():
                pri[np.fromiter(terminal_draws, dtype=np.int64)] = 0.0
                legal = legal[keep]

    if legal.size == 1:
        a0 = int(legal[0])
        p = np.zeros((POLICY_SIZE,), dtype=np.float32)
        p[a0] = 1.0
        return _finish(p, a0, root_q)

    cands, gumbels = _select_top_m_with_gumbel(
        legal=legal, pri=pri, sim_budget=sim_budget,
        topk=int(cfg.topk), add_noise=cfg.add_noise,
        gumbel_scale=float(cfg.gumbel_scale), rng=rng,
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
    legal = np.nonzero(pri > 0)[0].astype(int)
    visits = np.empty(legal.size, dtype=np.float64)
    qvalues = np.empty(legal.size, dtype=np.float64)
    for i, a in enumerate(legal):
        ch = root.children.get(int(a))
        n = 0 if ch is None else int(ch.N)
        visits[i] = float(n)
        qvalues[i] = (-ch.W / n) if (ch is not None and n > 0) else float(root_q)
    q_logits = _completed_q_transform(
        actions=legal,
        priors=pri[legal],
        visits=visits,
        qvalues=qvalues,
        raw_value=float(root_q),
        cfg=cfg,
        sigma_factor=_volatility_sigma_factor(root.vol, cfg),
        fpu_penalty=_volatility_fpu_penalty(root.vol, cfg),
    )
    q_by_action = {int(a): float(q_logits[i]) for i, a in enumerate(legal.tolist())}
    rem.sort(
        key=lambda a: (
            float(gmap.get(int(a), 0.0))
            + float(np.log(max(float(pri[int(a)]), 1e-12)))
            + q_by_action.get(int(a), 0.0)
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
    visits = np.empty(legal.size, dtype=np.float64)
    qvalues = np.empty(legal.size, dtype=np.float64)
    for i, a in enumerate(legal):
        ch = root.children.get(int(a))
        n = 0 if ch is None else int(ch.N)
        visits[i] = float(n)
        qvalues[i] = (-ch.W / n) if (ch is not None and n > 0) else float(root_q)
    logits_imp = np.log(np.maximum(pri[legal], 1e-12)) + _completed_q_transform(
        actions=legal,
        priors=pri[legal],
        visits=visits,
        qvalues=qvalues,
        raw_value=float(root_q),
        cfg=cfg,
        sigma_factor=_volatility_sigma_factor(root.vol, cfg),
        fpu_penalty=_volatility_fpu_penalty(root.vol, cfg),
    )
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
@overload
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
    per_game_simulations: list[int] | None = None,
    per_game_add_noise: list[bool] | None = None,
    per_game_gumbel_scale: list[float] | None = None,
    return_diagnostics: Literal[False] = False,
) -> GumbelManyResult: ...


@overload
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
    per_game_simulations: list[int] | None = None,
    per_game_add_noise: list[bool] | None = None,
    per_game_gumbel_scale: list[float] | None = None,
    return_diagnostics: Literal[True],
) -> GumbelManyDiagnosticsResult: ...


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
    per_game_simulations: list[int] | None = None,
    per_game_add_noise: list[bool] | None = None,
    per_game_gumbel_scale: list[float] | None = None,
    return_diagnostics: bool = False,
) -> GumbelManyResult | GumbelManyDiagnosticsResult:
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
        if return_diagnostics:
            return [], [], [], [], []
        return [], [], [], []

    sim_budget = max(1, int(cfg.simulations))
    if per_game_simulations is not None:
        if len(per_game_simulations) != n_boards:
            raise ValueError(
                f"per_game_simulations length {len(per_game_simulations)} does not match "
                f"boards length {n_boards}"
            )
        budget_remaining = [max(1, int(v)) for v in per_game_simulations]
    else:
        budget_remaining = [sim_budget] * n_boards

  # ── 1. Batch root evaluation + resolve leaf evaluator for phase 3 ────────
    pol_logits_batch, wdl_logits_batch, leaf_eval, root_vols = _resolve_root_logits(
        boards,
        model=model, evaluator=evaluator, device=device,
        cfg=cfg,
        pre_pol_logits=pre_pol_logits, pre_wdl_logits=pre_wdl_logits,
    )
    root_qs = [_wdl_to_q(wdl_logits_batch[i]) for i in range(n_boards)]

  # ── 2. Per-board root init + Gumbel candidate selection ──────────────────
    states: list[_BoardSearchState] = []
    for i, b in enumerate(boards):
        board_cfg = replace(
            cfg,
            add_noise=bool(per_game_add_noise[i]) if per_game_add_noise is not None else bool(cfg.add_noise),
            gumbel_scale=(
                float(per_game_gumbel_scale[i])
                if per_game_gumbel_scale is not None
                else float(cfg.gumbel_scale)
            ),
        )
        st = _init_board_search_state(
            b,
            pol_logits=pol_logits_batch[i],
            root_q=float(root_qs[i]),
            sim_budget=budget_remaining[i],
            cfg=board_cfg,
            rng=rng,
        )
        if root_vols is not None:
            st.root.vol = float(root_vols[i])
        states.append(st)

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
            _evaluate_and_backprop_leaves(leaf_nodes, leaf_paths, leaf_eval, cfg)

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
    diagnostics_out: list[dict[str, float] | None] = []
    for i, st in enumerate(states):
        if st.finished_probs is not None:
            probs_out.append(st.finished_probs)
            actions_out.append(int(st.finished_action or 0))
            values_out.append(float(st.finished_value if st.finished_value is not None else root_qs[i]))
            diagnostics_out.append(None)
        else:
            probs, action, value = _build_improved_policy_for_board(
                st, root_q=float(root_qs[i]), cfg=cfg, rng=rng,
            )
            probs_out.append(probs)
            actions_out.append(action)
            values_out.append(value)
            legal = np.nonzero(st.priors > 0)[0].astype(int)
            diagnostics_out.append(gumbel_policy_diagnostics(
                probs=probs, action=int(action), legal=legal, candidates=st.candidates,
            ))

        mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
        for a in st.root.children:
            mask[a] = True
        legal_masks_out.append(mask)

    if return_diagnostics:
        return probs_out, actions_out, values_out, legal_masks_out, diagnostics_out
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
