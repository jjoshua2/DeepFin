"""The C sequential-halving elimination must score with the SAME rule the
Python reference does.

Three implementations of one decision live in this repo, and before this file
existed only two of them agreed:

* ``mcts/gumbel.py`` ``_halve_remaining_for_board`` — the reference. It passes
  the FRESH network root value (``root_qs[i]``, fixed for the whole search)
  into ``_completed_q_transform`` as ``raw_value``, and builds the completed-Q
  vector over ALL LEGAL actions, so every unvisited legal action contributes
  ``mixed_value`` to the min/max normalization.
* ``mcts/gumbel_c.py`` ``_build_improved_policy_for_board`` — the C path's own
  final-policy reconstruction. Same rule, same fresh ``root_q``.
* ``mcts/_mcts_tree.c`` ``gss_score_and_halve`` — the C elimination. It used to
  read ``W[root]/N[root]``: the running tree average, which this very search
  keeps mutating. So the C ELIMINATED candidates against one baseline and then
  RETURNED a policy scored against another.

⚑ These tests do NOT read a survivor off a full production-shaped search and
compare it with the Python path's. That comparison is GREEN on the unfixed
code — measured over 4,320 board-runs (72 shapes x 60 random middlegames),
where the fix changed not one action, value or policy digest. ``raw_value``
reaches the ranking ONLY through ``mixed_value``'s effect on min/max, and with
a spread of child Q values ``mixed_value`` sits strictly inside that spread,
where it moves neither endpoint. A broad-search parity test is therefore
vacuous for this defect by construction: it would certify the bug it was
written to catch.

So the scenarios below are built rather than sampled, at the shape where the
rules provably differ — a root the net likes whose searched replies disagree
with it, which is exactly when the mix value escapes the children's spread and
starts setting an endpoint. ``simulations=2`` with 2 candidates gives one
halving round and spends the budget exactly, so the FINAL tree state is the
state ``gss_score_and_halve`` scored, and the reference can be recomputed from
outside the C.

Every test asserts, in code, that the rules it discriminates actually disagree
in its scenario, so a scenario that stops discriminating fails loudly instead
of passing for free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.lc0 import c_input_history_mode
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _completed_q_transform,
    _root_sigma_scale,
)

POLICY_SIZE = 4672

# Quiet middlegame, 30+ legal moves, nothing forcing at depth 1 — so the
# one-visit-per-candidate round cannot reach a terminal or a proven node (the C
# short-circuits those; the Python reference has no such concept).
_FEN = "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 b - - 0 9"


@dataclass(frozen=True)
class _Scenario:
    """One hand-built halving round.

    ``child_q`` is each candidate's Q from the ROOT's point of view after its
    single visit; ``cand_prior`` its root prior.
    """

    root_q: float
    child_q: tuple[float, float]
    cand_prior: tuple[float, float]


# The root eval says +0.90 and BOTH searched replies refute it. The fresh mix
# value then lands above both children and sets max_q; the stale one lands
# between them and does not — a different q_denom, a different winner. Same
# scenario also separates the min/max pool question, because the endpoint the
# two rules argue over is the imputed one.
_REFUTED_ROOT = _Scenario(
    root_q=0.90, child_q=(-0.85, -0.65), cand_prior=(0.45, 0.02),
)

# The root eval says +0.95 and both replies agree with it. Here the fresh mix
# value is strictly inside the children's spread and changes nothing, while a
# root node SEEDED low drags the stale mix value below both children and flips
# the survivor. Used for the "W[root]/N[root] is not an input" test.
_CONFIRMED_ROOT = _Scenario(
    root_q=0.95, child_q=(0.85, 0.90), cand_prior=(0.45, 0.02),
)


def _cfg() -> GumbelConfig:
    """Production root transform, no noise, no Python-only knobs."""
    return GumbelConfig(
        simulations=2, topk=2, temperature=0.0, add_noise=False, c_scale=0.1,
    )


def _wdl_logits_for_q(q: float) -> tuple[float, float, float]:
    """WDL logits whose softmax gives ``P(win) - P(loss) == q``."""
    draw = 0.02
    win = (1.0 - draw + q) / 2.0
    loss = 1.0 - draw - win
    assert win > 0.0, f"q={q} outside the representable band"
    assert loss > 0.0, f"q={q} outside the representable band"
    return (math.log(win), math.log(draw), math.log(loss))


def _dense_priors(legal: np.ndarray, scenario: _Scenario) -> np.ndarray:
    """Dense prior vector: pinned mass on the two candidates, rest shared."""
    pri = np.zeros(POLICY_SIZE, dtype=np.float64)
    pri[legal] = (1.0 - sum(scenario.cand_prior)) / float(legal.size - 2)
    pri[int(legal[0])] = scenario.cand_prior[0]
    pri[int(legal[1])] = scenario.cand_prior[1]
    return pri


@dataclass
class _Round:
    survivor: int
    tree: MCTSTree
    root_id: int
    priors: np.ndarray
    candidates: list[int]

    @property
    def stale_root_value(self) -> float:
        """``W[root]/N[root]`` — the baseline the C used to eliminate against."""
        return self.tree.node_q(self.root_id)


def _run_one_halving_round(scenario: _Scenario, *, root_w: float) -> _Round:
    """Drive ONE sequential-halving round through the real C state machine."""
    board = chess.Board(_FEN)
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices().astype(np.int32, copy=False)
    assert legal.size >= 8, "scenario needs unvisited legal actions in the pool"
    pri = _dense_priors(legal, scenario)
    cands = [int(legal[0]), int(legal[1])]

    tree = MCTSTree()
    root_id = tree.add_root(1, float(root_w))
    tree.expand(root_id, legal, pri[legal])

    enc = np.zeros((64, 175, 8, 8), dtype=np.float32)
    # Leaf i is candidate i's forced descent: the C builds one query per
    # remaining candidate, in candidate order, and at depth 1 the leaf IS the
    # candidate's own node. The realised child Q values are asserted in
    # ``test_scenario_realises_the_intended_tree_state``, so a change in that
    # order fails a test instead of silently re-labelling the scenario.
    wanted = [-q for q in scenario.child_q]  # leaf POV = -(root POV) at depth 1
    fed = 0

    cfg = _cfg()
    n_leaves = tree.start_gumbel_sims(
        [cb], np.array([root_id], dtype=np.int32), [cands],
        [np.zeros(POLICY_SIZE, dtype=np.float64)], [pri],
        np.array([cfg.simulations], dtype=np.int32),
        np.array([scenario.root_q], dtype=np.float64),
        float(cfg.c_scale), float(cfg.c_visit), float(cfg.c_puct),
        float(cfg.fpu_reduction), bool(cfg.full_tree), enc,
        0, 1, c_input_history_mode(cfg.input_history_encoding),
        None, float(cfg.q_visit_exp), 0, float(cfg.q_visit_floor),
        int(cfg.halving_div), float(cfg.c_visit_root), float(cfg.c_scale_root),
        float(cfg.q_visit_exp_root), 0,
    )
    while n_leaves is not None:
        n = int(n_leaves)
        pol = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        wdl = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            q = wanted[fed + i] if fed + i < len(wanted) else 0.0
            wdl[i] = _wdl_logits_for_q(q)
        fed += n
        n_leaves = tree.continue_gumbel_sims(pol, wdl)

    assert fed == len(wanted), f"expected {len(wanted)} leaf evals, got {fed}"
    remaining = tree.get_gumbel_remaining()[0]
    assert len(remaining) == 1, f"expected a single survivor, got {remaining}"
    return _Round(int(remaining[0]), tree, root_id, pri, cands)


def _reference_scores(
    rnd: _Round, *, raw_value: float, include_unvisited: bool = True,
) -> dict[int, float]:
    """``gumbel.py``'s halving score, recomputed from the C tree's own state."""
    cfg = _cfg()
    legal = np.nonzero(rnd.priors > 0.0)[0].astype(np.int64)
    actions, visits, qs = rnd.tree.get_children_q(rnd.root_id, 0.0)
    by_action = {int(a): (int(n), float(q)) for a, n, q in zip(actions, visits, qs)}
    vis = np.array([by_action.get(int(a), (0, 0.0))[0] for a in legal], dtype=np.float64)
    qv = np.array([by_action.get(int(a), (0, 0.0))[1] for a in legal], dtype=np.float64)

    if include_unvisited:
        q_logits = _completed_q_transform(
            actions=legal, priors=rnd.priors[legal], visits=vis, qvalues=qv,
            raw_value=float(raw_value), cfg=cfg, root=True,
        )
        table = {int(a): float(v) for a, v in zip(legal, q_logits)}
    else:
        # The mutant: normalize over the VISITED children only, dropping the
        # unvisited legal actions' imputed mix value from the min/max pool.
        seen = vis > 0.0
        lo, hi = float(qv[seen].min()), float(qv[seen].max())
        scale = _root_sigma_scale(max_visit=int(vis.max()), cfg=cfg)
        table = {
            int(a): scale * (float(q) - lo) / max(hi - lo, 1e-8)
            for a, q in zip(legal, qv)
        }
    return {
        int(a): math.log(max(float(rnd.priors[int(a)]), 1e-12)) + table[int(a)]
        for a in rnd.candidates
    }


def _argmax(scores: dict[int, float]) -> int:
    return max(scores, key=lambda a: scores[a])


def test_scenario_realises_the_intended_tree_state() -> None:
    """Precondition: the constructed round really produced the tree it claims."""
    for scenario in (_REFUTED_ROOT, _CONFIRMED_ROOT):
        rnd = _run_one_halving_round(scenario, root_w=scenario.root_q)
        actions, visits, qs = rnd.tree.get_children_q(rnd.root_id, 0.0)
        by_action = {int(a): (int(n), float(q)) for a, n, q in zip(actions, visits, qs)}
        for cand, want_q in zip(rnd.candidates, scenario.child_q):
            n, q = by_action[cand]
            assert n == 1, f"candidate {cand} got {n} visits, expected 1"
            assert q == pytest.approx(want_q, abs=1e-5), f"candidate {cand} Q={q}"
        unvisited = [a for a, (n, _q) in by_action.items() if n == 0]
        assert len(unvisited) >= 4, "scenario needs unvisited legal actions"
        assert rnd.survivor in rnd.candidates


def test_halving_uses_the_fresh_root_value_not_the_running_average() -> None:
    """The C elimination must score against ``root_qs``, not ``W[root]/N[root]``."""
    rnd = _run_one_halving_round(_REFUTED_ROOT, root_w=_REFUTED_ROOT.root_q)
    fresh = _argmax(_reference_scores(rnd, raw_value=_REFUTED_ROOT.root_q))
    stale = _argmax(_reference_scores(rnd, raw_value=rnd.stale_root_value))
    assert stale != fresh, (
        "scenario no longer discriminates the two baselines — the assertion "
        "below would pass for free; re-tune _REFUTED_ROOT"
    )
    assert rnd.survivor == fresh, (
        f"C halving kept {rnd.survivor}; the Python reference "
        f"(_completed_q_transform, raw_value=root_qs) keeps {fresh}. "
        f"W[root]/N[root]={rnd.stale_root_value:.6f} would keep {stale}."
    )


def test_unvisited_legal_actions_stay_in_the_normalization_pool() -> None:
    """Every unvisited legal action contributes ``mixed_value`` to min/max."""
    rnd = _run_one_halving_round(_REFUTED_ROOT, root_w=_REFUTED_ROOT.root_q)
    full = _argmax(_reference_scores(rnd, raw_value=_REFUTED_ROOT.root_q))
    visited_only = _argmax(
        _reference_scores(
            rnd, raw_value=_REFUTED_ROOT.root_q, include_unvisited=False,
        ),
    )
    assert visited_only != full, (
        "scenario no longer discriminates the min/max pool — re-tune _REFUTED_ROOT"
    )
    assert rnd.survivor == full, (
        f"C halving kept {rnd.survivor}; normalizing over ALL legal actions "
        f"keeps {full}, over the visited children only keeps {visited_only}"
    )


def test_halving_is_independent_of_the_root_nodes_stored_value() -> None:
    """``W[root]``/``N[root]`` is not an input to the elimination at all.

    Same ``root_qs``, same leaf values, same priors — only the value the root
    node was SEEDED with differs, and the search must not be able to tell.
    """
    hi = _run_one_halving_round(_CONFIRMED_ROOT, root_w=0.95)
    lo = _run_one_halving_round(_CONFIRMED_ROOT, root_w=-0.95)
    assert hi.stale_root_value != lo.stale_root_value, (
        "the two runs must actually differ in W[root]/N[root]"
    )
    stale_hi = _argmax(_reference_scores(hi, raw_value=hi.stale_root_value))
    stale_lo = _argmax(_reference_scores(lo, raw_value=lo.stale_root_value))
    assert stale_hi != stale_lo, (
        "scenario no longer discriminates the seed value — the assertion below "
        "would pass for free; re-tune _CONFIRMED_ROOT"
    )
    fresh = _argmax(_reference_scores(hi, raw_value=_CONFIRMED_ROOT.root_q))
    assert hi.survivor == lo.survivor == fresh, (
        f"survivor moved with the root node's seed value: {hi.survivor} "
        f"(seed +0.95) vs {lo.survivor} (seed -0.95); the fresh-root reference "
        f"keeps {fresh}"
    )
    hi_actions, hi_visits = hi.tree.get_children_visits(hi.root_id)
    lo_actions, lo_visits = lo.tree.get_children_visits(lo.root_id)
    assert list(hi_actions) == list(lo_actions)
    assert list(hi_visits) == list(lo_visits)
