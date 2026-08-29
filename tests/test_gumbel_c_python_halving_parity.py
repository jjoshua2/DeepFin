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
code — measured over 4,320 board-runs, see ``scripts/gumbel_halving_baseline_sweep.py``.
``raw_value`` moves ``mixed_value`` by only ``Δroot / (ΣN + 1)``, and it reaches
the ranking ONLY through what that does to ``min_q`` / ``max_q``. At
production's first halving round ``ΣN`` is already 64 (topk 16 x 4 visits), so
the shift is ``Δroot/65`` — far too small to move an endpoint against a spread
of child Q values. A broad-search parity test is therefore vacuous for this
defect by construction: it would certify the bug it was written to catch.

So the scenarios below are built rather than sampled, and they run at ``ΣN = 2``
— the maximum leverage the term can ever have. ``simulations=2`` with 2
candidates gives exactly one halving round and spends the budget exactly, so
the FINAL tree state IS the state ``gss_score_and_halve`` scored, and
``gumbel.py``'s ``_completed_q_transform`` can be recomputed from outside the C
and compared.

Both production ROOT TRANSFORMS are covered, because a scenario that
discriminates the two rules at one ``q_scale`` need not at the other:

* ``selfplay_linear_root`` — the committed ``GumbelConfig`` default, what
  training runs: linear, ``q_scale = c_scale * (c_visit + max_visit) = 5.10``.
* ``play_log_root`` — ``gumbel.PLAY_SEARCH_DEFAULTS``, what UCI / the arena /
  the puzzle eval run: LOG, ``q_scale = c_scale_root * log1p(c_visit_root +
  max_visit) = 47.63``.

⚑ ONE scenario cannot cover both, and that is measured, not assumed. The two
rules disagree only for ``q_scale`` inside a band, and the WIDEST achievable
band is a factor of **2.96** (max over a 77 x 77 x 77 Q-grid crossed with 8 x 8
priors); the two shapes are a factor of **9.34** apart. Hence one scenario per
shape, bound to it in ``_SHAPES``.

Every test asserts, in code, that the rules it discriminates actually disagree
in its scenario, so a scenario that stops discriminating fails loudly instead
of passing for free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.lc0 import c_input_history_mode
from chess_anti_engine.mcts import _mcts_tree as _mcts_tree_ext
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    GumbelConfig,
    _completed_q_transform,
    _root_sigma_scale,
)
from chess_anti_engine.mcts.gumbel_c import (
    _DELETED_Q_GLOBAL_SCALE,
    _DELETED_Q_VISIT_EXP,
    _DELETED_Q_VISIT_FLOOR,
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


# ── selfplay_linear_root scenarios (q_scale 5.10) ───────────────────────────
# The root eval says +0.90 and BOTH searched replies refute it.
_REFUTED_ROOT_LINEAR = _Scenario(
    root_q=0.90, child_q=(-0.85, -0.65), cand_prior=(0.45, 0.02),
)
# The root eval says +0.95 and both replies agree with it — the fresh mix value
# sits inside the children's spread and changes nothing, so only the root
# node's SEED can move the survivor.
_CONFIRMED_ROOT_LINEAR = _Scenario(
    root_q=0.95, child_q=(0.85, 0.90), cand_prior=(0.45, 0.02),
)

# ── play_log_root scenarios (q_scale 47.63) ─────────────────────────────────
# Same two stories at the shape a 9.3x larger q_scale needs: the sigma term is
# ~9x bigger, so the prior gap that has to sit between the two rules' Q terms is
# ~9x bigger too (0.45 vs 0.001 instead of 0.45 vs 0.02), and the children's Q
# values are closer together.
_REFUTED_ROOT_LOG = _Scenario(
    root_q=0.575, child_q=(-0.575, -0.550), cand_prior=(0.45, 0.001),
)
_CONFIRMED_ROOT_LOG = _Scenario(
    root_q=0.95, child_q=(0.925, 0.950), cand_prior=(0.45, 0.001),
)


@dataclass(frozen=True)
class _Shape:
    """A production root transform plus the scenarios that discriminate at it."""

    name: str
    overrides: dict[str, float]
    q_scale: float
    refuted: _Scenario
    confirmed: _Scenario

    def cfg(self) -> GumbelConfig:
        """``simulations=2`` (one halving round) on top of the shape's own knobs.

        ``topk`` is forced last and overrides whatever the shape's dict sets:
        the candidate list is passed to ``start_gumbel_sims`` explicitly here,
        so ``topk`` selects nothing and would only misdescribe the round.
        """
        base = GumbelConfig(temperature=0.0, add_noise=False)
        fields = set(base.__dataclass_fields__)
        applied = {k: v for k, v in self.overrides.items() if k in fields}
        assert applied.keys() == self.overrides.keys(), "unknown GumbelConfig field"
        applied["simulations"] = 2
        applied["topk"] = 2
        return replace(base, **applied)


_SHAPES = (
    _Shape(
        name="selfplay_linear_root",
        overrides={},
        q_scale=0.1 * (50.0 + 1.0),
        refuted=_REFUTED_ROOT_LINEAR,
        confirmed=_CONFIRMED_ROOT_LINEAR,
    ),
    _Shape(
        name="play_log_root",
        overrides=dict(PLAY_SEARCH_DEFAULTS),
        q_scale=7.0 * math.log1p(900.0 + 1.0),
        refuted=_REFUTED_ROOT_LOG,
        confirmed=_CONFIRMED_ROOT_LOG,
    ),
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
    cfg: GumbelConfig

    @property
    def stale_root_value(self) -> float:
        """``W[root]/N[root]`` — the baseline the C used to eliminate against."""
        return self.tree.node_q(self.root_id)


def _run_one_halving_round(
    shape: _Shape,
    scenario: _Scenario,
    *,
    root_w: float,
    stored_cand_prior: tuple[float, float] | None = None,
) -> _Round:
    """Drive ONE sequential-halving round through the real C state machine."""
    cfg = shape.cfg()
    board = chess.Board(_FEN)
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices().astype(np.int32, copy=False)
    assert legal.size >= 8, "scenario needs unvisited legal actions in the pool"
    pri = _dense_priors(legal, scenario)
    cands = [int(legal[0]), int(legal[1])]

    tree_pri = pri.copy()
    if stored_cand_prior is not None:
        tree_pri[legal] = (
            1.0 - sum(stored_cand_prior)
        ) / float(legal.size - 2)
        tree_pri[int(legal[0])] = stored_cand_prior[0]
        tree_pri[int(legal[1])] = stored_cand_prior[1]

    tree = MCTSTree()
    root_id = tree.add_root(1, float(root_w))
    tree.expand(root_id, legal, tree_pri[legal])

    planes = input_plane_count(cfg.input_extra_features)
    enc = np.zeros((64, planes, 8, 8), dtype=np.float32)
    # Leaf i is candidate i's forced descent: the C builds one query per
    # remaining candidate, in candidate order, and at depth 1 the leaf IS the
    # candidate's own node. The realised child Q values are asserted in
    # ``test_scenario_realises_the_intended_tree_state``, so a change in that
    # order fails a test instead of silently re-labelling the scenario.
    wanted = [-q for q in scenario.child_q]  # leaf POV = -(root POV) at depth 1
    fed = 0

    n_leaves = tree.start_gumbel_sims(
        [cb], np.array([root_id], dtype=np.int32), [cands],
        [np.zeros(POLICY_SIZE, dtype=np.float64)], [pri],
        np.array([cfg.simulations], dtype=np.int32),
        np.array([scenario.root_q], dtype=np.float64),
        float(cfg.c_scale), float(cfg.c_visit), float(cfg.c_puct),
        float(cfg.fpu_reduction), bool(cfg.full_tree), enc,
        0, 1, c_input_history_mode(cfg.input_history_encoding),
      # The three deleted descent q-knobs. This harness calls the C entry point
      # directly, so it must mirror what `gumbel_c` passes -- imported from the
      # consumer rather than re-typed, so a change there cannot leave this
      # harness silently searching a different shape than production.
        None, _DELETED_Q_VISIT_EXP, _DELETED_Q_GLOBAL_SCALE, _DELETED_Q_VISIT_FLOOR,
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
    return _Round(int(remaining[0]), tree, root_id, pri, cands, cfg)


def _root_children(rnd: _Round) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legal actions, their visits and their root-POV Q, aligned."""
    legal = np.nonzero(rnd.priors > 0.0)[0].astype(np.int64)
    actions, visits, qs = rnd.tree.get_children_q(rnd.root_id, 0.0)
    by_action = {int(a): (int(n), float(q)) for a, n, q in zip(actions, visits, qs)}
    vis = np.array([by_action.get(int(a), (0, 0.0))[0] for a in legal], dtype=np.float64)
    qv = np.array([by_action.get(int(a), (0, 0.0))[1] for a in legal], dtype=np.float64)
    return legal, vis, qv


def _mix_value(rnd: _Round, raw_value: float) -> float:
    """mctx's mix value, from the C tree's own state."""
    _legal, vis, qv = _root_children(rnd)
    seen = vis > 0.0
    pri = np.maximum(rnd.priors[np.nonzero(rnd.priors > 0.0)[0]], np.finfo(np.float64).tiny)
    sum_probs = float(pri[seen].sum())
    weighted_q = (
        float((pri[seen] * qv[seen] / sum_probs).sum())
        if sum_probs > 0.0 else float(raw_value)
    )
    n_total = float(vis.sum())
    return (float(raw_value) + n_total * weighted_q) / (n_total + 1.0)


def _reference_scores(
    rnd: _Round,
    *,
    raw_value: float,
    include_unvisited: bool = True,
    mix_priors: np.ndarray | None = None,
) -> dict[int, float]:
    """``gumbel.py``'s halving score, recomputed from the C tree's own state."""
    legal, vis, qv = _root_children(rnd)
    if include_unvisited:
        priors_for_mix = rnd.priors if mix_priors is None else mix_priors
        q_logits = _completed_q_transform(
            actions=legal, priors=priors_for_mix[legal], visits=vis, qvalues=qv,
            raw_value=float(raw_value), cfg=rnd.cfg, root=True,
        )
        table = {int(a): float(v) for a, v in zip(legal, q_logits)}
    else:
        # The mutant: normalize over the VISITED children only, dropping the
        # unvisited legal actions' imputed mix value from the min/max pool.
        seen = vis > 0.0
        lo, hi = float(qv[seen].min()), float(qv[seen].max())
        scale = _root_sigma_scale(max_visit=int(vis.max()), cfg=rnd.cfg)
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


_SHAPE_IDS = [s.name for s in _SHAPES]


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_shape_resolves_the_production_root_transform(shape: _Shape) -> None:
    """The shape under test really is the transform it claims to be."""
    resolved = _root_sigma_scale(max_visit=1, cfg=shape.cfg())
    assert resolved == pytest.approx(shape.q_scale, rel=1e-9), (
        f"{shape.name} resolves q_scale={resolved}, expected {shape.q_scale}"
    )


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_scenario_realises_the_intended_tree_state(shape: _Shape) -> None:
    """Precondition: the constructed rounds really produced the tree they claim."""
    for scenario in (shape.refuted, shape.confirmed):
        rnd = _run_one_halving_round(shape, scenario, root_w=scenario.root_q)
        actions, visits, qs = rnd.tree.get_children_q(rnd.root_id, 0.0)
        by_action = {int(a): (int(n), float(q)) for a, n, q in zip(actions, visits, qs)}
        for cand, want_q in zip(rnd.candidates, scenario.child_q):
            n, q = by_action[cand]
            assert n == 1, f"candidate {cand} got {n} visits, expected 1"
            assert q == pytest.approx(want_q, abs=1e-5), f"candidate {cand} Q={q}"
        unvisited = [a for a, (n, _q) in by_action.items() if n == 0]
        assert len(unvisited) >= 4, "scenario needs unvisited legal actions"
        assert rnd.survivor in rnd.candidates


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_the_two_rules_differ_by_q_denom_not_by_endpoint_membership(
    shape: _Shape,
) -> None:
    """Pins the MECHANISM the refuted-root scenarios exploit.

    Both the fresh and the stale mix value sit ABOVE both children, so both are
    ``max_q`` — endpoint MEMBERSHIP is identical and is not what separates the
    rules. What differs is how far above: ``q_denom``, and therefore how much of
    the sigma budget separates the two candidates. (The confirmed-root scenarios
    used by the independence test work the other way round — there a low seed
    drags the mix BELOW both children and it becomes ``min_q`` — which is why
    they are a different scenario and not a re-tune of this one.)
    """
    rnd = _run_one_halving_round(
        shape, shape.refuted, root_w=shape.refuted.root_q,
    )
    _legal, _vis, qv = _root_children(rnd)
    child_hi = max(shape.refuted.child_q)
    child_lo = min(shape.refuted.child_q)
    fresh_mix = _mix_value(rnd, shape.refuted.root_q)
    stale_mix = _mix_value(rnd, rnd.stale_root_value)
    assert fresh_mix > child_hi, f"fresh mix {fresh_mix} is not max_q"
    assert stale_mix > child_hi, f"stale mix {stale_mix} is not max_q"
    den_fresh = fresh_mix - child_lo
    den_stale = stale_mix - child_lo
    assert den_fresh > den_stale * 1.2, (
        f"q_denom barely moves ({den_fresh} vs {den_stale}); the scenario no "
        "longer exercises the mechanism this test names"
    )
    del qv


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_halving_uses_the_fresh_root_value_not_the_running_average(
    shape: _Shape,
) -> None:
    """The C elimination must score against ``root_qs``, not ``W[root]/N[root]``."""
    rnd = _run_one_halving_round(
        shape, shape.refuted, root_w=shape.refuted.root_q,
    )
    fresh = _argmax(_reference_scores(rnd, raw_value=shape.refuted.root_q))
    stale = _argmax(_reference_scores(rnd, raw_value=rnd.stale_root_value))
    assert stale != fresh, (
        f"[{shape.name}] scenario no longer discriminates the two baselines — "
        "the assertion below would pass for free; re-tune this shape's "
        "refuted-root scenario"
    )
    assert rnd.survivor == fresh, (
        f"[{shape.name}] C halving kept {rnd.survivor}; the Python reference "
        f"(_completed_q_transform, raw_value=root_qs) keeps {fresh}. "
        f"W[root]/N[root]={rnd.stale_root_value:.6f} would keep {stale}."
    )


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_unvisited_legal_actions_stay_in_the_normalization_pool(
    shape: _Shape,
) -> None:
    """Every unvisited legal action contributes ``mixed_value`` to min/max."""
    rnd = _run_one_halving_round(
        shape, shape.refuted, root_w=shape.refuted.root_q,
    )
    full = _argmax(_reference_scores(rnd, raw_value=shape.refuted.root_q))
    visited_only = _argmax(
        _reference_scores(
            rnd, raw_value=shape.refuted.root_q, include_unvisited=False,
        ),
    )
    assert visited_only != full, (
        f"[{shape.name}] scenario no longer discriminates the min/max pool — "
        "re-tune this shape's refuted-root scenario"
    )
    assert rnd.survivor == full, (
        f"[{shape.name}] C halving kept {rnd.survivor}; normalizing over ALL "
        f"legal actions keeps {full}, over the visited children only keeps "
        f"{visited_only}"
    )


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_halving_is_independent_of_the_root_nodes_stored_value(
    shape: _Shape,
) -> None:
    """``W[root]``/``N[root]`` is not an input to the elimination at all.

    Same ``root_qs``, same leaf values, same priors — only the value the root
    node was SEEDED with differs, and the search must not be able to tell.
    """
    scenario = shape.confirmed
    hi = _run_one_halving_round(shape, scenario, root_w=0.95)
    lo = _run_one_halving_round(shape, scenario, root_w=-0.95)
    assert hi.stale_root_value != lo.stale_root_value, (
        "the two runs must actually differ in W[root]/N[root]"
    )
    stale_hi = _argmax(_reference_scores(hi, raw_value=hi.stale_root_value))
    stale_lo = _argmax(_reference_scores(lo, raw_value=lo.stale_root_value))
    assert stale_hi != stale_lo, (
        f"[{shape.name}] scenario no longer discriminates the seed value — the "
        "assertion below would pass for free; re-tune this shape's "
        "confirmed-root scenario"
    )
    fresh = _argmax(_reference_scores(hi, raw_value=scenario.root_q))
    assert hi.survivor == lo.survivor == fresh, (
        f"[{shape.name}] survivor moved with the root node's seed value: "
        f"{hi.survivor} (seed +0.95) vs {lo.survivor} (seed -0.95); the "
        f"fresh-root reference keeps {fresh}"
    )
    hi_actions, hi_visits = hi.tree.get_children_visits(hi.root_id)
    lo_actions, lo_visits = lo.tree.get_children_visits(lo.root_id)
    assert list(hi_actions) == list(lo_actions)
    assert list(hi_visits) == list(lo_visits)


@pytest.mark.parametrize(
    ("shape", "scenario", "stored_cand_prior"),
    [
        (
            _SHAPES[0],
            _Scenario(
                root_q=-0.90,
                child_q=(-0.20, 0.00),
                cand_prior=(0.60, 0.05),
            ),
            (0.01, 0.90),
        ),
        (
            _SHAPES[1],
            _Scenario(
                root_q=-0.40,
                child_q=(0.18, 0.20),
                cand_prior=(0.04, 0.0004),
            ),
            (0.001, 0.90),
        ),
    ],
    ids=_SHAPE_IDS,
)
def test_halving_refreshes_carried_root_priors_from_the_current_search(
    shape: _Shape,
    scenario: _Scenario,
    stored_cand_prior: tuple[float, float],
) -> None:
    """A reused root's old edge priors must not weight this search's mixed Q.

    The tree is deliberately expanded with a prior distribution from an older
    search, while start_gumbel_sims receives a different CURRENT dense prior.
    Both production root transforms are hand-tuned so the two rules choose
    opposite survivors; the fixture therefore fails if the stale edge priors
    still reach gss_score_and_halve's weighted_q.
    """
    rnd = _run_one_halving_round(
        shape,
        scenario,
        root_w=scenario.root_q,
        stored_cand_prior=stored_cand_prior,
    )
    legal, _vis, _qv = _root_children(rnd)
    stale_priors = rnd.priors.copy()
    stale_priors[legal] = (
        1.0 - sum(stored_cand_prior)
    ) / float(legal.size - 2)
    stale_priors[rnd.candidates[0]] = stored_cand_prior[0]
    stale_priors[rnd.candidates[1]] = stored_cand_prior[1]

    current = _argmax(
        _reference_scores(
            rnd,
            raw_value=scenario.root_q,
            mix_priors=rnd.priors,
        ),
    )
    stale = _argmax(
        _reference_scores(
            rnd,
            raw_value=scenario.root_q,
            mix_priors=stale_priors,
        ),
    )
    assert current != stale, (
        f"[{shape.name}] fixture stopped discriminating current vs carried "
        "root priors; the production assertion below would pass for free"
    )
    assert rnd.survivor == current, (
        f"[{shape.name}] C halving kept {rnd.survivor}; current-search priors "
        f"keep {current}, while the carried tree priors keep {stale}"
    )


def test_the_loaded_extension_announces_its_halving_revision(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A running process must be able to say which halving rule its .so runs.

    Nothing else makes it observable: no signature changed, so a stale .so runs
    the OLD rule silently and a routine rebuild for an unrelated ``.c`` edit
    deploys the new one. The line is read from the loaded module, so it reports
    the artifact rather than the checkout — pinned here by driving it from a
    patched constant, which a hardcoded literal would survive.
    """
    from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod

    assert _mcts_tree_ext.GSS_HALVING_REV == 3

    monkeypatch.setattr(gumbel_c_mod, "_halving_rev_reported", False)
    gumbel_c_mod._report_halving_rev()
    line = capsys.readouterr().err
    assert "gss_halving_rev=3" in line
    assert "fresh root_qs + current-search root priors" in line

    # Once per process, not once per search.
    gumbel_c_mod._report_halving_rev()
    assert capsys.readouterr().err == ""

    # Revision 2 fixed the root-value baseline but still weighted carried
    # child Q with carried edge priors.
    monkeypatch.setattr(_mcts_tree_ext, "GSS_HALVING_REV", 2)
    monkeypatch.setattr(gumbel_c_mod, "_halving_rev_reported", False)
    gumbel_c_mod._report_halving_rev()
    rev2 = capsys.readouterr().err
    assert "gss_halving_rev=2" in rev2
    assert "carried root priors" in rev2

    # A pre-fix .so has no such constant, and that IS revision 1 — not unknown.
    monkeypatch.delattr(_mcts_tree_ext, "GSS_HALVING_REV", raising=False)
    monkeypatch.setattr(gumbel_c_mod, "_halving_rev_reported", False)
    gumbel_c_mod._report_halving_rev()
    legacy = capsys.readouterr().err
    assert "gss_halving_rev=1" in legacy
    assert "W[root]/N[root]" in legacy
