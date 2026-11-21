"""``GumbelConfig.root_full_width``: a full-width first round at the root.

Every searched legal root child without visits gets one simulation before
sequential halving; the topk candidates are then chosen by gumbel + log prior +
sigma(q) from those evaluations, and the unchanged halving schedule runs on
them. Each test below states the mutant it exists to catch.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel import GumbelConfig, _completed_q_transform, run_gumbel_root_many
from chess_anti_engine.mcts.gumbel_c import MCTSTree, buffer_full_flushes, run_gumbel_root_many_c
from chess_anti_engine.moves.encode import index_to_move_strict
from tests.test_c_only_gumbel_knobs import _HashEvaluator

_FENS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/pp2bppp/2n1bn2/3p4/3P4/2NBBN2/PP3PPP/R2Q1RK1 w - - 0 11",
    "8/5pk1/6p1/3R4/5P2/6P1/r5K1/8 b - - 1 40",
]
_SIMS = 64


def _cfg(**kw: Any) -> GumbelConfig:
    base = GumbelConfig(simulations=_SIMS, topk=16, add_noise=True, temperature=0.0, c_scale=0.1)
    return dataclasses.replace(base, **kw)


def _search(cfg: GumbelConfig, boards: list[chess.Board], *, seed: int = 7, tree: MCTSTree | None = None,
            root_ids: list[int] | None = None) -> tuple[Any, list[Any]]:
    stats: list[Any] = []
    out = run_gumbel_root_many_c(None, boards, device="cpu", rng=np.random.default_rng(seed), cfg=cfg,
                                 evaluator=_HashEvaluator(), tree=tree, root_node_ids=root_ids,
                                 root_stats_out=stats, vloss_weight=1)
    return out, stats


def _boards() -> list[chess.Board]:
    return [chess.Board(f) for f in _FENS]


def test_round0_visits_every_searched_child_and_is_counted() -> None:
    """Mutant: skip round 0 (or count it as zero) -> a child stays unvisited, or
    the visit total no longer equals sims + prelude."""
    _, on = _search(_cfg(root_full_width=True), _boards())
    for s in on:
        assert s is not None
        assert (s.visits > 0).all(), "an unvisited root child would be filled with root_q"
        assert s.prelude_visits == s.legal.size  # fresh roots: every child was unvisited
        assert int(s.visits.sum()) == _SIMS + s.prelude_visits
        assert s.start_visits == 0
  # Control: the same search with the knob off leaves children unvisited and
  # spends exactly the budget, so the assertions above can fail.
    _, off = _search(_cfg(), _boards())
    assert any((s.visits == 0).any() for s in off)
    assert all(s.prelude_visits == 0 and int(s.visits.sum()) == _SIMS for s in off)


def test_a_reused_root_only_evaluates_its_unvisited_children() -> None:
    """Mutant: visit every child (visited ones too) or count the plan instead of
    the tree -> the visit identity breaks on carried roots."""
    cfg = _cfg(root_full_width=True)
    tree = MCTSTree()
    boards = _boards()
    out, _ = _search(cfg, boards, tree=tree)
    actions, root_ids = out[1], out[5]
    nxt, rids = [], []
    for b, a, rid in zip(boards, actions, root_ids, strict=True):
        b2 = b.copy()
        b2.push(index_to_move_strict(int(a), b))
        nxt.append(b2)
        rids.append(tree.find_child(rid, int(a)))
    _, stats = _search(cfg, nxt, seed=8, tree=tree, root_ids=rids)
    carried = [s for s in stats if s is not None and s.start_visits > 0]
    assert carried, "no root was carried; the test would be vacuous"
    for s in stats:
        if s is None:
            continue
        assert (s.visits > 0).all()
        assert int(s.visits.sum()) == s.start_visits + s.prelude_visits + _SIMS
    assert all(s.prelude_visits < s.legal.size for s in carried)


def test_candidates_are_chosen_with_the_round0_q(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant: rank by gumbel + log prior only (drop sigma), or compute the new
    candidates without handing them to the halving call."""
    real = gumbel_c_mod._full_width_round0
    seen: list[dict[str, Any]] = []

    def spy(**kw: Any) -> None:
        real(**kw)
        tree = kw["tree"]
        for i, cands in enumerate(kw["candidates_per_board"]):
            prep = kw["root_prep"][i]
            if cands is None or prep is None:
                continue
            legal, pri, g = prep.legal_idx, kw["root_pri"][i][prep.legal_idx], kw["gumbels_per_board"][i][prep.legal_idx]
            root_q = float(kw["root_qs"][i])
            acts, vis, q = tree.get_children_q(kw["root_ids"][i], root_q)
            slot = {int(a): j for j, a in enumerate(acts)}
            visits = np.array([float(vis[slot[int(a)]]) for a in legal])
            qv = np.array([float(q[slot[int(a)]]) for a in legal])
            assert (visits == 1).all()  # fresh root: exactly the round-0 visit, nothing else yet
            sigma = _completed_q_transform(actions=legal, priors=pri, visits=visits, qvalues=qv,
                                           raw_value=root_q, cfg=kw["cfg"], root=True)
            base = g + np.log(np.maximum(pri, 1e-12))
            m = int(prep.m)
            seen.append({
                "board": i, "chosen": set(cands), "remaining": set(kw["remaining_per_board"][i]),
                "with_q": {int(legal[j]) for j in np.argsort(-(base + sigma), kind="stable")[:m]},
                "prior_only": {int(legal[j]) for j in np.argsort(-base, kind="stable")[:m]},
            })

    monkeypatch.setattr(gumbel_c_mod, "_full_width_round0", spy)
    _, stats = _search(_cfg(root_full_width=True, c_scale=1.0), _boards())
    assert seen
    for rec in seen:
        assert rec["chosen"] == rec["with_q"]
        assert rec["remaining"] == rec["chosen"]
  # Non-vacuous: the q term must actually change the set somewhere.
    assert any(rec["with_q"] != rec["prior_only"] for rec in seen)
  # The halving ran on the chosen set: at 64 sims / 16 candidates every
  # candidate gets a halving visit on top of its round-0 visit, nothing else does.
    for rec in seen:
        s = stats[rec["board"]]
        assert {int(a) for a, v in zip(s.legal, s.visits, strict=True) if v >= 2} == rec["chosen"]


def test_round0_packs_into_the_leaf_buffer_without_a_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant: send all round-0 children in one call -> the 3-board batch needs
    more rows than the 64-row buffer and the C registers a buffer-full flush."""
    monkeypatch.setattr(gumbel_c_mod, "leaf_buffer_rows", lambda n_boards, *, topk, pipelined: 64)
    boards = _boards()[:3]
    assert sum(b.legal_moves.count() for b in boards) > 64
    before = buffer_full_flushes()
    _, stats = _search(_cfg(root_full_width=True), boards)
    assert buffer_full_flushes() == before
    for s in stats:
        assert (s.visits > 0).all()
        assert int(s.visits.sum()) == _SIMS + s.prelude_visits


def test_the_python_reference_search_refuses_it() -> None:
    with pytest.raises(ValueError, match="root_full_width"):
        run_gumbel_root_many(None, [chess.Board()], device="cpu", rng=np.random.default_rng(0),
                             cfg=_cfg(root_full_width=True), evaluator=_HashEvaluator())


def _integer_digest(cfg: GumbelConfig) -> str:
    """Three plies with tree reuse; only integer search decisions are hashed
    (actions, supports, visit vectors, survivors, carried visits), so the pin is
    insensitive to float formatting but sees any change in what was searched."""
    h = hashlib.sha256()
    tree = MCTSTree()
    boards = _boards()
    rids: list[int] | None = None
    rng = np.random.default_rng(5)
    for _ply in range(3):
        stats: list[Any] = []
        out = run_gumbel_root_many_c(None, boards, device="cpu", rng=rng, cfg=cfg, evaluator=_HashEvaluator(),
                                     tree=tree, root_node_ids=rids, root_stats_out=stats, vloss_weight=1)
        actions, root_ids = out[1], out[5]
        for a, s in zip(actions, stats, strict=True):
            h.update(f"{int(a)};".encode())
            if s is not None:
                h.update(np.asarray(s.legal, np.int64).tobytes())
                h.update(np.asarray(s.visits, np.int64).tobytes())
                h.update(f"{int(s.survivor)};{int(s.start_visits)};".encode())
        nxt, nr = [], []
        for b, a, rid in zip(boards, actions, root_ids, strict=True):
            b2 = b.copy()
            b2.push(index_to_move_strict(int(a), b))
            nxt.append(b2)
            nr.append(tree.find_child(rid, int(a)) if rid >= 0 else -1)
        boards, rids = nxt, nr
    return h.hexdigest()


# Computed on origin/main 918785204 (before this knob existed) with the same
# script; the knob-off search must reproduce it exactly.
_MAIN_DIGEST = "814c97851b4ec3cbb7127bf4244dc5fba8c2bd4fb61e43434cc278da82b6940a"


def test_the_default_off_search_is_unchanged_from_main() -> None:
    """Mutant: any change to the knob-off path (round 0 always on, the refactored
    sequential driver reordering a call) moves the digest."""
    assert _integer_digest(_cfg()) == _MAIN_DIGEST
    assert _integer_digest(_cfg(root_full_width=True)) != _MAIN_DIGEST


def test_the_yaml_key_is_in_the_schema_and_validated() -> None:
    """Mutant: key missing from the selfplay schema (a launch would die on it),
    or the validator coercing a quoted string with bool() ("false" -> on)."""
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    flat = flatten_run_config_defaults({"selfplay": {"gumbel_root_full_width": True}})
    assert flat["gumbel_root_full_width"] is True
    assert TrialConfig.from_dict(flat).gumbel_root_full_width is True
    assert TrialConfig.from_dict({}).gumbel_root_full_width is False
    for bad in ("false", 1, None):
        with pytest.raises(ValueError, match="gumbel_root_full_width"):
            TrialConfig.from_dict({"gumbel_root_full_width": bad})
