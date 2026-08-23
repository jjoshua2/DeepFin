"""Transposition-table correctness for the C Gumbel search (audit W1 + W2).

The Gumbel path is the only consumer of the tree's transposition table, and a
hit does not merely reuse a value: it copies the donor node's child ACTION LIST
onto the new leaf and marks it expanded forever. So the key must imply "same
legal move set". ``CBoard.zobrist_hash`` does not — it is the repetition key and
deliberately ignores en passant — which injected illegal pawn captures into
production trees and dropped legal ones.

These tests fail on the pre-fix build (67 illegal + 3 missing children over the
same 32k-node sample) and pass after it.
"""
from __future__ import annotations

import dataclasses

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import (
    root_coverage_miss_count,
    run_gumbel_root_many_c,
)
from chess_anti_engine.moves.encode import move_to_index

POLICY_SIZE = 4672

# Same placement/castling/side-to-move, differing only in the ep right. White
# has a pawn on d5, so e5-e6 en passant is a real extra legal move.
_EP_FEN = "rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3"
_NO_EP_FEN = "rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3"

# (fen, sims, seed) triples that are each individually RED on the pre-fix build.
# A parametrisation that cannot fail is decoration, so the search shape is pinned
# per position rather than shared: the same FEN at a different seed produces a
# clean tree. Trailing number = corrupt nodes observed on `main` at that shape,
# re-derivable with the walk below against an un-patched extension.
_EP_RICH_CASES = (
    ("rnbqkb1r/pp2pppp/3p1n2/2pP4/8/2N2N2/PPP1PPPP/R1BQKB1R b KQkq - 0 5", 512, 2),   # 22
    ("r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4", 1024, 1),  # 2
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", 512, 3),       # 1
)


class _Ev:
    """Deterministic pseudo-random evaluator keyed on the encoded row."""

    def __init__(self, seed: int = 20260803) -> None:
        rng = np.random.default_rng(seed)
        self._pol = rng.standard_normal((1024, POLICY_SIZE)).astype(np.float32)
        self._wdl = rng.standard_normal((1024, 3)).astype(np.float32)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(n, -1)
        idx = np.array(
            [int.from_bytes(flat[i].tobytes()[:6], "little") % 1024 for i in range(n)],
            dtype=np.int64,
        )
        return self._pol[idx], self._wdl[idx]


def _cb(fen: str) -> CBoard:
    return CBoard.from_board(chess.Board(fen))


def _legal_set(board: chess.Board) -> set[int]:
    return {int(x) for x in CBoard.from_board(board).legal_move_indices()}


def test_zobrist_hash_stays_ep_blind() -> None:
    """The repetition hash is NOT what changed — it is persisted in resume
    records and matches python-chess is_repetition() semantics."""
    assert _cb(_EP_FEN).zobrist_hash == _cb(_NO_EP_FEN).zobrist_hash


def test_transposition_key_separates_positions_with_different_legal_moves() -> None:
    ep, no_ep = _cb(_EP_FEN), _cb(_NO_EP_FEN)
    # The two positions genuinely differ: d5xe6 e.p. is legal in one only.
    assert _legal_set(chess.Board(_EP_FEN)) != _legal_set(chess.Board(_NO_EP_FEN))
    assert ep.transposition_key != no_ep.transposition_key


def test_transposition_key_ignores_an_unusable_ep_right() -> None:
    """An ep square no pawn can capture on does not change the legal move set,
    so it must not split the table (that is the hit rate the exclusion bought)."""
    # 1. e4: ep square e3 is set, but Black has no pawn on d4/f4.
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    cb = CBoard.from_board(board)
    assert cb.transposition_key == cb.zobrist_hash


def test_transposition_key_is_stable_across_construction_paths() -> None:
    board = chess.Board(_NO_EP_FEN)
    board.push(chess.Move.from_uci("e2e4"))  # irrelevant push; recompute path
    pushed = CBoard.from_board(chess.Board(_EP_FEN))
    fresh = CBoard.from_board(chess.Board(_EP_FEN))
    assert pushed.transposition_key == fresh.transposition_key


def _search_tree(fen: str, sims: int, seed: int):
    cfg = GumbelConfig(
        simulations=sims, topk=16, c_scale=0.1, temperature=0.0, add_noise=False
    )
    res = run_gumbel_root_many_c(
        None,
        [chess.Board(fen)],
        device="cpu",
        rng=np.random.default_rng(seed),
        cfg=cfg,
        evaluator=_Ev(seed),
        target_batch=0,
        vloss_weight=1,
    )
    return res[4], int(res[5][0])


def _walk_violations(tree, root_id: int, root_board: chess.Board):
    """Every expanded node's child-action set must equal the legal move set of
    the position on its action path. Returns (nodes_checked, violations)."""
    violations: list[str] = []
    checked = 0
    stack = [(root_id, root_board)]
    while stack:
        node, board = stack.pop()
        if not tree.is_expanded(node):
            continue
        actions, _visits = tree.get_children_visits(node)
        got = {int(a) for a in actions}
        if not got:
            continue
        checked += 1
        legal = _legal_set(board)
        if got != legal:
            violations.append(
                f"node={node} fen={board.fen()} "
                f"extra={sorted(got - legal)} missing={sorted(legal - got)}"
            )
            continue  # an illegal action cannot be replayed to recurse
        i2m = {int(move_to_index(mv, board)): mv for mv in board.legal_moves}
        for a in got:
            child = tree.find_child(node, a)
            if child < 0:
                continue
            nb = board.copy(stack=True)
            nb.push(i2m[a])
            stack.append((child, nb))
    return checked, violations


@pytest.mark.parametrize(("fen", "sims", "seed"), _EP_RICH_CASES)
def test_expanded_nodes_carry_their_own_legal_moves(
    monkeypatch, fen: str, sims: int, seed: int
) -> None:
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    tree, rid = _search_tree(fen, sims=sims, seed=seed)
    checked, violations = _walk_violations(tree, rid, chess.Board(fen))
    assert checked > 100, "search produced too few expanded nodes to be a test"
    assert not violations, "\n".join(violations[:5])


def test_transposition_guard_runs_and_passes(monkeypatch) -> None:
    """The child-set verification is not decorative: it must actually execute
    on a search that transposes, and must accept every donor once the key
    carries ep rights."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _mcts_tree.tt_stats(reset=True)
    fen, sims, seed = _EP_RICH_CASES[0]
    _search_tree(fen, sims=sims, seed=seed)
    stats = _mcts_tree.tt_stats()
    assert stats["probe_hits"] > 0, "no transposition hit — test is not exercising the guard"
    assert stats["reuse"] == stats["probe_hits"]
    assert stats["reject"] == 0


def test_reused_root_coverage_check_runs_without_allowed_indices(monkeypatch) -> None:
    """Audit W2: the coverage check used to be short-circuited whenever
    ``allowed_root_indices_batch is None``, which selfplay — the only caller
    that carries a tree across plies — always passes."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = chess.Board(_EP_FEN)
    legal = np.array(sorted(_legal_set(board)), dtype=np.int32)
    assert legal.size > 2

    tree = _mcts_tree.MCTSTree()
    stale_root = tree.add_root(1, 0.0)
    # A root expanded with a child set that is missing a legal action — exactly
    # what a stale/transposed node looks like.
    subset = legal[:-1]
    tree.expand(
        stale_root,
        subset,
        np.full(subset.size, 1.0 / subset.size, dtype=np.float64),
    )

    before = root_coverage_miss_count()
    cfg = GumbelConfig(
        simulations=8, topk=4, c_scale=0.1, temperature=0.0, add_noise=False
    )
    res = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=cfg,
        evaluator=_Ev(1),
        tree=tree,
        root_node_ids=[stale_root],
        allowed_root_indices_batch=None,
        target_batch=0,
        vloss_weight=1,
    )
    new_root = int(res[5][0])
    assert new_root != stale_root, "deficient root was reused"
    assert root_coverage_miss_count() == before + 1
    actions, _ = tree.get_children_visits(new_root)
    assert {int(a) for a in actions} == set(legal.tolist())


# --- W2b: the gate is support EQUALITY, not coverage ---------------------
#
# A carried root may be a strict SUPERSET of the support the CURRENT search is
# allowed to touch: `allowed_root_indices_batch` (UCI `searchmoves`) and the
# winning-root terminal-draw prune both narrow `legal_idx` below the previous
# ply's expansion. `gss_score_and_halve` reads max_visit / total_visits /
# weighted_q / mixed_value / min_q / max_q off ALL of the root's children, so an
# excluded child still moves the Q transform every INCLUDED candidate is scored
# through — while the improved policy on top is built over the current support
# only. The superset gate accepted exactly that.

# Position with 37 legal moves; the first 10 policy indices are the support and
# the 11th is the carried-but-excluded child. Search shapes pinned per case
# (the house rule: a parametrisation that cannot fail is decoration) — the
# selfplay shape is the live yaml's `c_scale=0.1` linear root, the play shape is
# `gumbel.PLAY_SEARCH_DEFAULTS`' root-log split, which is what the UCI path that
# actually carries trees across moves runs.
_NARROW_FEN = "r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4"
_NARROW_SHAPES = ("selfplay_linear_root", "play_log_root")


def _narrow_cfg(shape: str) -> GumbelConfig:
    """Written as ``dataclasses.replace`` on one base rather than ``**dict``:
    a ``dict[str, float]`` splat widens every keyword to ``float`` and
    basedpyright rejects the ``int``/``bool``/``str`` fields it lands on."""
    base = GumbelConfig(
        simulations=128, topk=16, c_scale=0.1, temperature=0.0, add_noise=False
    )
    if shape == "selfplay_linear_root":
        return base
    return dataclasses.replace(
        base, c_scale=0.025, c_visit=50.0, c_visit_root=900.0,
        c_scale_root=7.0, q_visit_exp_root=-1.0,
    )


def _narrowed_support_tree(carried: list[int], excluded: int, visits: int):
    """A carried root over ``carried`` whose ``excluded`` child holds ``visits``
    visits at Q = +1 — the drawing child of a now-winning root, or the move a
    ``searchmoves`` list just dropped. Built by tree surgery rather than a first
    search so the two arms differ ONLY in whether the root is reused: the tree
    handed to both is byte-for-byte the same shape, and no transposition-table
    entry exists to make node ids matter."""
    tree = _mcts_tree.MCTSTree()
    rid = tree.add_root(1, 0.0)
    acts = np.array(sorted(carried), dtype=np.int32)
    tree.expand(rid, acts, np.full(acts.size, 1.0 / acts.size, dtype=np.float64))
    cid = tree.find_child(rid, excluded)
    assert cid >= 0
    path = np.array([rid, cid], dtype=np.int32)
    for _ in range(visits):
        tree.backprop(path, 1.0)
    return tree, rid


@pytest.mark.parametrize("shape", _NARROW_SHAPES)
def test_a_narrowed_root_does_not_inherit_the_excluded_child(
    monkeypatch, shape: str
) -> None:
    """RED on the pre-fix (superset) gate, in both search shapes.

    Arm: carried root over support+{excluded}, reused. Control: the SAME tree,
    same seed, same evaluator, same narrowed support — but ``root_node_ids=None``
    so the root is built over the current support alone, which is the Python
    reference's (``gumbel.py``, no tree reuse) semantic. The two must agree
    exactly. Pre-fix they do not: the played move flips between two INCLUDED
    moves, and the root's visit split across the included moves shifts with it —
    both decided by a child the current search says does not exist.

    Control chosen as the same populated tree rather than an empty one so the
    single manipulated variable is the reuse decision itself — post-fix the arm
    rebuilds into the identical node slot, so equality is exact, not approximate.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = chess.Board(_NARROW_FEN)
    legal = sorted(_legal_set(board))
    assert len(legal) > 11, "position must be wide enough to narrow"
    support = legal[:10]
    excluded = legal[10]
    carried = [*support, excluded]

    cfg = _narrow_cfg(shape)

    def _search(tree, root_ids):
        return run_gumbel_root_many_c(
            None, [chess.Board(_NARROW_FEN)], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg, evaluator=_Ev(1),
            tree=tree, root_node_ids=root_ids,
            allowed_root_indices_batch=[set(support)],
            target_batch=0, vloss_weight=1,
        )

    arm_tree, stale_root = _narrowed_support_tree(carried, excluded, visits=4000)
    misses_before = root_coverage_miss_count()
    arm = _search(arm_tree, [stale_root])
    arm_root = int(arm[5][0])

    assert root_coverage_miss_count() == misses_before + 1, (
        "the wider carried root was reused instead of rejected"
    )
    assert arm_root != stale_root
    arm_actions, arm_visits = arm_tree.get_children_visits(arm_root)
    assert {int(a) for a in arm_actions} == set(support), (
        "the rebuilt root must hold exactly the current support"
    )

    ctl_tree, _ = _narrowed_support_tree(carried, excluded, visits=4000)
    ctl = _search(ctl_tree, None)
    ctl_actions, ctl_visits = ctl_tree.get_children_visits(int(ctl[5][0]))

    # Played move first: it is the assertion with a chess meaning, and the one
    # the superset gate flips.
    assert int(arm[1][0]) == int(ctl[1][0]), (
        f"reused wide root played {int(arm[1][0])}, support-only root played "
        f"{int(ctl[1][0])} — the excluded child decided the halving"
    )
    assert dict(
        zip((int(a) for a in arm_actions), (int(v) for v in arm_visits), strict=True)
    ) == dict(
        zip((int(a) for a in ctl_actions), (int(v) for v in ctl_visits), strict=True)
    ), "the excluded child moved the visit split over the included moves"
    assert np.array_equal(arm[0][0], ctl[0][0]), "improved policy diverged"
    assert float(arm[2][0]) == float(ctl[2][0]), "root value diverged"


def test_the_root_support_gate_is_equality_not_coverage() -> None:
    """The predicate itself, in every direction. The old superset reading
    (``isin(actions, child_actions).all()`` behind a ``child_actions.size <
    actions.size`` early-out) agrees on the equal, reordered, wider and
    same-size-but-different rows — and disagrees on the two that matter: a
    NARROWED support, and the empty support its ``actions.size == 0`` early-out
    waved through."""
    tree = _mcts_tree.MCTSTree()
    rid = tree.add_root(1, 0.0)
    acts = np.array([7, 11, 23], dtype=np.int32)
    tree.expand(rid, acts, np.full(3, 1.0 / 3.0, dtype=np.float64))
    matches = gumbel_c_mod._expanded_root_matches_actions

    assert matches(tree, rid, np.array([7, 11, 23], dtype=np.int32))
    assert matches(tree, rid, np.array([23, 7, 11], dtype=np.int64)), "order-blind"
    assert not matches(tree, rid, np.array([7, 11], dtype=np.int32)), (
        "narrowed support: the ROOT is a superset — this is the bug"
    )
    assert not matches(tree, rid, np.array([7, 11, 23, 40], dtype=np.int32)), (
        "the root is missing an action we are about to search"
    )
    assert not matches(tree, rid, np.array([7, 11, 40], dtype=np.int32)), (
        "same size, different members — the size early-out must not decide it"
    )
    assert not matches(tree, rid, np.empty(0, dtype=np.int32)), (
        "an empty support matches only an empty child set"
    )


def test_selfplay_shaped_tree_carry_still_reuses_its_root(monkeypatch) -> None:
    """Positive direction of W2. Making the coverage check unconditional must not
    quietly disable tree carry: if reuse stopped happening altogether every other
    test here would still pass, and the only symptom would be a slower search
    that throws away its tree every ply.

    This walks a selfplay-shaped carry — persistent tree, root advanced with
    ``find_child`` on the played action, ``allowed_root_indices_batch=None``.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = chess.Board(
        "r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4"
    )
    tree = _mcts_tree.MCTSTree()
    cfg = GumbelConfig(
        simulations=64, topk=8, c_scale=0.1, temperature=0.0, add_noise=False
    )

    root_ids: list[int] | None = None
    misses_before = root_coverage_miss_count()
    reused_plies = 0
    for ply in range(5):
        res = run_gumbel_root_many_c(
            None,
            [board],
            device="cpu",
            rng=np.random.default_rng(7),
            cfg=cfg,
            evaluator=_Ev(7),
            tree=tree,
            root_node_ids=root_ids,
            allowed_root_indices_batch=None,
            target_batch=0,
            vloss_weight=1,
        )
        rid = int(res[5][0])
        if root_ids is not None:
            reused_plies += int(rid == root_ids[0])
        action = int(res[1][0])
        move = {int(move_to_index(mv, board)): mv for mv in board.legal_moves}[action]
        board.push(move)
        child = tree.find_child(rid, action)
        assert child >= 0, f"played action has no child node at ply {ply}"
        root_ids = [child]

    assert reused_plies == 4, (
        f"tree carry stopped reusing roots: only {reused_plies}/4 carried plies "
        "reused their root"
    )
    assert root_coverage_miss_count() == misses_before, (
        "the coverage check rejected a legitimately carried root"
    )


def _reset_guard_reporter(monkeypatch, *, misses: int = 0) -> None:
    """Clear the reporter's rate limiter and both counters."""
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_reported", (0, 0))
    monkeypatch.setattr(gumbel_c_mod, "_ROOT_COVERAGE_MISSES", misses)
    _mcts_tree.tt_stats(reset=True)


def _tiny_search() -> None:
    cfg = GumbelConfig(
        simulations=2, topk=2, c_scale=0.1, temperature=0.0, add_noise=False
    )
    run_gumbel_root_many_c(
        None,
        [chess.Board()],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=cfg,
        evaluator=_Ev(0),
        target_batch=0,
        vloss_weight=1,
    )


def test_guard_health_is_silent_when_nothing_fired(monkeypatch, capsys) -> None:
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch)
    _tiny_search()
    assert "[mcts]" not in capsys.readouterr().err


def test_root_coverage_miss_is_reported_on_the_production_path(
    monkeypatch, capsys
) -> None:
    """A counter no production path reads is the defect these guards exist to
    fix. The shared C-path entry point must actually emit the line."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch)

    board = chess.Board(_EP_FEN)
    legal = np.array(sorted(_legal_set(board)), dtype=np.int32)
    tree = _mcts_tree.MCTSTree()
    stale_root = tree.add_root(1, 0.0)
    subset = legal[:-1]
    tree.expand(
        stale_root,
        subset,
        np.full(subset.size, 1.0 / subset.size, dtype=np.float64),
    )
    cfg = GumbelConfig(
        simulations=8, topk=4, c_scale=0.1, temperature=0.0, add_noise=False
    )
    run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=_Ev(1), tree=tree, root_node_ids=[stale_root],
        allowed_root_indices_batch=None, target_batch=0, vloss_weight=1,
    )
    assert root_coverage_miss_count() == 1
    capsys.readouterr()

    # The reporter runs at search ENTRY, so the miss above is reported by the
    # NEXT search; clear the rate limiter the way 60s of wall clock would.
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    _tiny_search()
    err = capsys.readouterr().err
    assert "[mcts] search guards FIRED" in err, err
    assert "root_coverage_miss=1" in err, err


def test_tt_reject_is_reported_on_the_production_path(monkeypatch, capsys) -> None:
    """The reject counter reaches the operator too. Driven by patching the
    counter SOURCE the production reporter reads, so this pins the wiring — the
    ep-aware key means no real search can produce a reject to observe."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch)
    monkeypatch.setattr(
        gumbel_c_mod._mcts_tree_ext,
        "tt_stats",
        lambda *a, **k: {"probe_hits": 9, "reuse": 4, "reject": 5},
    )
    _tiny_search()
    err = capsys.readouterr().err
    assert "[mcts] search guards FIRED" in err, err
    assert "tt_donor_reject=5" in err, err


def test_guard_health_reports_once_per_change(monkeypatch, capsys) -> None:
    """Static counters must not spam every search."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch, misses=3)
    _tiny_search()
    assert "[mcts] search guards FIRED" in capsys.readouterr().err

    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    _tiny_search()
    assert "[mcts]" not in capsys.readouterr().err
