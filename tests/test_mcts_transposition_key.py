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

# Positions that reach ep-capable nodes during search, from the audit sweep.
_EP_RICH_FENS = (
    "rnbqkb1r/pp2pppp/3p1n2/2pP4/8/2N2N2/PPP1PPPP/R1BQKB1R b KQkq - 0 5",
    "r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
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


@pytest.mark.parametrize("fen", _EP_RICH_FENS)
def test_expanded_nodes_carry_their_own_legal_moves(monkeypatch, fen: str) -> None:
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    tree, rid = _search_tree(fen, sims=512, seed=2)
    checked, violations = _walk_violations(tree, rid, chess.Board(fen))
    assert checked > 100, "search produced too few expanded nodes to be a test"
    assert not violations, "\n".join(violations[:5])


def test_transposition_guard_runs_and_passes(monkeypatch) -> None:
    """The child-set verification is not decorative: it must actually execute
    on a search that transposes, and must accept every donor once the key
    carries ep rights."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _mcts_tree.tt_stats(reset=True)
    _search_tree(_EP_RICH_FENS[0], sims=512, seed=2)
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
