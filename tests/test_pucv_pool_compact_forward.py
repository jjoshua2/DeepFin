"""The PUCT pool forwards only the distinct non-terminal leaves of a batch.

Terminal rows (a twofold repetition, mate, stalemate: already backpropped by
the descent) used to go to the device as zeroed rows, and a leaf that several
descents of one batch selected (virtual loss does not prevent it) was
forwarded once per selection. Both are wasted rows: now only the first row of
each distinct non-terminal leaf is forwarded and its output is copied to the
later selections, so the integration -- and the tree -- must be unchanged.

The evaluator here computes each row's output from that row alone, one row at
a time, so the outputs cannot depend on the batch's size or composition; the
comparison then checks only the pool's plumbing (compaction and scatter).
"""
from __future__ import annotations

import random
import threading
from typing import Any

import chess
import numpy as np
import pytest

import chess_anti_engine.uci.multi_gpu_pucv_pool as pool_module
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.uci.multi_gpu_pucv_pool import MultiGpuPucvConfig, MultiGpuPucvPool

_PLANES = 146


class _RowwiseEvaluator:
    """Each row's outputs are a fixed function of that row's input alone."""

    n_slots = 2

    def __init__(self, max_batch: int = 64) -> None:
        rng = np.random.default_rng(1234)
        flat = _PLANES * 64
        self._w_pol = rng.standard_normal((flat, 4672)).astype(np.float32) * 0.05
        self._w_wdl = rng.standard_normal((flat, 3)).astype(np.float32) * 0.05
        self._bufs = [np.zeros((max_batch, _PLANES, 8, 8), dtype=np.float32) for _ in range(2)]
        self.forwarded: list[np.ndarray] = []

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self._bufs[slot][:bsz]

    def evaluate_inplace_async(self, bsz: int, *, slot: int = 0) -> tuple[np.ndarray, np.ndarray, None]:
        x = self._bufs[slot][:bsz].reshape(bsz, -1).copy()
        self.forwarded.append(x)
        pol = np.stack([row @ self._w_pol for row in x]) if bsz else np.zeros((0, 4672), np.float32)
        wdl = np.stack([row @ self._w_wdl for row in x]) if bsz else np.zeros((0, 3), np.float32)
        return pol.astype(np.float32), wdl.astype(np.float32), None


def _root(board: chess.Board) -> tuple[MCTSTree, int, CBoard]:
    tree = MCTSTree()
    tree.reserve(4096, 65536)
    cb = CBoard.from_board(board)
    rid = tree.add_root(0, 0.0)
    legal = cb.legal_move_indices().astype(np.int32)
    tree.expand(rid, legal, np.full(legal.size, 1.0 / legal.size, dtype=np.float64))
    return tree, rid, cb


def _snapshot(tree: MCTSTree, node: int, depth: int) -> list[Any]:
    """Visits and Q of every child down to ``depth`` plies, in action order."""
    actions, visits = tree.get_children_visits(node)
    out: list[Any] = []
    for a, v in zip(actions.tolist(), visits.tolist()):
        child = int(tree.find_child(node, int(a)))
        q = None if child < 0 else round(float(tree.node_q(child)), 12)
        sub = _snapshot(tree, child, depth - 1) if depth > 1 and child >= 0 else []
        out.append((int(a), int(v), q, sub))
    return out


def _search(board: chess.Board, *, sims: int, gather: int, compact: bool,
            monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], Any, _RowwiseEvaluator]:
    with monkeypatch.context() as m:
        if not compact:  # the previous behaviour: every row, terminal or repeated, forwarded
            m.setattr(pool_module, "_distinct_leaf_rows", lambda leaf_ids, is_term, n: (
                np.arange(n, dtype=np.int32), pool_module._NO_ROWS, pool_module._NO_ROWS))
        ev = _RowwiseEvaluator(max_batch=gather)
        pool = MultiGpuPucvPool(
            MultiGpuPucvConfig(n_gpus=1, gather=gather, vloss_weight=3, input_planes=_PLANES),
            evaluators=[ev],
        )
        try:
            tree, rid, cb = _root(board)
            pool.run(tree=tree, root_id=rid, root_cboard=cb, target_sims=sims,
                     stop_event=threading.Event())
            return _snapshot(tree, rid, 3), pool.last_stats(), ev
        finally:
            pool.close()


def _board(fen: str | None, moves: list[str]) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board


_POSITIONS = {
    "start": (None, []),
    "knight_shuffle": (None, ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]),
    "kr_vs_k_shuffle": ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", ["a1a2", "e8d8", "a2a1", "d8e8"]),
    "kq_vs_k_mating": ("7k/8/6K1/8/8/8/8/1Q6 w - - 0 1", []),
    "rook_endgame": ("8/5pk1/6p1/3R4/r4P2/6P1/5K2/8 w - - 0 40", ["d5d6", "a4a3", "d6d5", "a3a4"]),
}


@pytest.mark.parametrize("name", sorted(_POSITIONS))
def test_compacted_forward_builds_the_identical_tree(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board(*_POSITIONS[name])
    full, full_stats, full_ev = _search(board, sims=600, gather=32, compact=False, monkeypatch=monkeypatch)
    compact, stats, ev = _search(board, sims=600, gather=32, compact=True, monkeypatch=monkeypatch)
    assert compact == full
    assert stats.leaves == full_stats.leaves
    forwarded = sum(x.shape[0] for x in ev.forwarded)
    assert forwarded == stats.evaluated_rows
    assert forwarded + stats.duplicate_leaves <= sum(x.shape[0] for x in full_ev.forwarded)
    for x in ev.forwarded:  # no zeroed terminal row, no row twice in one forward
        assert np.any(x != 0, axis=1).all()
        assert np.unique(x, axis=0).shape[0] == x.shape[0]


def test_the_fixtures_exercise_both_kinds_of_wasted_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precondition for the identity test: terminal rows and duplicate
    selections both occur (else the comparison would be vacuous)."""
    _, shuffle_stats, _ = _search(_board(*_POSITIONS["kr_vs_k_shuffle"]), sims=600, gather=32,
                                  compact=True, monkeypatch=monkeypatch)
    _, start_stats, _ = _search(_board(*_POSITIONS["start"]), sims=600, gather=32,
                                compact=True, monkeypatch=monkeypatch)
    assert shuffle_stats.terminal_leaves > 0
    assert start_stats.duplicate_leaves > 0


def test_fuzz_random_games_with_repetitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Random play (biased toward shuffling back) from random openings; each
    position searched with and without compaction must give the same tree."""
    rng = random.Random(20261120)
    checked = 0
    for _ in range(12):
        board = chess.Board()
        for _ply in range(rng.randrange(4, 40)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=True):
                break
            back = [m for m in moves if board.move_stack and m.to_square == board.move_stack[-1].from_square]
            board.push(rng.choice(back) if back and rng.random() < 0.4 else rng.choice(moves))
        if board.is_game_over(claim_draw=True):
            continue
        gather = rng.choice([8, 16, 32, 64])
        full, _, _ = _search(board, sims=300, gather=gather, compact=False, monkeypatch=monkeypatch)
        compact, _, _ = _search(board, sims=300, gather=gather, compact=True, monkeypatch=monkeypatch)
        assert compact == full, board.fen()
        checked += 1
    assert checked >= 8


def test_distinct_leaf_rows_maps_each_duplicate_to_its_first_row() -> None:
    leaf_ids = np.array([7, 3, 7, 9, 3, 7, 5], dtype=np.int32)
    is_term = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    eval_rows, dup_rows, dup_first = pool_module._distinct_leaf_rows(leaf_ids, is_term, 7)
    assert eval_rows.tolist() == [0, 1, 3]
    assert dup_rows.tolist() == [2, 4, 5]
    assert dup_first.tolist() == [0, 1, 0]


def test_the_debug_info_line_reports_evals_terminal_and_duplicate_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool's per-search counters reach the `debug on` info string, so a
    timing pass can read unique evaluations apart from the visits `nps` counts."""
    from chess_anti_engine.uci.search import _pucv_stats_string

    _, stats, ev = _search(_board(*_POSITIONS["knight_shuffle"]), sims=600, gather=32,
                           compact=True, monkeypatch=monkeypatch)
    line = _pucv_stats_string(stats, pending_mode=0)
    assert line is not None
    forwarded = sum(x.shape[0] for x in ev.forwarded)
    assert f"evals={forwarded} " in line
    assert f"terminal={stats.terminal_leaves} " in line
    assert line.endswith(f"duplicate={stats.duplicate_leaves}")
    assert stats.terminal_leaves > 0


def test_scatter_reuses_the_buffer_and_leaves_nothing_stale() -> None:
    """The per-buffer output array is reused batch after batch (no fresh
    n x 4672 allocation); every row is rewritten each batch: evaluated rows
    from the forward, duplicates from their first row, terminal rows zeroed."""
    rng = np.random.default_rng(7)
    b: dict[str, np.ndarray] = {"leaf_ids": np.zeros(8, dtype=np.int32)}

    def batch(leaf_ids: list[int], is_term: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(leaf_ids)
        b["leaf_ids"][:n] = leaf_ids
        b["is_term"] = np.array(is_term, dtype=np.int8)
        eval_rows, b["dup_rows"], b["dup_first"] = pool_module._distinct_leaf_rows(b["leaf_ids"], b["is_term"], n)
        pol_m = rng.standard_normal((len(eval_rows), 4672)).astype(np.float32)
        wdl_m = rng.standard_normal((len(eval_rows), 3)).astype(np.float32)
        pol, wdl = pool_module._scatter_outputs(b, n, {}, eval_rows, (pol_m, wdl_m))
        return pol, wdl, pol_m, wdl_m

    batch([1, 2, 3, 4, 5, 6, 7, 8], [0] * 8)  # fills every row of the buffer
    first_buffer = b["out_pol"]
    pol, wdl, pol_m, wdl_m = batch([4, 9, 4, 5, 9, 6], [0, 0, 0, 1, 0, 1])
    assert b["out_pol"] is first_buffer
    assert np.shares_memory(pol, first_buffer)
    assert pol.shape == (6, 4672)
    assert wdl.shape == (6, 3)
    np.testing.assert_array_equal(pol[[0, 1]], pol_m)
    np.testing.assert_array_equal(wdl[[0, 1]], wdl_m)
    np.testing.assert_array_equal(pol[2], pol_m[0])
    np.testing.assert_array_equal(pol[4], pol_m[1])
    np.testing.assert_array_equal(wdl[4], wdl_m[1])
    assert not pol[[3, 5]].any()
    assert not wdl[[3, 5]].any()
