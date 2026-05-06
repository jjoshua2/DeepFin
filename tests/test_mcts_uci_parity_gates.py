from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many
from chess_anti_engine.mcts.puct import MCTSConfig, run_mcts_many
from chess_anti_engine.mcts.puct_c import run_mcts_many_c
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask, move_to_index
from chess_anti_engine.uci import search as uci_search
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

try:
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
except ImportError:  # pragma: no cover - extension absent
    run_gumbel_root_many_c = None


class _ZeroEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        batch = int(x.shape[0])
        return (
            np.zeros((batch, POLICY_SIZE), dtype=np.float32),
            np.zeros((batch, 3), dtype=np.float32),
        )


@dataclass
class _FakeTbProbe:
    _path: str = "fake-tb"
    hits: int = 0
    probes: int = 0
    max_pieces: int = 32

    def reset_counts(self) -> None:
        self.hits = 0
        self.probes = 0

    def apply(
        self,
        _leaf_cboards: list[Any],
        _wdl: np.ndarray,
        _indices: np.ndarray | None = None,
        solved_out: np.ndarray | None = None,
    ) -> int:
        if solved_out is not None:
            solved_out[:] = 0
        return 0


def _edge_case_boards() -> list[chess.Board]:
    single_legal = chess.Board("7k/6Q1/4K3/8/8/8/8/8 b - - 0 1")
    assert len(list(single_legal.legal_moves)) == 1

    checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert checkmate.is_checkmate()

    bare_kings = chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 0 1")
    assert bare_kings.is_game_over()
    assert bare_kings.result(claim_draw=True) == "1/2-1/2"

    return [
        chess.Board(),
        chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
        single_legal,
        checkmate,
        bare_kings,
    ]


def _root_logits(n: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((n, POLICY_SIZE), dtype=np.float32),
        np.zeros((n, 3), dtype=np.float32),
    )


def test_puct_c_and_python_root_masks_match_edge_cases() -> None:
    boards = _edge_case_boards()
    pre_pol, pre_wdl = _root_logits(len(boards))
    cfg = MCTSConfig(simulations=0, dirichlet_eps=0.0, temperature=0.0)

    py_probs, py_actions, py_values, py_masks = run_mcts_many(
        None,
        boards,
        device="cpu",
        rng=np.random.default_rng(7),
        cfg=cfg,
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )
    c_probs, c_actions, c_values, c_masks = run_mcts_many_c(
        None,
        boards,
        device="cpu",
        rng=np.random.default_rng(7),
        cfg=cfg,
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )

    assert py_actions == c_actions
    np.testing.assert_allclose(py_values, c_values, atol=0.0)
    for board, py_prob, c_prob, py_mask, c_mask in zip(
        boards, py_probs, c_probs, py_masks, c_masks, strict=True,
    ):
        expected_mask = (
            np.zeros((POLICY_SIZE,), dtype=np.bool_)
            if board.is_game_over()
            else legal_move_mask(board)
        )
        assert np.array_equal(py_mask, expected_mask)
        assert np.array_equal(c_mask, expected_mask)
        assert np.array_equal(py_mask, c_mask)
        np.testing.assert_allclose(py_prob, c_prob, atol=0.0)
        assert np.count_nonzero(py_prob[~py_mask]) == 0
        assert np.count_nonzero(c_prob[~c_mask]) == 0


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="gumbel_c extension not available")
def test_gumbel_c_and_python_root_masks_match_edge_cases() -> None:
    boards = _edge_case_boards()
    pre_pol, pre_wdl = _root_logits(len(boards))
    cfg = GumbelConfig(simulations=1, topk=8, temperature=0.0, add_noise=False)

    py_probs, py_actions, py_values, py_masks = run_gumbel_root_many(
        None,
        boards,
        device="cpu",
        rng=np.random.default_rng(11),
        cfg=cfg,
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )
    run_c = run_gumbel_root_many_c
    assert run_c is not None
    c_probs, c_actions, c_values, c_masks = run_c(
        None,
        boards,
        device="cpu",
        rng=np.random.default_rng(11),
        cfg=cfg,
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )[:4]

    assert py_actions == c_actions
    np.testing.assert_allclose(py_values, c_values, atol=0.0)
    for board, py_prob, c_prob, py_mask, c_mask in zip(
        boards, py_probs, c_probs, py_masks, c_masks, strict=True,
    ):
        expected_mask = (
            np.zeros((POLICY_SIZE,), dtype=np.bool_)
            if board.is_game_over()
            else legal_move_mask(board)
        )
        assert np.array_equal(py_mask, expected_mask)
        assert np.array_equal(c_mask, expected_mask)
        assert np.array_equal(py_mask, c_mask)
        assert np.count_nonzero(py_prob[~py_mask]) == 0
        assert np.count_nonzero(c_prob[~c_mask]) == 0


def test_uci_tb_solved_root_shortcuts_without_evaluating(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("8/8/8/8/8/8/4k3/R3K3 w - - 0 1")
    best = next(iter(board.legal_moves))
    calls: list[str] = []

    def fake_try_tb_root_move(probe_board: chess.Board, _path: str) -> tuple[chess.Move, int] | None:
        calls.append(probe_board.fen())
        if probe_board.is_game_over():
            return None
        return next(iter(probe_board.legal_moves)), 2

    monkeypatch.setattr(uci_search, "try_tb_root_move", fake_try_tb_root_move)
    evaluator = _ZeroEvaluator()
    worker = SearchWorker(evaluator, device="cpu", chunk_sims=4)
    worker.set_tb_probe(_FakeTbProbe())

    result = worker.run(
        board,
        stop_event=threading.Event(),
        deadline=Deadline(None),
        max_nodes=4,
    )

    assert evaluator.calls == 0
    assert calls
    assert result.bestmove_uci == best.uci()
    assert result.nodes == 1
    assert result.tbhits == 1
    assert result.score_cp == uci_search._TB_WIN_CP
    assert result.pv == (best.uci(),)


def test_uci_searchmoves_bypass_tb_shortcut_and_filter_bestmove(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = chess.Board()
    allowed_move = chess.Move.from_uci("e2e4")
    assert allowed_move in board.legal_moves
    allowed_idx = int(move_to_index(allowed_move, board))
    tb_calls = 0

    def fake_try_tb_root_move(_board: chess.Board, _path: str) -> tuple[chess.Move, int] | None:
        nonlocal tb_calls
        tb_calls += 1
        return next(iter(board.legal_moves)), 2

    monkeypatch.setattr(uci_search, "try_tb_root_move", fake_try_tb_root_move)
    evaluator = _ZeroEvaluator()
    worker = SearchWorker(
        evaluator,
        device="cpu",
        chunk_sims=1,
        gumbel_cfg=GumbelConfig(simulations=1, topk=8, temperature=0.0, add_noise=False),
    )
    worker.set_tb_probe(_FakeTbProbe())

    result = worker.run(
        board,
        stop_event=threading.Event(),
        deadline=Deadline(None),
        max_nodes=1,
        root_moves=(allowed_move.uci(),),
    )

    assert tb_calls == 0
    assert evaluator.calls > 0
    assert result.bestmove_uci == allowed_move.uci()
    assert result.pv[0] == allowed_move.uci()
    assert uci_search._allowed_root_indices(board, (allowed_move.uci(),)) == {allowed_idx}
