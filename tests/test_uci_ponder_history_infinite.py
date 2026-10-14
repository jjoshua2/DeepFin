from __future__ import annotations

import threading
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdGo, CmdPonderHit, CmdPosition, CmdStop, GoArgs
from chess_anti_engine.uci.search import SearchResult, SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


MOVES = ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 2


class RecordingEvaluator:
    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.started = threading.Event()

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        self.rows.extend(np.asarray(row).copy() for row in x)
        self.started.set()
        return np.zeros((len(x), POLICY_SIZE), np.float32), np.zeros((len(x), 3), np.float32)


class RecordingWorker(SearchWorker):
    def __init__(self, evaluator: RecordingEvaluator) -> None:
        super().__init__(evaluator, device='cpu', n_walkers=1, chunk_sims=4,
                         gumbel_cfg=GumbelConfig(topk=20, add_noise=False, temperature=0.0,
                                                input_history_encoding='lc0_root_legacy_meta'))
        self.boards: list[chess.Board] = []
        self.results: list[SearchResult] = []
        self.returned = threading.Event()

    def run(self, board: chess.Board, **kwargs: Any) -> SearchResult:
        self.boards.append(board.copy(stack=True))
        result = super().run(board, **kwargs)
        self.results.append(result)
        self.returned.set()
        return result


def test_real_ponderhit_preserves_threefold_and_canonical_root_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    evaluator = RecordingEvaluator()
    worker = RecordingWorker(evaluator)
    engine = Engine(worker)
    try:
        engine.dispatch(CmdPosition(None, MOVES))
        expected = engine._board.copy(stack=True)
        engine.dispatch(CmdGo(GoArgs(ponder=True, nodes=4)))
        assert evaluator.started.wait(3)
        engine.dispatch(CmdPonderHit())
        thread = engine._search_thread
        assert thread is not None
        thread.join(4)
        assert not thread.is_alive(), 'finite ponderhit must release without a stop command'
        assert len(worker.boards) == 2
        actual = worker.boards[1]
        assert actual.move_stack == expected.move_stack
        assert actual.is_repetition(3)
        expected_x = encode_cboard(CBoard.from_board(expected),
                                  input_history_encoding='lc0_root_legacy_meta',
                                  input_extra_features=worker._cfg.input_extra_features)
        actual_x = encode_cboard(CBoard.from_board(actual),
                                input_history_encoding='lc0_root_legacy_meta',
                                  input_extra_features=worker._cfg.input_extra_features)
        np.testing.assert_array_equal(actual_x, expected_x)
        assert any(np.array_equal(row, expected_x) for row in evaluator.rows)
        control = RecordingWorker(RecordingEvaluator())
        try:
            direct = control.run(expected, stop_event=threading.Event(), deadline=Deadline(None),
                                 max_nodes=4)
        finally:
            control.close()
        assert worker.results[1].nodes == direct.nodes == 0
        assert worker.results[1].root_declined == direct.root_declined
        assert worker.results[1].bestmove_uci == direct.bestmove_uci
        assert len([line for line in capsys.readouterr().out.splitlines()
                    if line.startswith('bestmove ')]) == 1
    finally:
        engine.close()


@pytest.mark.parametrize('position', [
    CmdPosition(None, MOVES),
    CmdPosition('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1', ()),
])
def test_actual_infinite_early_result_waits_for_gui_stop(
    position: CmdPosition, capsys: pytest.CaptureFixture[str],
) -> None:
    worker = RecordingWorker(RecordingEvaluator())
    engine = Engine(worker)
    try:
        engine.dispatch(position)
        engine.dispatch(CmdGo(GoArgs(infinite=True)))
        assert worker.returned.wait(3)
        thread = engine._search_thread
        assert thread is not None
        thread.join(0.1)
        before = capsys.readouterr().out
        assert thread.is_alive()
        assert not any(line.startswith('bestmove ') for line in before.splitlines())
        engine._stop_event.set()  # A worker failure latch is not a GUI stop.
        thread.join(0.1)
        assert thread.is_alive()
        engine.dispatch(CmdStop())
        thread.join(3)
        assert not thread.is_alive()
        assert len([line for line in capsys.readouterr().out.splitlines()
                    if line.startswith('bestmove ')]) == 1
    finally:
        engine.close()


def test_actual_ponderhit_without_popped_move_finishes() -> None:
    evaluator = RecordingEvaluator()
    worker = RecordingWorker(evaluator)
    engine = Engine(worker)
    try:
        engine.dispatch(CmdPosition(None, ()))
        engine.dispatch(CmdGo(GoArgs(ponder=True, nodes=4)))
        assert evaluator.started.wait(3)
        assert engine._popped_ponder_move is None
        engine.dispatch(CmdPonderHit())
        thread = engine._search_thread
        assert thread is not None
        thread.join(4)
        assert not thread.is_alive()
        assert len(worker.boards) == 2
        assert worker.boards[0].fen() == worker.boards[1].fen() == chess.STARTING_FEN
        assert worker.boards[1].move_stack == []
        assert worker.results[1].nodes == 4
    finally:
        engine.close()


@pytest.mark.parametrize('reason', ['cancelled', 'deadline'])
def test_ponder_transition_controls_preserve_history_without_dispatch(
    reason: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from chess_anti_engine.uci.time_manager import SearchLimits

    expected = chess.Board()
    for uci in MOVES:
        expected.push_uci(uci)
    ponder_board = expected.copy(stack=True)
    popped = ponder_board.pop()
    worker = RecordingWorker(RecordingEvaluator())
    engine = Engine(worker)
    fallback_boards: list[chess.Board] = []
    original = engine._bestmove_fallback

    def fallback(board: chess.Board, moves: tuple[str, ...] = ()) -> tuple[str, str]:
        fallback_boards.append(board.copy(stack=True))
        return original(board, moves)

    monkeypatch.setattr(engine, '_bestmove_fallback', fallback)
    try:
        engine._stop_event.set()
        engine._ponder_release.set()
        engine._ponderhit_event.set()
        engine._popped_ponder_move = popped
        engine._pending_real_limits = SearchLimits(max_nodes=4, deadline_ms=20)
        engine._ponderhit_at = time.monotonic() - (1.0 if reason == 'deadline' else 0.0)
        if reason == 'cancelled':
            engine._cancel_event.set()
        engine._run_search(SearchLimits(ponder=True, max_nodes=4), engine._search_gen, ponder_board)
        assert len(worker.boards) == 1
        assert len(fallback_boards) == 1
        assert fallback_boards[0].move_stack == expected.move_stack
        assert fallback_boards[0].is_repetition(3)
        assert engine._popped_ponder_move is None
        assert engine._pending_real_limits is None
        assert not engine._ponderhit_event.is_set()
    finally:
        engine.close()


def test_infinite_actual_worker_error_keeps_existing_fallback_until_gui_stop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingEvaluator(RecordingEvaluator):
        def evaluate_encoded(
            self, x: np.ndarray, relations: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            del x, relations
            engine._stop_event.set()
            self.started.set()
            raise RuntimeError('primary evaluator failure')

    evaluator = FailingEvaluator()
    worker = RecordingWorker(evaluator)
    engine = Engine(worker)
    try:
        engine.dispatch(CmdGo(GoArgs(infinite=True)))
        assert evaluator.started.wait(3)
        thread = engine._search_thread
        assert thread is not None
        thread.join(0.15)
        before = capsys.readouterr().out
        assert thread.is_alive()
        assert engine._stop_event.is_set()
        assert not engine._cancel_event.is_set()
        assert 'primary evaluator failure' in before
        assert not any(line.startswith('bestmove ') for line in before.splitlines())
        engine.dispatch(CmdStop())
        thread.join(3)
        assert not thread.is_alive()
        assert len([line for line in capsys.readouterr().out.splitlines()
                    if line.startswith('bestmove ')]) == 1
        assert engine.bestmove_fallback_used == 1
    finally:
        engine.close()
