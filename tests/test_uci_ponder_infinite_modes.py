from __future__ import annotations

from dataclasses import replace
import threading
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdGo, parse_command
from chess_anti_engine.uci.search import SearchResult, SearchWorker
from chess_anti_engine.uci.time_manager import limits_from_go


class NeutralEvaluator:
    def __init__(self) -> None:
        self.started = threading.Event()

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        self.started.set()
        return np.zeros((len(x), POLICY_SIZE), np.float32), np.zeros((len(x), 3), np.float32)


class RecordingWorker(SearchWorker):
    def __init__(self) -> None:
        self.evaluator = NeutralEvaluator()
        super().__init__(self.evaluator, device='cpu', n_walkers=1, chunk_sims=3,
                         gumbel_cfg=GumbelConfig(topk=4, add_noise=False, temperature=0.0))
        self.boards: list[chess.Board] = []
        self.budgets: list[int | None] = []
        self.results: list[SearchResult] = []
        self.started = [threading.Event(), threading.Event()]
        self.returned = [threading.Event(), threading.Event()]

    def run(self, board: chess.Board, **kwargs: Any) -> SearchResult:
        phase = len(self.boards)
        self.boards.append(board.copy(stack=True))
        self.budgets.append(kwargs.get('max_nodes'))
        self.started[phase].set()
        result = super().run(board, **kwargs)
        self.results.append(result)
        self.returned[phase].set()
        return result


def dispatch_line(engine: Engine, line: str) -> None:
    command = parse_command(line)
    assert command is not None
    engine.dispatch(command)


def bestmoves(output: str) -> list[str]:
    return [line.split()[1] for line in output.splitlines() if line.startswith('bestmove ')]


@pytest.mark.parametrize('line', [
    'go ponder infinite searchmoves e7e5',
    'go infinite ponder searchmoves e7e5',
    'go ponder infinite nodes 3 depth 4 movetime 30 searchmoves e7e5',
])
def test_mixed_wire_flags_select_ponder_before_post_hit_infinite(line: str) -> None:
    command = parse_command(line)
    assert isinstance(command, CmdGo)
    initial = limits_from_go(command.args, side_to_move_is_white=True)
    assert initial.ponder
    assert initial.is_open_ended()
    assert initial.deadline_ms is None
    assert initial.searchmoves == ('e7e5',)
    real = limits_from_go(replace(command.args, ponder=False), side_to_move_is_white=False)
    assert not real.ponder
    assert real.infinite
    assert real.max_nodes is None
    assert real.max_depth is None
    assert real.deadline_ms is None
    assert real.searchmoves == initial.searchmoves


@pytest.mark.parametrize('position', [
    'position startpos moves e2e4',
    'position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 e2e4',
])
@pytest.mark.parametrize('go', [
    'go ponder infinite searchmoves e7e5',
    'go infinite ponder searchmoves e7e5',
])
def test_actual_mixed_go_replays_history_after_early_warm_result(
    position: str, go: str, capsys: pytest.CaptureFixture[str],
) -> None:
    worker = RecordingWorker()
    engine = Engine(worker)
    try:
        dispatch_line(engine, position)
        expected = engine._board.copy(stack=True)
        dispatch_line(engine, go)
        assert worker.returned[0].wait(3), 'invalid pre-reply filter should return the real warm worker'
        thread = engine._search_thread
        assert thread is not None
        thread.join(0.05)
        assert thread.is_alive()
        assert bestmoves(capsys.readouterr().out) == []
        dispatch_line(engine, 'ponderhit')
        assert worker.started[1].wait(3), 'late ponderhit must start the real phase'
        assert worker.boards[1].fen() == expected.fen()
        assert worker.boards[1].move_stack == expected.move_stack
        assert worker.boards[1].turn == chess.BLACK
        assert worker.budgets[1] is None
        thread.join(0.05)
        assert thread.is_alive(), 'real infinite phase must still wait for GUI stop'
        assert bestmoves(capsys.readouterr().out) == []
        dispatch_line(engine, 'stop')
        thread.join(3)
        assert not thread.is_alive()
        assert bestmoves(capsys.readouterr().out) == ['e7e5']
        assert chess.Move.from_uci('e7e5') in expected.legal_moves
    finally:
        engine.close()


def test_actual_fen_mixed_go_hands_off_after_declined_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    worker = RecordingWorker()
    engine = Engine(worker)
    try:
        dispatch_line(engine, 'position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1')
        expected = engine._board.copy(stack=True)
        dispatch_line(engine, 'go ponder infinite searchmoves e2e4')
        assert worker.returned[0].wait(3)
        assert worker.results[0].root_declined is not None
        assert bestmoves(capsys.readouterr().out) == []
        dispatch_line(engine, 'ponderhit')
        assert worker.started[1].wait(3)
        assert worker.returned[1].wait(3)
        assert worker.boards[1].fen() == expected.fen()
        assert worker.boards[1].move_stack == []
        thread = engine._search_thread
        assert thread is not None
        thread.join(0.05)
        assert thread.is_alive()
        assert bestmoves(capsys.readouterr().out) == []
        dispatch_line(engine, 'stop')
        thread.join(3)
        assert not thread.is_alive()
        assert bestmoves(capsys.readouterr().out) == ['e2e4']
    finally:
        engine.close()


@pytest.mark.parametrize('position', ['position startpos', 'position startpos moves e2e4'])
def test_actual_normal_finite_ponder_hit_releases_without_stop(
    position: str, capsys: pytest.CaptureFixture[str],
) -> None:
    worker = RecordingWorker()
    engine = Engine(worker)
    try:
        dispatch_line(engine, position)
        expected = engine._board.copy(stack=True)
        move = 'e2e4' if expected.turn == chess.WHITE else 'e7e5'
        dispatch_line(engine, f'go ponder nodes 3 searchmoves {move}')
        assert worker.started[0].wait(3)
        assert bestmoves(capsys.readouterr().out) == []
        dispatch_line(engine, 'ponderhit')
        assert worker.started[1].wait(3)
        thread = engine._search_thread
        assert thread is not None
        thread.join(4)
        assert not thread.is_alive()
        assert worker.boards[1].fen() == expected.fen()
        assert worker.boards[1].move_stack == expected.move_stack
        assert worker.budgets[1] == 3
        assert worker.results[1].nodes == 3
        assert bestmoves(capsys.readouterr().out) == [move]
    finally:
        engine.close()


@pytest.mark.parametrize('go', ['go nodes 3 searchmoves e7e5', 'go infinite searchmoves e7e5'])
def test_actual_normal_go_modes_keep_current_root(
    go: str, capsys: pytest.CaptureFixture[str],
) -> None:
    worker = RecordingWorker()
    engine = Engine(worker)
    try:
        dispatch_line(engine, 'position startpos moves e2e4')
        expected = engine._board.copy(stack=True)
        dispatch_line(engine, go)
        assert worker.evaluator.started.wait(3)
        thread = engine._search_thread
        assert thread is not None
        if 'infinite' in go:
            thread.join(0.05)
            assert thread.is_alive()
            assert bestmoves(capsys.readouterr().out) == []
            dispatch_line(engine, 'stop')
        thread.join(4)
        assert not thread.is_alive()
        assert len(worker.boards) == 1
        assert worker.boards[0].move_stack == expected.move_stack
        assert worker.boards[0].fen() == expected.fen()
        assert bestmoves(capsys.readouterr().out) == ['e7e5']
    finally:
        engine.close()


@pytest.mark.parametrize('go', [
    'go ponder infinite searchmoves e7e5',
    'go ponder nodes 3 searchmoves e7e5',
])
def test_actual_stop_before_hit_cancels_warm_owner_and_late_hit_is_ignored(
    go: str, capsys: pytest.CaptureFixture[str],
) -> None:
    worker = RecordingWorker()
    engine = Engine(worker)
    try:
        dispatch_line(engine, 'position startpos moves e2e4')
        dispatch_line(engine, go)
        assert worker.returned[0].wait(3)
        assert bestmoves(capsys.readouterr().out) == []
        dispatch_line(engine, 'stop')
        thread = engine._search_thread
        if thread is not None:
            thread.join(3)
            assert not thread.is_alive()
        assert len(worker.boards) == 1
        assert bestmoves(capsys.readouterr().out) == [worker.results[0].bestmove_uci]
        dispatch_line(engine, 'ponderhit')
        assert len(worker.boards) == 1
        assert bestmoves(capsys.readouterr().out) == []
    finally:
        engine.close()
