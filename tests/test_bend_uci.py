"""Inexpensive UCI parsing/state contracts: no native build, search or inference."""
from __future__ import annotations

from threading import Event, Thread
import time

import chess
import pytest

from native.bend_engine.uci_probe.protocol import Active, Engine, Limits, Result, fallback, limits, position, unsigned


@pytest.mark.parametrize('text', ['-1', '+1', '1.0', '²', '１２', '', '4294967296', '0' * 11])
def test_invalid_integer(text: str) -> None:
    with pytest.raises(ValueError, match=r'integer|limit'):
        unsigned(text)


@pytest.mark.parametrize('text', ['position', 'startpos junk', 'fen 8/8/8/8/8/8/8/8 w - - 0 1',
                                  'fen 4k3/8/8/8/8/8/8/R3K3 w - - 256 1',
                                  'startpos moves e2e5', 'startpos moves 0000', 'fen incomplete'])
def test_invalid_position(text: str) -> None:
    with pytest.raises(ValueError, match=r"expected|invalid|unsupported|illegal|positive|missing|duplicate|infinite|clock"):
        position(text.split())


def test_position_preserves_history() -> None:
    b = position(['startpos', 'moves', 'g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'])
    assert len(b.move_stack) == 8
    assert b.is_repetition(3)
    assert b.fen().split()[0] == chess.Board().fen().split()[0]


@pytest.mark.parametrize('uci', ['a7a8q', 'a7a8r', 'a7a8b', 'a7a8n'])
def test_promotion_position(uci: str) -> None:
    b = position(('fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1 moves ' + uci).split())
    assert b.peek().uci() == uci


@pytest.mark.parametrize('text', ['nodes 0', 'depth 0', 'nodes -1', 'movetime', 'nodes 1 nodes 2',
                                  'ponder', 'searchmoves e2e4', 'infinite movetime 5', 'infinite infinite',
                                  'movestogo 0', 'winc 1', 'mate 1', 'btime 1000'])
def test_invalid_limits(text: str) -> None:
    with pytest.raises(ValueError, match=r"expected|invalid|unsupported|illegal|positive|missing|duplicate|infinite|clock"):
        limits(text.split(), chess.WHITE)


def test_bounds_are_explicit() -> None:
    assert limits(['nodes', '999', 'depth', '999'], True) == Limits(256, 32)
    assert limits(['movetime', '0'], True).seconds == 0
    assert limits(['infinite'], True).infinite
    assert limits(['wtime', '1000', 'btime', '2000', 'winc', '100', 'movestogo', '10'], True).seconds == 0.16
    assert limits(['wtime', '10', 'btime', '2000'], True).seconds == 0
    assert limits(['wtime', '1000', 'btime', '2000', 'binc', '100', 'movestogo', '10'], False).seconds == 0.26


class FakeBackend:
    def __init__(self):
        self.closed = self.interrupts = self.resets = 0
    def interrupt(self) -> None:
        self.interrupts += 1
    def close(self) -> None:
        self.closed += 1
    def new_game(self) -> None:
        self.resets += 1
    def search(self, board: chess.Board, bounds: Limits, cancelled: Event) -> Result:
        del bounds, cancelled
        return fallback(board, 'fake')


def active(engine: Engine, infinite: bool = False) -> Active:
    job = Active(engine.board.copy(stack=True), Limits(4, 2, infinite=infinite), Event(), time.monotonic(), Thread())
    engine.active = job
    return job


def test_handshake_options_and_ready_while_active() -> None:
    out: list[str] = []
    engine = Engine(FakeBackend(), out.append)
    engine.command('uci')
    assert 'uciok' in out
    assert len([s for s in out if s.startswith('option ')]) == 2
    job = active(engine)
    engine.command('isready')
    assert out[-1] == 'readyok'
    assert not job.cancel.is_set()


def test_invalid_position_does_not_change_board_or_history() -> None:
    engine = Engine(FakeBackend(), lambda _: None)
    engine.command('position startpos moves e2e4')
    original = engine.board
    engine.command('position startpos moves d2d4 e7e6 d1d8')
    assert engine.board is original
    assert engine.board.peek() == chess.Move.from_uci('e2e4')


def test_busy_commands_cannot_replace_active_root() -> None:
    out: list[str] = []
    backend = FakeBackend()
    engine = Engine(backend, out.append)
    job = active(engine)
    for line in ('position startpos moves e2e4', 'ucinewgame', 'setoption name BendDepth value 2', 'go nodes 4'):
        engine.command(line)
        assert 'search active' in out[-1]
    assert engine.active is job
    assert engine.depth == 4
    assert not engine.board.move_stack
    assert backend.resets == 0


@pytest.mark.parametrize('bad', ['0000', 'e2e5', 'claim', 'a7a8q'])
def test_invalid_backend_moves_are_never_published(bad: str) -> None:
    out: list[str] = []
    engine = Engine(FakeBackend(), out.append)
    active(engine)
    engine.complete(Result(bad))
    assert out[-1] == 'bestmove a2a3'
    assert any('invalid backend' in s for s in out)
    engine.complete(Result('e2e4'))
    assert sum(s.startswith('bestmove ') for s in out) == 1


def test_stop_completed_infinite_exactly_once() -> None:
    out: list[str] = []
    engine = Engine(FakeBackend(), out.append)
    job = active(engine, infinite=True)
    job.result = Result('e2e4', 4, 81)
    engine.command('stop')
    engine.command('stop')
    assert out.count('bestmove e2e4') == 1
    assert engine.active is None


def test_stop_then_bounded_hard_abort() -> None:
    backend = FakeBackend()
    engine = Engine(backend, lambda _: None)
    job = active(engine)
    engine.stop()
    assert job.cancel.is_set()
    job.cancelled_at = time.monotonic() - 1
    engine.tick()
    engine.tick()
    assert backend.interrupts == 1


def test_time_limit_triggers_cancel() -> None:
    engine = Engine(FakeBackend(), lambda _: None)
    job = active(engine)
    job.bounds = Limits(32, 4, seconds=0)
    engine.tick()
    assert job.cancel.is_set()


def test_setoption_and_reset() -> None:
    backend = FakeBackend()
    engine = Engine(backend, lambda _: None)
    engine.command('setoption name BendSimulations value 8')
    engine.command('setoption name BendDepth value 2')
    assert (engine.simulations, engine.depth) == (8, 2)
    engine.command('setoption name BendDepth value 33')
    assert engine.depth == 2
    engine.command('position startpos moves e2e4')
    engine.command('ucinewgame')
    assert not engine.board.move_stack
    assert backend.resets == 1


def test_checkmate_null_move() -> None:
    out: list[str] = []
    engine = Engine(FakeBackend(), out.append)
    engine.board = chess.Board('k7/1Q6/2K5/8/8/8/8/8 b - - 0 1')
    active(engine)
    engine.complete(Result('0000'))
    assert out[-1] == 'bestmove 0000'
