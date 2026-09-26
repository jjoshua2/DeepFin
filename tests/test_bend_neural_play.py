"""Cheap host rules, retirement and game-result tests; no export/search/forward."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import Mock

import chess
import chess.pgn
import numpy as np
import pytest

from native.bend_engine.neural_probe.game import (
    GameActor, GameEnd, GameSpec, adjudicate, game_end, pgn_text, retire_and_advance,
)
from native.bend_engine.neural_probe.batching import Batcher, Key
from native.bend_engine.session_probe.root_protocol import board_position, packed_move
from native.bend_engine.session_probe.run_probe import SENTINEL, Oracle, Peer, Reference

CYCLE = ('g1f3', 'g8f6', 'f3g1', 'f6g8')


def history(plies: int) -> chess.Board:
    b = chess.Board()
    for uci in (CYCLE * 4)[:plies]:
        b.push_uci(uci)
    return b


@pytest.mark.parametrize(('fen', 'reason', 'result'), [
    ('7k/6Q1/6K1/8/8/8/8/8 b - - 150 1', 'checkmate', '1-0'),
    ('7k/5Q2/6K1/8/8/8/8/8 b - - 150 1', 'stalemate', '1/2-1/2'),
    ('4k3/8/8/8/8/8/8/R3K3 w - - 150 1', 'seventyfive_moves', '1/2-1/2'),
    ('4k3/8/8/8/8/8/8/4K3 w - - 0 1', 'insufficient_material', '1/2-1/2'),
])
def test_automatic_result_precedes_ply_cap(fen: str, reason: str, result: str) -> None:
    end = game_end(chess.Board(fen), 'automatic', 4, 4)
    assert end == GameEnd(result, reason)


def test_ply_limit_is_not_a_draw() -> None:
    assert game_end(chess.Board(), 'automatic', 8, 8) == GameEnd('*', 'ply_limit')
    assert game_end(chess.Board(), 'automatic', 7, 8) is None


@pytest.mark.parametrize('plies', [7, 8])
def test_claimable_repetition_is_optional_and_has_intended_move_witness(plies: int) -> None:
    b = history(plies)
    before = b.fen(), list(b.move_stack)
    assert adjudicate(b, 'automatic') is None
    claim = adjudicate(b, 'claim_available')
    assert claim is not None
    assert claim.result == '1/2-1/2'
    assert claim.reason == 'threefold_repetition'
    assert claim.claimed
    assert (b.fen(), b.move_stack) == before
    if plies == 7:
        assert claim.intended_move is not None
        b.push_uci(claim.intended_move)
        assert b.is_repetition(3)
    else:
        assert claim.intended_move is None


def test_fivefold_is_automatic() -> None:
    assert adjudicate(history(16), 'automatic') == GameEnd('1/2-1/2', 'fivefold_repetition')


@pytest.mark.parametrize('clock', [99, 100])
def test_fifty_move_claim_is_optional(clock: int) -> None:
    b = chess.Board(f'4k3/8/8/8/8/8/8/R3K3 w - - {clock} 1')
    before = b.fen(), list(b.move_stack)
    assert adjudicate(b, 'automatic') is None
    claim = adjudicate(b, 'claim_available')
    assert claim is not None
    assert claim.reason == 'fifty_moves'
    assert claim.claimed
    assert (b.fen(), b.move_stack) == before
    if clock == 99:
        assert claim.intended_move is not None
        b.push_uci(claim.intended_move)
        assert b.is_fifty_moves()
    else:
        assert claim.intended_move is None


@pytest.mark.parametrize('limit', [0, -1, 129, True])
def test_invalid_play_budget(limit: int) -> None:
    with pytest.raises(ValueError, match='ply limit'):
        GameSpec('bad', chess.Board(), max_plies=limit)


@pytest.mark.parametrize(('script', 'policy', 'pattern'), [
    (('e2e5',), 'automatic', 'illegal'),
    (CYCLE * 4 + ('e2e4',), 'automatic', 'past a game result'),
    (CYCLE * 2 + ('e2e4',), 'claim_available', 'past a game result'),
])
def test_scripts_cannot_hide_illegal_or_post_game_moves(script, policy, pattern) -> None:
    with pytest.raises(ValueError, match=pattern):
        GameSpec('bad', chess.Board(), max_plies=32, claims=policy, scripted=script)


@pytest.mark.parametrize('board', [chess.Board(chess960=True), chess.Board.empty()])
def test_nonstandard_or_invalid_roots_rejected(board: chess.Board) -> None:
    with pytest.raises(ValueError, match='orthodox'):
        adjudicate(board, 'automatic')


def test_draw_policy_is_explicit() -> None:
    with pytest.raises(ValueError, match='draw-claim'):
        adjudicate(chess.Board(), 'guess')


def test_pgn_preserves_previous_history_and_unfinished_result() -> None:
    root = history(4)
    root.push_uci('e2e4')
    text = pgn_text(root, GameEnd('*', 'ply_limit'), 'test')
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None
    assert not game.errors
    assert game.headers['Result'] == '*'
    assert game.end().board().move_stack == root.move_stack
    assert game.end().board().fen(en_passant='fen') == root.fen(en_passant='fen')


class Wire:
    def __init__(self, result: str, board: chess.Board):
        pos = board_position(board)
        words = [v for x in pos[:8] for v in (x >> 32, x & 0xffffffff)] + list(pos[8:])
        self.lines = iter([result, 'board ' + ' '.join(map(str, words)), 'ready'])
        self.sent: list[str] = []

    def write(self, text: str) -> None:
        self.sent.append(text)

    def line(self) -> str:
        return next(self.lines)

    def expect(self, expected: str) -> None:
        assert self.line() == expected


def broker_request():
    broker = Batcher(4, 175, max_wait=0)
    broker.register(0, 1)
    old = Key(0, 1, 3, 0)
    broker.submit(old, np.zeros((1, 175, 8, 8)), np.array([0]), now=0, deadline=100)
    return broker, old


def test_live_pending_work_prevents_any_advance_wire() -> None:
    broker, _ = broker_request()
    peer = Wire('', chess.Board())
    with pytest.raises(ValueError, match='pending evaluator'):
        retire_and_advance(peer, broker, 0, chess.Board(), 1, 1804)
    assert peer.sent == []


@pytest.mark.parametrize('fault', ['identity', 'board', 'rejected', 'truncated'])
def test_bad_advance_does_not_commit_root_or_broker(fault: str) -> None:
    root = chess.Board()
    child = root.copy(stack=True)
    child.push_uci('e2e4')
    text = 'advance_result 1 2 1804 0 2'
    if fault == 'identity':
        text = 'advance_result 0 2 1804 0 2'
    elif fault == 'board':
        child = root
    elif fault == 'rejected':
        text, child = 'advance_result 1 2 1804 2 1', root
    elif fault == 'truncated':
        text = 'advance_result'
    broker = Batcher(4, 175)
    broker.register(0, 1)
    with pytest.raises(ValueError, match=r'identity|board|rejected|malformed'):
        retire_and_advance(Wire(text, child), broker, 0, root, 1, 1804)
    assert broker.epochs[0] == (1, 0)
    assert not root.move_stack


def test_cancelled_inflight_old_root_cannot_touch_new_root_request() -> None:
    root = history(4)
    move = chess.Move.from_uci('e2e4')
    child = root.copy(stack=True)
    child.push(move)
    key = packed_move(root, move)
    broker, old = broker_request()
    batch = broker.dispatch(0)
    assert batch is not None
    broker.cancel(old)
    assert broker.reserved == 1
    returned, epoch = retire_and_advance(Wire(f'advance_result 1 2 {key} 0 2', child), broker, 0, root, 1, key)
    assert epoch == 2
    assert returned.move_stack == child.move_stack
    assert len(root.move_stack) == 4
    broker.register(0, 3)
    new = Key(0, 3, 3, 0)  # deliberately reuse request/node IDs under the new root
    broker.submit(new, np.ones((1, 175, 8, 8)), np.array([0]), now=1, deadline=100)
    assert broker.complete(batch, np.zeros((4, 1858)), np.zeros((4, 3)), now=2) == []
    assert broker.pending[0].key == new
    assert broker.reserved == 1


@pytest.mark.parametrize('epoch', [0, SENTINEL - 1, SENTINEL])
def test_stale_or_exhausted_epoch_is_not_sent(epoch: int) -> None:
    broker = Batcher(4, 175)
    broker.register(0, epoch if epoch else 1)
    peer = Wire('', chess.Board())
    with pytest.raises(ValueError, match='epoch'):
        retire_and_advance(peer, broker, 0, chess.Board(), epoch, 1804)
    assert peer.sent == []


@pytest.mark.parametrize('stop', [1, 2, 3])
def test_incomplete_unscripted_search_never_plays_partial_best(stop: int) -> None:
    actor = GameActor.__new__(GameActor)
    actor.waiting, actor.session, actor.moves = None, 0, []
    actor.ref = Reference(board_position(chess.Board()), Oracle(Path("unused")), cap=32, depth=2, budget=4)
    actor.ref.stop = stop
    actor.spec = GameSpec('test', chess.Board())
    actor.after_search(Batcher(4, 175))
    assert actor.done
    assert actor.end is not None
    assert actor.end.result == '*'
    assert actor.moves == []


@pytest.mark.parametrize('fen', [
    '7k/6Q1/6K1/8/8/8/8/8 b - - 150 1',
    '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
    '4k3/8/8/8/8/8/8/4K3 w - - 0 1',
    '4k3/8/8/8/8/8/8/R3K3 w - - 150 1',
])
def test_terminal_root_sends_no_search_command(fen: str) -> None:
    from native.bend_engine.neural_probe.adapter import Encoding
    from chess_anti_engine.encoding import rep_fix
    peer = Mock(spec=Peer)
    peer.proc = Mock(pid=123)
    actor = GameActor(peer, GameSpec('terminal', chess.Board(fen)),
                      Encoding('lc0_root', 'v1', rep_fix.current() or False),
                      Oracle(Path('must-not-execute')), 0, 4)
    assert actor.done
    assert actor.results == []
    peer.write.assert_not_called()
