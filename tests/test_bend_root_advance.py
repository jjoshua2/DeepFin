"""Cheap ACK/host-history contracts; no native compilation, search or neural forward."""
from __future__ import annotations

import chess
import pytest

from native.bend_engine.session_probe.root_protocol import (
    advance_root, board_position, packed_move, parse_advance,
)


def board_line(board: chess.Board) -> str:
    pos = board_position(board)
    words = [v for x in pos[:8] for v in (x >> 32, x & 0xffffffff)] + list(pos[8:])
    return 'board ' + ' '.join(map(str, words))


class Peer:
    def __init__(self, result: str, board: chess.Board):
        self.lines = iter([result, board_line(board), 'ready'])
        self.sent: list[str] = []

    def write(self, text: str) -> None:
        self.sent.append(text)

    def line(self) -> str:
        return next(self.lines)

    def expect(self, expected: str) -> None:
        assert self.line() == expected


@pytest.mark.parametrize('line', [
    '', 'advanced 0 1 1 0 1', 'advance_result 0 1 1 0',
    'advance_result 0 1 1 0 1 2', 'advance_result 0 1 -1 0 1',
    'advance_result 0 1 4294967296 0 1', 'advance_result 0 1 x 0 1',
    'advance_result 0 0 1 0 0', 'advance_result 0 1 1 3 1',
    'advance_result 0 1 1 0 2', 'advance_result 2 2 1 0 2',
    'advance_result 0 1 1 1 0', 'advance_result 0 1 1 2 1',
    'advance_result 2 1 1 2 2',
])
def test_bad_advance_ack(line: str) -> None:
    with pytest.raises(ValueError, match=r'malformed|nondecimal|exceeds|advance'):
        parse_advance(line, board_line(chess.Board()))


@pytest.mark.parametrize('text', ['board', 'board -1', 'board x'])
def test_bad_advance_board(text: str) -> None:
    with pytest.raises(ValueError, match=r'malformed|nondecimal|position'):
        parse_advance('advance_result 0 1 1804 0 1', text)


def test_history_committed_only_after_complete_ack() -> None:
    root = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
        root.push_uci(uci)
    move = chess.Move.from_uci('e2e4')
    key = packed_move(root, move)
    child = root.copy(stack=True)
    child.push(move)
    peer = Peer(f'advance_result 7 8 {key} 0 8', child)
    result, reply = advance_root(peer, root, expected_epoch=7, new_epoch=8, key=key)
    assert result.fen() == child.fen()
    assert result.move_stack == child.move_stack
    assert len(root.move_stack) == 4
    assert peer.sent == [f'advance 7 8 {key:x}\n']
    assert reply.current_epoch == 8


@pytest.mark.parametrize('status', [1, 2])
def test_rejected_advance_does_not_commit(status: int) -> None:
    root = chess.Board()
    expected = 6 if status == 1 else 7
    peer = Peer(f'advance_result {expected} 8 0 {status} 7', root)
    result, reply = advance_root(peer, root, expected_epoch=expected, new_epoch=8, key=0)
    assert result is root
    assert reply.current_epoch == 7
    assert root.move_stack == []


@pytest.mark.parametrize('fault', ['expected', 'new', 'key', 'board', 'illegal'])
def test_bad_ack_never_commits_host_history(fault: str) -> None:
    root = chess.Board()
    child = root.copy(stack=True)
    child.push_uci('e2e4')
    key = packed_move(root, chess.Move.from_uci('e2e4'))
    expected, new, sent_key = 7, 8, key
    if fault == 'expected':
        expected = 6
    elif fault == 'new':
        new = 9
    elif fault == 'key':
        key += 1
    elif fault == 'board':
        child = root
    else:
        key = sent_key = 0
    peer = Peer(f'advance_result {expected} {new} {key} 0 {new}', child)
    with pytest.raises(ValueError, match=r'identity|board mismatch|illegal move'):
        advance_root(peer, root, expected_epoch=7, new_epoch=8, key=sent_key)
    assert root.move_stack == []


@pytest.mark.parametrize(('expected', 'new', 'key'), [
    (-1, 1, 0), (0, 0, 0), (0, 1, -1), (0, 1, 1 << 32),
    (True, 1, 0), (0, True, 0), (0, 1, False), (0, 1 << 32, 0),
])
def test_invalid_command_never_sent(expected: int, new: int, key: int) -> None:
    root = chess.Board()
    peer = Peer('', root)
    with pytest.raises(ValueError, match='U32'):
        advance_root(peer, root, expected_epoch=expected, new_epoch=new, key=key)
    assert peer.sent == []


def test_chess960_is_not_silently_accepted() -> None:
    root = chess.Board(chess960=True)
    peer = Peer('', root)
    with pytest.raises(ValueError, match='orthodox'):
        advance_root(peer, root, expected_epoch=0, new_epoch=1, key=1804)
    assert peer.sent == []
