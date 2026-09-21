"""Idle-boundary root advancement; retain host history only after a valid ACK.

This is the private probe protocol, not UCI. A transport/ACK failure leaves the
remote state uncertain: close that peer rather than retrying blindly. Search
configuration epochs and accepted advances share one monotonically increasing ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chess

from . import run_probe as sessions
from ..legal_probe import run_probe as rules


class Wire(Protocol):
    def write(self, text: str) -> None: ...
    def line(self) -> str: ...
    def expect(self, expected: str) -> None: ...


def board_position(board: chess.Board) -> rules.Position:
    return rules.fen_position(board.fen(en_passant='fen'))


def packed_move(board: chess.Board, move: chess.Move) -> int:
    return sessions.move_key((move.from_square, move.to_square,
                              move.promotion - 1 if move.promotion else 0,
                              2 if board.is_castling(move) else int(board.is_en_passant(move))))


@dataclass(frozen=True)
class AdvanceReply:
    expected_epoch: int
    new_epoch: int
    key: int
    status: int  # 0=accepted, 1=stale/non-increasing epoch, 2=not an exact legal key
    current_epoch: int
    board: rules.Position


def parse_advance(line: str, board_line: str) -> AdvanceReply:
    expected, new, key, status, current = sessions.numbers(line, 'advance_result', 5)
    if new == 0 or status not in (0, 1, 2):
        raise ValueError('invalid advance result status/epoch')
    if status == 0 and (current != new or new <= expected):
        raise ValueError('invalid accepted advance epochs')
    if status == 1 and expected == current and new > current:
        raise ValueError('inconsistent stale advance result')
    if status == 2 and (current != expected or new <= current):
        raise ValueError('inconsistent illegal-move advance result')
    return AdvanceReply(expected, new, key, status, current,
                        sessions.position(sessions.numbers(board_line, 'board', 19)))


def advance_root(peer: Wire, root: chess.Board, *, expected_epoch: int,
                 new_epoch: int, key: int) -> tuple[chess.Board, AdvanceReply]:
    """Request a move, validate the echoed transaction and full board, then commit.

    Neither the caller's board nor its pre-root history is mutated. Return the
    same board on semantic rejection; a successful result is a stack-preserving
    copy with exactly one move appended. No model/encoder regime is changed.
    """
    if root.chess960 or not root.is_valid():
        raise ValueError('advance requires a valid orthodox root')
    if any(type(x) is not int or not 0 <= x <= sessions.SENTINEL
           for x in (expected_epoch, new_epoch, key)) or new_epoch == 0:
        raise ValueError('advance command requires U32 words and a nonzero new epoch')
    peer.write(f'advance {expected_epoch:x} {new_epoch:x} {key:x}\n')
    reply = parse_advance(peer.line(), peer.line())
    if (reply.expected_epoch, reply.new_epoch, reply.key) != (expected_epoch, new_epoch, key):
        raise ValueError('advance acknowledgement identity mismatch')
    candidate = root
    if reply.status == 0:
        move = next((m for m in root.legal_moves if packed_move(root, m) == key), None)
        if move is None:
            raise ValueError('native advance accepted an illegal move key')
        candidate = root.copy(stack=True)
        candidate.push(move)
    if reply.board != board_position(candidate):
        raise ValueError('advance acknowledgement board mismatch')
    peer.expect('ready')
    return candidate, reply
