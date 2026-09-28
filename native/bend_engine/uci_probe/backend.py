"""UCI bridge to the existing native Bend session, without a shadow search oracle.

Single caller owns search and pipe reads. interrupt() only kills owned processes;
closing/reaping their streams happens in the owner thread after the read unwinds.
"""
from __future__ import annotations

from pathlib import Path
from threading import Event, Lock
from typing import Protocol

import chess

from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.root_protocol import advance_root, board_position, packed_move
from native.bend_engine.session_probe.search_draws import automatic_draw, draw_reply, reconstruct_leaf
from .protocol import Limits, Result, fallback


class Evaluator(Protocol):
    def evaluate(self, root: chess.Board, path: list[int], board: sessions.rules.Position,
                 actions: list[int]) -> tuple[list[float], list[float]]: ...
    def interrupt(self) -> None: ...
    def close(self) -> None: ...


class DiagnosticEvaluator:
    def evaluate(self, root: chess.Board, path: list[int], board: sessions.rules.Position,
                 actions: list[int]) -> tuple[list[float], list[float]]:
        del root, path
        return sessions.evaluation(board, actions)

    def interrupt(self) -> None:
        pass

    def close(self) -> None:
        pass


class NativeSearch:
    def __init__(self, binary: Path, evaluator: Evaluator):
        self.binary, self.evaluator = binary.resolve(), evaluator
        self.peer: sessions.Peer | None = None
        self.root: chess.Board | None = None
        self.epoch = 0
        self.lock = Lock()
        self.aborted = False

    def reset_peer(self) -> None:
        with self.lock:
            peer, self.peer = self.peer, None
        if peer is not None:
            peer.close()
        self.root, self.epoch = None, 0

    def interrupt(self) -> None:
        with self.lock:
            self.aborted = True
            if self.peer is not None and self.peer.proc.poll() is None:
                self.peer.proc.kill()
        self.evaluator.interrupt()

    def new_game(self) -> None:
        self.reset_peer()

    def close(self) -> None:
        self.reset_peer()
        self.evaluator.close()

    def bind(self, root: chess.Board) -> sessions.Peer:
        old = self.root
        if self.peer is not None and old is not None:
            prefix = root.move_stack[:len(old.move_stack)]
            if (old.root().fen(en_passant='fen') == root.root().fen(en_passant='fen')
                    and prefix == old.move_stack and self.epoch + len(root.move_stack) - len(old.move_stack) < 900):
                for move in root.move_stack[len(old.move_stack):]:
                    old, reply = advance_root(self.peer, old, expected_epoch=self.epoch,
                                              new_epoch=self.epoch + 1, key=packed_move(old, move))
                    if reply.status:
                        raise ValueError('native root advance rejected')
                    self.epoch = reply.current_epoch
                if old.fen(en_passant='fen') == root.fen(en_passant='fen'):
                    self.root = old
                    return self.peer
            self.reset_peer()
        peer = sessions.Peer(self.binary, board_position(root))
        with self.lock:
            self.peer = peer
            if self.aborted:
                peer.proc.kill()
        self.root = root.copy(stack=True)
        return peer

    def search(self, board: chess.Board, bounds: Limits, cancelled: Event) -> Result:
        with self.lock:
            self.aborted = False
        if cancelled.is_set():
            return fallback(board, 'stopped before search')
        try:
            peer = self.bind(board)
            self.epoch += 1
            peer.write(f'config {self.epoch:x} {bounds.simulations:x} 1000 {bounds.depth:x}\n')
            last_request = 0
            while True:
                line = peer.line()
                if line.startswith('result '):
                    epoch, done, count, stop, pending, sequence = sessions.numbers(line, 'result', 6)
                    if (epoch != self.epoch or not 1 <= count <= 4096 or done > bounds.simulations
                            or pending != 4096 or sequence < last_request or stop > 4):
                        raise ValueError('invalid native result identity/bounds')
                    rows = [sessions.numbers(peer.line(), 'node', 30) for _ in range(count)]
                    best = sessions.numbers(peer.line(), 'best', 1)[0]
                    peer.expect('ready')
                    if rows[0][5] != done or sessions.position(rows[0][11:]) != board_position(board):
                        raise ValueError('native result root mismatch')
                    if any(row[0] != i for i, row in enumerate(rows)):
                        raise ValueError('invalid native node ordering')
                    if stop in (3, 4):
                        raise ValueError('native search rejected evaluator reply')
                    reason = {0: 'native completed', 1: 'native arena limit', 2: 'native cancelled'}[stop]
                    if best == sessions.SENTINEL:
                        spare = fallback(board, reason + '; no native root choice')
                        return Result(spare.move, done, count, spare.reason)
                    move = next((m for m in board.legal_moves if packed_move(board, m) == best), None)
                    if move is None:
                        raise ValueError('native best is not a legal board action')
                    return Result(move.uci(), done, count, reason)
                epoch, request, node, count = sessions.numbers(line, 'eval', 4)
                if epoch != self.epoch or request <= last_request or node >= 4096 or not 1 <= count <= 256:
                    raise ValueError('invalid native evaluation identity/bounds')
                last_request = request
                supplied = sessions.position(sessions.numbers(peer.line(), 'board', 19))
                path = sessions.parse_path(peer.line())
                actions = [sessions.numbers(peer.line(), 'action', 1)[0] for _ in range(count)]
                peer.expect('end_eval')
                leaf = reconstruct_leaf(board, path, supplied, actions)
                if cancelled.is_set():
                    peer.write(f'reply {epoch:x} {request:x} {node:x} 2 0 3f800000 0 0\n')
                    continue
                if automatic_draw(leaf) is not None:
                    peer.write(draw_reply(epoch, request, node))
                    continue
                wdl, policy = self.evaluator.evaluate(board, path, supplied, actions)
                if cancelled.is_set():
                    peer.write(f'reply {epoch:x} {request:x} {node:x} 2 0 3f800000 0 0\n')
                    continue
                fields = [epoch, request, node, 0, *(sessions.bits(x) for x in wdl),
                          len(policy), *(sessions.bits(x) for x in policy)]
                peer.write('reply ' + ' '.join(f'{x:x}' for x in fields) + '\n')
        except Exception:
            self.close()
            raise
