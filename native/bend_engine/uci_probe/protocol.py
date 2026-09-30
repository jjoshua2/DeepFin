"""Bounded experimental UCI frontend. Protocol output has one owner.

No model or native build at import time. The backend runs off the input thread so
isready/stop remain responsive. The diagnostic tree's bounds are NOT lifted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
import time
from typing import Protocol, TextIO
from collections.abc import Callable

import chess

MAX_LINE = 16384
MAX_PLIES = 512


def unsigned(text: str, maximum: int = 2**31 - 1) -> int:
    if not text.isascii() or not text.isdecimal() or len(text) > 10:
        raise ValueError('expected a bounded unsigned decimal integer')
    value = int(text)
    if value > maximum:
        raise ValueError('integer exceeds supported limit')
    return value


def position(tokens: list[str]) -> chess.Board:
    if not tokens:
        raise ValueError('missing position')
    if tokens[0] == 'startpos':
        board, rest = chess.Board(), tokens[1:]
    elif tokens[0] == 'fen' and len(tokens) >= 7:
        board, rest = chess.Board(' '.join(tokens[1:7])), tokens[7:]
    else:
        raise ValueError('expected startpos or a complete six-field FEN')
    if not board.is_valid() or board.halfmove_clock > 255:
        raise ValueError('invalid orthodox position or unsupported rule50 clock')
    if rest:
        if rest[0] != 'moves' or len(rest) - 1 > MAX_PLIES:
            raise ValueError('expected at most 512 history moves')
        for text in rest[1:]:
            move = chess.Move.from_uci(text)
            if move not in board.legal_moves:
                raise ValueError('illegal history move: ' + text)
            board.push(move)
            if board.halfmove_clock > 255:
                raise ValueError('history exceeds supported rule50 clock')
    return board


@dataclass(frozen=True)
class Limits:
    simulations: int
    depth: int
    seconds: float | None = None
    infinite: bool = False


def limits(tokens: list[str], turn: bool, simulations: int = 32, depth: int = 4) -> Limits:
    fields: dict[str, int] = {}
    infinite = False
    i = 0
    while i < len(tokens):
        key = tokens[i]
        i += 1
        if key == 'infinite' and not infinite:
            infinite = True
            continue
        if key not in ('nodes', 'depth', 'movetime', 'wtime', 'btime', 'winc', 'binc', 'movestogo'):
            raise ValueError('unsupported go argument: ' + key)
        if key in fields or i == len(tokens):
            raise ValueError('duplicate or missing go value: ' + key)
        fields[key] = unsigned(tokens[i])
        i += 1
    if any(fields.get(key) == 0 for key in ('nodes', 'depth', 'movestogo')):
        raise ValueError('nodes, depth and movestogo must be positive')
    if infinite and fields:
        raise ValueError('infinite cannot be combined with finite limits')
    seconds = None
    if 'movetime' in fields:
        seconds = fields['movetime'] / 1000
    elif 'wtime' in fields or 'btime' in fields:
        side = 'w' if turn else 'b'
        if side + 'time' not in fields:
            raise ValueError('missing clock for side to move')
        remaining = fields[side + 'time']
        reserve = min(remaining, 20)
        allocation = remaining / fields.get('movestogo', 30) + fields.get(side + 'inc', 0) * 0.8
        seconds = max(0.0, min(remaining - reserve, allocation - reserve)) / 1000
    elif any(key in fields for key in ('winc', 'binc', 'movestogo')):
        raise ValueError('clock increments require a clock')
    return Limits(min(fields.get('nodes', simulations), 256), min(fields.get('depth', depth), 32), seconds, infinite)


@dataclass(frozen=True)
class Result:
    move: str
    simulations: int = 0
    allocated: int = 0
    reason: str = 'completed'


def fallback(board: chess.Board, reason: str) -> Result:
    # Used only without a completed native answer, and explicitly labeled as such.
    move = min((m.uci() for m in board.legal_moves), default='0000')
    return Result(move, reason=reason + '; unsearched legal fallback' if move != '0000' else reason)


class Backend(Protocol):
    def search(self, board: chess.Board, bounds: Limits, cancelled: Event) -> Result: ...
    def interrupt(self) -> None: ...
    def new_game(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class Active:
    board: chess.Board
    bounds: Limits
    cancel: Event
    started: float
    thread: Thread
    result: Result | None = None
    cancelled_at: float | None = None
    interrupted: bool = False


class Engine:
    def __init__(self, backend: Backend, emit: Callable[[str], None]):
        self.backend, self.emit = backend, emit
        self.board = chess.Board()
        self.simulations, self.depth = 32, 4
        self.active: Active | None = None
        self.events: Queue[tuple[str, object]] = Queue(maxsize=128)
        self.alive = True

    def info(self, message: str) -> None:
        self.emit('info string ' + ' '.join(message.split())[:1024])

    def complete(self, result: Result) -> None:
        job = self.active
        if job is None:
            return
        try:
            move = chess.Move.from_uci(result.move)
            valid = (result.move == '0000' and not any(job.board.legal_moves)) or move in job.board.legal_moves
        except ValueError:
            valid = False
        if not valid:
            result = fallback(job.board, 'invalid backend move rejected')
        ms = int((time.monotonic() - job.started) * 1000)
        self.emit(f'info nodes {result.simulations} time {ms}')
        self.info(f'{result.reason}; allocated={result.allocated}; nodes count completed simulations')
        self.emit('bestmove ' + result.move)
        self.active = None

    def stop(self) -> None:
        job = self.active
        if job is None:
            return
        if job.result is not None:
            self.complete(job.result)
        else:
            job.cancel.set()
            if job.cancelled_at is None:
                job.cancelled_at = time.monotonic()

    def tick(self) -> None:
        job = self.active
        if job is None:
            return
        now = time.monotonic()
        if job.bounds.seconds is not None and now - job.started >= job.bounds.seconds:
            self.stop()
        if self.active is job and job.cancelled_at is not None and now - job.cancelled_at >= 0.25 and not job.interrupted:
            # Interrupt only processes owned by this backend, never external jobs.
            job.interrupted = True
            self.backend.interrupt()

    def command(self, line: str) -> None:
        if len(line) > MAX_LINE:
            self.info('oversized UCI command rejected')
            return
        words = line.split()
        if not words:
            return
        cmd, args = words[0], words[1:]
        try:
            if cmd == 'uci':
                self.emit('id name DeepFin Bend Experimental')
                self.emit('id author DeepFin contributors')
                self.emit('option name BendSimulations type spin default 32 min 1 max 256')
                self.emit('option name BendDepth type spin default 4 min 1 max 32')
                self.emit('uciok')
            elif cmd == 'isready':
                self.emit('readyok')
            elif cmd == 'stop':
                self.stop()
            elif cmd == 'quit':
                self.alive = False
            elif cmd in ('position', 'ucinewgame', 'setoption', 'go'):
                if self.active is not None:
                    raise ValueError('search active; send stop and wait for bestmove first')
                if cmd == 'position':
                    self.board = position(args)  # transactional: errors retain old root/history
                elif cmd == 'ucinewgame':
                    self.backend.new_game()
                    self.board = chess.Board()
                elif cmd == 'setoption':
                    if len(args) != 4 or args[0] != 'name' or args[2] != 'value':
                        raise ValueError('expected setoption name NAME value INTEGER')
                    key = args[1].lower()
                    if key not in ('bendsimulations', 'benddepth'):
                        raise ValueError('unsupported option: ' + args[1])
                    maximum = 256 if key == 'bendsimulations' else 32
                    value = unsigned(args[3], maximum)
                    if value == 0:
                        raise ValueError('option value must be positive')
                    if key == 'bendsimulations':
                        self.simulations = value
                    else:
                        self.depth = value
                else:
                    bounds = limits(args, self.board.turn, self.simulations, self.depth)
                    board, cancelled = self.board.copy(stack=True), Event()
                    def search() -> None:
                        try:
                            result = self.backend.search(board, bounds, cancelled)
                        except Exception as error:
                            result = fallback(board, 'backend failure: ' + str(error))
                        self.events.put(('done', result))
                    thread = Thread(target=search, name='bend-uci-search', daemon=True)
                    self.active = Active(board, bounds, cancelled, time.monotonic(), thread)
                    self.info(f'bounded diagnostic PUCT: simulations<={bounds.simulations}, depth<={bounds.depth}, arena=4096; claims are GUI-owned')
                    thread.start()
            elif cmd not in ('debug', 'register'):
                self.info('unsupported command: ' + cmd)
        except ValueError as error:
            self.info('rejected ' + cmd + ': ' + str(error))
            if cmd == 'go' and self.active is None:
                # Invalid unsupported go still releases a GUI waiting for an answer.
                self.emit('bestmove ' + fallback(self.board, 'rejected go').move)

    def run(self, stream: TextIO) -> None:
        def read() -> None:
            while True:
                line = stream.readline(MAX_LINE + 2)
                if len(line) > MAX_LINE:
                    self.events.put(('quit', None))
                    return
                self.events.put(('command', line) if line else ('quit', None))
                if not line:
                    return
        Thread(target=read, name='bend-uci-input', daemon=True).start()
        try:
            while self.alive:
                try:
                    kind, data = self.events.get(timeout=0.01)
                except Empty:
                    self.tick()
                    continue
                if kind == 'command':
                    self.command(str(data))
                elif kind == 'quit':
                    self.alive = False
                elif kind == 'done' and self.active is not None:
                    assert isinstance(data, Result)
                    if self.active.bounds.infinite and not self.active.cancel.is_set():
                        self.active.result = data
                        self.info('bounded search finished; holding result until stop')
                    else:
                        self.complete(data)
                self.tick()
        finally:
            if self.active is not None:
                self.active.cancel.set()
                self.backend.interrupt()
                self.active.thread.join(timeout=15)
            self.backend.close()
