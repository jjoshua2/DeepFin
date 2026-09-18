#!/usr/bin/env python3
"""Opt-in persistent Bend PUCT/evaluator contract check. No NN or GPU claim."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import struct
import subprocess
import tempfile
from threading import Thread
import time

from native.bend_engine.bitboard_probe.run_probe import check_compiler
from native.bend_engine.legal_probe import run_probe as rules

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = rules.MODES
SENTINEL = (1 << 32) - 1


def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', x))[0]


def unbits(x: int) -> float:
    return struct.unpack('<f', struct.pack('<I', x))[0]


def move_key(move: rules.Move) -> int:
    src, dst, promotion, flag = move
    return src | dst << 6 | promotion << 12 | flag << 15


def numbers(line: str, tag: str, count: int | None = None) -> list[int]:
    tokens = line.split()
    if not tokens or tokens[0] != tag or (count is not None and len(tokens) != count + 1):
        raise ValueError(f'malformed {tag} record')
    if any(not t.isascii() or not t.isdecimal() for t in tokens[1:]):
        raise ValueError('nondecimal output')
    values = [int(t) for t in tokens[1:]]
    if any(x > SENTINEL for x in values):
        raise ValueError('output exceeds U32')
    return values


def position(words: list[int]) -> rules.Position:
    if len(words) != 19 or any(not 0 <= x <= SENTINEL for x in words):
        raise ValueError('invalid position words')
    if words[16] > 1 or words[17] > 15 or words[18] > 64:
        raise ValueError('invalid position metadata')
    return (*(words[i] << 32 | words[i+1] for i in range(0, 16, 2)), *words[16:])


@dataclass
class Node:
    board: rules.Position
    key: int = 0
    parent: int = 0
    depth: int = 0
    status: int = 0
    n: int = 0
    w: float = 0.0
    prior: float = 1.0
    value: float = 0.0
    first: int = 0
    count: int = 0


class Oracle:
    """Independent tree/scheduler, using actual CBoard for chess transitions."""
    def __init__(self, binary: Path, with_python_chess: bool = False):
        self.binary = binary
        self.with_python_chess = with_python_chess
        self.cache: dict[rules.Position, dict[int, rules.Position]] = {}
        self.terminal: dict[rules.Position, float] = {}

    def moves(self, board: rules.Position) -> dict[int, rules.Position]:
        if board not in self.cache:
            parsed, _ = rules.parse_moves(rules.run_binary(self.binary, board, mode=1))
            assert isinstance(parsed, dict)
            self.cache[board] = {move_key(m): p for m, p in parsed.items()}
            if self.with_python_chess:
                import chess
                b = chess.Board(None)
                for p in range(6):
                    for square in chess.scan_forward(board[p]):
                        b.set_piece_at(square, chess.Piece(p + 1, bool(board[6] & 1 << square)))
                b.turn = bool(board[8])
                b.castling_rights = sum(1 << sq for bit, sq in enumerate((7, 0, 63, 56)) if board[9] & 1 << bit)
                b.ep_square = None if board[10] == 64 else board[10]
                observed = {}
                for m in b.legal_moves:
                    flag = 2 if b.is_castling(m) else int(b.is_en_passant(m))
                    child = b.copy(stack=False)
                    child.push(m)
                    k = move_key((m.from_square, m.to_square, m.promotion - 1 if m.promotion else 0, flag))
                    observed[k] = rules.fen_position(child.fen(en_passant='fen'))
                if observed != self.cache[board]:
                    raise AssertionError('independent python-chess/CBoard disagreement')
            if not self.cache[board]:
                terminal = rules.run_binary(self.binary, board, mode=2).strip()
                if terminal not in ('checkmate', 'stalemate'):
                    raise AssertionError('invalid terminal oracle')
                self.terminal[board] = -1.0 if terminal == 'checkmate' else 0.0
        return self.cache[board]


class Reference:
    def __init__(self, board: rules.Position, oracle: Oracle, *, cap: int, depth: int, budget: int):
        self.nodes = [Node(board)]
        self.oracle = oracle
        self.cap, self.depth, self.budget = cap, depth, budget
        self.completed, self.seq, self.stop = 0, 1, 0

    def select(self) -> int:
        index = 0
        while self.nodes[index].status == 1:
            parent = self.nodes[index]
            children = list(range(parent.first, parent.first + parent.count))
            mass = 0.0
            for i in children:
                if self.nodes[i].n:
                    mass = f32(mass + self.nodes[i].prior)
            mean = f32(parent.w / max(1, parent.n))
            reduction = f32(0.25 if parent.depth == 0 else 0.15)
            fpu = f32(mean - f32(reduction * f32(math.sqrt(mass))))
            scale = f32(1.5 * f32(math.sqrt(max(1, parent.n))))
            def score(i: int, fpu: float = fpu, scale: float = scale) -> tuple[float, int]:
                a = self.nodes[i]
                q = -f32(a.w / a.n) if a.n else fpu
                u = f32(f32(scale * a.prior) / (1 + a.n))
                return f32(q + u), -a.key
            index = max(children, key=score)
        return index

    def backup(self, index: int, value: float) -> None:
        while True:
            a = self.nodes[index]
            a.n += 1
            a.w = f32(a.w + value)
            if index == 0:
                break
            index, value = a.parent, -value
        self.completed += 1

    def next(self) -> int | None:
        while not self.stop and self.completed < self.budget:
            index = self.select()
            a = self.nodes[index]
            if a.status >= 2:
                self.backup(index, a.value)
                continue
            moves = self.oracle.moves(a.board)
            if not moves:
                a.status, a.value = 2, self.oracle.terminal[a.board]
                self.backup(index, a.value)
                continue
            if a.depth < self.depth and len(self.nodes) + len(moves) > self.cap:
                self.stop = 1
                return None
            return index
        return None

    def accept(self, index: int, actions: list[int], wdl: list[float], policy: list[float]) -> None:
        a = self.nodes[index]
        total = f32(f32(wdl[0] + wdl[1]) + wdl[2])
        value = f32(f32(wdl[0] - wdl[2]) / total)
        a.value = value
        if a.depth >= self.depth:
            a.status = 3
        else:
            a.status, a.first, a.count = 1, len(self.nodes), len(actions)
            total_policy = 0.0
            for p in policy:
                total_policy = f32(total_policy + p)
            successors = self.oracle.moves(a.board)
            self.nodes.extend(Node(successors[k], key=k, parent=index, depth=a.depth + 1,
                                   prior=f32(p / total_policy)) for k, p in zip(actions, policy, strict=True))
        self.backup(index, value)
        self.seq += 1

    def check_snapshot(self, rows: list[list[int]], result: list[int], best: int, epoch: int) -> None:
        expected = [epoch, self.completed, len(self.nodes), self.stop, 4096, self.seq]
        if result != expected or len(rows) != len(self.nodes):
            raise AssertionError(f'session accounting mismatch: {result} != {expected}')
        for i, (row, a) in enumerate(zip(rows, self.nodes, strict=True)):
            if row[:6] != [i, a.parent, a.key, a.depth, a.status, a.n] or row[9:11] != [a.first, a.count]:
                raise AssertionError(f'node metadata mismatch: {i}: {row[:11]} vs {a}')
            for observed, expected_f in zip(row[6:9], (a.w, a.prior, a.value), strict=True):
                if not math.isfinite(unbits(observed)) or abs(unbits(observed) - expected_f) > 2e-6:
                    raise AssertionError(f'node float mismatch: {i}: {unbits(observed)} != {expected_f}')
            if position(row[11:]) != a.board:
                raise AssertionError(f'node board mismatch: {i}')
        root_children = [a for i, a in enumerate(self.nodes) if i and a.parent == 0]
        expected_best = min(root_children, key=lambda a: (-a.n, a.key)).key if root_children else SENTINEL
        if best != expected_best or self.nodes[0].n != self.completed:
            raise AssertionError('best-move/root visit mismatch')


class Peer:
    def __init__(self, binary: Path, board: rules.Position):
        with ExitStack() as resources:
            self.errors = resources.enter_context(tempfile.TemporaryFile(mode='w+'))
            self.proc = subprocess.Popen([str(binary), '--threads', '1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.errors, text=True, bufsize=1,
                                         env={**os.environ, 'BEND_NO_TELEMETRY': '1'})
            self.resources = resources.pop_all()
        assert self.proc.stdout
        assert self.proc.stdin
        self.queue: Queue[str | None] = Queue()
        def read() -> None:
            assert self.proc.stdout
            for line in self.proc.stdout:
                self.queue.put(line.rstrip('\n'))
            self.queue.put(None)
        self.reader = Thread(target=read, daemon=True)
        self.reader.start()
        try:
            self.write(rules.request(board, mode=1))
            self.expect('ready')
        except BaseException:
            self.close()
            raise

    def write(self, text: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def line(self) -> str:
        try:
            line = self.queue.get(timeout=10)
        except Empty as e:
            raise RuntimeError('native session exceeded reply deadline') from e
        if line is None:
            self.errors.seek(0)
            raise RuntimeError(f'unexpected native EOF: {self.errors.read()}')
        if len(line) > 4096:
            raise RuntimeError('oversized native output')
        return line

    def expect(self, expected: str) -> None:
        observed = self.line()
        if observed != expected:
            raise AssertionError(f'{observed!r} != {expected!r}')

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=5)
        self.reader.join(timeout=5)
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.stdout:
            self.proc.stdout.close()
        self.resources.close()

    def finish(self) -> None:
        self.write('config 0 0 0 0\n')
        self.expect('bye')
        if self.proc.wait(timeout=5) != 0:
            raise RuntimeError('native shutdown failed')


def evaluation(board: rules.Position, actions: list[int], variant: int = 0) -> tuple[list[float], list[float]]:
    """Test evaluator only: exact-quarter WDL and deterministic legal priors."""
    material = sum(weight * ((board[p] & board[6]).bit_count() - (board[p] & board[7]).bit_count())
                   for p, weight in enumerate((1, 3, 3, 5, 9)))
    material *= 1 if board[8] else -1
    wdl = [0.75, 0.25, 0.0] if material > 0 else ([0.0, 0.25, 0.75] if material < 0 else [0.25, 0.5, 0.25])
    priors = [((k * (3 if variant else 5) + variant) % 4 + 1) / 4 for k in actions]
    return wdl, priors


def session(peer: Peer, oracle: Oracle, board: rules.Position, *, epoch: int, budget: int = 24,
            cap: int = 4096, depth: int = 4, fault: str = '', fault_at: int = 3, variant: int = 0) -> dict[str, int | str]:
    peer.write(f'config {epoch:x} {budget:x} {cap:x} {depth:x}\n')
    ref = Reference(board, oracle, cap=cap, depth=depth, budget=budget)
    exchanges = 0
    while True:
        wanted = ref.next()
        line = peer.line()
        if wanted is None:
            result = numbers(line, 'result', 6)
            rows = [numbers(peer.line(), 'node', 30) for _ in ref.nodes]
            best = numbers(peer.line(), 'best', 1)[0]
            ref.check_snapshot(rows, result, best, epoch)
            peer.expect('ready')
            return {'completed': ref.completed, 'nodes': len(ref.nodes), 'exchanges': exchanges,
                    'max_depth': max(a.depth for a in ref.nodes), 'best': best, 'stop': ref.stop, 'fault': fault}
        header = numbers(line, 'eval', 4)
        if header[:3] != [epoch, ref.seq, wanted]:
            raise AssertionError(f'selection/request mismatch: {header}, expected {[epoch, ref.seq, wanted]}')
        supplied = position(numbers(peer.line(), 'board', 19))
        if supplied != ref.nodes[wanted].board:
            raise AssertionError('request carries wrong board')
        if not 1 <= header[3] <= 256:
            raise AssertionError('invalid legal action count')
        actions = [numbers(peer.line(), 'action', 1)[0] for _ in range(header[3])]
        peer.expect('end_eval')
        legal = oracle.moves(supplied)
        if len(set(actions)) != len(actions) or set(actions) != set(legal):
            raise AssertionError('request legal moves differ from CBoard')
        wdl, policy = evaluation(supplied, actions, variant)
        fields = [epoch, ref.seq, wanted, 0, *(bits(v) for v in wdl), len(policy), *(bits(v) for v in policy)]
        exchanges += 1
        if fault and exchanges == fault_at:
            if fault == 'cancel':
                fields[3] = 2
                ref.stop = 2
            elif fault == 'backend':
                fields[3] = 1
                ref.stop = 3
            else:
                ref.stop = 4
                if fault == 'epoch':
                    fields[0] -= 1
                elif fault == 'request':
                    fields[1] -= 1
                elif fault == 'node':
                    fields[2] = 4095
                elif fault == 'count':
                    fields[7] -= 1
                    fields.pop()
                elif fault == 'nan':
                    fields[4] = 0x7fc00000
                elif fault == 'infinity':
                    fields[8] = 0x7f800000
                elif fault == 'negative':
                    fields[8] = bits(-0.25)
                elif fault == 'zero_policy':
                    fields[8:] = [0] * len(policy)
                elif fault == 'bad_wdl':
                    fields[4:7] = [bits(0.0)] * 3
                elif fault == 'status':
                    fields[3] = 9
                else:
                    raise ValueError(f'unknown fault {fault}')
        else:
            ref.accept(wanted, actions, wdl, policy)
        peer.write('reply ' + ' '.join(f'{x:x}' for x in fields) + '\n')


def build(source: Path, directory: Path, bun: str, cc: str, modes: list[str]) -> dict[str, Path]:
    check_compiler(source)
    directory.mkdir(parents=True, exist_ok=True)
    generated = directory / 'session.c'
    rules.command([bun, str(source / 'bend2/main.ts'), str(HERE / 'main.bend'), '-o', str(generated)])
    binaries = {}
    for mode in modes:
        obj, binary = directory / f'support-{mode}.o', directory / f'session-{mode}'
        flags = MODES[mode]
        rules.command([cc, '-std=c11', '-O3', *flags, '-I', str(ROOT), '-c', str(rules.HERE / 'support.c'), '-o', str(obj)])
        rules.command([cc, '-std=c11', '-O3', '-ffp-contract=off', *flags, '-I', str(rules.HERE), str(generated), str(obj), '-pthread', '-lm', '-o', str(binary)])
        binaries[mode] = binary
    oracle = directory / 'cboard-reference'
    rules.command([cc, '-std=c11', '-O3', '-DLEGAL_ORACLE', '-I', str(ROOT), str(rules.HERE / 'support.c'), '-pthread', '-lm', '-o', str(oracle)])
    binaries['reference'] = oracle
    return binaries


def verify(binaries: dict[str, Path], with_python_chess: bool = False) -> dict[str, object]:
    oracle = Oracle(binaries['reference'], with_python_chess)
    fixtures = [(label, fen) for label, fen, _ in rules.CANONICAL[:2]] + [rules.EDGES[i] for i in (0, 3, 4, 5, 6, 14, 15, 16)]
    result = []
    start = rules.fen_position(rules.START)
    for mode, binary in binaries.items():
        if mode == 'reference':
            continue
        rows = []
        for label, fen in fixtures:
            board = rules.fen_position(fen)
            peer = Peer(binary, board)
            try:
                row = session(peer, oracle, board, epoch=1)
                rows.append({'fixture': label, **row})
                # Reset the arena in the same process, retaining immutable tables.
                again = session(peer, oracle, board, epoch=2)
                if again != row:
                    raise AssertionError('fresh epoch changes deterministic search')
                peer.finish()
            finally:
                peer.close()
        peer = Peer(binary, start)
        try:
            for epoch, fault in enumerate(('cancel', 'backend', 'epoch', 'request', 'node', 'count', 'nan',
                                           'infinity', 'negative', 'zero_policy', 'bad_wdl', 'status'), 1):
                row = session(peer, oracle, start, epoch=epoch, fault=fault)
                if row['completed'] != 2:
                    raise AssertionError('bad reply committed a simulation')
            # Recovery after failures on the same native process.
            rows.append({'fixture': 'recovery', **session(peer, oracle, start, epoch=13, budget=8)})
            rows.append({'fixture': 'capacity', **session(peer, oracle, start, epoch=14, cap=32)})
            rows.append({'fixture': 'cutoff', **session(peer, oracle, start, epoch=15, depth=1, budget=64)})
            changed = session(peer, oracle, start, epoch=16, variant=1)
            if changed['best'] == rows[0]['best']:
                raise AssertionError('changed evaluator policy did not affect best move')
            rows.append({'fixture': 'policy_change', **changed})
            rows.append({'fixture': 'capacity_before_eval', **session(peer, oracle, start, epoch=17, cap=1)})
            peer.finish()
        finally:
            peer.close()
        # Errors at the synchronous transport boundary must fail closed.
        invalid = [
            ('config 1 0 1000 4\n', 'invalid session config'),
            ('config 1 101 1000 4\n', 'invalid session config'),
            ('config 1 1 1001 4\n', 'invalid session config'),
            ('config 1 1 1000 21\n', 'invalid session config'),
            ('config 1 1 0 4\n', 'invalid session config'),
            ('config\n', 'invalid session config'),
            ('config 1 1 x 4\n', 'invalid transport word'),
            ('config 1 1 1000 4', 'unterminated transport record'),
            ('config 1 1 1000 4\n', 'unexpected evaluator EOF'),
            ('config 1 1 1000 4\nreply 1 1 0 0 0 0 0 1\n', 'invalid reply length'),
        ]
        for text, diagnostic in invalid:
            response = subprocess.run([str(binary), '--threads', '1'],
                                      input=rules.request(start, mode=1) + text,
                                      capture_output=True, text=True, timeout=10, check=False)
            if response.returncode != 2 or diagnostic not in response.stderr:
                raise AssertionError(f'wrong transport failure: {response.returncode}: {response.stderr}')
        peer = Peer(binary, start)
        try:
            session(peer, oracle, start, epoch=1, budget=1)
            peer.write('config 1 1 1000 4\n')
            code = peer.proc.wait(timeout=5)
            peer.errors.seek(0)
            if code != 2 or 'session epoch must increase' not in peer.errors.read():
                raise AssertionError('non-increasing epoch was accepted')
        finally:
            peer.close()
        result.append({'mode': mode, 'sessions': 38, 'control_or_bad_replies': 12,
                       'non_increasing_epoch_rejected': True,
                       'invalid_transport_requests_rejected': len(invalid), 'observations': rows})
    return {'results': result, 'oracle_positions': len(oracle.cache), 'python_chess': with_python_chess,
            'scope': 'persistent bounded PUCT + external test evaluator; not DeepFin NN/Gumbel/batching/UCI parity'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--modes', nargs='+', choices=list(MODES), default=list(MODES))
    parser.add_argument('--python-chess', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='bend-sessions-') as temp:
        report = verify(build(args.compiler_root, Path(temp), args.bun, args.cc, args.modes), args.python_chess)
    report['validation_seconds'] = time.monotonic() - start
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
