"""Opt-in real UCI client qualification; no work is added to ordinary pytest.

Always builds/checks the unchanged Bend sources. Optional exact-package arguments
exercise native neural UCI too; no model is downloaded or exported by this command.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
import tempfile
from threading import Thread
import time

import chess
import chess.engine

from native.bend_engine.session_probe import run_probe as sessions
from .backend import DiagnosticEvaluator, NativeSearch
from .protocol import Limits
from threading import Event

ROOT = Path(__file__).resolve().parents[3]


class Client:
    def __init__(self, command: list[str]):
        self.lines: Queue[str | None] = Queue()
        self.transcript: list[str] = []
        with ExitStack() as resources:
            self.errors = resources.enter_context(tempfile.TemporaryFile(mode='w+'))
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.errors, text=True, bufsize=1)
            self.resources = resources.pop_all()
        def read() -> None:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.lines.put(line.rstrip())
            self.lines.put(None)
        self.reader = Thread(target=read, daemon=True)
        self.reader.start()

    def send(self, text: str) -> None:
        assert self.proc.stdin is not None
        self.transcript.append('> ' + text)
        self.proc.stdin.write(text + '\n')
        self.proc.stdin.flush()

    def until(self, prefix: str, timeout: float = 15) -> str:
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.001, deadline - time.monotonic()))
            except Empty as error:
                raise AssertionError('UCI timeout waiting for ' + prefix) from error
            if line is None:
                self.errors.seek(0)
                raise AssertionError('early UCI exit: ' + self.errors.read())
            self.transcript.append(line)
            if line.startswith(prefix):
                return line
            if time.monotonic() >= deadline:
                raise AssertionError('UCI timeout waiting for ' + prefix)

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


def native_and_client(binary: Path, reference: Path) -> dict[str, object]:
    command = [sys.executable, '-m', 'native.bend_engine.uci_probe', '--diagnostic', '--bend-binary', str(binary)]
    oracle = sessions.Oracle(reference, with_python_chess=True)
    backend = NativeSearch(binary, DiagnosticEvaluator())
    compared = []
    try:
        with chess.engine.SimpleEngine.popen_uci(command, timeout=15) as engine:
            engine.configure({'BendDepth': 2, 'BendSimulations': 4})
            board = chess.Board()
            pid = None
            for _ in range(8):
                # Independent diagnostic reference + direct native result + UCI result.
                peer = sessions.Peer(binary, sessions.rules.fen_position(board.fen(en_passant='fen')))
                try:
                    expected = sessions.session(peer, oracle, sessions.rules.fen_position(board.fen(en_passant='fen')), epoch=1, budget=4, depth=2)
                    peer.finish()
                finally:
                    peer.close()
                direct = backend.search(board, Limits(4, 2), Event())
                assert backend.peer is not None
                if pid is None:
                    pid = backend.peer.proc.pid
                assert backend.peer.proc.pid == pid
                result = engine.play(board, chess.engine.Limit(nodes=4))
                assert result.move is not None
                from native.bend_engine.session_probe.root_protocol import packed_move
                assert packed_move(board, result.move) == expected['best']
                assert result.move.uci() == direct.move
                assert result.move in board.legal_moves
                compared.append(result.move.uci())
                board.push(result.move)
            # Arbitrary root replacement, both promotion sides, EP and terminal roots.
            for fen in ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
                        '4k3/8/8/8/8/8/p7/4K3 b - - 0 1',
                        '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
                        'k7/1Q6/2K5/8/8/8/8/8 b - - 0 1',
                        'k7/8/1QK5/8/8/8/8/8 b - - 0 1'):
                root = chess.Board(fen)
                result = engine.play(root, chess.engine.Limit(nodes=4), game=object())
                assert result.move is not None
                assert (result.move in root.legal_moves) if any(root.legal_moves) else result.move == chess.Move.null()
            # Standard client's infinite analysis + isready + stop lifecycle.
            analysis = engine.analysis(chess.Board())
            engine.ping()
            analysis.stop()
            stopped = analysis.wait()
            assert stopped.move is not None
            assert stopped.move in chess.Board().legal_moves
            clocked = engine.play(chess.Board(), chess.engine.Limit(white_clock=1, black_clock=1))
            assert clocked.move is not None
            assert clocked.move in chess.Board().legal_moves
    finally:
        backend.close()
    client = Client(command)
    try:
        client.send('uci')
        client.until('uciok')
        client.send('setoption name BendSimulations value 4')
        client.send('go infinite')
        client.until('info string bounded search finished')
        assert not any(x.startswith('bestmove ') for x in client.transcript)
        client.send('position startpos moves e2e4')
        client.until('info string rejected position: search active')
        client.send('isready')
        client.until('readyok', 2)
        client.send('stop')
        assert chess.Move.from_uci(client.until('bestmove ').split()[1]) in chess.Board().legal_moves
        client.send('stop')
        client.send('position startpos moves e2e4')
        client.send('go nodes 4')
        root = chess.Board()
        root.push_uci('e2e4')
        assert chess.Move.from_uci(client.until('bestmove ').split()[1]) in root.legal_moves
        client.send('quit')
        assert client.proc.wait(timeout=5) == 0
        assert sum(x.startswith('bestmove ') for x in client.transcript) == 2
        return {'reference_compared_plies': compared, 'special_or_terminal_roots': 5,
                'raw_transcript': client.transcript, 'normal_shutdown': True}
    finally:
        client.close()


SLOW = '''
from threading import Event
from native.bend_engine.uci_probe.backend import DiagnosticEvaluator
from native.bend_engine.uci_probe.__main__ import main
wake = Event()
original = DiagnosticEvaluator.evaluate
def evaluate(self, *args):
    print("deliberately pending evaluator", file=__import__('sys').stderr, flush=True)
    wake.wait(10)
    return original(self, *args)
DiagnosticEvaluator.evaluate = evaluate
DiagnosticEvaluator.interrupt = lambda self: wake.set()
main()
'''


def stopped_evaluator(binary: Path) -> dict[str, object]:
    client = Client([sys.executable, '-c', SLOW, '--diagnostic', '--bend-binary', str(binary)])
    try:
        client.send('uci')
        client.until('uciok')
        client.send('position startpos moves e2e4')
        client.send('go nodes 256')
        deadline = time.monotonic() + 5
        while True:
            client.errors.seek(0)
            if 'deliberately pending' in client.errors.read():
                break
            if time.monotonic() > deadline:
                raise AssertionError('slow evaluator was not reached')
            time.sleep(0.01)
        client.send('isready')
        client.until('readyok', 2)
        start = time.monotonic()
        client.send('stop')
        answer = client.until('bestmove ', 5)
        elapsed = time.monotonic() - start
        board = chess.Board()
        board.push_uci('e2e4')
        assert chess.Move.from_uci(answer.split()[1]) in board.legal_moves
        assert any('unsearched legal fallback' in line for line in client.transcript)
        client.send('position startpos moves d2d4')
        client.send('go nodes 4')
        next_answer = client.until('bestmove ')
        board = chess.Board()
        board.push_uci('d2d4')
        assert chess.Move.from_uci(next_answer.split()[1]) in board.legal_moves
        client.send('quit')
        assert client.proc.wait(timeout=5) == 0
        assert sum(x.startswith('bestmove ') for x in client.transcript) == 2
        return {'held_evaluator_stop_seconds': elapsed, 'transcript': client.transcript}
    finally:
        client.close()


def neural_client(binary: Path, checkpoint: Path, package: Path, worker: Path) -> dict[str, object]:
    from .evaluator import PackageEvaluator
    evaluator = PackageEvaluator(checkpoint, package, worker, weights='model', device='cpu', device_index=0, batch=4)
    backend = NativeSearch(binary, evaluator)
    command = [sys.executable, '-m', 'native.bend_engine.uci_probe', '--bend-binary', str(binary),
               '--checkpoint', str(checkpoint), '--reuse-package', str(package),
               '--worker-binary', str(worker), '--device', 'cpu', '--batch', '4']
    board = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 2:
        board.push_uci(uci)
    moves = []
    try:
        with chess.engine.SimpleEngine.popen_uci(command, timeout=30) as engine:
            engine.configure({'BendDepth': 2})
            for _ in range(4):
                expected = backend.search(board, Limits(4, 2), Event())
                result = engine.play(board, chess.engine.Limit(nodes=4), info=chess.engine.INFO_ALL)
                if result.move is None or result.move.uci() != expected.move:
                    raise AssertionError('neural UCI move differs from direct native session')
                if 'fallback' in result.info.get('string', '') or expected.simulations != 4:
                    raise AssertionError('neural verification silently used a fallback')
                moves.append(result.move.uci())
                board.push(result.move)
            engine.ping()
        return {'scope': 'CPU static batch4 UCI/direct native comparison; not strength or GPU',
                'pre_root_plies': 8, 'played': moves, 'package_sha256': evaluator.worker.manifest['sha256'] if evaluator.worker else None}
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=list(sessions.MODES))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reuse-package', type=Path)
    parser.add_argument('--worker-binary', type=Path)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    provided = [args.checkpoint, args.reuse_package, args.worker_binary]
    if any(provided) and not all(provided):
        parser.error('neural qualification requires checkpoint, reuse-package and worker-binary together')
    with tempfile.TemporaryDirectory(prefix='bend-uci-') as directory:
        binaries = sessions.build(args.compiler_root, Path(directory), args.bun, args.cc, args.modes)
        report: dict[str, object] = {mode: {'client': native_and_client(binaries[mode], binaries['reference']),
                         'stop': stopped_evaluator(binaries[mode])} for mode in args.modes}
        if args.checkpoint:
            mode = 'native' if 'native' in binaries else args.modes[0]
            report['neural'] = neural_client(binaries[mode], args.checkpoint, args.reuse_package, args.worker_binary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
