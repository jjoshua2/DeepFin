"""External oracle/client only. Never imported or invoked by the native engine.

Pass --command chroot EMPTY_ROOT /deepfin-bend --threads 1 to run exactly the
same tests with no Python, Bun, shell, libraries or data files in the engine root.
"""
from __future__ import annotations

import argparse
import json
import queue
import random
import subprocess
import threading
import time
from pathlib import Path

import chess
import chess.engine


class Client:
    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        self.input = self.proc.stdin
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.errors: list[str] = []

        def read() -> None:
            assert self.proc.stdout
            for line in self.proc.stdout:
                self.lines.put(line.rstrip('\n'))
            self.lines.put(None)

        def errors() -> None:
            assert self.proc.stderr
            self.errors.extend(self.proc.stderr)

        self.reader = threading.Thread(target=read, daemon=True)
        self.err_reader = threading.Thread(target=errors, daemon=True)
        self.reader.start()
        self.err_reader.start()
        self.send('uci\nisready\n')
        intro = self.until('readyok')
        assert 'uciok' in intro
        assert any('Bend standalone' in x for x in intro)

    def send(self, text: str) -> None:
        self.input.write(text)
        self.input.flush()

    def until(self, prefix: str, timeout: float = 10) -> list[str]:
        deadline = time.monotonic() + timeout
        result = []
        while True:
            line = self.lines.get(timeout=max(.001, deadline - time.monotonic()))
            if line is None:
                raise AssertionError('engine EOF: ' + ''.join(self.errors))
            result.append(line)
            if line.startswith(prefix):
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(result)

    def sync(self, command: str) -> list[str]:
        self.send(command + '\nisready\n')
        return self.until('readyok')

    def dump(self) -> list[str]:
        self.send('d\n')
        return self.until('info string state_end')

    def go(self, board: chess.Board, nodes: int = 8) -> str:
        self.send(f'go nodes {nodes}\n')
        rows = self.until('bestmove ')
        assert any(x.startswith(f'info nodes {nodes} ') for x in rows), rows
        move = rows[-1].split()[1]
        if any(board.legal_moves):
            assert chess.Move.from_uci(move) in board.legal_moves, (board.fen(), rows)
        else:
            assert move == '0000'
        return move

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send('quit\n')
                self.proc.wait(timeout=5)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.input.close()
            self.reader.join(timeout=2)
            self.err_reader.join(timeout=2)
            assert self.proc.stdout is not None
            assert self.proc.stderr is not None
            self.proc.stdout.close()
            self.proc.stderr.close()
        assert self.proc.returncode == 0, self.errors
        assert not self.errors, self.errors


def words(b: chess.Board) -> str:
    bitboards = (b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                 b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK])
    result = [str(v) for bb in bitboards for v in (bb >> 32, bb & 0xffffffff)]
    rights = sum(1 << i for i, sq in enumerate((7, 0, 63, 56)) if b.castling_rights & (1 << sq))
    return ' '.join([*result, str(int(b.turn)), str(rights), str(64 if b.ep_square is None else b.ep_square)])


def assert_state(c: Client, board: chess.Board) -> None:
    observed = c.dump()
    expected = [f'info string state {words(board)} {board.halfmove_clock} {board.fullmove_number} {len(board.move_stack)}']
    b = board.copy(stack=True)
    while b.move_stack:
        move = b.pop()
        expected.append(f'info string history {words(b)} {b.halfmove_clock} {b.fullmove_number} {move.uci()}')
    expected.append('info string state_end')
    assert observed == expected, (observed, expected)


FENS = [
    chess.STARTING_FEN,
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
    '4k3/8/8/8/8/8/p7/4K3 b - - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1',
    'k7/1Q6/2K5/8/8/8/8/8 b - - 0 1',
    'k7/2Q5/2K5/8/8/8/8/8 b - - 0 1',
]


def verify(command: list[str]) -> dict[str, object]:
    c = Client(command)
    transitions = searches = invalid = 0
    try:
        # All legal children in selected special positions; compare FULL boards,
        # both clocks and each historical board/move, not only legal bestmove.
        for fen in FENS:
            b = chess.Board(fen)
            assert b.is_valid(), fen
            assert c.sync('position fen ' + fen) == ['readyok']
            assert_state(c, b)
            c.go(b)
            searches += 1
            for move in list(b.legal_moves):
                child = b.copy(stack=True)
                child.push(move)
                assert c.sync('position fen ' + fen + ' moves ' + move.uci()) == ['readyok']
                assert_state(c, child)
                transitions += 1
        # Increasing played history, and independent rewinds to the same root.
        b, rng = chess.Board(), random.Random(20260920)
        for _ in range(40):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))
            text = ' '.join(m.uci() for m in b.move_stack)
            assert c.sync('position startpos moves ' + text) == ['readyok']
            assert_state(c, b)
            c.go(b, 4)
            searches += 1
        # Parser rejections are transactions, including an illegal late move.
        prior = c.dump()
        bad = ['position', 'position startpos bogus', 'position startpos moves e2e4 e7e5 a1a8',
               'position startpos moves e2e4q', 'position fen 8/8/8/8/8/8/8/8 w - - 0 1',
               'position fen ' + chess.STARTING_FEN.replace(' w ', ' x '),
               'position fen ' + chess.STARTING_FEN.replace('KQkq', 'KK'),
               'position fen ' + chess.STARTING_FEN.replace(' - ', ' e3 '),
               'position fen ' + chess.STARTING_FEN.replace(' 0 1', ' 0 0'),
               'position fen ' + chess.STARTING_FEN.replace(' 0 1', ' 4294967296 1'),
               'position fen ' + chess.STARTING_FEN.replace(' 0 1', ' 1000001 1'),
               'position fen ' + chess.STARTING_FEN.replace('pppppppp', '9'),
               'position startpos moves ' + 'e2e4 ' * 513]
        for text in bad:
            rows = c.sync(text)
            assert any('invalid position' in x for x in rows), rows
            assert c.dump() == prior
            invalid += 1
        for limits in ('nodes 0', 'nodes 257', 'nodes 4294967296', 'nodes -1', 'nodes 2 nodes 3',
                       'depth 33', 'depth', 'wtime 1000', 'infinite movetime 1', 'movetime 1 infinite'):
            rows = c.sync('go ' + limits)
            assert any('invalid go' in x for x in rows), rows
            assert c.dump() == prior
            invalid += 1
        assert any('invalid or overlong' in x for x in c.sync('x' * 5000))
        assert any('invalid or overlong' in x for x in c.sync('position\x00 startpos'))
        # The same protocol loop remains responsive to partial lines while searching.
        c.sync('position startpos')
        c.send('go infinite nodes 256 depth 32\nisrea')
        time.sleep(.05)
        c.send('dy\n')
        assert c.until('readyok') == ['readyok']  # no premature bestmove
        assert any('busy' in x for x in c.sync('position startpos moves e2e4'))
        start = time.monotonic()
        c.send('stop\nstop\nisready\n')
        rows = c.until('readyok')
        stop_seconds = time.monotonic() - start
        assert sum(x.startswith('bestmove ') for x in rows) == 1, rows
        assert_state(c, chess.Board())
        c.sync('position startpos moves e2e4')
        b = chess.Board()
        b.push_uci('e2e4')
        c.go(b)
        # Perft is an explicit synchronous diagnostic, never a periodic benchmark.
        counts = []
        for fen, depth, expected in ((FENS[0], 3, 8902), (FENS[1], 3, 97862), (FENS[2], 4, 43238)):
            c.sync('position fen ' + fen)
            rows = c.sync(f'perft {depth}')
            assert f'info string perft {depth} 0 {expected}' in rows, rows
            counts.append(expected)
        c.sync('ucinewgame')
        assert_state(c, chess.Board())
        return {'positions': len(FENS), 'exact_children': transitions, 'searched_roots': searches + 1,
                'rejected_transactions': invalid, 'perft_counts': counts,
                'partial_line_ready_and_single_stop': True, 'observed_stop_seconds': stop_seconds}
    finally:
        c.close()


def real_client(command: list[str]) -> list[str]:
    """External standard UCI client, independent of our raw-line verifier."""
    moves: list[str] = []
    with chess.engine.SimpleEngine.popen_uci(command, timeout=10) as engine:
        board = chess.Board()
        for _ in range(8):
            result = engine.play(board, chess.engine.Limit(nodes=4))
            assert result.move is not None
            assert result.move in board.legal_moves
            moves.append(result.move.uci())
            board.push(result.move)
        engine.ping()
    return moves


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    result = verify(args.command)
    result['real_uci_client_moves'] = real_client(args.command)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
