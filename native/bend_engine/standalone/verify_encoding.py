"""External oracle for the Bend-owned 112-plane block, never an engine dependency.

The complete float32 payload is compared bit-for-bit with the independent Python
encoder and, when requested, the existing CBoard encoder. No neural inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.lc0 import encode_lc0_full
from .verify import Client, assert_state
from .verify_rules import fixtures, position

LAYOUTS = ('lc0_root', 'lc0_root_legacy_meta')


def read_encoding(c: Client, layout: str, moves: list[str] | None = None) -> np.ndarray:
    command = 'encode_history ' + layout
    if moves:
        command += ' moves ' + ' '.join(moves)
    c.send(command + '\n')
    lines = c.until('info string history_encoding_end', timeout=30)
    assert lines[0] == f'info string history_encoding {layout} 112 8 8 repfix=1 partial_input', lines[:1]
    assert len(lines) == 114
    words = []
    for plane, line in enumerate(lines[1:-1]):
        fields = line.split()
        assert fields[:4] == ['info', 'string', 'history_plane', str(plane)]
        assert len(fields) == 68
        row = [int(v) for v in fields[4:]]
        assert all(0 <= v <= 0xffffffff for v in row)
        words.extend(row)
    result = np.array(words, dtype=np.uint32).view(np.float32).reshape(112, 8, 8)
    assert np.isfinite(result).all()
    return result


def expanded_fixtures() -> list[tuple[str, chess.Board]]:
    result = fixtures()
    # Exhaust every normalized halfmove fraction, not a loose float tolerance.
    result += [(f'fraction-{n}', chess.Board(f'4k3/8/8/8/8/8/8/R3K3 b - - {n} 1'))
               for n in range(101)]
    # Single-sided rights make us/them AND king/queen ordering observable.
    for rights in ('K', 'Q', 'k', 'q', 'Kq', 'Qk', 'KQkq', '-'):
        for side in ('w', 'b'):
            result.append((f'rights-{rights}-{side}', chess.Board(f'r3k2r/8/8/8/8/8/8/R3K2R {side} {rights} - 17 1')))
    for side, fen, stem in [
        ('white', '4k3/P7/8/8/8/8/8/4K3 w - - 2 1', 'a7a8'),
        ('black', '4k3/8/8/8/8/8/p7/4K3 b - - 2 1', 'a2a1'),
    ]:
        for piece in 'nbrq':
            board = chess.Board(fen)
            board.push_uci(stem + piece)
            result.append((f'{side}-promotion-{piece}', board))
    # A later pawn move must not erase repetition flags of older visible frames.
    board = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 3:
        board.push_uci(move)
    for move in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1e2', 'g8f6', 'e1g1']:
        board.push_uci(move)
        result.append(('after-irreversible-' + move, board.copy(stack=True)))
    return result


def c_representable(board: chess.Board) -> bool:
    # CBoard clocks are uint8; don't turn a wrapped reference into an oracle.
    return board.halfmove_clock <= 255 and all(s.halfmove_clock <= 255 for s in board._stack)


def verify(command: list[str], *, require_c: bool = False) -> dict[str, object]:
    cboard_type = None
    if require_c:
        from chess_anti_engine.encoding._lc0_ext import CBoard
        rep_fix.apply(True)
        cboard_type = CBoard
    client = Client(command)
    count = c_count = c_skipped = paths = rejected = 0
    digest = hashlib.sha256()

    def compare(board: chess.Board, layout: str, observed: np.ndarray) -> None:
        nonlocal count, c_count, c_skipped
        expected = encode_lc0_full(board, input_history_encoding=layout)
        np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))
        digest.update(observed.astype('<f4', copy=False).tobytes())
        count += 1
        if cboard_type is not None:
            if c_representable(board):
                cb = cboard_type.from_board(board)
                c_input = cb.encode_full(LAYOUTS.index(layout) + 1, 34)[:112]
                np.testing.assert_array_equal(observed.view(np.uint32), c_input.view(np.uint32))
                c_count += 1
            else:
                c_skipped += 1

    try:
        boards = expanded_fixtures()
        for label, board in boards:
            assert board.is_valid(), label
            assert client.sync(position(board)) == ['readyok'], label
            for layout in LAYOUTS:
                compare(board, layout, read_encoding(client, layout))
            assert_state(client, board)

        # Hypothetical descendants do not become the played root. Missing history
        # is not fabricated; exact comparison covers each slot independently.
        root = chess.Board()
        for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
            root.push_uci(move)
        assert client.sync(position(root)) == ['readyok']
        lines = [
            ['g1f3', 'g8f6', 'b1c3', 'b8c6'],
            ['b1c3', 'b8c6', 'g1f3', 'g8f6'],
            ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1e2'],
            ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 8,
        ]
        same_board = []
        for continuation in lines:
            board = root.copy(stack=True)
            for move in continuation:
                board.push_uci(move)
            for layout in LAYOUTS:
                tensor = read_encoding(client, layout, continuation)
                compare(board, layout, tensor)
                paths += 1
                if layout == LAYOUTS[0]:
                    same_board.append((board, tensor))
            assert_state(client, root)
        assert same_board[0][0].fen(en_passant='fen') == same_board[1][0].fen(en_passant='fen')
        assert not np.array_equal(same_board[0][1][:104], same_board[1][1][:104])
        for args in ['', 'legacy', 'lc0_root v1', 'lc0_root_legacy_meta repfix=0',
                     'lc0_root garbage', 'lc0_root moves e2e5',
                     'lc0_root moves e2e4 e7e5 a1a8',
                     'lc0_root moves ' + ' '.join(['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 8 + ['g1f3'])]:
            rows = client.sync('encode_history ' + args)
            assert len(rows) == 2, rows
            assert rows[0].startswith('info string encode_history requires '), rows
            assert rows[-1] == 'readyok'
            assert_state(client, root)
            rejected += 1
        # A busy request is rejected without altering the active search or root.
        client.send('go infinite nodes 4\nencode_history lc0_root\nisready\n')
        rows = client.until('readyok')
        assert any('busy;' in x for x in rows)
        assert not any('history_encoding ' in x for x in rows)
        client.send('stop\n')
        client.until('bestmove ')
        assert_state(client, root)
        assert client.sync('ucinewgame') == ['readyok']
        compare(chess.Board(), LAYOUTS[0], read_encoding(client, LAYOUTS[0]))
        assert_state(client, chess.Board())
    finally:
        client.close()
    return {
        'scope': 'Bend-authored 112-plane block only; no classical features, policy mapping or neural inference',
        'status': 'passed', 'layouts': LAYOUTS, 'repetition_fix': True,
        'position_fixtures': len(boards), 'tensor_comparisons_python': count,
        'tensor_comparisons_c': c_count, 'c_clock_range_skips': c_skipped,
        'hypothetical_path_comparisons': paths, 'rejected_requests': rejected,
        'float_values_checked_python': count * 112 * 64, 'equality': 'exact F32 bits, no tolerance',
        'ordered_tensor_sha256': digest.hexdigest(), 'numpy_version': np.__version__,
        'python_chess_version': chess.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-c', action='store_true', help='require the existing CBoard reference extension outside the engine')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.command:
        parser.error('--command requires an executable')
    report = verify(args.command, require_c=args.require_c)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
