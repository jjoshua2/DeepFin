"""External exact policy-map oracle; never called by the Bend executable.

No neural model is loaded. The reference tables come from the existing project
Python module; optional CBoard checks independently cover all legal full IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess

from chess_anti_engine.moves import encode as reference
from .verify import Client, FENS, assert_state
from .verify_rules import fixtures, position

INVALID = 0xffffffff


def key(board: chess.Board, move: chess.Move) -> int:
    promotion = move.promotion - 1 if move.promotion else 0
    flag = 2 if board.is_castling(move) else int(board.is_en_passant(move))
    return move.from_square | (move.to_square << 6) | (promotion << 12) | (flag << 15)


def entry(board: chess.Board, move: chess.Move) -> tuple[str, int, int, int]:
    full = reference.move_to_index(move, board)
    return move.uci(), key(board, move), full, int(reference.FULL_TO_COMPACT_POLICY[full])


def parse_entry(line: str) -> tuple[str, int, int, int]:
    fields = line.split()
    assert fields[:3] == ['info', 'string', 'policy_move'], line
    assert len(fields) == 7, line
    return fields[3], int(fields[4]), int(fields[5]), int(fields[6])


def tables(client: Client) -> str:
    client.send('policy tables\n')
    rows = client.until('info string policy_end', timeout=30)
    assert rows[0] == 'info string policy_tables az_4672=4672 lc0_1858=1858 invalid=4294967295'
    assert len(rows) == 1 + 4672 + 1858 + 4096 + 1
    f2c = [int(v) if v >= 0 else INVALID for v in reference.FULL_TO_COMPACT_POLICY]
    for i, line in enumerate(rows[1:4673]):
        assert line == f'info string policy_full {i} {f2c[i]} {reference.MIRROR_POLICY_MAP[i]}', line
    for i, line in enumerate(rows[4673:6531]):
        assert line == f'info string policy_compact {i} {reference.COMPACT_TO_FULL_POLICY[i]} {reference.COMPACT_MIRROR_POLICY_MAP[i]}', line
    # Check every square-pair, including all non-queen/knight geometries and nulls.
    board = chess.Board()
    for i, line in enumerate(rows[6531:-1]):
        move = chess.Move(i // 64, i % 64)
        try:
            expected = reference.move_to_index(move, board)
        except ValueError:
            expected = INVALID
        assert line == f'info string policy_pair {i} {expected}', line
    assert sum(v != INVALID for v in f2c) == 1858
    assert len(set(map(int, reference.COMPACT_TO_FULL_POLICY))) == 1858
    return hashlib.sha256(('\n'.join(rows) + '\n').encode()).hexdigest()


def promotion_fixtures() -> list[chess.Board]:
    roots = []
    for source_file in range(8):
        for delta in (-1, 0, 1):
            target_file = source_file + delta
            if not 0 <= target_file < 8:
                continue
            board = chess.Board(None)
            board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.H6, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(chess.square(source_file, 6), chess.Piece(chess.PAWN, chess.WHITE))
            if delta:
                board.set_piece_at(chess.square(target_file, 7), chess.Piece(chess.ROOK, chess.BLACK))
            assert board.is_valid()
            for root in (board, board.mirror()):
                for promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    move = chess.Move(chess.square(source_file, 6 if root.turn else 1),
                                      chess.square(target_file, 7 if root.turn else 0), promotion)
                    assert move in root.legal_moves, (root.fen(), move)
                roots.append(root)
    return roots


def verify(command: list[str], *, require_c: bool = False) -> dict[str, object]:
    cboard_type = None
    if require_c:
        from chess_anti_engine.encoding._lc0_ext import CBoard
        cboard_type = CBoard
    client = Client(command)
    positions = moves = reverse = c_moves = rejected = 0
    promoted: set[tuple[int, int, int, int]] = set()
    digest = hashlib.sha256()

    def reject(command: str) -> None:
        nonlocal rejected
        rows = client.sync(command)
        assert rows == ['info string policy error: invalid request or no exact legal move; root preserved', 'readyok'], rows
        rejected += 1

    def compare(board: chess.Board, roundtrip: bool = False) -> list[tuple[str, int, int, int]]:
        nonlocal positions, moves, reverse, c_moves
        assert client.sync(position(board)) == ['readyok']
        rows = client.sync('policy legal')
        expected = sorted(entry(board, m) for m in board.legal_moves)
        assert rows[0] == f'info string policy_legal {len(expected)}', rows[:1]
        assert rows[-2:] == ['info string policy_end', 'readyok']
        observed = sorted(parse_entry(row) for row in rows[1:-2])
        assert observed == expected, (board.fen(), observed, expected)
        assert len({r[2] for r in observed}) == len({r[3] for r in observed}) == len(expected)
        assert all(0 <= r[2] < 4672 and 0 <= r[3] < 1858 for r in observed)
        digest.update(json.dumps(observed, separators=(',', ':')).encode())
        positions += 1
        moves += len(observed)
        if cboard_type is not None:
            cb = cboard_type.from_board(board)
            assert sorted(map(int, cb.legal_move_indices())) == sorted(r[2] for r in observed)
            c_moves += len(observed)
        if roundtrip:
            for wanted in expected:
                uci, private, full, compact = wanted
                for query in (f'policy encode {uci}', f'policy key {private}',
                              f'policy decode az_4672 {full}', f'policy decode lc0_1858 {compact}'):
                    reply = client.sync(query)
                    assert len(reply) == 3, reply
                    assert reply[-2:] == ['info string policy_end', 'readyok'], reply
                    assert parse_entry(reply[0]) == wanted
                    reverse += 1
                move = chess.Move.from_uci(uci)
                if move.promotion:
                    promoted.add((int(board.turn), move.from_square, move.to_square, move.promotion))
                if board.is_castling(move) or board.is_en_passant(move) or move.promotion:
                    # Identical endpoints with wrong/missing flags must NOT select a move.
                    reject(f'policy key {move.from_square | (move.to_square << 6)}')
        for space, size, legal_ids in [('az_4672', 4672, {r[2] for r in expected}),
                                       ('lc0_1858', 1858, {r[3] for r in expected})]:
            absent = next(i for i in range(size) if i not in legal_ids)
            reject(f'policy decode {space} {absent}')
        assert_state(client, board)
        return observed

    try:
        table_hash = tables(client)
        for fen in FENS:
            compare(chess.Board(fen), roundtrip=True)
        for board in promotion_fixtures():
            compare(board, roundtrip=True)
        for _, board in fixtures():
            compare(board)
        # Different histories reaching one board must map to the SAME policy slots.
        history_rows = []
        for path in ('g1f3 g8f6 b1c3 b8c6', 'b1c3 b8c6 g1f3 g8f6'):
            board = chess.Board()
            for move in path.split():
                board.push_uci(move)
            history_rows.append(compare(board))
        assert history_rows[0] == history_rows[1]
        root = chess.Board()
        assert client.sync(position(root)) == ['readyok']
        for text in ['', 'legal extra', 'tables extra', 'decode', 'decode lc0_1858',
                     'decode bogus 0', 'decode az_4672 4672', 'decode lc0_1858 1858',
                     'decode lc0_1858 2047', 'decode az_4672 8191', 'decode lc0_1858 -1',
                     'decode az_4672 4294967295', 'decode lc0_1858 4294967294',
                     'decode az_4672 4294967296', 'decode lc0_1858 0 extra',
                     'key 4294967294', 'key 4294967295', 'key 4294967296', 'key -1',
                     'key 67340', 'encode e2e4q', 'encode 0000', 'encode e2e5', 'encode E2E4']:
            reject('policy ' + text)
            assert_state(client, root)
        client.send('go infinite nodes 4\npolicy legal\nisready\n')
        rows = client.until('readyok')
        assert any('busy;' in line for line in rows)
        assert not any('policy_move' in line for line in rows)
        client.send('stop\n')
        client.until('bestmove ')
        assert_state(client, root)
        assert client.sync('ucinewgame') == ['readyok']
        compare(chess.Board())
    finally:
        client.close()
    # Every file, forward/capture direction, promotion type and color is represented.
    assert len(promoted) == 176, len(promoted)
    return {'status': 'passed', 'scope': 'Bend policy maps and exact legal resolution; no neural forward',
            'full_slots': 4672, 'compact_slots': 1858, 'square_pairs': 4096,
            'scalar_table_values': 2 * 4672 + 2 * 1858 + 4096,
            'table_sha256': table_hash, 'position_fixtures': positions,
            'legal_move_comparisons': moves, 'c_legal_move_comparisons': c_moves,
            'exact_resolution_checks': reverse, 'rejected_requests': rejected,
            'distinct_promotion_moves': len(promoted), 'history_independence': True,
            'ordered_legal_sha256': digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-c', action='store_true')
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
