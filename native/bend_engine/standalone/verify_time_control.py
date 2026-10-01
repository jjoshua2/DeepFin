"""External UCI clock qualification of the actual native Bend engine.

Python is an oracle and standard UCI client only. It does not provide the engine
with parsed clocks, time allocations, legal moves, features or search results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.engine

from .verify import Client, assert_state


def search(c: Client, board: chess.Board, limits: str, expected_ms: int,
           *, zero_work: bool = False) -> dict[str, object]:
    c.send('go ' + limits + '\n')
    rows = c.until('bestmove ')
    records = [json.loads(row.removeprefix('info string neural_work '))
               for row in rows if row.startswith('info string neural_work ')]
    assert len(records) == 1, rows
    record = records[0]
    assert record['movetime_ms'] == expected_ms, rows
    if zero_work:
        assert record['completed_simulations'] == 0, rows
        assert record['dispatched_real_rows'] == 0, rows
        assert any('unsearched legal fallback' in row for row in rows), rows
    move = chess.Move.from_uci(rows[-1].split()[1])
    assert move in board.legal_moves, (board.fen(), rows)
    assert_state(c, board)
    return {'limits': limits, 'allocated_ms': expected_ms,
            'completed_simulations': record['completed_simulations'], 'move': move.uci()}


def verify(command: list[str]) -> dict[str, object]:
    c = Client(command)
    checks: list[dict[str, object]] = []
    invalid = 0
    try:
        white = chess.Board()
        black = white.copy()
        black.push_uci('e2e4')
        cases = [
            (white, 'wtime 1000 btime 2000 winc 100 binc 200 movestogo 10 nodes 2', 175),
            (black, 'wtime 1000 btime 2000 winc 100 binc 200 movestogo 10 nodes 2', 350),
            (white, 'nodes 2 wtime 1000', 33),
            (black, 'btime 1000 nodes 2', 33),
            (white, 'wtime 500 movestogo 1000 nodes 2', 1),
            (white, 'wtime 86400000 winc 86400000 movestogo 1 nodes 2', 60000),
            (white, 'wtime 1000 movetime 7 nodes 2', 7),
            (white, 'movetime 7 wtime 1000 nodes 2', 7),
            (white, 'wtime 0 btime 1000 nodes 256', 0),
            (white, 'wtime 50 winc 86400000 nodes 256', 0),
            (black, 'btime 0 wtime 1000 nodes 256', 0),
        ]
        for board, limits, ms in cases:
            moves = ' '.join(move.uci() for move in board.move_stack)
            c.sync('position startpos' + (' moves ' + moves if moves else ''))
            checks.append(search(c, board, limits, ms, zero_work=ms == 0))
        c.sync('position startpos moves g1f3 g8f6 f3g1 f6g8')
        prior = c.dump()
        for limits in (
                'wtime', 'wtime -1', 'wtime 4294967296', 'wtime 86400001',
                'wtime 1000 wtime 2000', 'wtime 1000 winc 86400001',
                'wtime 1000 movestogo 0', 'wtime 1000 movestogo 1001',
                'wtime 1000 movestogo 1 movestogo 2', 'btime 1000', 'winc 100',
                'wtime 1000 infinite', 'infinite wtime 1000',
                'nodes wtime 1000 4', 'movetime wtime 1000 4',
                'wtime nodes 4 1000', 'wtime 1000 nodes', 'wtime 1000 nonsense'):
            rows = c.sync('go ' + limits)
            assert any('invalid go' in row for row in rows), rows
            assert not any(row.startswith('bestmove ') for row in rows), rows
            assert c.dump() == prior
            invalid += 1
        # isready and stop still operate during a clocked search. Consume exactly
        # one decision whether the bounded tree finished before stop or after it.
        c.sync('ucinewgame')
        c.send('go wtime 86400000 btime 86400000\nisready\nstop\nstop\nisready\n')
        rows = c.until('readyok') + c.until('readyok')
        assert sum(row.startswith('bestmove ') for row in rows) == 1, rows
        assert_state(c, white)
    finally:
        c.close()

    moves: list[str] = []
    with chess.engine.SimpleEngine.popen_uci(command, timeout=10) as engine:
        board = chess.Board()
        for _ in range(8):
            result = engine.play(board, chess.engine.Limit(
                white_clock=.2, black_clock=.2, white_inc=.01,
                black_inc=.02, remaining_moves=10))
            assert result.move is not None and result.move in board.legal_moves
            moves.append(result.move.uci())
            board.push(result.move)
        engine.ping()
    return {'schema': 'deepfin.bend-uci-clocks.v1', 'qualified': True,
            'allocation_cases': checks, 'rejected_transactions': invalid,
            'clocked_ready_stop_exactly_once': True, 'real_uci_client_moves': moves,
            'scope': 'native material UCI clocks; no trained-model, CUDA or hard-latency claim'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'qualified': False, 'status': 'started'}) + '\n')
    result = verify(args.command)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
