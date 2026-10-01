"""Actual native UCI session-resource qualification; Python is an external client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine

from .verify import Client, assert_state
from .verify_time_control import require_assertions


def result(c: Client, limits: str, capacity: int, milliseconds: int = 0) -> dict[str, Any]:
    c.send('go ' + limits + '\n')
    rows = c.until('bestmove ')
    resources = [json.loads(row.removeprefix('info string tree_resources '))
                 for row in rows if row.startswith('info string tree_resources ')]
    work = [json.loads(row.removeprefix('info string neural_work '))
            for row in rows if row.startswith('info string neural_work ')]
    assert len(resources) == len(work) == 1, rows
    assert resources[0]['capacity'] == capacity, rows
    assert 1 <= resources[0]['used'] <= capacity, rows
    assert work[0]['movetime_ms'] == milliseconds, rows
    assert chess.Move.from_uci(rows[-1].split()[1]) in chess.Board().legal_moves, rows
    assert_state(c, chess.Board())
    return {'limits': limits, 'resources': resources[0], 'work': work[0]}


def verify(command: list[str]) -> dict[str, Any]:
    require_assertions()
    c = Client(command)
    observations: list[dict[str, Any]] = []
    rejected = 0
    try:
        rows = c.sync('uci')
        assert 'option name TreeNodes type spin default 4096 min 4096 max 65536' in rows
        assert 'option name Move Overhead type spin default 50 min 0 max 5000' in rows
        observations.append(result(c, 'nodes 2', 4096))
        for capacity in (8193, 65536, 4096):
            rows = c.sync(f'setoption name TreeNodes value {capacity}')
            assert any(f'TreeNodes={capacity}' in row for row in rows), rows
            observations.append(result(c, 'nodes 2', capacity))
        c.sync('setoption name TreeNodes value 8193')
        c.sync('setoption name Move Overhead value 75')
        c.sync('position startpos moves e2e4')
        c.sync('ucinewgame')
        observations.append(result(c, 'wtime 100 movestogo 1 nodes 2', 8193, 25))
        prior = c.dump()
        for text in (
                'setoption', 'setoption name', 'setoption name TreeNodes',
                'setoption name TreeNodes value', 'setoption name TreeNodes value 4095',
                'setoption name TreeNodes value 65537', 'setoption name TreeNodes value -1',
                'setoption name TreeNodes value 4294967296', 'setoption name TreeNodes value 8192 extra',
                'setoption name Move Overhead value -1', 'setoption name Move Overhead value 5001',
                'setoption name Move Overhead value 4294967296', 'setoption name Unknown value 1',
                'setoption name Threads value 2', 'setoption name Ponder value true'):
            rows = c.sync(text)
            assert any('invalid setoption' in row for row in rows), rows
            assert c.dump() == prior
            observations.append(result(c, 'wtime 100 movestogo 1 nodes 2', 8193, 25))
            rejected += 1
        # A queued option cannot alter the current Running search.
        c.send('go nodes 1 infinite\nsetoption name TreeNodes value 65536\nisready\n')
        rows = c.until('readyok')
        assert any('busy' in row for row in rows), rows
        assert not any(row.startswith('bestmove ') for row in rows), rows
        c.send('stop\nstop\nisready\n')
        rows = c.until('readyok')
        assert sum(row.startswith('bestmove ') for row in rows) == 1, rows
        observations.append(result(c, 'wtime 100 movestogo 1 nodes 2', 8193, 25))
        for overhead, remaining, expected in ((0, 100, 100), (100, 100, 0),
                                               (5000, 5001, 1), (5000, 5000, 0)):
            c.sync(f'setoption name Move Overhead value {overhead}')
            report = result(c, f'wtime {remaining} movestogo 1 nodes 2', 8193, expected)
            if expected == 0:
                assert report['work']['completed_simulations'] == 0
                assert report['work']['dispatched_real_rows'] == 0
            observations.append(report)
        observations.append(result(c, 'wtime 100 movetime 7 nodes 2', 8193, 7))
        c.sync('setoption name Move Overhead value 50')
        # At depth one the bounded tree can finish >256 simulations without
        # exhausting its arena; this proves the new explicit bound is effective.
        report = result(c, 'nodes 257 depth 1', 8193)
        assert report['work']['completed_simulations'] == 257, report
        observations.append(report)
        report = result(c, 'wtime 0 nodes 65536', 8193)
        assert report['work']['completed_simulations'] == 0, report
        observations.append(report)
        rows = c.sync('go nodes 65537')
        assert any('invalid go' in row for row in rows), rows
    finally:
        c.close()
    with chess.engine.SimpleEngine.popen_uci(command, timeout=10) as engine:
        engine.configure({'TreeNodes': 8193, 'Move Overhead': 0})
        played = engine.play(chess.Board(), chess.engine.Limit(nodes=257, depth=1))
        assert played.move is not None and played.move in chess.Board().legal_moves
        engine.ping()
    return {'schema': 'deepfin.bend-uci-resources.v1', 'qualified': True,
            'rejected_options': rejected, 'observations': observations,
            'standard_client': True,
            'scope': 'native session resources; no trained-model or performance claim'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'qualified': False, 'status': 'started'}) + '\n')
    report = verify(args.command)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
