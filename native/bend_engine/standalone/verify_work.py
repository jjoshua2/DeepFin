"""Opt-in runtime checks for a compiled work_probe and the actual UCI engine.

No model required; use verify_neural.py separately for actual tensor/forward
trace reconciliation. Never imported by the engine or ordinary pytest.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from native.bend_engine.standalone.verify import Client
from scripts.bench_neural_search import PREFIX, parse_report


def verify_probe(binary: Path) -> None:
    result = subprocess.run([str(binary.resolve()), '--threads', '1'], capture_output=True,
                            text=True, timeout=30, check=True)
    assert not result.stderr
    rows = [json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines()]
    assert len(rows) == 2
    first, second = rows
    assert first['executed_real_rows'] == 2
    assert first['accepted_neural_rows'] == first['rejected_rows'] == first['completed_simulations'] == 1
    assert first['useful_eps'] == 1
    assert first['executed_eps'] == 2
    assert first['phase_seconds']['encoding'] == .008
    assert first['phase_seconds']['backup'] == .014
    assert second['completed_simulations'] == 4
    assert second['executed_real_rows'] == second['accepted_neural_rows'] == 0
    assert second['useful_eps'] is None
    assert all(value is None for value in second['phase_seconds'].values())


def verify_engine(command: list[str]) -> None:
    client = Client(command)
    try:
        for profile in ('', ' profile'):
            assert client.sync('position startpos') == ['readyok']
            client.send('go nodes 4 depth 2' + profile + '\n')
            report = parse_report(client.until('bestmove '), kind='evals', budget=4)
            assert report['counters']['completed_simulations'] == 4
            assert report['counters']['executed_real_rows'] == 0  # material is never NN work
            assert not report['comparable']
            assert (report['counters']['phase_seconds']['backup'] is not None) == bool(profile)
        # A completed infinite search holds its final metrics with bestmove.
        # No extra output may leak into an intervening isready response.
        client.send('go infinite nodes 1\n')
        time.sleep(.05)
        client.send('isready\n')
        assert client.until('readyok') == ['readyok']
        client.send('stop\nstop\nisready\n')
        held = parse_report(client.until('readyok'), kind='evals', budget=1)
        assert held['counters']['completed_simulations'] == 1
        assert held['counters']['executed_real_rows'] == 0
        # An eval-only request on the material product cannot fake an NN budget.
        client.send('go evals 3\n')
        report = parse_report(client.until('bestmove '), kind='evals', budget=3)
        assert not report['comparable']
        assert report['counters']['completed_simulations'] == report['counters']['executed_real_rows'] == 0
        for command_text in ('go evals 0', 'go evals 3 evals 4', 'go profile profile', 'go evals'):
            reply = client.sync(command_text)
            assert any('unsupported or invalid go' in line for line in reply), reply
        # A profile does not change deterministic material move selection.
        assert client.sync('position startpos') == ['readyok']
        client.send('go nodes 4\n')
        baseline = client.until('bestmove ')[-1]
        client.send('go nodes 4 profile\n')
        assert client.until('bestmove ')[-1] == baseline
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    verify_probe(args.probe)
    verify_engine(args.command)
    print(json.dumps({'status': 'passed', 'scope': 'compiled accounting and material UCI wiring'}))


if __name__ == '__main__':
    main()
