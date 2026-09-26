"""Opt-in real-model reuse/quiet-mode checks against a diagnostic reference report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

from .verify import Client
from .verify_neural import fixtures
from .verify_rules import position


_AUDIT_PATTERN = r'native-buffer-audit calls=(\d+) input_changes=(\d+) output_changes=(\d+) input_tensor_allocations=(\d+)'


class AuditClient(Client):
    """Keep the reference client's shutdown checks, allowing only one audit line."""

    def close(self) -> None:
        try:
            super().close()
        except AssertionError:
            # The inherited client asserts empty stderr after shutdown/join.
            # This opt-in test deliberately requests one structured audit record;
            # nonzero exit or any additional stderr must still fail.
            if (self.proc.returncode != 0 or len(self.errors) != 1
                    or re.fullmatch(_AUDIT_PATTERN, self.errors[0].rstrip('\n')) is None):
                raise


def verify(command: list[str], package: Path, reference: Path) -> dict[str, object]:
    expected = json.loads(reference.read_text())['results'][:14]
    roots = fixtures()
    # Repeat A after unrelated B, invalid position and ucinewgame; the same cache
    # lives throughout. Include terminal, castling, EP, promotion and both histories.
    order = [0, 1, 0, *range(2, 14), 0]
    env = os.environ.copy()
    env['DEEPFIN_BEND_MODEL_PACKAGE'] = str(package.resolve())
    env['DEEPFIN_BEND_BUFFER_AUDIT'] = '1'
    env.pop('DEEPFIN_BEND_NATIVE_DIAGNOSTICS', None)
    with tempfile.TemporaryDirectory(prefix='deepfin-reuse-') as tmp:
        raw = []
        results = []
        traces = []
        for diag in [None, '1']:
            trace = Path(tmp) / ('quiet.bin' if diag is None else 'diagnostic.bin')
            run_env = {**env, 'DEEPFIN_BEND_MODEL_TRACE': str(trace)}
            if diag is not None:
                run_env['DEEPFIN_BEND_NATIVE_DIAGNOSTICS'] = diag
            with patch.dict(os.environ, run_env, clear=True):
                client = AuditClient(command)
            last = 0
            try:
                for n, i in enumerate(order):
                    assert client.sync(position(roots[i])) == ['readyok']
                    if n == 2:
                        invalid = client.sync('position startpos moves e2e5\n')
                        assert any('invalid position; previous root preserved' in line for line in invalid)
                    client.send('go nodes 4 depth 2\n')
                    lines = client.until('bestmove ', timeout=60)
                    metrics = [json.loads(s.removeprefix('info string neural_work ')) for s in lines
                               if s.startswith('info string neural_work ')]
                    assert len(metrics) == 1
                    m = metrics[0]
                    want = expected[i]
                    assert lines[-1] == 'bestmove ' + want['bestmove']
                    assert m['accepted_neural_rows'] == m['executed_real_rows'] == m['forward_calls'] == want['native_calls']
                    assert m['completed_simulations'] == want['completed']
                    count = sum(s.startswith('info string native_reply ') for s in lines)
                    assert count == (0 if diag is None else want['native_calls'])
                    assert sum(s.startswith('info string native_path ') for s in lines) == count
                    last += want['native_calls']
                    results.append({'diagnostics': diag is not None, 'fixture': i,
                                    'bestmove': want['bestmove'], 'native_calls': want['native_calls']})
                    if n == 2:
                        assert client.sync('ucinewgame\n') == ['readyok']
            finally:
                client.close()
            assert client.proc.returncode == 0
            stderr = ''.join(client.errors)
            audit = re.findall(_AUDIT_PATTERN, stderr)
            assert audit == [(str(last), '0', '0', '1')], stderr
            raw.append({'calls': last, 'input_changes': 0, 'output_changes': 0, 'input_tensor_allocations': 1})
            traces.append(trace.read_bytes())
        assert traces[0] == traces[1], 'quiet mode changed actual input/output bytes'
    invalid_env = {**env, 'DEEPFIN_BEND_NATIVE_DIAGNOSTICS': 'invalid'}
    invalid_env.pop('DEEPFIN_BEND_MODEL_TRACE', None)
    bad = subprocess.run(command, input='uci\n', text=True, capture_output=True, env=invalid_env, timeout=30, check=False)
    assert bad.returncode == 2
    assert 'uciok' not in bad.stdout
    return {'status': 'passed', 'searches': len(results), 'audits': raw,
            'quiet_diagnostic_traces_bit_exact': True, 'invalid_diagnostics_rejected': True,
            'results': results, 'scope': 'real CPU fixture functional reuse; no speed or strength conclusion'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    report = verify(args.command, args.package, args.reference)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
