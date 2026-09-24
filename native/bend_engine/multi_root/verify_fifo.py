"""Exact list/FIFO scheduler equivalence with a deterministic native callback.

This checks chronological dispatch/reply traces, all final trees and all non-time
accounting. It is not a model, benchmark or arbitrary-interleaving proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .verify import environment, parse

TIME_FIELDS = frozenset(('wall_seconds', 'useful_eps', 'executed_eps',
                         'gathering_seconds', 'backend_and_transport_seconds',
                         'normalization_and_backup_seconds'))
POSITIONS = (
    'startpos', 'startpos moves e2e4',
    'fen k7/1Q6/2K5/8/8/8/8/8 b - - 150 1',
    'startpos moves g1f3 g8f6 b1c3 b8c6',
    'startpos moves b1c3 b8c6 g1f3 g8f6',
    'fen 4k3/8/8/8/8/8/8/R3K3 w - - 150 1',
)
CASES = (
    ('single', POSITIONS[:1], 4, 2, 0),
    ('mixed-three', (POSITIONS[2], POSITIONS[0], POSITIONS[1]), 4, 2, 0),
    ('mixed-sixteen', tuple(POSITIONS[i % len(POSITIONS)] for i in range(16)), 4, 2, 0),
    ('neural-cap', (POSITIONS[0], POSITIONS[3], POSITIONS[4]), 256, 32, 3),
    ('terminal-only', (POSITIONS[2], POSITIONS[5]), 4, 2, 3),
)


def semantic_view(parsed: dict[str, Any]) -> dict[str, Any]:
    """Only explicitly measured time values are excluded; new fields stay visible."""
    return {'root_order': list(parsed['roots']), 'roots': parsed['roots'],
            'nodes': parsed['nodes'], 'events': parsed['events'],
            'work': {k: v for k, v in parsed['work'].items() if k not in TIME_FIELDS}}


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> str:
    left, right = semantic_view(reference), semantic_view(candidate)
    if left != right:
        differing = [key for key in left if left[key] != right[key]]
        raise ValueError('FIFO scheduler differs: ' + ', '.join(differing))
    return hashlib.sha256(json.dumps(right, sort_keys=True).encode()).hexdigest()


def probe_expected() -> str:
    lines = []
    for size in (0, 1, 3, 16):
        for batch in (0, 1, 2, 4, 16, 17):
            count = min(size, batch)
            order = list(range(1, size + 1))
            for _ in range(7):
                order = order[count:] + order[:count]
            for mask in (0, 2, 8, 65536, 65546):
                live = 1  # Preserve the caller's accumulated bit too.
                for root in order:
                    if not mask & (1 << root):
                        live |= 1 << root
                lines.extend((f'case {size} {batch} {mask}', f'state {live} {size}'))
                lines.extend(f'root {root} 0 4096 1 {2 if mask & (1 << root) else 0} 0' for root in order)
    return '\n'.join(lines) + '\n'


def check_probe(text: str) -> None:
    if text != probe_expected():
        raise ValueError('FIFO adapter order, mask or ownership-state mismatch')


def run(binary: Path, positions: tuple[str, ...], batch: int, sims: int, depth: int,
        budget: int, asynchronous: bool) -> dict[str, Any]:
    env = {**environment(), 'DEEPFIN_COHORT_ASYNC': str(int(asynchronous))}
    command = [str(binary.resolve()), '--threads', '1', '--',
               str(sims), str(depth), str(budget), '1', *positions]
    result = subprocess.run(command, env=env, input='', text=True, capture_output=True,
                            timeout=90, check=False)
    if result.returncode or result.stderr:
        raise RuntimeError(f'scheduler exited {result.returncode}: {result.stderr}')
    return parse(result.stdout, len(positions), batch, sims, budget, True,
                 asynchronous=asynchronous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    observations = []
    report: dict[str, Any] = {'status': 'failed', 'batch': args.batch,
                              'scope': 'deterministic full-scheduler semantic equivalence',
                              'observations': observations}
    try:
        for asynchronous in (False, True):
            for name, positions, sims, depth, budget in CASES:
                reference = run(args.reference, positions, args.batch, sims, depth, budget, asynchronous)
                candidate = run(args.candidate, positions, args.batch, sims, depth, budget, asynchronous)
                digest = compare(reference, candidate)
                observations.append({'case': name, 'asynchronous': asynchronous,
                                     'roots': len(positions), 'semantic_sha256': digest,
                                     'events': len(candidate['events']),
                                     'nodes': sum(len(nodes) for nodes in candidate['nodes'].values())})
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
