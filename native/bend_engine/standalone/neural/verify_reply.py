"""External softmax/corrupt-output oracle for the actual compiled Bend module."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def verify(command: list[str]) -> dict[str, object]:
    run = subprocess.run(command, check=True, text=True, capture_output=True, timeout=30)
    assert not run.stderr, run.stderr
    rows = run.stdout.splitlines()
    assert len(rows) == 18
    maximum_error = 0.0
    rejected = 0
    for case, row in enumerate(rows):
        fields = row.split()
        assert fields[0] == 'reply_case'
        values = list(map(int, fields[1:]))
        mode, epoch, request, node, status, w, d, l, count, *priors = values
        assert (mode, epoch, request, node) == (case, 9, 13, 31)
        assert len(priors) == count
        valid = case in (0, 1, 2, 3, 4, 14, 15)
        if not valid:
            assert status == 1 and count == 0
            rejected += 1
            continue
        assert status == 0
        raw = np.zeros(1861, dtype=np.float32)
        raw[[0, 1857, 1858, 1859, 1860]] = [-1, 2, 1, 2, -3]
        if case == 1:
            raw[1] = 1e20  # Illegal giant logit must not affect legal normalization.
        elif case == 2:
            raw.fill(-3e38)
        elif case == 3:
            raw.fill(3e38)
            raw[0] = -3e38
        elif case == 4:
            raw.fill(0)
        slots = list(range(256)) if case == 15 else ([0] if case == 14 else [0, 1857])
        assert count == len(slots)
        for bits, logits in (([w, d, l], raw[1858:]), (priors, raw[slots])):
            actual = np.asarray(bits, dtype=np.uint32).view(np.float32)
            delta = logits.astype(np.float64) - float(logits.max())
            e = np.exp(delta)
            expected = e / e.sum()
            assert np.isfinite(actual).all() and (actual >= 0).all()
            np.testing.assert_allclose(actual, expected, atol=6e-7, rtol=2e-5)
            maximum_error = max(maximum_error, float(np.abs(actual - expected).max()))
    return {'status': 'passed', 'cases': len(rows), 'rejected': rejected,
            'max_absolute_probability_error': maximum_error,
            'scope': 'actual Bend legal-softmax/WDL and malformed-output contracts, no model forward'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    report = verify(args.command)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
