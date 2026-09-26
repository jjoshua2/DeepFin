"""External oracle for the native Bend logit boundary; no engine dependency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def verify(command: list[str]) -> dict[str, object]:
    p = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
    rows = [list(map(int, line.split())) for line in p.stdout.splitlines()]
    assert len(rows) == 11
    def softmax(values: list[float]) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        exps = np.exp(x - x.max())
        return (exps / exps.sum()).astype(np.float32)
    observed = []
    for mode, row in enumerate(rows):
        assert len(row) == 8
        assert row[0] == mode
        x = np.asarray(row[3:], dtype=np.uint32).view(np.float32)
        assert np.isfinite(x).all()
        if mode < 3:
            assert row[1:3] == [0, 2]
            wdl = softmax([-2.0, 0.0, 2.0]) if mode < 2 else softmax([0.0, 0.0, 0.0])
            policy = softmax([-2.0, 2.0]) if mode < 2 else np.array([0.0, 1.0], dtype=np.float32)
            np.testing.assert_allclose(x[:3], wdl, atol=2e-7, rtol=3e-6)
            np.testing.assert_allclose(x[3:], policy, atol=2e-7, rtol=3e-6)
            observed.append(x)
        else:
            assert row[1:3] == [1, 0]
            np.testing.assert_array_equal(x, np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(observed[0].view(np.uint32), observed[1].view(np.uint32))
    return {'status': 'passed', 'cases': 11, 'accepted_cases': 3, 'rejected_cases': 8,
            'covers': ['legal-only softmax despite huge illegal logits', 'constant-shift invariance',
                       'extreme finite inputs', 'NaN', 'positive and negative infinity',
                       'short and long outputs', 'out-of-range policy slot', 'wrong move flags',
                       'empty legal mapping'], 'rows': rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    if not args.command:
        p.error('--command requires executable')
    result = verify(args.command)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
