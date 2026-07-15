"""Benchmark the redundant selected-row copy in replay input mirroring."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import numpy as np


def _mirror(x: np.ndarray, mask: np.ndarray, *, extra_copy: bool) -> np.ndarray:
    out = np.array(x, copy=True, order="C")
    selected = out[mask, :, :, ::-1]
    out[mask] = selected.copy() if extra_copy else selected
    rows = np.flatnonzero(mask)
    for a, b in ((104, 105), (106, 107)):
        tmp = out[rows, a, :, :].copy()
        out[rows, a, :, :] = out[rows, b, :, :]
        out[rows, b, :, :] = tmp
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--planes", type=int, default=175)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    rng = np.random.default_rng(20260715)
    x = rng.normal(size=(args.rows, args.planes, 8, 8)).astype(np.float16)
    mask = rng.random(args.rows) < 0.5

    samples = {"reference": [], "candidate": []}
    checksums = {"reference": set(), "candidate": set()}
    for round_idx in range(args.rounds + 2):
        order = (
            ("reference", True), ("candidate", False)
        ) if round_idx % 2 == 0 else (
            ("candidate", False), ("reference", True)
        )
        for name, extra_copy in order:
            start = time.perf_counter()
            result = x
            for _ in range(args.iterations):
                result = _mirror(x, mask, extra_copy=extra_copy)
            elapsed = time.perf_counter() - start
            if not result.flags.c_contiguous or any(stride <= 0 for stride in result.strides):
                raise AssertionError(f"invalid output layout: {result.strides}")
            checksums[name].add(hashlib.sha256(result.tobytes()).hexdigest())
            if round_idx >= 2:
                samples[name].append(elapsed)

    if len(checksums["reference"] | checksums["candidate"]) != 1:
        raise AssertionError(f"checksum mismatch: {checksums}")
    reference = statistics.median(samples["reference"])
    candidate = statistics.median(samples["candidate"])
    print(f"reference_seconds={reference:.9f}")
    print(f"candidate_seconds={candidate:.9f}")
    print(f"ratio={candidate / reference:.6f}")
    print(f"checksum={next(iter(checksums['reference']))}")


if __name__ == "__main__":
    main()
