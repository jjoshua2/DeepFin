"""Benchmark replay policy mirroring with and without float32 widening."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import numpy as np

from chess_anti_engine.moves.encode import (
    COMPACT_MIRROR_POLICY_INV,
    MIRROR_POLICY_INV,
    mirror_policy_batch,
)


def _reference(arrays: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    out = []
    for src in arrays:
        dst = np.array(src, copy=True, order="C")
        mirrored = mirror_policy_batch(dst[mask])
        dst[mask] = mirrored.astype(src.dtype, copy=False)
        out.append(dst)
    return out


def _candidate(arrays: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    out = []
    for src in arrays:
        dst = np.array(src, copy=True, order="C")
        mirror_map = COMPACT_MIRROR_POLICY_INV if src.shape[1] == 1858 else MIRROR_POLICY_INV
        dst[mask] = dst[mask][:, mirror_map]
        out.append(dst)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--width", type=int, default=1858)
    parser.add_argument("--fields", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    rng = np.random.default_rng(20260715)
    arrays = [
        rng.normal(size=(args.rows, args.width)).astype(np.float16)
        for _ in range(args.fields)
    ]
    mask = rng.random(args.rows) < 0.5

    samples = {"reference": [], "candidate": []}
    checksums = {"reference": set(), "candidate": set()}
    for round_idx in range(args.rounds + 2):
        order = (
            ("reference", _reference), ("candidate", _candidate)
        ) if round_idx % 2 == 0 else (
            ("candidate", _candidate), ("reference", _reference)
        )
        for name, fn in order:
            start = time.perf_counter()
            result = []
            for _ in range(args.iterations):
                result = fn(arrays, mask)
            elapsed = time.perf_counter() - start
            digest = hashlib.sha256()
            for value in result:
                digest.update(value.tobytes())
                digest.update(value.dtype.str.encode())
            checksums[name].add(digest.hexdigest())
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
