"""Benchmark separate versus coalesced loss scalar materialization."""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def _reference(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = [key for key in losses if key != "total"]
    values = torch.stack([losses[key] for key in keys]).tolist()
    out = dict(zip(keys, values, strict=True))
    out["loss"] = float(losses["total"].item())
    return out


def _coalesced(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = list(losses)
    values = torch.stack([losses[key] for key in keys]).tolist()
    out = {
        ("loss" if key == "total" else key): value
        for key, value in zip(keys, values, strict=True)
    }
    return out


def _time(fn, losses: dict[str, torch.Tensor], iterations: int) -> tuple[float, float]:
    checksum = 0.0
    start = time.perf_counter()
    for _ in range(iterations):
        checksum += sum(fn(losses).values())
    return time.perf_counter() - start, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    losses = {
        **{f"component_{i}": torch.tensor(i / 17.0) for i in range(args.components)},
        "total": torch.tensor(3.25),
    }

    samples = {"reference": [], "coalesced": []}
    checksums: set[float] = set()
    for round_idx in range(args.rounds + 2):
        order = (
            ("reference", _reference), ("coalesced", _coalesced)
        ) if round_idx % 2 == 0 else (
            ("coalesced", _coalesced), ("reference", _reference)
        )
        for name, fn in order:
            elapsed, checksum = _time(fn, losses, args.iterations)
            checksums.add(checksum)
            if round_idx >= 2:
                samples[name].append(elapsed)

    if len(checksums) != 1:
        raise AssertionError(f"checksum mismatch: {checksums}")
    reference = statistics.median(samples["reference"])
    coalesced = statistics.median(samples["coalesced"])
    print(f"reference_seconds={reference:.9f}")
    print(f"coalesced_seconds={coalesced:.9f}")
    print(f"ratio={coalesced / reference:.6f}")
    print(f"checksum={next(iter(checksums)):.9f}")


if __name__ == "__main__":
    main()
