"""Measure the throughput overhead of delayed-GIL-reacquisition telemetry."""
from __future__ import annotations

import argparse
import statistics
import time

from chess_anti_engine.utils.gil_probe import GilContentionProbe


def _work(iterations: int) -> int:
    value = 0x1234_5678_9ABC_DEF0
    for i in range(iterations):
        value ^= (value << 7) & 0xFFFF_FFFF_FFFF_FFFF
        value ^= value >> 9
        value = (value + i * 0x9E37_79B1) & 0xFFFF_FFFF_FFFF_FFFF
    return value


def _calibrate(seconds: float) -> int:
    iterations = 100_000
    while True:
        start = time.perf_counter()
        _work(iterations)
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            return iterations
        iterations = max(iterations + 1, int(iterations * seconds / max(elapsed, 1e-9) * 1.05))


def _timed(iterations: int, probe: GilContentionProbe | None) -> tuple[float, int, float]:
    if probe is not None:
        probe.reset()
    start = time.perf_counter()
    digest = _work(iterations)
    elapsed = time.perf_counter() - start
    sample_rate = 0.0 if probe is None else probe.read_and_reset().sample_rate
    return elapsed, digest, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--interval-ms", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.rounds <= 0 or args.interval_ms <= 0.0:
        raise SystemExit("all arguments must be positive")

    iterations = _calibrate(float(args.seconds))
    control: list[float] = []
    probed: list[float] = []
    hashes: set[int] = set()
    sample_rates: list[float] = []
    interval_s = float(args.interval_ms) / 1000.0
    for round_idx in range(int(args.rounds)):
        order = (False, True) if round_idx % 2 == 0 else (True, False)
        for enabled in order:
            if enabled:
                # Construct outside the timed region, but close it before the
                # control run so the baseline truly has no probe thread.
                with GilContentionProbe(interval_s=interval_s) as probe:
                    elapsed, digest, sample_rate = _timed(iterations, probe)
            else:
                elapsed, digest, sample_rate = _timed(iterations, None)
            hashes.add(digest)
            throughput = iterations / elapsed
            (probed if enabled else control).append(throughput)
            if enabled:
                sample_rates.append(sample_rate)

    control_median = statistics.median(control)
    probed_median = statistics.median(probed)
    ratio = probed_median / control_median
    sample_rate = statistics.median(sample_rates)
    print(f"iterations={iterations} rounds={args.rounds}")
    print(f"control_iterations_s={control_median:.1f}")
    print(f"probed_iterations_s={probed_median:.1f}")
    print(f"throughput_ratio={ratio:.6f}")
    print(f"median_probe_samples_s={sample_rate:.1f}")
    print(f"hashes_stable={len(hashes) == 1}")


if __name__ == "__main__":
    main()
