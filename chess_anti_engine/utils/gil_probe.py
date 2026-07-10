"""Low-overhead delayed-GIL-reacquisition telemetry.

The probe sleeps on a daemon thread (releasing the GIL), then measures how late
Python resumes after the requested wake time. The delay is an upper bound on GIL
contention because it also includes OS timer and scheduler latency. It is useful
when paired with phase/thread wall-time telemetry, but is not an exact GIL-wait
percentage.
"""
from __future__ import annotations

import bisect
import math
import threading
import time
from dataclasses import dataclass

_BUCKET_UPPER_MS = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, math.inf)


@dataclass(frozen=True, slots=True)
class GilDelayStats:
    samples: int
    elapsed_s: float
    sample_rate: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    over_1ms_pct: float
    over_5ms_pct: float


def _bucket_percentile(counts: tuple[int, ...], samples: int, q: float, max_ms: float) -> float:
    if samples <= 0:
        return 0.0
    target = max(1, math.ceil(float(q) * samples))
    cumulative = 0
    for count, upper in zip(counts, _BUCKET_UPPER_MS, strict=True):
        cumulative += int(count)
        if cumulative >= target:
            return float(max_ms if math.isinf(upper) else upper)
    return float(max_ms)


class GilContentionProbe:
    """Sample delayed interpreter reacquisition from one daemon thread."""

    def __init__(self, *, interval_s: float = 0.010) -> None:
        self.interval_s = max(0.001, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._counts = [0] * len(_BUCKET_UPPER_MS)
        self._samples = 0
        self._sum_ms = 0.0
        self._max_ms = 0.0
        self._reset_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="gil-delay-probe",
        )
        self._thread.start()

    def _record_delay(self, delay_s: float) -> None:
        delay_ms = max(0.0, float(delay_s) * 1000.0)
        bucket = bisect.bisect_left(_BUCKET_UPPER_MS, delay_ms)
        bucket = min(bucket, len(_BUCKET_UPPER_MS) - 1)
        with self._lock:
            self._counts[bucket] += 1
            self._samples += 1
            self._sum_ms += delay_ms
            self._max_ms = max(self._max_ms, delay_ms)

    def _run(self) -> None:
        target = time.perf_counter() + self.interval_s
        while not self._stop.wait(self.interval_s):
            observed = time.perf_counter()
            self._record_delay(observed - target)
            target = observed + self.interval_s

    def reset(self) -> None:
        with self._lock:
            self._counts = [0] * len(_BUCKET_UPPER_MS)
            self._samples = 0
            self._sum_ms = 0.0
            self._max_ms = 0.0
            self._reset_at = time.perf_counter()

    def read_and_reset(self) -> GilDelayStats:
        now = time.perf_counter()
        with self._lock:
            counts = tuple(self._counts)
            samples = int(self._samples)
            total_ms = float(self._sum_ms)
            max_ms = float(self._max_ms)
            elapsed_s = max(0.0, now - self._reset_at)
            self._counts = [0] * len(_BUCKET_UPPER_MS)
            self._samples = 0
            self._sum_ms = 0.0
            self._max_ms = 0.0
            self._reset_at = now
        over_1 = sum(counts[bisect.bisect_right(_BUCKET_UPPER_MS, 1.0):])
        over_5 = sum(counts[bisect.bisect_right(_BUCKET_UPPER_MS, 5.0):])
        return GilDelayStats(
            samples=samples,
            elapsed_s=elapsed_s,
            sample_rate=float(samples / max(elapsed_s, 1e-9)),
            mean_ms=float(total_ms / max(samples, 1)),
            p50_ms=_bucket_percentile(counts, samples, 0.50, max_ms),
            p95_ms=_bucket_percentile(counts, samples, 0.95, max_ms),
            p99_ms=_bucket_percentile(counts, samples, 0.99, max_ms),
            max_ms=max_ms,
            over_1ms_pct=float(100.0 * over_1 / max(samples, 1)),
            over_5ms_pct=float(100.0 * over_5 / max(samples, 1)),
        )

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, 4.0 * self.interval_s))

    def __enter__(self) -> GilContentionProbe:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
