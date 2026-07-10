from __future__ import annotations

import time

import pytest

from chess_anti_engine.utils.gil_probe import GilContentionProbe


def test_gil_probe_summarizes_recorded_delays() -> None:
    with GilContentionProbe(interval_s=1.0) as probe:
        probe.reset()
        probe._record_delay(0.0002)
        probe._record_delay(0.0020)
        probe._record_delay(0.0080)
        stats = probe.read_and_reset()

    assert stats.samples == 3
    assert stats.mean_ms > 3.0
    assert stats.p50_ms == 2.0
    assert stats.p95_ms == 10.0
    assert stats.p99_ms == 10.0
    assert stats.max_ms == 8.0
    assert stats.over_1ms_pct == pytest.approx(2 / 3 * 100.0)
    assert stats.over_5ms_pct == pytest.approx(1 / 3 * 100.0)


def test_gil_probe_daemon_collects_and_closes() -> None:
    probe = GilContentionProbe(interval_s=0.001)
    time.sleep(0.02)
    stats = probe.read_and_reset()
    probe.close()
    probe.close()

    assert stats.samples > 0
    assert stats.sample_rate > 0.0
