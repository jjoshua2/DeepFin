"""H7 NaN admission only: pure configuration and unstarted construction."""
import math

import pytest

from chess_anti_engine.broker_hang import (
    BOOT_HANG_ABORT_ENV, HANG_ABORT_ENV, WORKER_HANG_ABORT_ENV,
    DEFAULT_WORKER_BOOT_HANG_ABORT_S, BrokerHangWatchdog,
    resolve_boot_hang_abort_seconds, resolve_hang_abort_seconds,
    resolve_worker_boot_hang_abort_seconds, should_hang_abort,
)


def test_broker_nan_cli_is_rejected():
    with pytest.raises(ValueError, match='NaN'):
        resolve_hang_abort_seconds(float('nan'), env={})


@pytest.mark.parametrize('raw', ['nan', 'NaN', ' NaN '])
def test_broker_nan_env_is_rejected(raw):
    with pytest.raises(ValueError, match='NaN'):
        resolve_hang_abort_seconds(10., env={HANG_ABORT_ENV: raw})
    with pytest.raises(ValueError, match='NaN'):
        resolve_boot_hang_abort_seconds(10., env={BOOT_HANG_ABORT_ENV: raw})


@pytest.mark.parametrize(('steady', 'boot'), [(float('nan'), None), (10., float('nan'))])
def test_unstarted_watchdog_rejects_nan(steady, boot):
    with pytest.raises(ValueError, match='NaN'):
        BrokerHangWatchdog(threshold_s=steady, boot_threshold_s=boot)


def test_boot_nan_steady_or_default_is_rejected():
    with pytest.raises(ValueError, match='NaN'):
        resolve_boot_hang_abort_seconds(float('nan'), env={})
    with pytest.raises(ValueError, match='NaN'):
        resolve_boot_hang_abort_seconds(10., float('nan'), env={})


@pytest.mark.parametrize('raw', ['nan', 'NaN', ' NaN ', 'banana'])
def test_worker_invalid_env_keeps_documented_default_fallback(raw, caplog):
    seconds = resolve_worker_boot_hang_abort_seconds(env={WORKER_HANG_ABORT_ENV: raw})
    assert seconds == DEFAULT_WORKER_BOOT_HANG_ABORT_S
    assert WORKER_HANG_ABORT_ENV in caplog.text
    assert should_hang_abort(armed=True, oldest_inflight_age_s=seconds + 1, threshold_s=seconds)


@pytest.mark.parametrize('value', [0., -1., 20., float('inf'), -float('inf')])
def test_existing_non_nan_threshold_semantics_are_preserved(value):
    assert resolve_hang_abort_seconds(value, env={}) == value
    assert resolve_worker_boot_hang_abort_seconds(env={WORKER_HANG_ABORT_ENV: str(value)}) == value
    wd = BrokerHangWatchdog(threshold_s=value)
    assert wd.threshold_s == value
    assert wd.boot_threshold_s == value
    assert wd._thread is None
    assert resolve_boot_hang_abort_seconds(value, env={}) == (0. if value <= 0 else 1800.)
    assert not math.isnan(wd.threshold_s)


def test_disabled_boot_ignores_inactive_override_and_valid_env_overrides_cli():
    assert resolve_boot_hang_abort_seconds(0., env={BOOT_HANG_ABORT_ENV: 'nan'}) == 0.
    assert resolve_hang_abort_seconds(float('nan'), env={HANG_ABORT_ENV: '10'}) == 10.
