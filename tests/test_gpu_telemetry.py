"""Telemetry subprocesses are mocked; these tests never query a GPU."""
from __future__ import annotations

import subprocess
from typing import Any

import pytest

from scripts import gpu_telemetry as tool


COMMAND = ['/fixture/nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']


def harness(monkeypatch, outcomes):
    calls: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    checks: list[None] = []
    def run(argv, **kwargs):
        assert argv == COMMAND
        assert kwargs['capture_output'] is True
        assert kwargs['text'] is True
        assert kwargs['check'] is False
        assert 0 < kwargs['timeout'] <= 3
        calls.append(kwargs)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(argv, outcome, ' 1742\n', 'diagnostic')
    monkeypatch.setattr(tool.subprocess, 'run', run)
    return calls, failures, checks


def test_transient_failure_reports_then_returns_current_sample(monkeypatch):
    calls, failures, checks = harness(monkeypatch, [3, 0])
    assert tool.query(COMMAND, check_budget=lambda: checks.append(None), on_failure=failures.append) == '1742'
    assert len(calls) == 2
    assert len(checks) == 4
    assert [(x['attempt'], x['returncode'], x['stdout'], x['stderr']) for x in failures] == [
        (1, 3, ' 1742\n', 'diagnostic')]


def test_repeated_transport_error_never_returns_stale_stdout(monkeypatch):
    calls, failures, checks = harness(monkeypatch, [3, 3, 3])
    with pytest.raises(subprocess.CalledProcessError) as error:
        tool.query(COMMAND, check_budget=lambda: checks.append(None), on_failure=failures.append)
    assert error.value.returncode == 3
    assert len(calls) == len(failures) == 3
    assert len(checks) == 6


def test_timeout_bytes_are_reported_and_retry_is_bounded(monkeypatch):
    timeout = subprocess.TimeoutExpired(COMMAND, 3, output=b'partial\xff', stderr=b'timed out')
    calls, failures, checks = harness(monkeypatch, [timeout, timeout, timeout])
    with pytest.raises(subprocess.TimeoutExpired):
        tool.query(COMMAND, check_budget=lambda: checks.append(None), on_failure=failures.append)
    assert len(calls) == len(failures) == 3
    assert failures[0]['stdout'] == 'partial\ufffd'
    assert failures[0]['stderr'] == 'timed out'
    assert failures[0]['kind'] == 'timeout'


@pytest.mark.parametrize('outcome', [1, 2, 4, OSError('binary missing')])
def test_other_failures_are_not_retried(monkeypatch, outcome):
    calls, failures, checks = harness(monkeypatch, [outcome])
    with pytest.raises((subprocess.CalledProcessError, OSError)):
        tool.query(COMMAND, check_budget=lambda: checks.append(None), on_failure=failures.append)
    assert len(calls) == len(failures) == 1
    assert len(checks) == 2


@pytest.mark.parametrize(('stop_at', 'outcomes'), [(1, []), (2, [3]), (2, [0])])
def test_budget_stop_prevents_retry_or_success(monkeypatch, stop_at, outcomes):
    calls, failures, checks = harness(monkeypatch, outcomes)
    def budget():
        checks.append(None)
        if len(checks) == stop_at:
            raise RuntimeError('operation STOP')
    with pytest.raises(RuntimeError, match='operation STOP'):
        tool.query(COMMAND, check_budget=budget, on_failure=failures.append)
    assert len(calls) == len(outcomes)
    assert len(failures) == (1 if outcomes == [3] else 0)


def test_total_budget_clips_remaining_attempt(monkeypatch):
    now = [0.0]
    timeouts = []
    failures = []
    monkeypatch.setattr(tool.time, 'monotonic', lambda: now[0])
    def run(argv, **kwargs):
        timeouts.append(kwargs['timeout'])
        now[0] += kwargs['timeout']
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
    monkeypatch.setattr(tool.subprocess, 'run', run)
    with pytest.raises(TimeoutError, match='total deadline'):
        tool.query(COMMAND, check_budget=lambda: None, on_failure=failures.append, total_timeout=4)
    assert timeouts == [3, 1]
    assert len(failures) == 2


def test_numeric_limits_remain_callers_responsibility(monkeypatch):
    calls, failures, checks = harness(monkeypatch, [0])
    sample = tool.query(COMMAND, check_budget=lambda: checks.append(None), on_failure=failures.append)
    assert int(sample) > 1000
    assert len(calls) == 1
    assert not failures


@pytest.mark.parametrize('kwargs', [{'attempts': 4}, {'attempts': True}, {'attempt_timeout': 4},
                                    {'total_timeout': 10}, {'total_timeout': float('nan')}])
def test_invalid_bounds_do_not_query(monkeypatch, kwargs):
    calls, failures, checks = harness(monkeypatch, [])
    with pytest.raises(ValueError, match=r'attempt|timeout'):
        tool.query(COMMAND, check_budget=lambda: checks.append(None), on_failure=failures.append, **kwargs)
    assert not calls
