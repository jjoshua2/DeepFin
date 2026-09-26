"""Bounded synchronous recovery for transient GPU telemetry transport failures.

This retries a query, never a workload. Callers retain numeric limits, STOP checks
and the enclosing operation's deadline. Every failed query is reported before a
retry or exception; no previous sample is returned as a substitute.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import subprocess
import time


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value or ''


def _arguments(argv: Sequence[object]) -> list[str]:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError('argv must be a nonempty argument sequence')
    result = []
    for item in argv:
        if not isinstance(item, str):
            raise ValueError('argv must contain strings')
        result.append(item)
    return result


def query(
    argv: Sequence[str],
    *,
    check_budget: Callable[[], None],
    on_failure: Callable[[dict[str, object]], None],
    attempts: int = 3,
    attempt_timeout: float = 3.0,
    total_timeout: float = 9.2,
) -> str:
    """Return stripped stdout or fail after at most three bounded attempts.

Only return code 3 and subprocess timeouts are retryable. Failure callbacks run
before the post-attempt budget check so a STOP/deadline exception cannot hide the
failed query. A callback exception stops processing immediately.
    """
    command = _arguments(argv)
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError('attempts must be an integer in [1, 3]')
    if not math.isfinite(attempt_timeout) or not 0 < attempt_timeout <= 3:
        raise ValueError('attempt timeout must be positive and at most 3 seconds')
    if not math.isfinite(total_timeout) or not 0 < total_timeout <= 9.2:
        raise ValueError('total timeout must be positive and at most 9.2 seconds')
    started = time.monotonic()
    deadline = started + total_timeout
    for attempt in range(1, attempts + 1):
        check_budget()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('GPU telemetry total deadline exhausted')
        failure: subprocess.CalledProcessError | subprocess.TimeoutExpired | OSError
        retryable = False
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False,
                                    timeout=min(attempt_timeout, remaining))
        except subprocess.TimeoutExpired as exc:
            failure = exc
            retryable = True
            record: dict[str, object] = {'attempt': attempt, 'kind': 'timeout', 'returncode': None,
                'stdout': _text(exc.stdout), 'stderr': _text(exc.stderr)}
        except OSError as exc:
            failure = exc
            record = {'attempt': attempt, 'kind': 'launch_error', 'returncode': None,
                      'stdout': '', 'stderr': str(exc)}
        else:
            if result.returncode == 0:
                check_budget()
                if time.monotonic() >= deadline:
                    raise TimeoutError('GPU telemetry total deadline exhausted')
                return result.stdout.strip()
            failure = subprocess.CalledProcessError(result.returncode, command,
                                                    output=result.stdout, stderr=result.stderr)
            retryable = result.returncode == 3
            record = {'attempt': attempt, 'kind': 'exit', 'returncode': result.returncode,
                      'stdout': _text(result.stdout), 'stderr': _text(result.stderr)}
        record['elapsed_seconds'] = time.monotonic() - started
        on_failure(record)
        check_budget()
        if not retryable or attempt == attempts:
            raise failure
        if time.monotonic() >= deadline:
            raise TimeoutError('GPU telemetry total deadline exhausted') from failure
    raise AssertionError('unreachable telemetry retry state')
