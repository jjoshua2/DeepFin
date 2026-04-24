from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterable


def _list_matching_pids(
    *,
    module: str,
    required_terms: Iterable[str],
    ps_output: str | None = None,
    exclude_pids: Iterable[int] = (),
) -> list[int]:
    terms = [str(term) for term in required_terms if str(term)]
    excluded = {int(pid) for pid in exclude_pids}
    excluded.add(int(os.getpid()))

    if ps_output is None:
        try:
            ps_output = subprocess.check_output(
                ["ps", "-eo", "pid=,args="],
                text=True,
            )
        except (subprocess.CalledProcessError, OSError):
  # ps unavailable or failed — nothing we can match against
            return []

    matches: list[int] = []
    for raw_line in str(ps_output).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_raw, _, cmd = line.partition(" ")
        try:
            pid = int(pid_raw)
        except ValueError:
  # header row or malformed ps output — skip
            continue
        if pid in excluded:
            continue
        if module not in cmd:
            continue
        if not all(term in cmd for term in terms):
            continue
        matches.append(pid)
    return matches


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, *, timeout_s: float = 5.0) -> bool:
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
  # not our process or kernel refused — treat as failure
        return False

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if not _pid_exists(int(pid)):
            return True
        time.sleep(0.1)

    try:
        os.kill(int(pid), signal.SIGKILL)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
  # not our process or kernel refused — treat as failure
        return False
    return not _pid_exists(int(pid))


def terminate_matching_processes(
    *,
    module: str,
    required_terms: Iterable[str],
    exclude_pids: Iterable[int] = (),
    timeout_s: float = 5.0,
) -> list[int]:
    return [
        int(pid)
        for pid in _list_matching_pids(
            module=module,
            required_terms=required_terms,
            exclude_pids=exclude_pids,
        )
        if _terminate_pid(pid, timeout_s=float(timeout_s))
    ]

