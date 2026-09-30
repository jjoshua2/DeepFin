"""Small host-side memory guard; CUDA virtual memory is deliberately not capped."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time

STARTUP_GIB = 48
RUNNING_GIB = 32


def require_available(gib: int) -> int:
    values = [int(line.split()[1]) * 1024 for line in
              Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')]
    if len(values) != 1 or values[0] < gib * 1024**3:
        raise RuntimeError(f'Linux available memory below {gib} GiB or unavailable')
    return values[0]


def _group_alive(pgid: int) -> bool:
    # A time wrapper can exit before its coordinator. Zombies cannot execute
    # cleanup; inspect live members rather than just the Popen leader.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
            return True
    return False


def cleanup_owned(child: subprocess.Popen) -> None:
    """Stop a start_new_session coordinator group, allowing nested cleanup."""
    try:
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError('refusing to signal an unowned coordinator group')
    except ProcessLookupError:
        pass
    for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 5)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace
        while True:
            child.poll()
            if not _group_alive(child.pid):
                child.wait(timeout=5)
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
    raise RuntimeError('owned coordinator process group survived cleanup')


def wait_guarded(child: subprocess.Popen, *, interval: float = 2.0) -> int:
    """Wait on a newly owned coordinator group; allow its nested-stage cleanup on failure."""
    if interval <= 0:
        raise ValueError('guard interval must be positive')
    try:
        while child.poll() is None:
            require_available(RUNNING_GIB)
            try:
                return child.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass
        return child.wait()
    except BaseException:
        cleanup_owned(child)
        raise
