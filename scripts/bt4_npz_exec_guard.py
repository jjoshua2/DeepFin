"""One-game exec guard: bind owner, arm deadline, then replace this process.

The owner launches this file directly, without a Python preexec callback.
No archive path is available until the guarded child starts.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import sys


def main() -> None:
    if sys.flags.optimize != 0:
        os._exit(127)
    fields = ("BT4_EXPECTED_OWNER_PID", "BT4_CHILD_WALL_SECONDS",
              "BT4_INTERPRETER_PATH")
    if len(sys.argv) < 3 or any(name not in os.environ for name in fields):
        os._exit(127)
    try:
        owner = int(os.environ[fields[0]])
        seconds = int(os.environ[fields[1]])
    except ValueError:
        os._exit(127)
    interpreter = os.environ[fields[2]]
    if (owner <= 1 or not 0 < seconds <= 600 or
            not Path(interpreter).is_absolute() or
            os.getppid() != owner):
        os._exit(127)
    libc = ctypes.CDLL(None, use_errno=True)
    if (libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or
            os.getppid() != owner):
        os._exit(127)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(seconds)
    os.execv(interpreter, (interpreter, *sys.argv[1:]))


if __name__ == "__main__":
    main()
