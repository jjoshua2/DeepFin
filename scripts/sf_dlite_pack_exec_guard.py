"""Apply owned-unit death signal and alarm in a fresh exec, before worker exec."""
from __future__ import annotations
import ctypes
import os
import signal
import sys

def main() -> None:
    if sys.flags.optimize or len(sys.argv) < 5:
        raise ValueError("unoptimized guard needs owner, wall and executable command")
    owner_pid, wall_seconds = int(sys.argv[1]), int(sys.argv[2])
    command = sys.argv[3:]
    if owner_pid <= 1 or not 1 <= wall_seconds <= 1800:
        raise ValueError("owned-unit owner/wall outside reviewed bounds")
    if os.getppid() != owner_pid:
        raise RuntimeError("owner changed before death signal")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
    if os.getppid() != owner_pid:
        raise RuntimeError("owner changed during death signal setup")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(wall_seconds)
    os.execv(command[0], command)

if __name__ == "__main__":
    main()
