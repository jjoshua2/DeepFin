"""Run ONLY by `test_conftest_keeps_cuda_probe_off_the_device.py`, as an explicit
path (`python -m pytest tests/_cuda_probe_inner.py`), so that pytest loads
`tests/conftest.py` exactly as it does for the suite. The leading underscore
keeps it out of normal collection (`python_files` is `test_*.py`)."""
from __future__ import annotations

import os


def test_is_available_leaves_the_device_unopened() -> None:
    import torch

    torch.cuda.is_available()
    dxg = os.stat("/dev/dxg").st_rdev
    held = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.stat(f"/proc/self/fd/{fd}").st_rdev == dxg:
                held.append(fd)
        except OSError:
            pass
    assert held == [], (held, os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"))
