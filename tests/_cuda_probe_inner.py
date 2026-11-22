"""Run ONLY by `test_conftest_keeps_cuda_probe_off_the_device.py`, as an explicit
path (`python -m pytest tests/_cuda_probe_inner.py`), so that pytest loads
`tests/conftest.py` exactly as it does for the suite. The leading underscore
keeps it out of normal collection (`python_files` is `test_*.py`).

The outer test removes `PYTORCH_NVML_BASED_CUDA_CHECK` from this process's
environment, so the variable is here only if the conftest set it: asserted
everywhere, CI included. The /dev/dxg scan needs the node, so it runs only
where `CAE_TEST_DXG_NODE` (the outer test's choice: the real node, or a
missing path standing in for a host without one) exists."""
from __future__ import annotations

import os


def test_the_conftest_routes_is_available_through_nvml() -> None:
    assert os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"
    import torch

    torch.cuda.is_available()
    node = os.environ.get("CAE_TEST_DXG_NODE", "/dev/dxg")
    if not os.path.exists(node):
        return
    dxg = os.stat(node).st_rdev
    held = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.stat(f"/proc/self/fd/{fd}").st_rdev == dxg:
                held.append(fd)
        except OSError:
            pass
    assert held == [], held
