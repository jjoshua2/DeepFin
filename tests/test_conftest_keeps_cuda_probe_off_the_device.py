"""`tests/conftest.py` sets `PYTORCH_NVML_BASED_CUDA_CHECK=1` so that a test's
`torch.cuda.is_available()` never opens the real device. This proves that call
only: a CPU `backward()` still opens /dev/dxg through the C++ autograd engine
(accepted; see the comment beside the line in `tests/conftest.py`). On WSL2 the default
check (`cudaGetDeviceCount` -> `cuInit`) opens /dev/dxg and holds it for the
life of the process even with `CUDA_VISIBLE_DEVICES` empty.

Proved in a SUBPROCESS started the way the suite starts python: `python -m
pytest` on an explicit inner test file under `tests/`, so pytest itself loads
`tests/conftest.py` (rootdir, ini file and conftest discovery are the suite's),
with `CUDA_VISIBLE_DEVICES=""` and the variable REMOVED from the child's
environment, so only the conftest can set it. The inner test asserts the
variable is "1" everywhere (CI included); its /dev/dxg fd scan runs only where
the node exists. Both regimes run here: the real node, and a missing path
standing in for a host without one (CI), so deleting the conftest line fails
on every host."""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("node", [
    pytest.param("/dev/dxg", id="host-node",
                 marks=pytest.mark.skipif(not os.path.exists("/dev/dxg"), reason="no /dev/dxg (not WSL2)")),
    pytest.param("/nonexistent/dxg", id="no-node"),
])
def test_is_available_under_the_suites_conftest_leaves_dev_dxg_unopened(tmp_path, node) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_NVML_BASED_CUDA_CHECK"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CAE_TEST_DXG_NODE"] = node
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         f"--basetemp={tmp_path / 'inner'}", f"--junitxml={tmp_path / 'inner.xml'}",
         "tests/_cuda_probe_inner.py"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    suite = ET.parse(tmp_path / "inner.xml").getroot().find("testsuite")
    assert suite is not None
    counts = {k: int(suite.get(k, "0")) for k in ("tests", "failures", "errors", "skipped")}
    assert counts == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}, counts
