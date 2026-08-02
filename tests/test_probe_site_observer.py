"""The out-of-band buffer observer must load under the interpreter we run.

``scripts/probe_site/sitecustomize.py`` is the instrument the ledger cites for
"``deterministic_refresh=True`` reached the real call site". It was previously
named ``usercustomize.py``, which ``site.py`` imports only when
``site.ENABLE_USER_SITE`` is true -- true for ``/usr/bin/python3``, FALSE for
this project's ``.venv``. Under the documented
``PYTHONPATH=. python3 scripts/...`` invocation it therefore printed nothing,
and "no ``[probe_site]`` line" was indistinguishable from "no buffer was
constructed": an absence that reads as evidence.

These tests run the probe under ``sys.executable`` -- whatever interpreter is
running the suite, venv included -- so a rename back to ``usercustomize.py``,
or any other change that stops it loading, fails here.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PROBE_DIR = _REPO / "scripts" / "probe_site"

_BUILD_A_BUFFER = """
import tempfile
from pathlib import Path
import numpy as np
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

buf = DiskReplayBuffer(
    100, shard_dir=Path(tempfile.mkdtemp()) / "replay",
    rng=np.random.default_rng(0), read_only=False,
    shuffle_cap=10, shard_size=10, deterministic_refresh=True,
)
buf.close()
"""


def _run(body: str, **extra_env: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO), str(_PROBE_DIR)])
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", body],
        env=env, capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"probe subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_probe_site_loads_under_this_interpreter() -> None:
    # The banner prints at IMPORT time, before the probe touches anything that
    # could fail, so its absence means the probe did not load at all.
    out = _run("pass")
    assert "[probe_site] observer INSTALLED" in out, (
        "scripts/probe_site did not load under sys.executable "
        f"({sys.executable}); site.ENABLE_USER_SITE is "
        f"{__import__('site').ENABLE_USER_SITE}. A usercustomize-named probe "
        "is a silent no-op in a venv -- the module must be sitecustomize.py."
    )


def test_probe_announces_that_no_buffer_was_constructed() -> None:
    # Silence must not be readable as "no buffer": the exit line says so.
    out = _run("pass")
    assert "buffers=[]" in out
    assert "NO buffer was constructed" in out


def test_probe_reports_the_flag_of_a_real_buffer() -> None:
    out = _run(_BUILD_A_BUFFER)
    assert "deterministic_refresh=True prefetch_thread=False" in out
    assert "buffers=[True]" in out


def test_probe_can_report_the_other_value() -> None:
    # The counterfactual arm. Without this the observation would be a check
    # that cannot fail: a probe able to print only ``True`` proves nothing
    # about the script it is observing.
    out = _run(_BUILD_A_BUFFER, PROBE_FORCE_RACE="1")
    assert "deterministic_refresh=False prefetch_thread=True" in out
    assert "buffers=[False]" in out
