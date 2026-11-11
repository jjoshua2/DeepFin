"""`train.sh restart` / `salvage-restart` must run the C-extension gate BEFORE
stopping the live job.

start() runs the gate too, but by then stop() has already taken training
down, so a rejected extension build turned a restart into an outage. These
tests execute the functions and the command dispatch EXTRACTED FROM THE REAL
FILE, with stop/start/the checker stubbed: running the real stop() here would
signal the production job through the shared /tmp PID file.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

TRAIN_SH = Path(__file__).resolve().parents[1] / "scripts" / "train.sh"


def _function(name: str) -> str:
    lines = TRAIN_SH.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(name)}\(\) \{{$", ln)]
    assert len(starts) == 1, f"{name}() definition not unique: {starts}"
    end = next(i for i in range(starts[0] + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[starts[0]:end + 1])


def _dispatch() -> str:
    lines = TRAIN_SH.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith('case "${1:-status}" in')]
    assert len(starts) == 1
    end = next(i for i in range(starts[0], len(lines)) if lines[i] == "esac")
    return "\n".join(lines[starts[0]:end + 1])


_STUBS = """
check_c_extensions() { echo check >> "$CALL_LOG"; return "$CHECK_RC"; }
stop() { echo stop >> "$CALL_LOG"; }
start() { echo "start $*" >> "$CALL_LOG"; }
"""


def _run_command(tmp_path: Path, argv: list[str], *, check_rc: int) -> tuple[int, list[str]]:
    log = tmp_path / "calls.log"
    log.write_text("")
    script = "\n".join([
        "set -e",
        _STUBS,
        _function("restart_training"),
        _function("salvage_restart"),
        "set -- " + " ".join(f"'{a}'" for a in argv),
        _dispatch(),
    ])
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, timeout=30, check=False,
        env={"PATH": "/usr/bin:/bin", "CALL_LOG": str(log), "CHECK_RC": str(check_rc)},
    )
    return proc.returncode, log.read_text().splitlines()


def _salvage_pool(tmp_path: Path) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "manifest.json").write_text("{}")
    return pool


@pytest.mark.parametrize("command", ["restart", "salvage-restart"])
def test_failing_extension_gate_leaves_the_running_job_untouched(
    tmp_path: Path, command: str,
) -> None:
    argv = [command] + ([str(_salvage_pool(tmp_path))] if command == "salvage-restart" else [])

    rc, calls = _run_command(tmp_path, argv, check_rc=1)

    assert rc != 0
    assert calls == ["check"], f"{command} must not stop/start after a failed gate: {calls}"


@pytest.mark.parametrize("command", ["restart", "salvage-restart"])
def test_passing_extension_gate_runs_before_stop(tmp_path: Path, command: str) -> None:
    argv = [command] + ([str(_salvage_pool(tmp_path))] if command == "salvage-restart" else [])

    rc, calls = _run_command(tmp_path, argv, check_rc=0)

    assert rc == 0
    assert [c.split()[0] for c in calls] == ["check", "stop", "start"]


def test_start_keeps_its_own_extension_gate() -> None:
    assert "    check_c_extensions\n" in _function("start") + "\n"


def _run_check(tmp_path: Path, *, skip: bool) -> subprocess.CompletedProcess[str]:
    marker = tmp_path / "python3.called"
    script = "\n".join([
        f"python3() {{ touch '{marker}'; return 1; }}",
        _function("check_c_extensions"),
        "check_c_extensions",
    ])
    env = {"PATH": "/usr/bin:/bin"}
    if skip:
        env["TRAIN_SKIP_C_EXT_CHECK"] = "1"
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30,
        check=False, env=env,
    )


def test_check_failure_names_the_bypass_train_sh_honours(tmp_path: Path) -> None:
    proc = _run_check(tmp_path, skip=False)

    assert proc.returncode == 1
    assert (tmp_path / "python3.called").exists()
    assert "TRAIN_SKIP_C_EXT_CHECK=1" in proc.stderr


def test_bypass_skips_the_checker(tmp_path: Path) -> None:
    proc = _run_check(tmp_path, skip=True)

    assert proc.returncode == 0
    assert not (tmp_path / "python3.called").exists()
