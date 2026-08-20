"""The fail-closed guard on scripts/recover_stall.sh (#177, 2026-08-20).

On 2026-08-20 the watchdog cleared a live pause window's marker on an age
bound, and recover_stall.sh — whose only marker check lived in its CALLER —
killed the deliberately-parked trial and restarted production beside the job
the pause protected. The guard now lives in the script itself: an
intentional-stop or pause marker refuses the teardown (exit 7) unless
``--force`` is passed, and only ``--force`` may remove a pause marker.

⚑ Every test here drives the REAL script inside a sandbox where the teardown
path is harmless by construction: ``pkill``/``nvidia-smi`` resolve to PATH
shims, the pidfile names a sacrificial ``sleep``, and ``./scripts/train.sh``
is a recording stub. The proceed-path test exists precisely so the refusal
tests' "nothing was killed" assertions are non-vacuous — the same harness
demonstrably kills when the guard lets it through.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECOVER_SH = REPO / "scripts" / "recover_stall.sh"


class _Sandbox:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "repo"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts" / "recover_stall.sh").write_text(RECOVER_SH.read_text())

        train_sh = self.root / "scripts" / "train.sh"
        train_sh.write_text('#!/bin/bash\necho "$@" >> train_sh_invoked\nexit 0\n')
        train_sh.chmod(0o755)

        shim_bin = tmp_path / "bin"
        shim_bin.mkdir()
        pkill = shim_bin / "pkill"
        pkill.write_text('#!/bin/bash\necho "$@" >> pkill_invoked\nexit 1\n')
        pkill.chmod(0o755)
        smi = shim_bin / "nvidia-smi"
        smi.write_text("#!/bin/bash\nexit 0\n")
        smi.chmod(0o755)

        self.stop_marker = tmp_path / "intentional_stop"
        self.pause_txt = tmp_path / "pause.txt"
        self.pidfile = tmp_path / "trainer.pid"
        self.sacrifice = subprocess.Popen(["sleep", "600"])
        self.pidfile.write_text(f"{self.sacrifice.pid}\n")
        self.env = {
            "PATH": f"{shim_bin}:{os.environ.get('PATH', '')}",
            "RECOVER_ROOT": str(self.root),
            "RECOVER_STOP_MARKER": str(self.stop_marker),
            "RECOVER_PAUSE_TXT": str(self.pause_txt),
            "RECOVER_PIDFILE": str(self.pidfile),
        }

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "scripts/recover_stall.sh", *args],
            cwd=str(self.root), env=self.env,
            capture_output=True, text=True, check=False, timeout=60,
        )

    def sacrifice_alive(self) -> bool:
        # A killed child is a zombie until reaped; poll() reaps and reports.
        time.sleep(0.2)
        return self.sacrifice.poll() is None

    def cleanup(self) -> None:
        if self.sacrifice.poll() is None:
            self.sacrifice.terminate()
        self.sacrifice.wait()

    @property
    def log(self) -> str:
        f = self.root / "scratchpad" / "recover_stall.log"
        return f.read_text() if f.exists() else ""

    def teardown_ran(self) -> bool:
        return (self.root / "pkill_invoked").exists() or (
            self.root / "train_sh_invoked"
        ).exists()


def test_a_pause_marker_refuses_the_teardown(tmp_path: Path) -> None:
    """⚑ THE INCIDENT'S GUARD: a held pause.txt means an operator parked the
    stack on purpose, and the by-hand path must refuse before any kill."""
    sb = _Sandbox(tmp_path)
    try:
        sb.pause_txt.write_text("pause_window.sh pid=12345 started=x\n")
        r = sb.run()
        assert r.returncode == 7, (r.returncode, r.stdout, r.stderr, sb.log)
        assert "REFUSING" in sb.log, sb.log
        assert not sb.teardown_ran(), "the teardown ran despite the pause marker"
        assert sb.sacrifice_alive(), "the guarded path killed the trainer pid"
        assert sb.pause_txt.exists(), "the refusal path removed the marker"
    finally:
        sb.cleanup()


def test_an_intentional_stop_marker_refuses_the_teardown(tmp_path: Path) -> None:
    """`train.sh stop` writes this marker BEFORE killing precisely so recovery
    tooling does not undo the operator — the guard the caller had, now local."""
    sb = _Sandbox(tmp_path)
    try:
        sb.stop_marker.write_text("")
        r = sb.run()
        assert r.returncode == 7, (r.returncode, r.stdout, r.stderr, sb.log)
        assert not sb.teardown_ran()
        assert sb.sacrifice_alive()
    finally:
        sb.cleanup()


def test_no_markers_proceeds_and_the_harness_really_tears_down(tmp_path: Path) -> None:
    """The sanctioned wedge path (STALLED: no markers) must be untouched — and
    this is the proof the refusal tests' negatives can fail: the same sandbox
    kills the pidfile pid, pkills the patterns, and restarts via train.sh."""
    sb = _Sandbox(tmp_path)
    try:
        r = sb.run()
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.log)
        assert (sb.root / "pkill_invoked").exists(), "pkill never ran"
        assert (sb.root / "train_sh_invoked").read_text().strip() == "start"
        assert not sb.sacrifice_alive(), "the pidfile pid survived the teardown"
        assert not sb.pidfile.exists(), "the pidfile survived"
    finally:
        sb.cleanup()


def test_force_overrides_the_guard_and_removes_the_pause_marker(tmp_path: Path) -> None:
    """--force is the explicit operator override: it proceeds past a held
    marker AND is the only path that deletes one."""
    sb = _Sandbox(tmp_path)
    try:
        sb.pause_txt.write_text("pause_window.sh pid=12345 started=x\n")
        r = sb.run("--force")
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.log)
        assert (sb.root / "train_sh_invoked").exists()
        assert not sb.pause_txt.exists(), "--force must clear the marker"
    finally:
        sb.cleanup()
