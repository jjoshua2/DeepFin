"""Every path that can (re)START training must consult the intentional-stop marker.

`train.sh stop` touches `/tmp/chess_training.intentional_stop` BEFORE it kills,
so nothing in the stack resurrects a run the operator stopped on purpose. That
contract lived in the CALLER: `watchdog_loop.sh` tests the marker before it
invokes `recover_stall.sh`, which covered the watchdog route and left the
BY-HAND route — `./scripts/recover_stall.sh`, the mode that script's own header
invites — with no check at all. `scripts/watchdog_pbt.sh` had none on either
side: nothing calls it, so there was no caller to hold one, and its restart
branch launches `chess_anti_engine.run` directly.

The guard now lives in `scripts/intentional_stop_guard.sh` and is called by the
operation itself, so a future caller cannot be written without it.

⚑ Every test here drives the REAL scripts inside a sandbox where the restart
path is harmless by construction: `pkill` / `nvidia-smi` / `pgrep` / `python3`
resolve to PATH shims, the pidfile names a sacrificial `sleep`, and
`./scripts/train.sh` is a recording stub. The PROCEED-path tests exist precisely
so the refusal tests' "nothing was killed / nothing was launched" assertions are
non-vacuous — the same harness demonstrably kills and launches when the guard
lets it through.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
GUARD_SH = SCRIPTS / "intentional_stop_guard.sh"
RECOVER_SH = SCRIPTS / "recover_stall.sh"
WATCHDOG_PBT_SH = SCRIPTS / "watchdog_pbt.sh"

# The refusal status. Distinct from 1 (generic), 2 (usage) and the watchdog's
# own 3/5/6, so a caller can tell "refused on a marker" from any other failure.
REFUSE_EXIT = 7


def _stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


class _RecoverSandbox:
    """A throwaway repo root holding the real recover_stall.sh + the guard lib."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "repo"
        (self.root / "scripts").mkdir(parents=True)
        for src in (RECOVER_SH, GUARD_SH):
            (self.root / "scripts" / src.name).write_text(src.read_text())

        _stub(
            self.root / "scripts" / "train.sh",
            '#!/bin/bash\necho "$@" >> train_sh_invoked\nexit 0\n',
        )

        shim_bin = tmp_path / "bin"
        shim_bin.mkdir()
        # `exit 1` mirrors real pkill's "no process matched", so the production
        # `pkill ... && log` chain takes the same branch it does live.
        _stub(shim_bin / "pkill", '#!/bin/bash\necho "$@" >> pkill_invoked\nexit 1\n')
        _stub(shim_bin / "nvidia-smi", "#!/bin/bash\nexit 0\n")

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


# ── recover_stall.sh: the BY-HAND path ──────────────────────────────────────


def test_the_by_hand_path_refuses_on_an_intentional_stop_marker(tmp_path: Path) -> None:
    """⚑ THE REGRESSION. `./scripts/recover_stall.sh` with no arguments IS the
    by-hand path, and it used to go straight to the SIGKILLs and `train.sh
    start` — undoing a stop an operator asked for. The refusal must NAME THE
    MARKER PATH, because "refused" without "which file" leaves the operator
    guessing what to remove.
    """
    sb = _RecoverSandbox(tmp_path)
    try:
        sb.stop_marker.write_text("stopped for the lc0 control window\n")
        r = sb.run()
        assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, sb.log)
        assert "REFUSING" in sb.log, sb.log
        assert str(sb.stop_marker) in sb.log, (
            f"the refusal did not name the marker path {sb.stop_marker}:\n{sb.log}"
        )
        assert "--ignore-intentional-stop" in sb.log, (
            f"the refusal did not say how to override deliberately:\n{sb.log}"
        )
        assert not sb.teardown_ran(), "the teardown ran despite the stop marker"
        assert sb.sacrifice_alive(), "the guarded path killed the trainer pid"
        assert sb.stop_marker.exists(), "the refusal path removed the marker"
    finally:
        sb.cleanup()


def test_the_by_hand_path_refuses_on_a_pause_marker(tmp_path: Path) -> None:
    """The 2026-08-20 incident's marker: a held pause.txt means an operator
    parked the stack on purpose, and the refusal must precede any kill."""
    sb = _RecoverSandbox(tmp_path)
    try:
        sb.pause_txt.write_text("pause_window.sh pid=12345 started=x\n")
        r = sb.run()
        assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, sb.log)
        assert str(sb.pause_txt) in sb.log, sb.log
        assert not sb.teardown_ran()
        assert sb.sacrifice_alive()
        assert sb.pause_txt.exists(), "the refusal path removed the marker"
    finally:
        sb.cleanup()


def test_the_override_flag_proceeds_and_logs_what_it_is_overriding(
    tmp_path: Path,
) -> None:
    """The override is explicit, loud, and names the marker AND its contents —
    a line an operator can paste into an incident note. It is also the only
    path that deletes a pause marker."""
    sb = _RecoverSandbox(tmp_path)
    try:
        sb.stop_marker.write_text("")
        sb.pause_txt.write_text("pause_window.sh pid=12345 started=x\n")
        r = sb.run("--ignore-intentional-stop")
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.log)
        assert "OVERRIDE" in sb.log, sb.log
        assert str(sb.stop_marker) in sb.log, sb.log
        assert str(sb.pause_txt) in sb.log, sb.log
        assert "pid=12345" in sb.log, (
            f"the override did not report what the marker HELD:\n{sb.log}"
        )
        assert "REFUSING" not in sb.log, sb.log
        assert (sb.root / "train_sh_invoked").read_text().strip() == "start"
        assert not sb.sacrifice_alive(), "the override did not tear the stack down"
        assert not sb.pause_txt.exists(), "the override must clear the pause marker"
    finally:
        sb.cleanup()


def test_no_marker_is_unchanged_and_the_harness_really_tears_down(
    tmp_path: Path,
) -> None:
    """The sanctioned wedge path (STALLED: no markers) must be untouched — and
    this is the proof the refusal tests' negatives CAN fail: the same sandbox
    kills the pidfile pid, pkills the patterns, and restarts via train.sh."""
    sb = _RecoverSandbox(tmp_path)
    try:
        r = sb.run()
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.log)
        assert "REFUSING" not in sb.log, sb.log
        assert "OVERRIDE" not in sb.log, sb.log
        assert (sb.root / "pkill_invoked").exists(), "pkill never ran"
        assert (sb.root / "train_sh_invoked").read_text().strip() == "start"
        assert not sb.sacrifice_alive(), "the pidfile pid survived the teardown"
        assert not sb.pidfile.exists(), "the pidfile survived"
    finally:
        sb.cleanup()


def test_an_unknown_argument_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    """⚑ THE REPO'S SIGNATURE DEFECT, applied to argv. A typo'd override
    (`--ignore-intentional_stop`) that fell through to the default would read to
    the operator as "the guard did not fire" while the script recorded "no
    override was requested". Reject it, and reject it BEFORE the teardown."""
    sb = _RecoverSandbox(tmp_path)
    try:
        r = sb.run("--ignore-intentional_stop")
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "unknown argument" in r.stderr, r.stderr
        assert not sb.teardown_ran(), "a bad argv still tore the stack down"
        assert sb.sacrifice_alive()
    finally:
        sb.cleanup()


def test_the_force_alias_is_accepted(tmp_path: Path) -> None:
    """`--force` is the spelling the live branch shipped (42e72d6cb). It is kept
    as an alias so an operator's muscle memory does not hit a usage error mid
    incident — and it must be as loud as the canonical spelling."""
    sb = _RecoverSandbox(tmp_path)
    try:
        sb.stop_marker.write_text("stopped on purpose\n")
        r = sb.run("--force")
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.log)
        assert "OVERRIDE" in sb.log, sb.log
        assert str(sb.stop_marker) in sb.log, sb.log
    finally:
        sb.cleanup()


# ── watchdog_pbt.sh: the THIRD route, which has no caller to guard it ────────


class _PbtSandbox:
    """watchdog_pbt.sh restarts by launching `chess_anti_engine.run` directly,
    so the observable is a python3 shim's recording, not a train.sh stub."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "repo"
        (self.root / "scripts").mkdir(parents=True)
        for src in (WATCHDOG_PBT_SH, GUARD_SH):
            (self.root / "scripts" / src.name).write_text(src.read_text())
        _stub(self.root / "scripts" / "monitor_pbt.sh", "#!/bin/bash\nexit 0\n")

        shim_bin = tmp_path / "bin"
        shim_bin.mkdir()
        self.launched = self.root / "run_launched"
        # No training is running, so the restart branch is the one under test.
        _stub(shim_bin / "pgrep", "#!/bin/bash\nexit 1\n")
        _stub(
            shim_bin / "python3",
            f'#!/bin/bash\necho "$@" >> {self.launched}\nexit 0\n',
        )

        self.stop_marker = tmp_path / "intentional_stop"
        self.pause_txt = tmp_path / "pause.txt"
        self.log = tmp_path / "monitor.log"
        self.env = {
            "PATH": f"{shim_bin}:{os.environ.get('PATH', '')}",
            "CHESS_ROOT": str(self.root),
            "TRAIN_MONITOR_LOG": str(self.log),
            "WATCHDOG_PBT_STOP_MARKER": str(self.stop_marker),
            "WATCHDOG_PBT_PAUSE_TXT": str(self.pause_txt),
            "WATCHDOG_PBT_MAX_ITERS": "1",
            "WATCHDOG_INTERVAL_SECONDS": "0",
        }

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "scripts/watchdog_pbt.sh", *args],
            cwd=str(self.root), env=self.env,
            capture_output=True, text=True, check=False, timeout=60,
        )

    @property
    def text(self) -> str:
        return self.log.read_text() if self.log.exists() else ""

    def run_launched(self) -> bool:
        return self.launched.exists()


def test_watchdog_pbt_refuses_to_resurrect_a_deliberately_stopped_run(
    tmp_path: Path,
) -> None:
    """⚑ THE THIRD ROUTE. Nothing invokes this script, so there was never a
    caller holding the check — it launched `chess_anti_engine.run` on its own
    authority at the next poll, whatever the operator had asked for."""
    sb = _PbtSandbox(tmp_path)
    sb.stop_marker.write_text("stopped for the nightly audit\n")
    r = sb.run()
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text
    assert "REFUSING" in out, out
    assert str(sb.stop_marker) in out, (
        f"the refusal did not name the marker path:\n{out}"
    )
    assert not sb.run_launched(), "it started training against the stop marker"


def test_watchdog_pbt_override_proceeds_and_says_so(tmp_path: Path) -> None:
    sb = _PbtSandbox(tmp_path)
    sb.stop_marker.write_text("stopped for the nightly audit\n")
    r = sb.run("--ignore-intentional-stop")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text
    assert "OVERRIDE" in out, out
    assert str(sb.stop_marker) in out, out
    assert sb.run_launched(), "the override did not restart training"
    assert "--mode tune --resume" in sb.launched.read_text()


def test_watchdog_pbt_with_no_marker_is_unchanged(tmp_path: Path) -> None:
    """The non-vacuity control for the refusal above: the same sandbox launches
    when no marker is present."""
    sb = _PbtSandbox(tmp_path)
    r = sb.run()
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text
    assert "REFUSING" not in out, out
    assert "OVERRIDE" not in out, out
    assert sb.run_launched(), "the unguarded case stopped restarting"


# ── the guard is defined ONCE, and every copy of the path agrees ─────────────


def test_neither_script_redefines_the_guard_locally() -> None:
    """A private copy of the refusal is a copy that drifts. `ratchet_common.sh`
    exists for the same reason, and `tests/test_ratchet_search_shape.py` pins
    the same property for the ratchet pair."""
    for script in (RECOVER_SH, WATCHDOG_PBT_SH):
        text = script.read_text()
        assert "intentional_stop_guard.sh" in text, (
            f"{script.name} no longer sources the shared guard"
        )
        assert not re.search(r"^intentional_stop_guard\s*\(\)", text, re.M), (
            f"{script.name} defines its own copy of the guard"
        )


def test_every_copy_of_the_stop_marker_path_agrees() -> None:
    """Four files carry the marker literal for reasons of their own. A path edit
    that lands in one and not the others silently disarms a guard, so pin them
    equal here rather than discovering it during an incident.
    """
    guard = re.search(
        r'^CAE_STOP_MARKER_DEFAULT="([^"]+)"$', GUARD_SH.read_text(), re.M
    )
    assert guard is not None, "the shared default is no longer a plain literal"
    canonical = guard.group(1)

    found = {"scripts/intentional_stop_guard.sh": canonical}
    for rel, pattern in (
        ("scripts/train.sh", r'^STOP_MARKER="([^"$]+)"$'),
        ("scripts/watchdog_loop.sh", r'^MARKER="\$\{WATCHDOG_MARKER:-([^}]+)\}"$'),
        ("scripts/train_watchdog.py", r'^DEFAULT_STOP_MARKER = Path\("([^"]+)"\)$'),
    ):
        m = re.search(pattern, (REPO / rel).read_text(), re.M)
        assert m is not None, f"{rel}: could not read its stop-marker literal"
        found[rel] = m.group(1)

    assert len(set(found.values())) == 1, f"stop-marker path drifted: {found}"
