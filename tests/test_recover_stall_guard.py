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

Four properties beyond "a marker refuses", each from a review finding, each with
its own mutant:

* THE PAUSE SET IS NOT ONE FILE. `_resolve_pause_marker_paths` honours
  `tc.pause_file` and per-trial `trial_dir/pause.txt`; `find_pause_txt` walks the
  tune dir recursively. A guard that read only the root marker let a per-trial
  pause through and recovery SIGKILLed a deliberately parked run.
* A REFUSAL MUST NOT BURN THE COOLDOWN. `watchdog_loop.sh` stamps before it
  calls, so an exit-7 refusal used to suppress recovery for two hours with no
  recovery having happened.
* AN OVERRIDE IS ONE RESTART, NOT A SESSION. `watchdog_pbt.sh` polls for days.
* `[ -e ]` IS FALSE FOR A DANGLING SYMLINK, so such a marker silently permitted
  recovery while the comment above it claimed fail-closed.

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

import pytest

from tests.script_loading import load_script_module

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
GUARD_SH = SCRIPTS / "intentional_stop_guard.sh"
RECOVER_SH = SCRIPTS / "recover_stall.sh"
WATCHDOG_PBT_SH = SCRIPTS / "watchdog_pbt.sh"
WATCHDOG_LOOP_SH = SCRIPTS / "watchdog_loop.sh"
WATCHDOG_PY = SCRIPTS / "train_watchdog.py"

# The refusal status. Distinct from 1 (generic), 2 (usage) and the watchdog's
# own 3/5/6, so a caller can tell "refused on a marker" from any other failure.
REFUSE_EXIT = 7


def _stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


# ── recover_stall.sh ────────────────────────────────────────────────────────


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

        # A real tune tree: the root marker and a per-trial one live here, and
        # the script scans it recursively the way find_pause_txt does.
        self.tune_dir = self.root / "runs" / "pbt2_small" / "tune"
        self.trial_dir = self.tune_dir / "train_trial_a_00000"
        self.trial_dir.mkdir(parents=True)
        self.root_pause = self.tune_dir / "pause.txt"
        self.trial_pause = self.trial_dir / "pause.txt"

        self.stop_marker = tmp_path / "intentional_stop"
        self.pause_file = tmp_path / "configured_pause"
        self.pidfile = tmp_path / "trainer.pid"
        self.sacrifice = subprocess.Popen(["sleep", "600"])
        self.pidfile.write_text(f"{self.sacrifice.pid}\n")
        self.env = {
            "PATH": f"{shim_bin}:{os.environ.get('PATH', '')}",
            "RECOVER_ROOT": str(self.root),
            "RECOVER_STOP_MARKER": str(self.stop_marker),
            "RECOVER_TUNE_DIR": str(self.tune_dir),
            # Set (to empty) by default: that skips the yaml read, so the tests
            # never depend on a config file existing in the sandbox. The
            # pause_file wiring gets its own test, which sets it non-empty.
            "RECOVER_PAUSE_FILE": "",
            "RECOVER_PIDFILE": str(self.pidfile),
        }

    def run(self, *args: str, **env_extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "scripts/recover_stall.sh", *args],
            cwd=str(self.root), env={**self.env, **env_extra},
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


@pytest.fixture
def recover(tmp_path: Path):
    sb = _RecoverSandbox(tmp_path)
    try:
        yield sb
    finally:
        sb.cleanup()


def test_the_by_hand_path_refuses_on_an_intentional_stop_marker(recover) -> None:
    """⚑ THE REGRESSION. `./scripts/recover_stall.sh` with no arguments IS the
    by-hand path, and it used to go straight to the SIGKILLs and `train.sh
    start` — undoing a stop an operator asked for. The refusal must NAME THE
    MARKER PATH, because "refused" without "which file" leaves the operator
    guessing what to remove.
    """
    recover.stop_marker.write_text("stopped for the lc0 control window\n")
    r = recover.run()
    assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, recover.log)
    assert "REFUSING" in recover.log, recover.log
    assert str(recover.stop_marker) in recover.log, (
        f"the refusal did not name the marker path:\n{recover.log}"
    )
    assert "--ignore-intentional-stop" in recover.log, (
        f"the refusal did not say how to override deliberately:\n{recover.log}"
    )
    assert not recover.teardown_ran(), "the teardown ran despite the stop marker"
    assert recover.sacrifice_alive(), "the guarded path killed the trainer pid"
    assert recover.stop_marker.exists(), "the refusal path removed the marker"


def test_the_by_hand_path_refuses_on_the_root_pause_marker(recover) -> None:
    """The 2026-08-20 incident's marker: a held pause.txt means an operator
    parked the stack on purpose, and the refusal must precede any kill."""
    recover.root_pause.write_text("pause_window.sh pid=12345 started=x\n")
    r = recover.run()
    assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, recover.log)
    assert str(recover.root_pause) in recover.log, recover.log
    assert not recover.teardown_ran()
    assert recover.sacrifice_alive()
    assert recover.root_pause.exists(), "the refusal path removed the marker"


def test_a_PER_TRIAL_pause_marker_refuses_recovery(recover) -> None:
    """⚑ REVIEW FINDING P1-1. The guard passed only the ROOT pause.txt, but
    `_resolve_pause_marker_paths` also honours `trial_dir/pause.txt` and
    `find_pause_txt` walks the tune dir RECURSIVELY. With the root marker absent
    and a per-trial one held, the old guard saw nothing and recovery SIGKILLed a
    deliberately parked run.
    """
    assert not recover.root_pause.exists(), "the root marker must be ABSENT here"
    recover.trial_pause.write_text("pause_window.sh pid=999 started=x\n")
    r = recover.run()
    assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, recover.log)
    assert str(recover.trial_pause) in recover.log, (
        f"the refusal did not name the per-trial marker:\n{recover.log}"
    )
    assert not recover.teardown_ran(), "a per-trial pause did not stop the teardown"
    assert recover.sacrifice_alive()
    assert recover.trial_pause.exists()


def test_a_CONFIGURED_pause_file_refuses_recovery(recover) -> None:
    """⚑ REVIEW FINDING P1-1, the half a filesystem scan cannot reach.
    `pause_file` is a live-yaml key `_resolve_pause_marker_paths` honours
    wherever it points — outside the tune dir included."""
    recover.pause_file.write_text("graceful restart in progress\n")
    r = recover.run(RECOVER_PAUSE_FILE=str(recover.pause_file))
    assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, recover.log)
    assert str(recover.pause_file) in recover.log, recover.log
    assert not recover.teardown_ran()
    assert recover.sacrifice_alive()


def test_a_DANGLING_SYMLINK_marker_refuses_recovery(recover) -> None:
    """⚑ REVIEW FINDING P2-5. `[ -e ]` FOLLOWS the link and is FALSE when the
    target is gone, so a dangling-symlink marker read as "no marker" and
    silently permitted recovery — while the comment above it claimed
    fail-closed. `-L` is the half that sees the link itself.
    """
    recover.stop_marker.symlink_to(recover.root / "no" / "such" / "target")
    assert not recover.stop_marker.exists(), "the fixture is not dangling"
    assert recover.stop_marker.is_symlink()
    r = recover.run()
    assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, recover.log)
    assert str(recover.stop_marker) in recover.log, recover.log
    assert "dangling symlink" in recover.log, (
        f"the refusal did not say the marker was dangling:\n{recover.log}"
    )
    assert not recover.teardown_ran()
    assert recover.sacrifice_alive()


def test_a_DIRECTORY_marker_refuses_recovery(recover) -> None:
    """The other odd shape `-e` does cover. Pinned so the `-L` fix above cannot
    be "simplified" into dropping it."""
    recover.stop_marker.mkdir()
    r = recover.run()
    assert r.returncode == REFUSE_EXIT, (r.returncode, r.stdout, r.stderr, recover.log)
    assert "a directory" in recover.log, recover.log
    assert not recover.teardown_ran()


def test_the_override_proceeds_and_clears_the_WHOLE_pause_set(recover) -> None:
    """The override is explicit, loud, and names each marker AND its contents —
    a line an operator can paste into an incident note. ⚑ It must clear EVERY
    pause marker, not just the root one: leaving the per-trial marker would park
    the restarted trial while the log claimed a successful recovery."""
    recover.stop_marker.write_text("")
    recover.root_pause.write_text("pause_window.sh pid=12345 started=x\n")
    recover.trial_pause.write_text("pause_window.sh pid=999 started=y\n")
    r = recover.run("--ignore-intentional-stop")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, recover.log)
    assert "OVERRIDE" in recover.log, recover.log
    assert str(recover.stop_marker) in recover.log, recover.log
    assert "pid=12345" in recover.log, (
        f"the override did not report what the marker HELD:\n{recover.log}"
    )
    assert "REFUSING" not in recover.log, recover.log
    assert (recover.root / "train_sh_invoked").read_text().strip() == "start"
    assert not recover.sacrifice_alive(), "the override did not tear the stack down"
    assert not recover.root_pause.exists(), "the root pause marker survived"
    assert not recover.trial_pause.exists(), (
        "the PER-TRIAL pause marker survived the override — the restarted trial "
        "parks on it while the log says recovery succeeded"
    )


def test_no_marker_is_unchanged_and_the_harness_really_tears_down(recover) -> None:
    """The sanctioned wedge path (STALLED: no markers) must be untouched — and
    this is the proof the refusal tests' negatives CAN fail: the same sandbox
    kills the pidfile pid, pkills the patterns, and restarts via train.sh."""
    r = recover.run()
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, recover.log)
    assert "REFUSING" not in recover.log, recover.log
    assert "OVERRIDE" not in recover.log, recover.log
    assert (recover.root / "pkill_invoked").exists(), "pkill never ran"
    assert (recover.root / "train_sh_invoked").read_text().strip() == "start"
    assert not recover.sacrifice_alive(), "the pidfile pid survived the teardown"
    assert not recover.pidfile.exists(), "the pidfile survived"


def test_the_override_flag_with_no_marker_does_not_claim_an_override(recover) -> None:
    """A run that passed the flag and met no marker did not restart "against a
    deliberate stop"; logging that it did would put a false incident line in the
    operator's record."""
    r = recover.run("--force")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, recover.log)
    assert "OVERRIDE" not in recover.log, recover.log
    assert (recover.root / "train_sh_invoked").exists()


def test_an_unknown_argument_is_rejected_rather_than_ignored(recover) -> None:
    """⚑ THE REPO'S SIGNATURE DEFECT, applied to argv. A typo'd override
    (`--ignore-intentional_stop`) that fell through to the default would read to
    the operator as "the guard did not fire" while the script recorded "no
    override was requested". Reject it, and reject it BEFORE the teardown."""
    r = recover.run("--ignore-intentional_stop")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "unknown argument" in r.stderr, r.stderr
    assert not recover.teardown_ran(), "a bad argv still tore the stack down"
    assert recover.sacrifice_alive()


def test_the_force_alias_is_accepted(recover) -> None:
    """`--force` is the spelling the live branch shipped (42e72d6cb). It is kept
    as an alias so an operator's muscle memory does not hit a usage error mid
    incident — and it must be as loud as the canonical spelling."""
    recover.stop_marker.write_text("stopped on purpose\n")
    r = recover.run("--force")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, recover.log)
    assert "OVERRIDE" in recover.log, recover.log
    assert str(recover.stop_marker) in recover.log, recover.log


# ── the enumeration shares the pause machinery's own instrument ─────────────


def test_the_bash_enumeration_covers_what_find_pause_txt_finds(tmp_path: Path) -> None:
    """⚑ A GUARD MUST SHARE THE CRITERION'S INSTRUMENT. `find_pause_txt` is what
    produces the PAUSED-HELD/STALLED verdict; the guard's own scan must never be
    narrower than it, or the verdict says "paused" while the guard says "go".

    Superset, not equality: the guard deliberately also counts a dangling
    symlink and a directory, which `find_pause_txt`'s `is_file()` rejects.
    Stricter is the safe direction — a guard can only ever refuse more.
    """
    wd = load_script_module("train_watchdog.py", "train_watchdog")

    tune = tmp_path / "tune"
    (tune / "train_trial_a_00000").mkdir(parents=True)
    (tune / "train_trial_b_00001" / "nested").mkdir(parents=True)

    for present in (
        [tune / "pause.txt"],
        [tune / "train_trial_a_00000" / "pause.txt"],
        [tune / "train_trial_b_00001" / "nested" / "pause.txt"],
        [tune / "pause.txt", tune / "train_trial_a_00000" / "pause.txt"],
    ):
        for p in tune.rglob("pause.txt"):
            p.unlink()
        for p in present:
            p.write_text("held\n")

        found = wd.find_pause_txt(tune)
        assert found is not None, f"fixture broken for {present}"

        out = subprocess.run(
            ["bash", "-c", f'. "{GUARD_SH}"; cae_pause_markers "{tune}" ""'],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.split()
        assert str(found) in out, (
            f"find_pause_txt saw {found} and the guard's enumeration did not: {out}"
        )
        assert {str(p) for p in present} <= set(out), (present, out)


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

        self.tune_dir = self.root / "runs" / "pbt2_small" / "tune"
        self.trial_dir = self.tune_dir / "train_trial_a_00000"
        self.trial_dir.mkdir(parents=True)
        self.root_pause = self.tune_dir / "pause.txt"
        self.trial_pause = self.trial_dir / "pause.txt"

        self.stop_marker = tmp_path / "intentional_stop"
        self.log = tmp_path / "monitor.log"
        self.env = {
            "PATH": f"{shim_bin}:{os.environ.get('PATH', '')}",
            "CHESS_ROOT": str(self.root),
            "TRAIN_MONITOR_LOG": str(self.log),
            "WATCHDOG_PBT_STOP_MARKER": str(self.stop_marker),
            "WATCHDOG_PBT_TUNE_DIR": str(self.tune_dir),
            "WATCHDOG_PBT_MAX_ITERS": "1",
            "WATCHDOG_PBT_RESTART_SETTLE": "0",
            "WATCHDOG_INTERVAL_SECONDS": "0",
        }

    def run(self, *args: str, **env_extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "scripts/watchdog_pbt.sh", *args],
            cwd=str(self.root), env={**self.env, **env_extra},
            capture_output=True, text=True, check=False, timeout=60,
        )

    @property
    def text(self) -> str:
        return self.log.read_text() if self.log.exists() else ""

    def launches(self) -> int:
        if not self.launched.exists():
            return 0
        return len([ln for ln in self.launched.read_text().splitlines() if ln.strip()])


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
    assert str(sb.stop_marker) in out, f"the refusal did not name the marker:\n{out}"
    assert sb.launches() == 0, "it started training against the stop marker"


def test_watchdog_pbt_refuses_a_PER_TRIAL_pause_too(tmp_path: Path) -> None:
    """⚑ REVIEW FINDING P1-1 on the third route: the same recursive set."""
    sb = _PbtSandbox(tmp_path)
    sb.trial_pause.write_text("pause_window.sh pid=999 started=x\n")
    r = sb.run()
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text
    assert str(sb.trial_pause) in out, out
    assert sb.launches() == 0


def test_watchdog_pbt_override_does_NOT_override_a_pause(tmp_path: Path) -> None:
    """⚑ REVIEW FINDING P1-2, and the deliberate asymmetry with
    recover_stall.sh. This branch routes through neither recover_stall's
    teardown nor `train.sh start`, so it cannot clear a pause marker correctly.
    Overriding one would launch a trial that parks at its own pause check within
    seconds while the log said "Restarted with PID N" — a SILENT WEDGE, and a
    worse outcome than refusing, because the log asserts the opposite of what
    happened. So the override reaches the STOP marker only, and says why.
    """
    sb = _PbtSandbox(tmp_path)
    sb.stop_marker.write_text("stopped on purpose\n")
    sb.root_pause.write_text("pause_window.sh pid=12345 started=x\n")
    r = sb.run("--ignore-intentional-stop")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text
    assert "REFUSING" in out, out
    assert "does NOT override a PAUSE" in out, (
        f"it refused without explaining that the override does not reach a pause:\n{out}"
    )
    assert sb.launches() == 0, "it launched a trial that would park on the pause marker"
    assert sb.root_pause.exists(), "it deleted a pause marker it cannot clear safely"


def test_watchdog_pbt_override_of_a_STOP_marker_proceeds(tmp_path: Path) -> None:
    sb = _PbtSandbox(tmp_path)
    sb.stop_marker.write_text("stopped for the nightly audit\n")
    r = sb.run("--ignore-intentional-stop")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text
    assert "OVERRIDE" in out, out
    assert str(sb.stop_marker) in out, out
    assert sb.launches() == 1, "the override did not restart training"
    assert "--mode tune --resume" in sb.launched.read_text()


def test_watchdog_pbt_override_is_consumed_after_one_restart(tmp_path: Path) -> None:
    """⚑ REVIEW FINDING P1-4. This loop runs for DAYS in a tmux pane. A flag
    parsed once at launch and left set turns a single "yes, bring it back up"
    into standing permission to override every LATER operator stop — the flag
    outliving the intent that justified it. Two polls: the first overrides and
    launches, the operator stops again, and the second must REFUSE.
    """
    sb = _PbtSandbox(tmp_path)
    sb.stop_marker.write_text("first stop\n")
    r = sb.run("--ignore-intentional-stop", WATCHDOG_PBT_MAX_ITERS="2")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text

    assert "OVERRIDE" in out, out
    assert "override consumed" in out, (
        f"the authorization was never consumed:\n{out}"
    )
    assert "REFUSING" in out, (
        "the SECOND poll honoured the same one-shot authorization — a launch-time "
        f"flag became permanent license:\n{out}"
    )
    assert sb.launches() == 1, (
        f"expected exactly one authorized restart, got {sb.launches()}:\n{out}"
    )


def test_watchdog_pbt_with_no_marker_is_unchanged(tmp_path: Path) -> None:
    """The non-vacuity control for the refusals above: the same sandbox launches
    when no marker is present."""
    sb = _PbtSandbox(tmp_path)
    r = sb.run()
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, sb.text)
    out = r.stdout + sb.text
    assert "REFUSING" not in out, out
    assert "OVERRIDE" not in out, out
    assert sb.launches() == 1, "the unguarded case stopped restarting"


# ── watchdog_loop.sh: a refusal must not burn the anti-flap cooldown ─────────


def _loop_sandbox(tmp_path: Path):
    """A STALLED verdict driving the REAL loop against the REAL recover_stall.

    The marker that makes recover_stall refuse is a `pause_file` OUTSIDE the
    tune dir: `find_pause_txt` cannot see it, so the watchdog still returns
    STALLED (exit 3) and the loop still fires — which is exactly the race that
    makes rc=7 reachable on the watchdog route at all.
    """
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for src in (WATCHDOG_LOOP_SH, WATCHDOG_PY, RECOVER_SH, GUARD_SH):
        (root / "scripts" / src.name).write_text(src.read_text())
    _stub(root / "scripts" / "train.sh",
          '#!/bin/bash\necho "$@" >> train_sh_invoked\nexit 0\n')

    shim_bin = tmp_path / "bin"
    shim_bin.mkdir()
    _stub(shim_bin / "pkill", '#!/bin/bash\necho "$@" >> pkill_invoked\nexit 1\n')
    _stub(shim_bin / "nvidia-smi", "#!/bin/bash\nexit 0\n")

    tune = root / "runs" / "pbt2_small" / "tune"
    (tune / "train_trial_a_00000").mkdir(parents=True)
    (tune / "train_trial_a_00000" / "progress.csv").write_text("h\n1\n2\n")
    return root, tune, shim_bin


def _run_loop(root: Path, tmp_path: Path, shim_bin: Path, **env_extra: str) -> str:
    """One pass against a PRE-SEEDED flatness clock.

    STALLED needs `minutes_flat > 90`, which two back-to-back passes cannot
    produce. The state file is the watchdog's own cross-invocation memory
    (`rows` + `wall_time` of last growth), so seeding it with the trial's
    current row count and a `wall_time` 100 minutes ago is exactly "this trial
    has been flat for 100 minutes" — no clock mocking, no sleeping.
    """
    import json

    alive = subprocess.Popen(["sleep", "600"])
    pidfile = root / "trainer.pid"
    pidfile.write_text(f"{alive.pid}\n")
    logf = tmp_path / "watchdog.log"
    alertf = tmp_path / "watchdog_alerts.log"
    statef = tmp_path / "wd_state.json"
    statef.write_text(json.dumps({"rows": 2, "wall_time": time.time() - 100 * 60}))
    env = {
        "PATH": f"{shim_bin}:{os.environ.get('PATH', '')}",
        "WATCHDOG_ROOT": str(root),
        "WATCHDOG_MAX_ITERS": "1",
        "WATCHDOG_EVERY": "1",
        "WATCHDOG_STATEF": str(tmp_path / "wd_state.json"),
        "WATCHDOG_MARKER": str(tmp_path / "no_such_stop_marker"),
        "WATCHDOG_LOGF": str(logf),
        "WATCHDOG_ALERTF": str(alertf),
        "WATCHDOG_LAST_ALERT_F": str(tmp_path / "last_alert"),
        "WATCHDOG_LAST_ESCALATE_F": str(tmp_path / "last_escalate"),
        "WATCHDOG_RECOVER_STAMP": str(tmp_path / "recover_stamp"),
        "TRAIN_PIDFILE": str(pidfile),
        "RECOVER_ROOT": str(root),
        "RECOVER_STOP_MARKER": str(tmp_path / "no_such_stop_marker"),
        "RECOVER_TUNE_DIR": str(root / "runs" / "pbt2_small" / "tune"),
        "RECOVER_PAUSE_FILE": "",
        "RECOVER_PIDFILE": str(pidfile),
        "PYTHONDONTWRITEBYTECODE": "1",
        **env_extra,
    }
    try:
        r = subprocess.run(
            ["bash", "scripts/watchdog_loop.sh"],
            cwd=str(root), capture_output=True, text=True,
            check=False, timeout=180, env=env,
        )
        out = r.stdout + r.stderr
    finally:
        if alive.poll() is None:
            alive.terminate()
        alive.wait()
    return (
        out
        + (logf.read_text() if logf.exists() else "")
        + (alertf.read_text() if alertf.exists() else "")
    )


def test_a_refused_recovery_does_not_burn_the_cooldown_stamp(tmp_path: Path) -> None:
    """⚑ REVIEW FINDING P1-3. `watchdog_loop.sh` writes RECOVER_STAMP BEFORE it
    calls recover_stall (deliberately: if the loop dies mid-recovery the
    anti-flap bound must already be armed). A guard refusal then cost a full
    RECOVER_COOLDOWN_S of suppression for a recovery that NEVER HAPPENED — the
    next two hours of genuine stalls would report "SUPPRESSED (re-stall within
    cooldown of last recovery)" with no last recovery to speak of.
    """
    root, _tune, shim_bin = _loop_sandbox(tmp_path)
    blocker = tmp_path / "configured_pause"
    blocker.write_text("pause_window.sh pid=4242 started=x\n")
    stamp = tmp_path / "recover_stamp"

    out = _run_loop(root, tmp_path, shim_bin, RECOVER_PAUSE_FILE=str(blocker))

    assert "AUTO-RECOVER FIRING" in out, f"the stall path never fired:\n{out}"
    assert "REFUSED" in out, f"recover_stall did not refuse:\n{out}"
    assert "cooldown stamp rolled back" in out, out
    assert not stamp.exists(), (
        "a refusal armed the 2h anti-flap cooldown for a recovery that never "
        f"happened:\n{out}"
    )
    assert not (root / "train_sh_invoked").exists(), "it restarted anyway"


def test_a_refused_recovery_restores_a_PREVIOUS_stamp_exactly(tmp_path: Path) -> None:
    """The rollback is an undo, not a delete: a stamp from a real earlier
    recovery must survive a later refusal untouched."""
    root, _tune, shim_bin = _loop_sandbox(tmp_path)
    blocker = tmp_path / "configured_pause"
    blocker.write_text("held\n")
    stamp = tmp_path / "recover_stamp"
    # Older than the 2h cooldown, so the loop still reaches the FIRING branch.
    previous = str(int(time.time()) - 99_999)
    stamp.write_text(previous)

    out = _run_loop(root, tmp_path, shim_bin, RECOVER_PAUSE_FILE=str(blocker))

    assert "REFUSED" in out, out
    assert stamp.read_text().strip() == previous, (
        f"the rollback rewrote an unrelated earlier stamp: {stamp.read_text()!r}"
    )


def test_the_loop_still_recovers_when_nothing_refuses(tmp_path: Path) -> None:
    """⚑ THE NON-VACUITY CONTROL for both cooldown tests: with no blocking
    marker the same sandbox fires, recovers, and DOES arm the cooldown."""
    root, _tune, shim_bin = _loop_sandbox(tmp_path)
    stamp = tmp_path / "recover_stamp"

    out = _run_loop(root, tmp_path, shim_bin)

    assert "AUTO-RECOVER FIRING" in out, out
    assert "REFUSED" not in out, out
    assert (root / "train_sh_invoked").exists(), f"recovery never restarted:\n{out}"
    assert stamp.exists(), "a successful recovery did not arm the cooldown"


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


def test_the_refusal_status_agrees_between_the_guard_and_the_loop() -> None:
    """⚑ `watchdog_loop.sh` rolls its cooldown stamp back on EXACTLY this
    status. It deliberately does not source the guard library — a long-running
    loop must not gain a new way to die at startup — so the two constants are
    pinned here instead. Drift means the rollback silently stops firing and
    every refusal burns two hours of suppression again.
    """
    a = re.search(r"^INTENTIONAL_STOP_EXIT=(\d+)$", GUARD_SH.read_text(), re.M)
    b = re.search(r"^EXIT_RECOVER_REFUSED=(\d+)$", WATCHDOG_LOOP_SH.read_text(), re.M)
    assert a is not None, "the guard no longer defines INTENTIONAL_STOP_EXIT"
    assert b is not None, "the loop no longer defines EXIT_RECOVER_REFUSED"
    assert a.group(1) == b.group(1), (
        f"refusal status drifted: guard={a.group(1)} loop={b.group(1)}"
    )
