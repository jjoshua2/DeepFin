"""scripts/watchdog_loop.sh must alert ONCE per episode, not once per poll.

The 2026-08-08 outage appended 27 identical ``STOPPED`` lines to the alert log
across 4h32m and nobody read any of them. The detector was never the weak link;
the delivery was. These tests pin the delivery, driving the real loop through
its test seams (``WATCHDOG_ROOT`` / ``WATCHDOG_MAX_ITERS`` / path overrides)
against a stub ``train_watchdog.py`` whose verdicts are scripted per poll.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOOP = REPO / "scripts" / "watchdog_loop.sh"

# One stub verdict per poll, read line-by-line from a scripted file. Mirrors the
# real CLI contract: one "watchdog: STATE ..." line on stdout, state-coded exit.
_STUB = """#!/usr/bin/env python3
import sys
from pathlib import Path
plan = Path(__file__).with_name("plan.txt").read_text().split("\\n")
cur = Path(__file__).with_name("cursor.txt")
i = int(cur.read_text()) if cur.exists() else 0
state = plan[i].strip() if i < len(plan) else plan[-1].strip()
cur.write_text(str(i + 1))
codes = {"OK": 0, "STOPPED": 1, "PAUSED-HELD": 2, "STALLED": 3, "CRASHED": 5}
print(f"watchdog: {state} pid=4242 rows=249 minutes_flat=0.0")
sys.exit(codes[state])
"""


def _run(tmp_path: Path, plan: list[str], *, marker: bool = False,
         notify: str | None = None) -> tuple[list[str], list[str]]:
    """Drive the real loop for len(plan) polls; return (alert_lines, log_lines)."""
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "scratchpad").mkdir()
    stub = root / "scripts" / "train_watchdog.py"
    stub.write_text(_STUB)
    (root / "scripts" / "plan.txt").write_text("\n".join(plan))
    # recover_stall.sh must exist and be inert: a STALLED plan entry would
    # otherwise try to restart real training from a test.
    (root / "scripts" / "recover_stall.sh").write_text("#!/bin/bash\nexit 0\n")

    marker_path = tmp_path / "marker"
    if marker:
        marker_path.write_text("")

    env = dict(os.environ)
    env.update(
        WATCHDOG_ROOT=str(root),
        WATCHDOG_MAX_ITERS=str(len(plan)),
        WATCHDOG_EVERY="0",
        WATCHDOG_MARKER=str(marker_path),
        WATCHDOG_LOGF="scratchpad/wd.log",
        WATCHDOG_ALERTF="scratchpad/wd_alerts.log",
        WATCHDOG_STATEF=str(tmp_path / "state.json"),
        WATCHDOG_LAST_ALERT_F=str(tmp_path / "last_alert"),
        WATCHDOG_AUTO_RECOVER="0",
    )
    if notify is not None:
        env["WATCHDOG_NOTIFY_CMD"] = notify

    subprocess.run(["bash", str(LOOP)], env=env, check=True, timeout=120,
                   capture_output=True)

    def _lines(name: str) -> list[str]:
        p = root / "scratchpad" / name
        if not p.exists():
            return []
        return [ln for ln in p.read_text().splitlines() if ln.strip()]

    return _lines("wd_alerts.log"), _lines("wd.log")


def test_persisting_crash_alerts_once_not_once_per_poll(tmp_path: Path) -> None:
    """THE REGRESSION: 27 polls of the same crash must yield ONE alert line."""
    alerts, log = _run(tmp_path, ["CRASHED"] * 27)
    assert len(log) == 27, "every poll must still be logged in full"
    assert len(alerts) == 1, f"expected 1 alert, got {len(alerts)}:\n" + "\n".join(alerts)
    assert "CRASHED" in alerts[0]


def test_recovery_rearms_so_the_next_episode_alerts(tmp_path: Path) -> None:
    """CRASHED -> OK -> CRASHED is TWO episodes and must produce TWO alerts.

    A de-duplicator that never re-arms goes silent exactly once: on the second
    failure. That is a worse bug than the spam it replaces.
    """
    alerts, _ = _run(tmp_path, ["CRASHED", "CRASHED", "OK", "CRASHED", "CRASHED"])
    assert len(alerts) == 2, "\n".join(alerts)


def test_state_change_between_non_ok_states_alerts_again(tmp_path: Path) -> None:
    """STALLED -> CRASHED is news even though both are non-OK."""
    alerts, _ = _run(tmp_path, ["STALLED", "STALLED", "CRASHED", "CRASHED"])
    assert len(alerts) == 2
    assert "STALLED" in alerts[0]
    assert "CRASHED" in alerts[1]


def test_intentional_stop_marker_suppresses_all_alerts(tmp_path: Path) -> None:
    """`train.sh stop` sets the marker; a deliberate multi-hour stop is silent."""
    alerts, log = _run(tmp_path, ["STOPPED"] * 10, marker=True)
    assert alerts == []
    assert len(log) == 10, "suppressed alerts are still fully logged"


def test_ok_only_never_alerts(tmp_path: Path) -> None:
    alerts, _ = _run(tmp_path, ["OK"] * 5)
    assert alerts == []


def test_notify_cmd_fires_once_per_episode_and_is_failsoft(tmp_path: Path) -> None:
    """Notifier runs on transitions only, and a failing notifier cannot kill the loop."""
    receipts = tmp_path / "receipts.txt"
    notifier = tmp_path / "notify.sh"
    notifier.write_text(f"#!/bin/bash\necho \"$1\" >> {receipts}\nexit 1\n")  # always fails
    notifier.chmod(0o755)

    alerts, log = _run(tmp_path, ["CRASHED"] * 4 + ["OK"] + ["CRASHED"] * 3,
                       notify=str(notifier))
    assert len(log) == 8, "the loop survived a notifier that exits non-zero"
    assert len(alerts) == 2
    fired = [ln for ln in receipts.read_text().splitlines() if ln.strip()]
    assert len(fired) == 2, f"notifier should fire per episode, got {fired}"
