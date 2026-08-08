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
# Two plan entries model output the real tool never means to produce:
#   SILENT    -- nothing at all on either stream, rc=137 (the watchdog's own
#                python was SIGKILLed, e.g. by the OOM killer)
#   TRACEBACK -- a multi-line traceback, rc=1
_STUB = """#!/usr/bin/env python3
import sys
from pathlib import Path
plan = Path(__file__).with_name("plan.txt").read_text().split("\\n")
cur = Path(__file__).with_name("cursor.txt")
i = int(cur.read_text()) if cur.exists() else 0
state = plan[i].strip() if i < len(plan) else plan[-1].strip()
cur.write_text(str(i + 1))
if state == "SILENT":
    sys.exit(137)
if state == "TRACEBACK":
    sys.stderr.write("Traceback (most recent call last):\\n"
                     "  File \\"train_watchdog.py\\", line 1, in <module>\\n"
                     "ValueError: boom\\n")
    sys.exit(1)
codes = {"OK": 0, "STOPPED": 1, "PAUSED-HELD": 2, "STALLED": 3, "CRASHED": 5}
print(f"watchdog: {state} pid=4242 rows=249 minutes_flat=0.0")
sys.exit(codes[state])
"""


def _run(tmp_path: Path, plan: list[str], *, marker: bool = False,
         notify: str | None = None, auto_recover: bool = False,
         alertf: str = "scratchpad/wd_alerts.log",
         ) -> tuple[list[str], list[str]]:
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
        WATCHDOG_ALERTF=alertf,
        WATCHDOG_STATEF=str(tmp_path / "state.json"),
        WATCHDOG_LAST_ALERT_F=str(tmp_path / "last_alert"),
        WATCHDOG_LAST_ESCALATE_F=str(tmp_path / "last_escalate"),
        # Never let a test touch the real /tmp recovery stamp: writing it would
        # suppress a genuine production auto-recovery for the 2h cooldown.
        WATCHDOG_RECOVER_STAMP=str(tmp_path / "last_recover"),
        WATCHDOG_AUTO_RECOVER="1" if auto_recover else "0",
    )
    if notify is not None:
        env["WATCHDOG_NOTIFY_CMD"] = notify

    subprocess.run(["bash", str(LOOP)], env=env, check=True, timeout=120,
                   capture_output=True)

    def _lines(rel: str) -> list[str]:
        p = root / rel
        if not p.exists():
            return []
        return [ln for ln in p.read_text().splitlines() if ln.strip()]

    return _lines(alertf), _lines("scratchpad/wd.log")


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


def test_notify_cmd_accepts_a_command_with_arguments(tmp_path: Path) -> None:
    """`notify-send training` is the form train_watchdog.py's docstring advertises.

    A bare ``"$CMD" "$LINE"`` looks for an executable literally named
    "notify-send training", fails, and the ``|| true`` swallows the error — a
    notifier that is configured, accepted, and silently never fires.
    """
    receipts = tmp_path / "receipts.txt"
    notifier = tmp_path / "notify.sh"
    notifier.write_text(f"#!/bin/bash\necho \"$*\" >> {receipts}\n")
    notifier.chmod(0o755)

    alerts, _ = _run(tmp_path, ["CRASHED"] * 3, notify=f"{notifier} --urgent")
    assert len(alerts) == 1
    fired = [ln for ln in receipts.read_text().splitlines() if ln.strip()]
    assert len(fired) == 1, f"multi-word notifier never fired: {fired}"
    assert fired[0].startswith("--urgent "), fired[0]
    assert "CRASHED" in fired[0]


def test_a_silent_watchdog_alerts_and_keeps_alerting_across_episodes(
    tmp_path: Path,
) -> None:
    """If the watchdog itself is SIGKILLed it emits NOTHING. That must be loud.

    Regression guard for the dedupe's worst failure mode: an empty state string
    compares equal to the empty "no previous alert" value, so the first poll is
    suppressed — and because suppression writes no key, every later poll is
    suppressed too. Silence about a dead watchdog, forever.
    """
    alerts, log = _run(tmp_path, ["SILENT"] * 6)
    assert len(log) == 6
    assert len(alerts) == 1, f"expected 1 alert, got {len(alerts)}:\n" + "\n".join(alerts)
    assert "UNPARSEABLE-rc137" in alerts[0]
    assert "no output" in alerts[0], "the alert must say WHY it is empty"

    # And the episode boundary still works when the watchdog comes back.
    alerts2, _ = _run(tmp_path / "b", ["SILENT", "SILENT", "OK", "SILENT"])
    assert len(alerts2) == 2, "\n".join(alerts2)


def test_a_traceback_yields_one_single_line_state(tmp_path: Path) -> None:
    """A multi-line traceback must become ONE alert line with a ONE-line key.

    Two ways this goes wrong: `awk '{print $2}'` builds a multi-line "state" that
    can never compare equal to the stored key (so every poll alerts), and an
    unflattened append makes one alert read as N — in a file whose entire
    contract is that every line is news.
    """
    alerts, _ = _run(tmp_path, ["TRACEBACK"] * 5)
    assert len(alerts) == 1, "\n".join(alerts)
    assert "ValueError: boom" in alerts[0], "the flattened line keeps the detail"
    key = (tmp_path / "last_alert").read_text()
    assert "\n" not in key, f"dedupe key is multi-line: {key!r}"
    assert key == "(most", key  # field 2 of the traceback's first line


def test_a_failed_append_is_retried_not_deduped_away(tmp_path: Path) -> None:
    """Arming the dedupe key on an UNDELIVERED alert loses the whole episode.

    $ALERTF and the key file live on different filesystems (repo disk vs tmpfs),
    so the append can fail on its own — a full disk, a missing directory — while
    the key write succeeds. Arm only after the append lands.
    """
    alerts, log = _run(tmp_path, ["CRASHED"] * 4,
                       alertf="scratchpad/nosuchdir/wd_alerts.log")
    assert alerts == [], "the append could not have succeeded"
    assert len(log) == 4
    key = tmp_path / "last_alert"
    assert not key.exists(), (
        "dedupe key armed for an alert that was never delivered — the next "
        "poll would be suppressed and the episode lost entirely"
    )


def test_auto_recover_escalation_is_deduped_too(tmp_path: Path) -> None:
    """A stall persisting through the cooldown is ONE piece of news, not one per poll.

    Un-gated, this branch alone reproduces the 27-identical-lines pattern.
    """
    # First STALLED poll fires recovery (stamp absent) and writes the stamp;
    # every later poll lands inside the cooldown and escalates.
    alerts, _ = _run(tmp_path, ["STALLED"] * 12, auto_recover=True)
    firing = [ln for ln in alerts if "AUTO-RECOVER FIRING" in ln]
    suppressed = [ln for ln in alerts if "AUTO-RECOVER SUPPRESSED" in ln]
    assert len(firing) == 1, "\n".join(alerts)
    assert len(suppressed) == 1, (
        f"escalation repeated {len(suppressed)}x across 11 in-cooldown polls:\n"
        + "\n".join(suppressed)
    )
    assert "NEEDS HUMAN" in suppressed[0]
