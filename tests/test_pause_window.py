"""``scripts/pause_window.sh``: pause training, drain selfplay, run a job, resume.

Every test here corresponds to a way the OBVIOUS version of this script is
silently wrong, three of which were found by executing the procedure by hand on
2026-08-09 (docs/experiment_ledger.md, "the ack-gated pause window"):

* a stale ``.paused_<dead_trial>.ack`` makes an ack wait fire INSTANTLY, so the
  workers are killed while ``_revive_fleet`` is still live and get relaunched;
* the drain must not start before the ack, for the same reason;
* the marker must be released on EVERY path -- a script that dies holding it
  leaves production parked indefinitely.

The tests drive the real script with ``pgrep``/``pkill`` stubbed on PATH, so
they exercise its actual control flow rather than a paraphrase of it.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pause_window.sh"
TRAIN_SH = REPO / "scripts" / "train.sh"
TRIAL_ID = "abc12_00000"

# Generous, because selfplay workers run at nice 15 and CI is not a quiet
# machine; every wait below polls at 1s and these are ceilings, not sleeps.
ACK_WAIT = 40


def _sandbox(tmp_path: Path, *, workers: bool = False) -> tuple[Path, Path, Path]:
    """Build a fake work_dir + a stub bin dir. Returns (work_dir, bin, calls)."""
    work = tmp_path / "runs" / "x"
    tune = work / "tune"
    (tune / f"train_trial_{TRIAL_ID}_0_lr=0.0000_2026-01-01_00-00-00").mkdir(parents=True)
    (work / "server" / "trials" / TRIAL_ID / "workers" / "worker_00" / "selfplay_resume").mkdir(
        parents=True,
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    ack = tune / f".paused_{TRIAL_ID}.ack"

    # pgrep: reports one fake pid until pkill runs, then nothing -- so the
    # script's drain-wait loop terminates the way it would in production.
    (bin_dir / "pgrep").write_text(
        "#!/bin/sh\n"
        f'if [ "{int(workers)}" = "1" ] && [ ! -f "{tmp_path}/killed" ]; then echo 424242; exit 0; fi\n'
        "exit 1\n",
    )
    # ⚑ RECORD THE ACK'S CONTENT, NOT ITS EXISTENCE.
    # This stub used to ask only `[ -e "$ack" ]`. That is the same defect the
    # script it tests exists to prevent, one level up: with the stale-ack
    # removal DELETED, the pre-existing ack satisfies the wait instantly, pkill
    # fires immediately, and an existence check still sees an ack and writes
    # "pkill after_ack" -- the exact string the passing assertion demanded. A
    # reviewer deleted `pause_window.sh:111-114` and the test SURVIVED.
    # The content distinguishes them: the trial's real ack says `next_iter=7`,
    # every stale one planted below says something else.
    (bin_dir / "pkill").write_text(
        "#!/bin/sh\n"
        f'if [ -e "{ack}" ]; then echo "pkill ack:$(tr -d \'\\n\' < "{ack}")" >> "{calls}"; '
        f'else echo "pkill NO_ack" >> "{calls}"; fi\n'
        f'touch "{tmp_path}/killed"\n'
        "exit 0\n",
    )
    for f in ("pgrep", "pkill"):
        (bin_dir / f).chmod(0o755)
    return work, bin_dir, calls


def _run(work: Path, bin_dir: Path, *cmd: str, timeout: int = 90, **kw: str):
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAE_PAUSE_POLL_SECONDS"] = "1"
  # Most sandboxes here run with `workers=False` because what they exercise is
  # the marker/ack/trap machinery, not the drain. Zero matched workers is now a
  # REFUSAL (a pattern that stopped matching is indistinguishable from a fleet
  # that is down, and the expensive reading is the wrong one), so those cases
  # opt out explicitly. A caller passing this key wins -- the gate's own tests
  # set it to "0" to exercise the default.
    env["CAE_PAUSE_ALLOW_NO_WORKERS"] = "1"
    env.update(kw)
    return subprocess.run(
        [str(SCRIPT), "--work-dir", str(work), "--trial-id", TRIAL_ID, "--", *cmd],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO, check=False,
    )


def _ack_after(tune: Path, delay: float) -> subprocess.Popen[bytes]:
    """Write the trial's ack a moment from now, as the trial itself would."""
    ack = tune / f".paused_{TRIAL_ID}.ack"
    return subprocess.Popen(
        ["sh", "-c", f'sleep {delay}; echo "trial={TRIAL_ID} next_iter=7" > "{ack}"'],
    )


def test_the_worker_pattern_matches_train_sh_exactly() -> None:
    """THE SINGLE-SOURCE GUARANTEE. train.sh defines the pattern inside a shell
    function, so it cannot be sourced; this script therefore carries its own
    copy. A copy edited in one place and not the other is how the drain silently
    starts matching nothing -- the same defect
    `test_the_worker_pattern_is_defined_once` exists to prevent inside train.sh.
    """
    train = re.findall(r"local wpat='([^']+)'", TRAIN_SH.read_text())
    assert len(train) == 1, f"train.sh no longer defines wpat exactly once: {train}"
    ours = re.findall(r"^WORKER_PATTERN='([^']+)'", SCRIPT.read_text(), re.M)
    assert ours == train, f"pattern drifted: pause_window={ours} train.sh={train}"


@pytest.mark.parametrize("workers", [True, False])
def test_the_marker_is_released_even_when_the_command_fails(
    tmp_path: Path, workers: bool,
) -> None:
    """Leaving production parked is the one unrecoverable mistake here, so the
    release is a trap, not a trailing line. The command's exit code must still
    propagate, or a failing job inside a window would look like a success."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=workers)
    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "sh", "-c", "exit 17")
    finally:
        proc.wait()
    assert r.returncode == 17, r.stderr
    assert not (work / "tune" / "pause.txt").exists(), "marker survived a failed command"


def test_the_drain_never_starts_before_the_trial_has_parked(tmp_path: Path) -> None:
    """THE ORDERING TEST, and the reason the ack gate exists at all.

    `_revive_fleet` runs INSIDE the ingest phase, so it is inert only while the
    trial is parked in `_wait_if_paused`. A drain that fires early is undone by
    the driver relaunching every worker, and the window then measures a
    contended machine while reporting success.
    """
    work, bin_dir, calls = _sandbox(tmp_path, workers=True)
    proc = _ack_after(work / "tune", 4.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
    assert r.returncode == 0, r.stderr
    assert calls.read_text().strip() == f"pkill ack:trial={TRIAL_ID} next_iter=7", (
        f"the drain raced the pause: {calls.read_text()!r}"
    )


def test_a_stale_ack_from_a_DIFFERENT_trial_does_not_satisfy_the_wait(
    tmp_path: Path,
) -> None:
    """⚑ THE TRAP. `_clear_pause_acks` runs in a `finally`, which a hard kill
    skips, so dead trials really do leave `.paused_<id>.ack` behind -- one was
    found in the live tune dir on 2026-08-09. A wait polling for "an ack exists"
    is satisfied instantly and kills the workers while revive is live.
    """
    work, bin_dir, calls = _sandbox(tmp_path, workers=True)
    (work / "tune" / ".paused_dead99_00000.ack").write_text("trial=dead99_00000\n")
    proc = _ack_after(work / "tune", 4.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
    assert r.returncode == 0, r.stderr
    assert calls.read_text().strip() == f"pkill ack:trial={TRIAL_ID} next_iter=7", (
        f"a foreign trial's stale ack was accepted as this trial's park signal: "
        f"{calls.read_text()!r}"
    )


def test_a_preexisting_ack_for_this_trial_is_ignored_until_it_is_refreshed(
    tmp_path: Path,
) -> None:
    """Same trap, harder case: the stale ack carries OUR trial id, so an
    id-aware poll still fires instantly.

    ⚑ IGNORED, NOT DELETED. An earlier revision removed it as "stale by
    construction". It is not stale by construction -- a previous window can
    release the marker while the trial is still parked, leaving a LIVE ack, and
    deleting that destroys the signal `graceful_restart.py` uses as its primary
    pause detector. The rule is freshness (mtime >= our marker's), the same one
    `_pause_ack_files` already applies.
    """
    work, bin_dir, calls = _sandbox(tmp_path, workers=True)
    (work / "tune" / f".paused_{TRIAL_ID}.ack").write_text("trial=old next_iter=1\n")
    proc = _ack_after(work / "tune", 4.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
    assert r.returncode == 0, r.stderr
    recorded = calls.read_text().strip()
    assert recorded == f"pkill ack:trial={TRIAL_ID} next_iter=7", (
        f"a pre-existing ack for this trial was trusted instead of discarded: {recorded!r}"
    )
  # Explicit, because this is the assertion the earlier existence-only version
  # could not make: the ack the drain saw must be the FRESH one.
    assert "next_iter=1" not in recorded, (
        "the drain fired on the STALE ack -- the removal at pause_window.sh:111 is inert"
    )


def test_an_ack_that_never_arrives_aborts_WITHOUT_draining(tmp_path: Path) -> None:
    """If the trial cannot park (crashed, wedged, or mid-restart) the safe move
    is to abort. Draining anyway would kill the fleet with revive live, which is
    strictly worse than not having tried."""
    work, bin_dir, calls = _sandbox(tmp_path, workers=True)
    r = _run(work, bin_dir, "true", CAE_PAUSE_ACK_TIMEOUT="3")
    assert r.returncode != 0
    assert "no ack" in r.stderr.lower()
    assert not calls.exists(), "aborted on timeout but still killed the workers"
    assert not (work / "tune" / "pause.txt").exists(), "aborted while holding the marker"


def test_the_job_does_not_run_when_the_pause_never_lands(tmp_path: Path) -> None:
    """The job must be gated on the pause, not merely ordered after it: a
    ratchet that runs anyway is the contended measurement this whole script
    exists to stop, and it would be recorded as if it were clean."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    ran = tmp_path / "ran"
    r = _run(work, bin_dir, "touch", str(ran), CAE_PAUSE_ACK_TIMEOUT="3")
    assert r.returncode != 0
    assert not ran.exists(), "the job ran despite training never pausing"


def test_it_refuses_when_a_marker_is_already_present(tmp_path: Path) -> None:
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    marker = work / "tune" / "pause.txt"
    marker.write_text("someone else\n")
    ran = tmp_path / "ran"
    r = _run(work, bin_dir, "touch", str(ran))
    assert r.returncode != 0
    assert "already exists" in r.stderr
    assert not ran.exists()
    assert marker.read_text() == "someone else\n", (
        "refused, but clobbered the other window's marker on the way out"
    )


def test_the_resume_baseline_is_taken_before_the_drain(tmp_path: Path) -> None:
    """`resumed_inflight_games` is a TOTAL, so without a pre-drain count of
    selfplay_resume/ it cannot be attributed to this drain -- the gap in the
    2026-08-09 run, where 224 resumed against 93 banked."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    d = work / "server" / "trials" / TRIAL_ID / "workers" / "worker_00" / "selfplay_resume"
    for i in range(3):
        (d / f"g{i}.npz").write_bytes(b"x")
    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
    assert r.returncode == 0, r.stderr
    assert "BEFORE drain" in r.stdout
    assert re.search(r"selfplay_resume\s+3", r.stdout), (
        f"pre-drain resume count not reported:\n{r.stdout}"
    )


def test_the_script_is_executable_and_passes_a_syntax_check() -> None:
    assert os.access(SCRIPT, os.X_OK), "not executable"
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_it_does_not_leave_the_marker_behind_when_interrupted(tmp_path: Path) -> None:
    """SIGINT/SIGTERM (operator Ctrl-C, a CI timeout) must release too."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAE_PAUSE_POLL_SECONDS"] = "1"
  # This test builds its own Popen rather than going through `_run`, so it
  # needs the same opt-out: what it exercises is the signal trap, and the
  # no-worker refusal would fire before the marker ever appears.
    env["CAE_PAUSE_ALLOW_NO_WORKERS"] = "1"
    p = subprocess.Popen(
        [str(SCRIPT), "--work-dir", str(work), "--trial-id", TRIAL_ID, "--", "sleep", "60"],
        env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ack = work / "tune" / f".paused_{TRIAL_ID}.ack"
    deadline = time.monotonic() + ACK_WAIT
    while not (work / "tune" / "pause.txt").exists():
        assert time.monotonic() < deadline, "marker never appeared"
        time.sleep(0.2)
    ack.write_text(f"trial={TRIAL_ID} next_iter=7\n")
    time.sleep(3)
    p.terminate()
    p.wait(timeout=30)
    assert not (work / "tune" / "pause.txt").exists(), "interrupted while holding the marker"


def test_the_ratchet_does_not_wrap_its_own_arenas() -> None:
    """The wrapper belongs around the WHOLE ratchet, not around each arena.

    Two reasons, both load-bearing: one pause covers both series (the ~35 min
    2x200 measurement), and `daily_gate_ratchet.sh` writes each arena's stdout
    to a log whose `[arena] Elo:` lines an outcome parser reads -- wrapper
    chatter inside that redirect would be parsed as arena output. The three
    tests below execute the loop's side of this; only the negative (the ratchet
    must NOT wrap itself) has no run to observe it.
    """
    ratchet = (REPO / "scripts" / "daily_gate_ratchet.sh").read_text()
    assert "pause_window.sh" not in ratchet, (
        "the ratchet wraps its own arenas: that pauses twice and pollutes the "
        "per-arena logs the outcome parser reads"
    )


# ── The loop's routing, EXECUTED ─────────────────────────────────────────────
# These replace a source-text assertion that `ratchet_loop.sh` CONTAINS the
# string "pause_window.sh -- bash scripts/daily_gate_ratchet.sh". It was green
# while the wiring was broken: `bash <missing script>` exits 127, the loop read
# that as an ordinary ratchet failure, and three tests in
# test_ratchet_search_shape.py went red in CI. A grep for the call cannot see
# whether the call WORKS.

_PAUSE_STUB = """#!/usr/bin/env bash
# Records its argv, then runs the job, so the assertion is about what the loop
# actually invoked rather than about the text of the loop.
printf '%s\\n' "$*" >> "$PAUSE_CALLS"
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
[ "$1" = "--" ] && shift
"$@"
"""


def _ratchet_sandbox(tmp_path: Path, *, wrapper: bool):
    """A ratchet sandbox, optionally carrying a stub pause_window.sh."""
    from tests.test_ratchet_search_shape import _sandbox

    root = _sandbox(tmp_path)
    calls = root / "pause_calls.txt"
    if wrapper:
        stub = root / "scripts" / "pause_window.sh"
        stub.write_text(_PAUSE_STUB)
        stub.chmod(0o755)
    return root, calls


def _pause_calls(calls: Path) -> list[str]:
    return calls.read_text().splitlines() if calls.exists() else []


def test_the_loop_routes_the_whole_ratchet_through_the_window_by_default(tmp_path: Path) -> None:
    """Default ON, ONE window, and the whole ratchet inside it."""
    from tests.test_ratchet_search_shape import _csv_rows, _run_one_poll

    root, calls = _ratchet_sandbox(tmp_path, wrapper=True)
    rc, out = _run_one_poll(root, env={"PAUSE_CALLS": str(calls)})

    assert rc == 0, f"the poll must still succeed through the wrapper:\n{out}"
  # Repo-RELATIVE, like `ratchet_common.sh`'s own `${TRAIN_WORK_DIR:-runs/pbt2_small}`:
  # the wrapper inherits the cwd that ratchet_common.sh already cd'd into.
    assert _pause_calls(calls) == [
        "--work-dir runs/pbt2_small -- bash scripts/daily_gate_ratchet.sh",
    ], (
        f"the wrapper was invoked {_pause_calls(calls)!r}; it must wrap the whole "
        f"ratchet exactly once:\n{out}"
    )
    assert len(_csv_rows(root)) == 1, f"the row must still be written:\n{out}"


def test_a_missing_wrapper_degrades_to_running_beside_training(tmp_path: Path) -> None:
    """⚑ THE CI REGRESSION. `bash <missing>` exits 127, which `ratchet_outcome`
    reads as an ordinary failure -- so the day gets no strength row, no give-up
    stamp, and a retry every poll until midnight. The comment in the loop
    promised this "degrades to the old contended behaviour"; until the presence
    test existed, it did not. Loud, and still measured."""
    from tests.test_ratchet_search_shape import _csv_rows, _loop_state, _run_one_poll

    root, calls = _ratchet_sandbox(tmp_path, wrapper=False)
    rc, out = _run_one_poll(root, env={"PAUSE_CALLS": str(calls)})

    assert rc == 0, f"a missing wrapper must not cost the day's reading:\n{out}"
    assert len(_csv_rows(root)) == 1, f"the ratchet must still have run:\n{out}"
    assert _loop_state(root)[0] != "", f"the day must be stamped as read:\n{out}"
    assert "not readable" in out, f"the degradation must be LOUD:\n{out}"


def test_the_window_can_be_switched_off(tmp_path: Path) -> None:
    """CAE_RATCHET_PAUSE_WINDOW=0 is the documented escape hatch; a flag that
    cannot be observed to turn anything off is not an escape hatch."""
    from tests.test_ratchet_search_shape import _csv_rows, _run_one_poll

    root, calls = _ratchet_sandbox(tmp_path, wrapper=True)
    rc, out = _run_one_poll(
        root, env={"PAUSE_CALLS": str(calls), "CAE_RATCHET_PAUSE_WINDOW": "0"},
    )

    assert rc == 0, out
    assert _pause_calls(calls) == [], f"the window ran despite being switched off:\n{out}"
    assert len(_csv_rows(root)) == 1, f"the ratchet must still run beside training:\n{out}"


# ── The drain as a GATE ──────────────────────────────────────────────────────
# Review finding: draining nothing exited 0. The wrapper then paid the full
# production pause and handed back exactly the contended arena it exists to
# prevent -- and `ratchet_outcome` stamped the day, so the row was filed as a
# clean strength reading. Three ways in, three gates.


def test_a_pgrep_usage_error_is_not_read_as_no_workers(tmp_path: Path) -> None:
    """`|| true` erased pgrep's status. A dropped `--` makes the pattern look
    like options (`pgrep -f "-m chess..."` => "invalid option -- 'm'", rc 2),
    which then reads as "nothing to drain" -- the mutation a reviewer made,
    which SURVIVED because the stubs ignored their arguments. rc>=2 is now
    fatal, and it fires before the marker is ever touched."""
    work, bin_dir, calls = _sandbox(tmp_path, workers=True)
    (bin_dir / "pgrep").write_text(
        "#!/bin/sh\necho \"pgrep: invalid option -- 'm'\" >&2\nexit 2\n",
    )
    (bin_dir / "pgrep").chmod(0o755)

    ran = tmp_path / "ran"
    r = _run(work, bin_dir, "touch", str(ran), CAE_PAUSE_ALLOW_NO_WORKERS="0")

    assert r.returncode != 0, "a broken worker pattern was accepted"
    assert "pattern is broken" in r.stderr, r.stderr
    assert not ran.exists(), "the job ran without any drain having happened"
    assert not (work / "tune" / "pause.txt").exists(), "production was parked for nothing"
    assert not calls.exists(), "pkill was called despite pgrep failing"


def test_zero_matched_workers_refuses_before_touching_the_marker(tmp_path: Path) -> None:
    """While training runs there are always workers. Zero matches means the
    fleet is down OR the pattern has drifted from its argv, and only the
    caller can tell which -- so refuse, and refuse BEFORE the pause so being
    wrong costs nothing. `CAE_PAUSE_ALLOW_NO_WORKERS=1` is the way to say
    "I know, run it anyway"."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    ran = tmp_path / "ran"
    r = _run(work, bin_dir, "touch", str(ran), CAE_PAUSE_ALLOW_NO_WORKERS="0")

    assert r.returncode != 0
    assert "no selfplay workers matched" in r.stderr, r.stderr
    assert not ran.exists()
    assert not (work / "tune" / "pause.txt").exists(), (
        "refused, but only after parking production"
    )


def test_a_worker_that_survives_sigterm_stops_the_job(tmp_path: Path) -> None:
    """The other half of the same gate. A worker that ignores SIGTERM still
    holds its server lease and still plays; the old code logged "continuing
    anyway" and ran the job beside it. The check reads the pids taken at
    BASELINE rather than re-running pgrep, so a pattern that stopped matching
    cannot report a clean drain either."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
  # A real process, so `kill -0` has something true to say. pkill is stubbed,
  # so nothing actually signals it -- which is precisely the case under test.
    victim = subprocess.Popen(["sleep", "120"])
    (bin_dir / "pgrep").write_text(f"#!/bin/sh\necho {victim.pid}\nexit 0\n")
    (bin_dir / "pgrep").chmod(0o755)

    ran = tmp_path / "ran"
    ack = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "touch", str(ran), CAE_PAUSE_DRAIN_TIMEOUT="2")
    finally:
        ack.wait()
        victim.terminate()
        victim.wait()

    assert r.returncode != 0, "an undrained fleet was treated as drained"
    assert "survived SIGTERM" in r.stderr, r.stderr
    assert not ran.exists(), "the job ran beside a live worker"
    assert not (work / "tune" / "pause.txt").exists(), "died holding the marker"


def test_the_double_dash_is_required_by_the_REAL_pgrep() -> None:
    """⚑ PINS THE PREMISE THE STUB CANNOT.

    The gate above proves "rc>=2 aborts". That is only worth having if a
    realistic mistake actually produces rc>=2, and the stubs in this file
    ignore their arguments -- which is exactly why a reviewer's mutation
    dropping `--` SURVIVED the whole suite. So ask the real binary.

    The pattern begins with `-m`, so without the separator pgrep parses it as
    options. Run against the actual `WORKER_PATTERN` this script uses, not a
    paraphrase of it.
    """
    pattern = re.findall(r"^WORKER_PATTERN='([^']+)'", SCRIPT.read_text(), re.M)[0]

    without = subprocess.run(
        ["pgrep", "-f", pattern], capture_output=True, text=True, check=False,
    )
    assert without.returncode >= 2, (
        "pgrep tolerated a leading-dash pattern without `--`; the abort this "
        f"suite relies on would never fire (rc={without.returncode})"
    )
    assert "invalid option" in without.stderr, without.stderr

    with_sep = subprocess.run(
        ["pgrep", "-f", "--", pattern], capture_output=True, text=True, check=False,
    )
    assert with_sep.returncode in (0, 1), (
        "with `--` pgrep must either match (0) or not match (1), never error: "
        f"rc={with_sep.returncode} {with_sep.stderr!r}"
    )


def test_a_never_refreshed_ack_aborts_FAST_instead_of_holding_the_marker(
    tmp_path: Path,
) -> None:
    """⚑ NB1, BOUNDED. Window A releases the marker while the trial is still
    inside `_wait_if_paused` (it re-reads only every `pause_poll_seconds`,
    production default 60). Window B then finds a live ack that will never be
    rewritten, because `_wait_if_paused` guards on an `announced` flag.

    Freshness alone would make B hold the marker for the full
    `CAE_PAUSE_ACK_TIMEOUT` -- 1800s of parked production, per poll, all day.
    So a pre-existing ack shortens the deadline and the message names the
    cause. The test proves BOTH: it aborts, and it aborts on the short clock.
    """
    work, bin_dir, calls = _sandbox(tmp_path, workers=True)
    (work / "tune" / f".paused_{TRIAL_ID}.ack").write_text(
        f"trial={TRIAL_ID} next_iter=1\n",
    )
    ran = tmp_path / "ran"

    started = time.monotonic()
    r = _run(
        work, bin_dir, "touch", str(ran),
        CAE_PAUSE_STALE_ACK_TIMEOUT="3", CAE_PAUSE_ACK_TIMEOUT="600",
    )
    elapsed = time.monotonic() - started

    assert r.returncode != 0
    assert "will not re-ack" in r.stderr, r.stderr
    assert not ran.exists(), "ran the job on a pause it never confirmed"
    assert not calls.exists(), "drained on a stale ack"
    assert not (work / "tune" / "pause.txt").exists(), "aborted holding the marker"
    assert elapsed < 60, (
        f"took {elapsed:.0f}s -- it waited on CAE_PAUSE_ACK_TIMEOUT (600) rather "
        "than the short stale-ack clock, so production stays parked"
    )
