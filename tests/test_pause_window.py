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
import shlex
import subprocess
import time
from pathlib import Path

import pytest

from tests.script_loading import load_script_module

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
    """The drain does not know what was already in the resume dirs.

    ⚑ NOT because `resumed_inflight_games` is a total -- an earlier version of
    this docstring said so and it is wrong. `finalize.py` increments it once per
    FINALIZED game that carried `resumed_from_disk`, so it is per-ingest and
    decays as the backlog clears (0 on every row through iter 567 on 2026-08-09,
    then 224, 456, 477, ... 1, back to 0 at iter 586: 2,963 games against 93
    banked). Nothing can be subtracted from a file count. The baseline is worth
    taking because it bounds how much of that backlog PRE-EXISTED, which is the
    only way the 32x gap could ever be explained.

    ⚑ AND IT MUST BE TAKEN BEFORE THE DRAIN, which the old assertions could not
    see: they only checked that the count was printed and that it said 3, and
    both survive the block being moved after the drain -- the files are still
    there. Two things fix that. The pkill stub now ADDS files, the way a real
    suspend does, so a post-drain baseline reports 8 rather than 3; and the
    baseline line must appear before the marker is even set.
    """
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    d = work / "server" / "trials" / TRIAL_ID / "workers" / "worker_00" / "selfplay_resume"
    for i in range(3):
        (d / f"g{i}.npz").write_bytes(b"x")
    # The drain banks in-flight games INTO this directory; a baseline taken
    # after it would count them as though they had been there all along.
    (bin_dir / "pkill").write_text(
        (bin_dir / "pkill").read_text().replace(
            "exit 0\n",
            f'for i in 3 4 5 6 7; do : > "{d}/banked_$i.npz"; done\nexit 0\n',
        ),
    )
    (bin_dir / "pkill").chmod(0o755)

    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
    assert r.returncode == 0, r.stderr
    assert "BEFORE drain" in r.stdout
    assert re.search(r"selfplay_resume\s+3\b", r.stdout), (
        "the resume baseline reports a count taken AFTER the drain banked its "
        f"games (expected 3, the pre-drain state):\n{r.stdout}"
    )
    assert "selfplay_resume 8" not in r.stdout, (
        f"the baseline block ran after the drain:\n{r.stdout}"
    )
    # Ordering, independently of the count: the baseline is taken before
    # ANYTHING is signalled, so it cannot follow the marker.
    assert r.stdout.index("BEFORE drain") < r.stdout.index("marker set"), (
        f"the baseline was taken after production was parked:\n{r.stdout}"
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
    rc = p.wait(timeout=30)
    assert not (work / "tune" / "pause.txt").exists(), "interrupted while holding the marker"
  # ⚑ 130, NOT the wrapper-failure code. This is the ONE path where the EXIT
  # trap is still armed while a status other than a setup failure is in flight,
  # so it is the only place the `PHASE=job` guard is observable -- deleting
  # `PHASE=job` SURVIVED every other test, because the normal path disarms the
  # trap with `trap - EXIT` before exiting. An interrupt reported as "the
  # wrapper failed" would be counted against CAE_RATCHET_PAUSE_MAX_FAILS and
  # would disable the window for the rest of the day, for an operator Ctrl-C.
    assert rc == 130, (
        f"an interrupt exited {rc}; 130 is the documented interrupt status and "
        "anything else is read by the loop as a wrapper failure"
    )


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
  # ⚑ AND --trial-id, WHICH IS LOAD-BEARING. Without it the wrapper runs its own
  # selector and can resolve a DIFFERENT trial from the one this poll's `iter`,
  # `ck_ready` and snapshot came from -- production parked to measure a trial
  # that is not the one being parked. The value is the directory the LOOP chose,
  # so the two agree by construction rather than by both being right.
    assert _pause_calls(calls) == [
        "--work-dir runs/pbt2_small --trial-id train_trial_x "
        "-- bash scripts/daily_gate_ratchet.sh",
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


# ── NB4: the worker log-file parse, and its THREE outcomes ───────────────────
# The parse was `grep -A1 -- '--log-file' | tail -1`, a weaker reimplementation
# of train.sh:266's anchored `awk '/^--log-file$/{getline; print; exit}'`. It is
# weaker three ways -- it matches any argv element CONTAINING the string,
# `-A1` prints the line after EVERY match so `tail -1` silently takes the last,
# and a failed parse writes nothing to $OFFSETS -- and all three end in the same
# place: the suspend-evidence block is skipped ENTIRELY, so a window in which
# nothing was banked prints exactly what a clean one prints.

_SUSPEND = "selfplay resume: suspended games=93 records=6378 skipped=0"


def _worker_with_argv(tmp_path: Path, *extra: str) -> int:
    """A real process whose /proc/<pid>/cmdline carries `extra`. Returns its pid.

    Real, because the parse reads /proc -- a stub could only test a paraphrase
    of the thing under test.

    ⚑ DOUBLE-FORKED ON PURPOSE. As a direct child of pytest it would sit as an
    unreaped ZOMBIE after the drain kills it, `kill -0` on a zombie SUCCEEDS,
    and the script's survived-SIGTERM gate would fire on every one of these
    tests. Backgrounded from a bash that then exits, it is reparented to init
    and really does disappear.
    """
    pidfile = tmp_path / f"worker_{len(extra)}_{abs(hash(extra)) % 10**6}.pid"
    subprocess.run(
        ["bash", "-c",
         'python3 -c "import time; time.sleep(300)" '
         f'{shlex.join(extra)} & echo $! > {shlex.quote(str(pidfile))}'],
        check=True,
    )
    return int(pidfile.read_text().strip())


def _reap(pid: int) -> None:
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def _drain_stub(bin_dir: Path, tmp_path: Path, victim: int, *, append_to: Path | None = None):
    """pgrep reports `victim`; pkill really kills it and appends the suspend
    line to `append_to`, the way a worker's own handler writes it."""
    (bin_dir / "pgrep").write_text(
        "#!/bin/sh\n"
        f'if [ ! -f "{tmp_path}/killed" ]; then echo {victim}; exit 0; fi\nexit 1\n',
    )
    write = f'printf "%s\\n" "{_SUSPEND}" >> "{append_to}"\n' if append_to else ""
    (bin_dir / "pkill").write_text(
        f'#!/bin/sh\n{write}kill -TERM {victim} 2>/dev/null\ntouch "{tmp_path}/killed"\nexit 0\n',
    )
    for f in ("pgrep", "pkill"):
        (bin_dir / f).chmod(0o755)


def test_the_log_file_parse_is_anchored_like_train_sh(tmp_path: Path) -> None:
    """⚑ THE MUTATION-KILLER, and it needs the flags in THIS order.

    With `--log-file <path> --log-file-level debug`, `grep -A1 -- '--log-file'`
    matches BOTH elements and `tail -1` returns "debug" -- not a file, so the
    offset is dropped and the whole evidence block silently disappears. The
    anchored awk stops at the first exact `--log-file` and returns the path.
    (Reverse the two flags and the loose parse happens to work, which is why an
    order-blind test would not have caught this.)
    """
    logf = tmp_path / "worker.log"
    logf.write_text("old generation, from an hours-old reco restart\n" + _SUSPEND + "\n")
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    victim = _worker_with_argv(tmp_path, "--log-file", str(logf), "--log-file-level", "debug")
    _drain_stub(bin_dir, tmp_path, victim, append_to=logf)

    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
        _reap(victim)

    assert r.returncode == 0, r.stderr
    assert "banked: suspended games=93 records=6378 skipped=0" in r.stdout, (
        f"the log-file parse missed the worker's log:\n{r.stdout}"
    )
    # The offset is load-bearing too: the SAME line exists before the drain, and
    # reading the whole file would report an hours-old restart as this drain's
    # proof. One match, not two.
    assert r.stdout.count("banked: suspended games=93") == 1, r.stdout


def test_a_worker_with_no_log_file_says_so_LOUDLY(tmp_path: Path) -> None:
    """worker.py defaults --log-file to None and every volunteer launch in
    README.md omits it, so this is reachable in normal use. train.sh calls it a
    THIRD state ("no evidence to read"), not a loss -- and the failure mode
    being fixed is that it printed NOTHING, which is what success looks like."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    victim = _worker_with_argv(tmp_path, "--log-file-level", "debug")
    _drain_stub(bin_dir, tmp_path, victim)

    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
        _reap(victim)

    assert r.returncode == 0, r.stderr
    assert "no --log-file in its argv" in r.stdout, (
        f"a worker with no log path was passed over in silence:\n{r.stdout}"
    )
    assert "NO suspend evidence available" in r.stdout, (
        f"nothing could be read and the run reported it like a clean one:\n{r.stdout}"
    )


def test_a_missing_log_file_is_distinguished_from_a_missing_flag(tmp_path: Path) -> None:
    """train.sh keeps these apart and so must this: one is a launch that never
    asked for a log, the other is a log that has gone missing under us."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    victim = _worker_with_argv(tmp_path, "--log-file", str(tmp_path / "not_there.log"))
    _drain_stub(bin_dir, tmp_path, victim)

    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
        _reap(victim)

    assert r.returncode == 0, r.stderr
    assert "log file does not exist" in r.stdout, r.stdout
    assert "no --log-file in its argv" not in r.stdout, (
        f"the two failure states were collapsed into one:\n{r.stdout}"
    )
    assert "NO suspend evidence available" in r.stdout, r.stdout


# ── NB8c: an interrupt must tear down the JOB'S PROCESS GROUP ────────────────


def test_an_interrupt_kills_the_jobs_GRANDCHILDREN_too(tmp_path: Path) -> None:
    """⚑ ONE SIGTERM TO THE DIRECT CHILD IS NOT A TEARDOWN.

    The job is `bash daily_gate_ratchet.sh`, which runs the arena under
    `timeout`. Signalling only the child kills the wrapper shell and leaves the
    ARENA running: the marker is then released, training resumes onto a GPU that
    still has a 16-concurrent arena on it, and 600s later the next poll opens a
    second window and a SECOND arena -- which CLAUDE.md forbids outright
    (paired/compiled arenas OOMed training twice).

    So the handler kills the process group and WAITS for it. The grandchild here
    stands in for the arena.

    ⚑ AND IT MUST BE PROMPT, which is why the deadline below is asserted.
    Mutating the group SIGTERM back to a single `kill -TERM "$CHILD"` SURVIVED
    the first version of this test: the escalation path is group-scoped too, so
    the grandchild still died -- a full `CAE_PAUSE_JOB_KILL_TIMEOUT` later, with
    production parked throughout. The timeout is therefore set well ABOVE the
    time this assertion allows, so "it got there in the end via SIGKILL" fails.
    """
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    gc_pid_file = tmp_path / "grandchild.pid"
    job = f'sleep 300 & echo $! > "{gc_pid_file}"; wait'

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAE_PAUSE_POLL_SECONDS"] = "1"
    env["CAE_PAUSE_ALLOW_NO_WORKERS"] = "1"
    env["CAE_PAUSE_JOB_KILL_TIMEOUT"] = "90"
    p = subprocess.Popen(
        [str(SCRIPT), "--work-dir", str(work), "--trial-id", TRIAL_ID,
         "--", "bash", "-c", job],
        env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + ACK_WAIT
        while not (work / "tune" / "pause.txt").exists():
            assert time.monotonic() < deadline, "marker never appeared"
            time.sleep(0.2)
        (work / "tune" / f".paused_{TRIAL_ID}.ack").write_text(
            f"trial={TRIAL_ID} next_iter=7\n",
        )
        deadline = time.monotonic() + ACK_WAIT
        while not gc_pid_file.exists():
            assert time.monotonic() < deadline, "the job never started"
            time.sleep(0.2)
        gc = int(gc_pid_file.read_text().strip())

        p.terminate()
        torn_down = time.monotonic()
        p.wait(timeout=30)
        elapsed = time.monotonic() - torn_down
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()

    assert elapsed < 25, (
        f"the teardown took {elapsed:.0f}s with CAE_PAUSE_JOB_KILL_TIMEOUT=90: "
        "the SIGTERM did not reach the job's process group and it was the "
        "SIGKILL escalation that eventually got there, with production parked "
        "for the whole wait"
    )
    assert not (work / "tune" / "pause.txt").exists(), "interrupted holding the marker"
    # Give the group kill a moment to be reaped, then insist it is gone.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(gc, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        os.kill(gc, 9)
        pytest.fail(
            f"the job's grandchild {gc} outlived the interrupt: training was "
            "resumed with the arena still running",
        )


# ── The trial id, and the work dir ───────────────────────────────────────────


def test_an_unparseable_trial_dir_is_refused_immediately(tmp_path: Path) -> None:
    """⚑ `sed` PRINTS ITS INPUT UNCHANGED WHEN THE PATTERN DOES NOT MATCH.

    So a trial-dir naming change does not fail the parse, it returns the whole
    directory name as the "trial id". The ack path becomes
    `.paused_train_trial_....ack`, which nothing will ever write, and the script
    waits out the FULL ACK_TIMEOUT holding the marker before dying with a
    message about the trial not parking. This asserts both halves: the right
    message, and that it costs nothing (an unvalidated build would sit on the
    600s clock below).
    """
    work = tmp_path / "runs" / "x"
    tune = work / "tune"
    (tune / "train_trial_nonsense").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pgrep").write_text("#!/bin/sh\necho 424242\nexit 0\n")
    (bin_dir / "pkill").write_text("#!/bin/sh\nexit 0\n")
    for f in ("pgrep", "pkill"):
        (bin_dir / f).chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAE_PAUSE_POLL_SECONDS"] = "1"
    env["CAE_PAUSE_ACK_TIMEOUT"] = "600"
    started = time.monotonic()
    r = subprocess.run(
        [str(SCRIPT), "--work-dir", str(work), "--", "true"],
        capture_output=True, text=True, timeout=90, env=env, cwd=REPO, check=False,
    )
    elapsed = time.monotonic() - started

    assert r.returncode != 0
    assert "could not parse a trial id" in r.stderr, r.stderr
    assert elapsed < 30, (
        f"took {elapsed:.0f}s: it accepted the garbage id and waited on "
        "CAE_PAUSE_ACK_TIMEOUT for an ack nothing will ever write"
    )
    assert not (work / "tune" / "pause.txt").exists(), "parked production for nothing"


def test_a_well_formed_trial_dir_still_resolves(tmp_path: Path) -> None:
    """The positive control for the validation above: a real trial-dir name
    must still resolve, or the guard is just a refusal to work."""
    work, bin_dir, calls = _sandbox(tmp_path, workers=True)
    proc = _ack_after(work / "tune", 2.0)
    try:
        env_free = subprocess.run(
            [str(SCRIPT), "--work-dir", str(work), "--", "true"],
            capture_output=True, text=True, timeout=90, cwd=REPO, check=False,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                 "CAE_PAUSE_POLL_SECONDS": "1"},
        )
    finally:
        proc.wait()
    assert env_free.returncode == 0, env_free.stderr
    assert f"trial={TRIAL_ID}" in env_free.stdout, env_free.stdout
    assert calls.read_text().strip().startswith("pkill ack:"), calls.read_text()


def test_the_default_work_dir_matches_the_ratchets_single_source() -> None:
    """⚑ A THIRD HARDCODED DEFAULT IS HOW THE TREES DIVERGE.

    `ratchet_common.sh` exists because ratchet_loop.sh and daily_gate_ratchet.sh
    had drifted on exactly this and wrote a row whose iter column named a
    checkpoint from the other tree. This script is a third participant; the loop
    now passes --work-dir explicitly (NB2), but the default is still reachable
    by hand, and a default that points at a tree nobody trains in would refuse
    every window with "no such tune dir".
    """
    ours = re.findall(r'^WORK_DIR="([^"]+)"', SCRIPT.read_text(), re.M)
    common = (REPO / "scripts" / "ratchet_common.sh").read_text()
    theirs = re.findall(r'^WORK_DIR="\$\{TRAIN_WORK_DIR:-([^}]+)\}"', common, re.M)
    assert len(ours) == 1, f"pause_window.sh no longer defines WORK_DIR once: {ours}"
    assert len(theirs) == 1, f"ratchet_common.sh no longer defines WORK_DIR once: {theirs}"
    assert ours == theirs, (
        f"pause_window.sh defaults to {ours} while the ratchet's single source "
        f"says {theirs}: a hand-run window would pause the wrong tree"
    )


def test_a_worker_REVIVED_during_the_ack_wait_stops_the_job(tmp_path: Path) -> None:
    """⚑ B1: THE OTHER HALF OF THE DRAIN GATE, and the half that runs the arena.

    The baseline pids are captured BEFORE the marker and before an ack wait that
    can run `CAE_PAUSE_ACK_TIMEOUT` (1800s default). `_revive_fleet` lives inside
    the ingest phase, so it is live for exactly that wait: a worker that dies and
    comes back there has a pid in NO baseline. With the gate checking only the
    baseline, every baseline pid is dead, the survivor check finds nothing, and
    the job runs beside a live selfplay worker with rc=0 -- which
    `ratchet_outcome` then stamps as a clean strength reading.

    An independent reviewer reproduced precisely this against the real script.
    The drain-wait loop even PRINTS `workers still alive after Ns` first, and the
    old gate never consulted it: a value accepted and then ignored.

    The stub reproduces the revive: `pgrep` reports the OLD pid until `pkill`
    runs, and a NEW live pid forever after. `pkill` signals nothing, so the
    revived worker ignores SIGTERM the way a wedged one would.
    """
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    revived = subprocess.Popen(["sleep", "120"])
    try:
        (bin_dir / "pgrep").write_text(
            "#!/bin/sh\n"
            f'if [ -f "{tmp_path}/killed" ]; then echo {revived.pid}; exit 0; fi\n'
            "echo 424242\nexit 0\n",
        )
        (bin_dir / "pkill").write_text(
            f'#!/bin/sh\ntouch "{tmp_path}/killed"\nexit 0\n',
        )
        for f in ("pgrep", "pkill"):
            (bin_dir / f).chmod(0o755)

        ran = tmp_path / "the_arena_ran"
        ack = _ack_after(work / "tune", 2.0)
        try:
            r = _run(work, bin_dir, "touch", str(ran), CAE_PAUSE_DRAIN_TIMEOUT="2")
        finally:
            ack.wait()
    finally:
        revived.terminate()
        revived.wait()

    assert not ran.exists(), (
        "THE JOB RAN BESIDE A LIVE WORKER. The baseline pid died, so the "
        "survivor check passed, and the worker revived during the ack wait was "
        "never looked for -- a contended arena, about to be filed as clean"
    )
    assert r.returncode != 0, "an undrained fleet exited 0"
    assert "still match" in r.stderr, r.stderr
    assert not (work / "tune" / "pause.txt").exists(), "died holding the marker"


def test_a_pgrep_usage_error_AFTER_the_drain_is_also_fatal(tmp_path: Path) -> None:
    """The post-drain pgrep needs its status checked for the same reason the
    pre-drain one does: rc>=2 means the pattern is broken, so "nothing matched"
    is not a measurement of anything."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    (bin_dir / "pgrep").write_text(
        "#!/bin/sh\n"
        f'if [ -f "{tmp_path}/killed" ]; then echo "pgrep: invalid option" >&2; exit 2; fi\n'
        "echo 424242\nexit 0\n",
    )
    (bin_dir / "pkill").write_text(f'#!/bin/sh\ntouch "{tmp_path}/killed"\nexit 0\n')
    for f in ("pgrep", "pkill"):
        (bin_dir / f).chmod(0o755)

    ran = tmp_path / "ran"
    ack = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "touch", str(ran), CAE_PAUSE_DRAIN_TIMEOUT="2")
    finally:
        ack.wait()

    assert r.returncode != 0
    assert "after the drain" in r.stderr, r.stderr
    assert not ran.exists(), "ran the job on an unverifiable drain"


# ── B3: the producer/consumer contract, ROUND-TRIPPED ────────────────────────
# `pause_window.sh` WRITES `pid=<n> started=<iso>`; `train_watchdog.py` READS it
# with `_PAUSE_OWNER_RE`. Every `pid=` elsewhere in this suite is a literal typed
# into a test file, so the two halves were pinned only by both files happening to
# contain the same four characters. An independent reviewer's mutant changing
# `pid=%s` to `owner=%s` IN THE SCRIPT ALONE survived all 166 tests -- and it
# makes every abandoned window PERMANENTLY unrecoverable (unowned marker =>
# `_abandoned_reason` returns None => PAUSED-HELD forever => production parked
# until a human notices). Same defect class as the worker pattern, which this PR
# already closed with a byte-identity test; the same technique was available here
# and was not used.
#
# So: run the real script far enough to create a real marker, then hand that file
# to the real consumer. No literal is shared between the two sides.

_WATCHDOG = load_script_module("train_watchdog.py", "train_watchdog_contract")


def test_the_marker_the_script_writes_is_one_the_watchdog_can_own(tmp_path: Path) -> None:
    """The whole PAUSE-ABANDONED mechanism in one round trip."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    marker = work / "tune" / "pause.txt"

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAE_PAUSE_POLL_SECONDS"] = "1"
    env["CAE_PAUSE_ALLOW_NO_WORKERS"] = "1"
    p = subprocess.Popen(
        [str(SCRIPT), "--work-dir", str(work), "--trial-id", TRIAL_ID,
         "--", "sleep", "60"],
        env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + ACK_WAIT
        while not marker.exists():
            assert p.poll() is None, "the script exited before writing a marker"
            assert time.monotonic() < deadline, "marker never appeared"
            time.sleep(0.2)
        text = marker.read_text()
        mtime = marker.stat().st_mtime
    finally:
        p.terminate()
        p.wait(timeout=60)

    # 1. The consumer finds an owner in what the producer wrote.
    owner, age = _WATCHDOG.parse_pause_marker(text, mtime=mtime, now=time.time())
    assert owner is not None, (
        f"the watchdog cannot find an owner in the marker the script writes; "
        f"every abandoned window is now permanently unrecoverable.\n{text!r}"
    )
    # 2. And that owner is the process that actually held it.
    assert owner == p.pid, (
        f"the marker names pid {owner} but the window was pid {p.pid}; the "
        f"watchdog would judge liveness of the wrong process.\n{text!r}"
    )
    # 3. `started=` parses, so the age bound has a clock. (mtime is the
    #    fallback, so a broken `started=` would hide behind it here -- pass a
    #    deliberately wrong mtime and require the answer to come from the text.)
    _, age_from_text = _WATCHDOG.parse_pause_marker(
        text, mtime=mtime - 86_400, now=time.time(),
    )
    assert age_from_text is not None, "`started=` did not parse and no mtime fallback"
    assert age_from_text < 60, (
        f"`started=` did not parse, so the age fell back to the (deliberately "
        f"day-old) mtime: {age_from_text} minutes"
    )

    # 4. The verdict a LIVE window produces, from that real marker: hands off.
    snap = _WATCHDOG.ProgressSnapshot(
        pid=os.getpid(), pid_alive=True, pause_txt=str(marker),
        rows=10, rows_prev=10, minutes_flat=5.0,
        pause_owner_pid=owner, pause_owner_alive=True, pause_age_minutes=age,
    )
    assert _WATCHDOG.decide(snap, stall_minutes=90.0).state == _WATCHDOG.STATE_PAUSED_HELD

    # 5. ...and the verdict once its owner is gone (the script is dead now).
    dead = _WATCHDOG.ProgressSnapshot(
        pid=os.getpid(), pid_alive=True, pause_txt=str(marker),
        rows=10, rows_prev=10, minutes_flat=5.0,
        pause_owner_pid=owner,
        pause_owner_alive=_WATCHDOG.pid_is_alive(owner),
        pause_age_minutes=age,
    )
    assert _WATCHDOG.decide(dead, stall_minutes=90.0).state == (
        _WATCHDOG.STATE_PAUSE_ABANDONED
    ), "a marker whose window has exited is not recoverable; it will be held forever"


def test_the_watchdogs_owner_pattern_matches_what_the_script_prints() -> None:
    """The static half, in the same shape as
    `test_the_worker_pattern_matches_train_sh_exactly`: the regex must match the
    literal `printf` format, so a rename on either side is a red test rather
    than a silently unrecoverable window."""
    fmt = re.findall(r"printf '(pause_window\.sh [^']*)\\n'", SCRIPT.read_text())
    assert len(fmt) == 1, f"the marker is no longer written by one printf: {fmt}"
    rendered = fmt[0].replace("%s", "12345", 1)
    assert _WATCHDOG._PAUSE_OWNER_RE.search(rendered), (
        f"train_watchdog._PAUSE_OWNER_RE cannot read the marker format "
        f"pause_window.sh writes: {fmt[0]!r}"
    )


def test_the_wrapper_failure_code_is_not_in_the_ratchets_vocabulary() -> None:
    """⚑ N2: THE SECOND INSTANCE OF THE EXIT-5 COLLISION, found the same way.

    The wrapper's `die` code was 3, justified by a comment claiming "the
    ratchet's vocabulary is 0 / 1 / 5". `ratchet_common.sh` -- the file that
    comment was paraphrasing -- says in as many words: *"5 avoids 1 (retryable),
    2 (usage), 3 (the arena's own no-pairs status)"*. 3 was already spoken for.
    Latent (`daily_gate_ratchet.sh` returns exactly 0/1/2/5 today), but it is the
    identical mistake as claiming exit 5 after #371 had taken it for CRASHED: a
    code chosen by reading part of the space.

    The taken set is DERIVED FROM THE FILES, not restated here, so a status
    someone adds later cannot silently collide with this one. Mutating
    `PAUSE_WINDOW_FAILED_RC` to 1 -- which makes every routine RETRY (the
    "checkpoint not ready yet" path every fresh restart passes through) count as
    a pause-window failure, silently disabling the window for the rest of the day
    after two polls -- fails here.
    """
    common = (REPO / "scripts" / "ratchet_common.sh").read_text()
    loop = (REPO / "scripts" / "ratchet_loop.sh").read_text()
    script = SCRIPT.read_text()

    retry = int(re.findall(r"^RATCHET_EXIT_RETRY=(\d+)", common, re.M)[0])
    no_retry = int(re.findall(r"^RATCHET_EXIT_NO_RETRY=(\d+)", common, re.M)[0])
    # The codes ratchet_common.sh's own comment names as already spoken for.
    documented = {int(n) for n in re.findall(r"^# 5 avoids .*", common, re.M)[0:1]
                  for n in re.findall(r"(\d+) \(", re.findall(r"^# 5 avoids .*", common, re.M)[0])}
    ours = int(re.findall(r"^WRAPPER_FAILED_RC=(\d+)", script, re.M)[0])
    theirs = int(re.findall(r"^PAUSE_WINDOW_FAILED_RC=(\d+)", loop, re.M)[0])

    assert ours == theirs, (
        f"pause_window.sh dies with {ours} but ratchet_loop.sh counts {theirs}: "
        "a wrapper failure would be filed as an ordinary ratchet failure and "
        "dodge CAE_RATCHET_PAUSE_MAX_FAILS entirely"
    )
    taken = {0, retry, no_retry, 2} | documented
    assert ours not in taken, (
        f"the wrapper's failure code {ours} is already taken: "
        f"{sorted(taken)} (0=rows written, {retry}=RETRY, {no_retry}=NO_RETRY, "
        f"2=usage/cd, plus ratchet_common.sh's own 'avoids' list). This is how "
        "exit 5 collided with CRASHED"
    )
    assert documented, "ratchet_common.sh's 'avoids' comment no longer parses; re-point this test"


def test_a_setup_failure_reports_the_wrapper_code_not_the_ratchets_retry(
    tmp_path: Path,
) -> None:
    """⚑ N4. Under `set -e`, a `mktemp`/`stat` failure before the marker exits 1
    — the ratchet's RETRY — so the loop logs "FAILED, retry next poll", does NOT
    count it against `CAE_RATCHET_PAUSE_MAX_FAILS`, and churns every 600s until
    midnight. The fail cap exists to bound exactly that.

    Forced here by making `mktemp` fail, which is the real-world case (full
    /tmp) rather than a paraphrase of it.
    """
    work, bin_dir, _ = _sandbox(tmp_path, workers=True)
    (bin_dir / "mktemp").write_text("#!/bin/sh\necho 'mktemp: failed' >&2\nexit 1\n")
    (bin_dir / "mktemp").chmod(0o755)

    ran = tmp_path / "ran"
    r = _run(work, bin_dir, "touch", str(ran))

    expected = int(re.findall(r"^WRAPPER_FAILED_RC=(\d+)", SCRIPT.read_text(), re.M)[0])
    assert r.returncode == expected, (
        f"a setup failure exited {r.returncode}, not {expected}; the loop reads "
        "that as the ratchet's own retryable status and never counts it against "
        f"the fail cap.\n{r.stderr}"
    )
    assert not ran.exists()
    assert not (work / "tune" / "pause.txt").exists()


def test_it_does_not_clear_a_marker_that_is_no_longer_ours(tmp_path: Path) -> None:
    """⚑ N6. `rm -f "$MARKER"` unconditionally means an operator who replaces the
    marker DURING our window — the obvious way to hold the pause past the arena —
    silently loses their pause when the job ends. The marker names its owner
    exactly so this is decidable. Leaving a pause up is recoverable; resuming
    against the operator's stated intent is not."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    marker = work / "tune" / "pause.txt"
    mine = tmp_path / "replaced"
    job = f'cat "{marker}" > "{mine}"; echo "operator holds this" > "{marker}"'

    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "bash", "-c", job)
    finally:
        proc.wait()

    assert r.returncode == 0, r.stderr
    assert "pid=" in mine.read_text(), "the window's own marker had no owner line"
    assert marker.exists(), (
        "the operator's replacement marker was deleted: their deliberate pause "
        f"ended the moment our job did.\n{r.stdout}"
    )
    assert marker.read_text() == "operator holds this\n"
    assert "no longer ours" in r.stdout, f"it went quietly:\n{r.stdout}"


def test_it_still_clears_its_OWN_marker(tmp_path: Path) -> None:
    """The positive control for the guard above. If the ownership check is
    wrong in the other direction the script leaves production parked on every
    run, which is the one unrecoverable mistake it exists to avoid."""
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    proc = _ack_after(work / "tune", 2.0)
    try:
        r = _run(work, bin_dir, "true")
    finally:
        proc.wait()
    assert r.returncode == 0, r.stderr
    assert not (work / "tune" / "pause.txt").exists(), (
        f"the window did not clear its own marker: production stays parked\n{r.stdout}"
    )
    assert "marker cleared" in r.stdout


def test_a_failure_AFTER_the_marker_also_reports_the_wrapper_code(
    tmp_path: Path,
) -> None:
    """⚑ THE OTHER SIDE OF THE TRAP HANDOFF, and it was a surviving mutant.

    `trap release EXIT` REPLACES the setup trap, so a `set -e` death between the
    marker and the job would release the marker correctly and then report 1 —
    the ratchet's RETRY — dodging `CAE_RATCHET_PAUSE_MAX_FAILS` exactly as N4
    describes. The first version of this fix composed the two traps
    (`release_then_remap`) but only tested the setup half, so deleting the remap
    from the composite SURVIVED.

    Forced with a read-only tune dir, which is the reachable real case (a full
    or read-only filesystem): every pre-flight check passes, and then
    `printf > "$MARKER"` fails.
    """
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    tune = work / "tune"
    ran = tmp_path / "ran"
    tune.chmod(0o555)
    try:
        r = _run(work, bin_dir, "touch", str(ran))
    finally:
        tune.chmod(0o755)

    expected = int(re.findall(r"^WRAPPER_FAILED_RC=(\d+)", SCRIPT.read_text(), re.M)[0])
    assert not ran.exists(), "the job ran despite the marker never being written"
    assert r.returncode == expected, (
        f"a failure after the trap handoff exited {r.returncode}, not {expected}: "
        "the loop reads that as the ratchet's own retryable status and it never "
        f"counts against the fail cap.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    assert not (tune / "pause.txt").exists(), "left a marker behind"


def test_the_ledgers_yardstick_selects_the_trial_by_data_not_by_name(
    tmp_path: Path,
) -> None:
    """⚑ B2: THE PRE-REGISTERED YARDSTICK WAS A GATE THAT COULD NOT FAIL.

    It selected the trial with `sorted(glob("train_trial_*"))[-1]` —
    LEXICOGRAPHIC. Against the live tune dir `'3' < 'd'`, so it read the dead
    `train_trial_d2003_..._2026-08-05` (last row 08-06) rather than the live
    `379f6`: the median came from a cold-start trial, it reported 7 stretched
    iterations on night ONE, and its output was STATIONARY — five nights produce
    byte-identical results, so the kill rule could not fire in either direction.
    Found by an independent reviewer EXECUTING it.

    CLAUDE.md protocol #1 requires that command, so it is extracted from the
    ledger and RUN here against a fixture built to punish the old selector: the
    live trial sorts FIRST by name and is newest by data.
    """
    ledger = (REPO / "docs" / "experiment_ledger.md").read_text()
    block = re.search(
        r"### Yardstick \(pre-registered, ONE deciding command\).*?```bash\n"
        r"PYTHONPATH=\. python3 - <<'PY'\n(.*?)\nPY\n```",
        ledger, re.S,
    )
    assert block is not None, "the yardstick's code block no longer parses; re-point this test"
    code = block.group(1)

    tune = tmp_path / "runs" / "pbt2_small" / "tune"
    # '3...' sorts BEFORE 'd...', so lexicographic selection picks the dead one.
    live = tune / "train_trial_379f6_00000_0_lr=0.0000_2026-08-06_23-51-06"
    dead = tune / "train_trial_d2003_00000_0_lr=0.0000_2026-08-05_12-07-56"
    for d in (dead, live):
        d.mkdir(parents=True)
    (dead / "result.json").write_text(
        '{"training_iteration": 1, "time_this_iter_s": 500.0, "timestamp": 1000000}\n',
    )
    time.sleep(0.02)
    (live / "result.json").write_text(
        '{"training_iteration": 900, "time_this_iter_s": 250.0, "timestamp": 2000000}\n',
    )
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "runs" / "arena_results.jsonl").write_text("")

    r = subprocess.run(
        ["python3", "-c", code], cwd=tmp_path,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert r.returncode == 0, f"the pre-registered command does not run:\n{r.stderr}"
    assert "379f6" in r.stdout, (
        "the yardstick selected the DEAD trial: its median, its threshold and "
        f"its verdict all describe a run that stopped days ago.\n{r.stdout}"
    )
    assert "d2003" not in r.stdout, r.stdout


# ── N7: the trial selector must TRACK THE LIVE TRIAL, not merely return one ──
# B2 was a "latest trial" selector that silently picked a DEAD trial and
# returned plausible, stationary numbers; it was bounded and logged right up
# until someone executed it. `resolve_trial_id` was the same shape with a
# different wrong rule -- DIRECTORY mtime, which moves whenever any entry is
# created or removed inside the directory, so Ray writing `checkpoint_NNNNNN/`
# under a dead trial floats it above the live one. Picking wrong costs a full
# CAE_PAUSE_ACK_TIMEOUT (1800s) of parked production per poll, twice, before the
# fail cap stops it.
#
# So these do not assert "it returned an id". They build trees where the WRONG
# rule has a clear preference, and require the live trial to win anyway.

_TRIAL_SUFFIX = "_0_lr=0.0000_2026-01-01_00-00-00"


def _selector_sandbox(tmp_path: Path, live: str, dead: str):
    """A tune dir with two trials; returns (work, bin_dir, live_dir, dead_dir).

    ⚑ THE LIVE TRIAL IS NAMED TO SORT *BEFORE* THE DEAD ONE (`aaa..` vs `zzz..`),
    so the selector's final name tiebreak actively pulls toward the WRONG answer.
    The first version of these fixtures used `live..`/`dead..`, which sorts the
    right way -- and with `stat -c %Y`'s whole-second mtimes the two files tied,
    so the name decided and three mutations of the mtime ranking SURVIVED. A
    fixture must not encode the answer it is checking.
    """
    work = tmp_path / "runs" / "x"
    tune = work / "tune"
    live_d = tune / f"train_trial_{live}{_TRIAL_SUFFIX}"
    dead_d = tune / f"train_trial_{dead}{_TRIAL_SUFFIX}"
    for d in (dead_d, live_d):
        d.mkdir(parents=True)
    (work / "server" / "trials" / live / "workers" / "worker_00" / "selfplay_resume").mkdir(
        parents=True,
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pgrep").write_text("#!/bin/sh\nexit 1\n")
    (bin_dir / "pkill").write_text("#!/bin/sh\nexit 0\n")
    for f in ("pgrep", "pkill"):
        (bin_dir / f).chmod(0o755)
    return work, bin_dir, live_d, dead_d


def _resolve(work: Path, bin_dir: Path, live_id: str):
    """Run the real script with NO --trial-id, acking as `live_id` would.

    End-to-end on purpose: if the selector picks the other trial, the ack it
    waits for is one nothing will ever write, and it dies on the clock -- which
    is exactly the production failure, not a paraphrase of it.
    """
    ack = work / "tune" / f".paused_{live_id}.ack"
    writer = subprocess.Popen(
        ["sh", "-c", f'sleep 2; echo "trial={live_id} next_iter=7" > "{ack}"'],
    )
    try:
        env = dict(os.environ)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["CAE_PAUSE_POLL_SECONDS"] = "1"
        env["CAE_PAUSE_ALLOW_NO_WORKERS"] = "1"
        env["CAE_PAUSE_ACK_TIMEOUT"] = "12"
        return subprocess.run(
            [str(SCRIPT), "--work-dir", str(work), "--", "true"],
            capture_output=True, text=True, timeout=90, env=env, cwd=REPO, check=False,
        )
    finally:
        writer.wait()


def test_the_trial_selector_prefers_the_populated_trial_over_a_touched_one(
    tmp_path: Path,
) -> None:
    """⚑ The `ls -1dt` case, minimal. The dead trial has NO data and the NEWEST
    directory mtime — which is what a checkpoint dir appearing under it does."""
    work, bin_dir, live_d, dead_d = _selector_sandbox(tmp_path, "aaa11_00000", "zzz91_00000")
    (live_d / "result.json").write_text('{"training_iteration": 900}\n')
    time.sleep(1.1)
    (dead_d / "checkpoint_000001").mkdir()      # bumps the DIRECTORY mtime
    assert dead_d.stat().st_mtime > live_d.stat().st_mtime, "fixture does not test anything"

    r = _resolve(work, bin_dir, "aaa11_00000")

    assert "trial=aaa11_00000" in r.stdout, (
        "the selector picked the trial with no data because its DIRECTORY was "
        f"touched more recently; the ack it waits for will never be written.\n{r.stdout}"
    )
    assert r.returncode == 0, f"it never parked -- wrong trial id:\n{r.stdout}\n{r.stderr}"


def test_the_trial_selector_prefers_FRESH_DATA_over_a_fresher_directory(
    tmp_path: Path,
) -> None:
    """⚑ The realistic case, and the one directory mtime gets exactly backwards:
    BOTH trials are populated, the dead one's DIRECTORY is newer (Ray created a
    checkpoint dir under it), the live one's DATA is newer. Ranking by directory
    mtime picks the dead trial; ranking by the data file picks the live one."""
    work, bin_dir, live_d, dead_d = _selector_sandbox(tmp_path, "aaa22_00000", "zzz92_00000")
    (dead_d / "result.json").write_text('{"training_iteration": 104}\n')
    time.sleep(1.1)
    (live_d / "result.json").write_text('{"training_iteration": 900}\n')   # newest DATA
    time.sleep(1.1)
    (dead_d / "checkpoint_000002").mkdir()                                 # newest DIR
    assert dead_d.stat().st_mtime > live_d.stat().st_mtime
    assert (live_d / "result.json").stat().st_mtime > (dead_d / "result.json").stat().st_mtime

    r = _resolve(work, bin_dir, "aaa22_00000")

    assert "trial=aaa22_00000" in r.stdout, (
        "the selector ranked by directory mtime: it chose the trial that stopped "
        f"writing data, and would hold the marker for the full ack timeout.\n{r.stdout}"
    )
    assert r.returncode == 0, f"it never parked -- wrong trial id:\n{r.stdout}\n{r.stderr}"


def test_the_trial_selector_matches_the_watchdogs_rule_on_the_same_tree(
    tmp_path: Path,
) -> None:
    """⚑ ONE RULE, TWO IMPLEMENTATIONS, PINNED EQUAL — the technique this PR
    already used for the worker pattern and for the marker format.

    `train_watchdog.newest_trial_dir` is the Python original; this is a bash
    re-implementation of it. Agreement is asserted on a tree built to separate
    them (populated-vs-not AND data-mtime-vs-dir-mtime at once), so the two
    cannot drift into disagreeing about which trial is live.
    """
    work, _bin, live_d, dead_d = _selector_sandbox(tmp_path, "aaa33_00000", "zzz93_00000")
    empty = work / "tune" / f"train_trial_zzz99_00000{_TRIAL_SUFFIX}"
    empty.mkdir()                                            # newest of all, no data
    (dead_d / "progress.csv").write_text("h\n1\n")
    time.sleep(1.1)
    (live_d / "result.json").write_text('{"training_iteration": 900}\n')
    time.sleep(1.1)
    (dead_d / "checkpoint_000003").mkdir()

    theirs = _WATCHDOG.newest_trial_dir(work / "tune")
    assert theirs is not None
    r = _resolve(work, _bin, "aaa33_00000")

    ours = re.search(r"trial=(\S+)", r.stdout)
    assert ours is not None, r.stdout
    assert theirs.name.startswith(f"train_trial_{ours.group(1)}_"), (
        f"pause_window.sh selected {ours.group(1)} while "
        f"train_watchdog.newest_trial_dir selected {theirs.name}: two rules for "
        "'the live trial', which is how one of them comes to be wrong unnoticed"
    )
    assert ours.group(1) == "aaa33_00000", r.stdout


def test_two_trials_written_in_the_SAME_SECOND_are_still_ordered(
    tmp_path: Path,
) -> None:
    """⚑ `stat -c %Y` IS WHOLE SECONDS, and a restart writes both trials' first
    rows well inside one second. With `%Y` they tie, the comparison falls
    through to the NAME, and which trial counts as live is decided
    alphabetically — while `newest_trial_dir` compares float `st_mtime` and gets
    it right, so the two implementations of one rule silently disagree.

    Disclosed as a survivor first: the other selector tests use 1.1s gaps, which
    makes them blind to exactly this. `%.9Y` was justified by parity with the
    Python rule and by nothing executable until this test existed.
    """
    probe_a, probe_b = tmp_path / "a", tmp_path / "b"
    probe_a.write_text("x")
    time.sleep(0.05)
    probe_b.write_text("x")
    if probe_a.stat().st_mtime == probe_b.stat().st_mtime:
        pytest.skip("filesystem has 1-second mtime granularity; nothing to order")

    work, bin_dir, live_d, dead_d = _selector_sandbox(tmp_path, "aaa44_00000", "zzz94_00000")
    (dead_d / "result.json").write_text('{"training_iteration": 104}\n')
    time.sleep(0.05)                       # same whole second, later nanosecond
    (live_d / "result.json").write_text('{"training_iteration": 900}\n')
    assert (live_d / "result.json").stat().st_mtime > (dead_d / "result.json").stat().st_mtime

    r = _resolve(work, bin_dir, "aaa44_00000")

    assert "trial=aaa44_00000" in r.stdout, (
        "two trials written in the same second tied, so the name decided which "
        f"one was 'live' — and the name was chosen to be wrong.\n{r.stdout}"
    )
    assert r.returncode == 0, f"it never parked -- wrong trial id:\n{r.stdout}\n{r.stderr}"


# ── #3: the fail cap, which is one of the ledger's three KILL criteria ───────
# `data/ratchet/pause_window_fails` is read at ratchet_loop.sh:190, written at
# :224-231, and consulted at :171 to stop asking for a window after
# CAE_RATCHET_PAUSE_MAX_FAILS. Nothing executed any of it: deleting the counting
# left every test green, and no test file so much as named the state file. A
# KILL rule that reads a counter nothing proves ever increments is this PR's own
# subject turned on the PR -- so the counter is now driven end to end, through
# the real loop, with a wrapper that really does fail.

_FAILING_STUB = """#!/usr/bin/env bash
# Records the call, then fails the way pause_window.sh's `die` does.
printf '%s\\n' "$*" >> "$PAUSE_CALLS"
exit 7
"""


def _fails_file(root: Path) -> str:
    p = root / "data" / "ratchet" / "pause_window_fails"
    return p.read_text().strip() if p.exists() else ""


def test_the_pause_fail_cap_counts_up_and_then_stops_asking(tmp_path: Path) -> None:
    """⚑ THE CAP, EXECUTED. Three polls against a wrapper that always exits 7:
    the first two are counted and produce no row, and the THIRD gives up on the
    window and takes the contended reading, so the day still gets measured.

    Without this the day would retry every 600s until midnight, setting and
    holding the marker each time, and `loop_health.ratchet_gap_alerts()` keys on
    attempts.csv so it would be invisible too.
    """
    from tests.test_ratchet_search_shape import _csv_rows, _run_one_poll

    root, calls = _ratchet_sandbox(tmp_path, wrapper=True)
    stub = root / "scripts" / "pause_window.sh"
    stub.write_text(_FAILING_STUB)
    stub.chmod(0o755)
    env = {"PAUSE_CALLS": str(calls), "CAE_RATCHET_PAUSE_MAX_FAILS": "2"}

    rc1, out1 = _run_one_poll(root, env=env)
    assert rc1 != 0, f"a wrapper that exits 7 must not read as success:\n{out1}"
    assert _csv_rows(root) == [], f"a row was written despite no ratchet running:\n{out1}"
    assert _fails_file(root).endswith(" 1"), (
        f"the first wrapper failure was not counted: {_fails_file(root)!r}\n{out1}"
    )

    rc2, out2 = _run_one_poll(root, env=env)
    assert rc2 != 0, (
        f"the SECOND wrapper failure read as success, so the day would look "
        f"measured while no ratchet ran:\n{out2}"
    )
    assert _fails_file(root).endswith(" 2"), (
        f"the counter did not increment on the second failure: {_fails_file(root)!r}\n{out2}"
    )
    assert len(_pause_calls(calls)) == 2, _pause_calls(calls)

    # Third poll: the cap is reached, so the loop stops asking for a window.
    rc3, out3 = _run_one_poll(root, env=env)
    assert rc3 == 0, f"the capped poll must still measure the day:\n{out3}"
    assert len(_pause_calls(calls)) == 2, (
        f"it asked for a window after the cap: {_pause_calls(calls)}\n{out3}"
    )
    assert "for the rest of" in out3, f"the fallback must be loud:\n{out3}"
    assert len(_csv_rows(root)) == 1, f"the contended reading was not taken:\n{out3}"


def test_a_successful_window_clears_the_fail_counter(tmp_path: Path) -> None:
    """The other direction, or the cap is a one-way ratchet that eventually
    disables the window permanently on a run that has since recovered."""
    from tests.test_ratchet_search_shape import _run_one_poll

    root, calls = _ratchet_sandbox(tmp_path, wrapper=True)
    stub = root / "scripts" / "pause_window.sh"
    stub.write_text(_FAILING_STUB)
    stub.chmod(0o755)
    env = {"PAUSE_CALLS": str(calls), "CAE_RATCHET_PAUSE_MAX_FAILS": "5"}
    _run_one_poll(root, env=env)
    assert _fails_file(root).endswith(" 1")

    stub.write_text(_PAUSE_STUB)          # the wrapper works again
    stub.chmod(0o755)
    rc, out = _run_one_poll(root, env=env)

    assert rc == 0, out
    assert _fails_file(root) == "", (
        f"a successful window left the failure count standing: {_fails_file(root)!r}\n{out}"
    )


def test_only_the_WRAPPERS_failures_are_counted_against_the_cap(tmp_path: Path) -> None:
    """⚑ The reason the wrapper's exit code had to leave the ratchet's
    vocabulary. A ratchet that legitimately returns RETRY (exit 1 — the
    "checkpoint has no trainer.pt yet" path every fresh restart passes through)
    must NOT burn the budget that exists to stop production being parked for
    nothing. With the codes shared, two ordinary retries disabled the window for
    the rest of the day."""
    from tests.test_ratchet_search_shape import _run_one_poll

    root, calls = _ratchet_sandbox(tmp_path, wrapper=True)
    stub = root / "scripts" / "pause_window.sh"
    # Runs the job, which fails the way a no-pairs ratchet does: RETRY, not 7.
    stub.write_text(_PAUSE_STUB)
    stub.chmod(0o755)

    rc, out = _run_one_poll(
        root, mode="nopairs", env={"PAUSE_CALLS": str(calls)},
    )

    assert rc != 0, out
    assert _fails_file(root) == "", (
        f"a retryable RATCHET failure was counted against the PAUSE budget: "
        f"{_fails_file(root)!r}\n{out}"
    )


def test_all_three_trial_selectors_agree_on_the_same_tree(tmp_path: Path) -> None:
    """⚑ #4: N7 WAS FIXED IN ONE PLACE OUT OF THREE, WHICH WAS WORSE THAN NOT
    FIXING IT.

    `ratchet_loop.sh` and `daily_gate_ratchet.sh` still used `ls -td`, and the
    loop's `ck_ready` read that unfixed selector thirty lines above the fixed
    one in the same file. So under exactly the scenario N7 documents, the
    wrapper parked the LIVE trial while `iter`, `MIN_ITER`, `ck_ready` and the
    snapshot all came from the DEAD one: production parked to measure a trial
    that is not the one being parked. Latent only because today's mtimes happen
    to order correctly.

    The rule now lives once, in `ratchet_common.sh` — the file that exists
    precisely because these two scripts had drifted on which tree to read. This
    asserts the shell function and the Python original agree on a tree built to
    separate them.
    """
    tune = tmp_path / "runs" / "pbt2_small" / "tune"
    live = tune / "train_trial_aaa55_00000_0_lr=0.0000_2026-01-01_00-00-00"
    dead = tune / "train_trial_zzz95_00000_0_lr=0.0000_2026-01-01_00-00-00"
    for d in (dead, live):
        d.mkdir(parents=True)
    (dead / "result.json").write_text('{"training_iteration": 104}\n')
    time.sleep(1.1)
    (live / "result.json").write_text('{"training_iteration": 900}\n')
    time.sleep(1.1)
    (dead / "checkpoint_000009").mkdir()        # newest DIRECTORY, dead trial
    assert dead.stat().st_mtime > live.stat().st_mtime, "fixture does not test anything"

    common = REPO / "scripts" / "ratchet_common.sh"
    r = subprocess.run(
        ["bash", "-c",
         f'RATCHET_ROOT={shlex.quote(str(tmp_path))} . {shlex.quote(str(common))}; '
         "ratchet_newest_trial_dir"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert r.returncode == 0, r.stderr
    picked = Path(r.stdout.strip().rstrip("/")).name

    theirs = _WATCHDOG.newest_trial_dir(tune)
    assert theirs is not None
    assert picked == theirs.name, (
        f"ratchet_common.sh picked {picked} while train_watchdog.newest_trial_dir "
        f"picked {theirs.name}: two rules for 'the live trial'"
    )
    assert picked == live.name, (
        f"the shared selector chose the trial that stopped writing data ({picked}); "
        "the ratchet would snapshot a dead checkpoint and the wrapper would park "
        "a different trial from the one being measured"
    )


def test_the_SHARED_selector_orders_two_trials_written_in_the_same_second(
    tmp_path: Path,
) -> None:
    """⚑ DISCLOSED SURVIVOR, then closed. Mutating `%.9Y` -> `%Y` in
    `ratchet_common.sh` survived the whole suite.

    `test_two_trials_written_in_the_SAME_SECOND_are_still_ordered` covers
    `pause_window.sh:resolve_trial_id` and only that — it goes through
    `_resolve`. But the copy that `ratchet_loop.sh` and `daily_gate_ratchet.sh`
    actually call is `ratchet_newest_trial_dir`, and every test touching it used
    1.1s gaps, which whole-second mtimes order correctly. So the sub-second
    resolution was justified by a comment on the production path and pinned only
    on the wrapper path — the same one-of-N gap that #4 exists to close, one
    level down.

    A restart writes both trials' first rows well inside one second.
    """
    probe_a, probe_b = tmp_path / "pa", tmp_path / "pb"
    probe_a.write_text("x")
    time.sleep(0.05)
    probe_b.write_text("x")
    if probe_a.stat().st_mtime == probe_b.stat().st_mtime:
        pytest.skip("filesystem has 1-second mtime granularity; nothing to order")

    tune = tmp_path / "runs" / "pbt2_small" / "tune"
    live = tune / "train_trial_aaa66_00000_0_lr=0.0000_2026-01-01_00-00-00"
    dead = tune / "train_trial_zzz96_00000_0_lr=0.0000_2026-01-01_00-00-00"
    for d in (dead, live):
        d.mkdir(parents=True)
    (dead / "result.json").write_text('{"training_iteration": 104}\n')
    time.sleep(0.05)                       # same whole second, later nanosecond
    (live / "result.json").write_text('{"training_iteration": 900}\n')
    same_second = int((dead / "result.json").stat().st_mtime) == int(
        (live / "result.json").stat().st_mtime
    )
    if not same_second:
        pytest.skip("the two writes straddled a second boundary; retry")

    common = REPO / "scripts" / "ratchet_common.sh"
    r = subprocess.run(
        ["bash", "-c",
         f'RATCHET_ROOT={shlex.quote(str(tmp_path))} . {shlex.quote(str(common))}; '
         "ratchet_newest_trial_dir"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert r.returncode == 0, r.stderr
    assert Path(r.stdout.strip().rstrip("/")).name == live.name, (
        "two trials written in the same second tied on whole-second mtime, so "
        "the NAME decided which one is live -- and the name was chosen to be "
        f"wrong. Got {r.stdout.strip()!r}. ratchet_common.sh needs `stat -c %.9Y`"
    )


def test_neither_ratchet_script_still_ranks_trials_by_directory_mtime() -> None:
    """The static half. `ls -td` on a trial glob is the removed rule; a
    re-introduction anywhere in the family is a red test, not a latent
    disagreement waiting for the mtimes to line up badly.

    ⚑ Includes `ratchet_common.sh`: it is where the rule now lives, so it is the
    one file where a regression would be invisible to the two call sites.
    """
    for name in ("ratchet_loop.sh", "daily_gate_ratchet.sh", "pause_window.sh",
                 "ratchet_common.sh"):
        text = (REPO / "scripts" / name).read_text()
        offenders = [
            ln for ln in text.splitlines()
            if "train_trial_" in ln and re.search(r"\bls\s+-[a-z]*t", ln)
            and not ln.lstrip().startswith("#")
        ]
        assert not offenders, (
            f"{name} ranks trials by DIRECTORY mtime again: {offenders}. "
            "Use ratchet_newest_trial_dir (or resolve_trial_id) -- a directory's "
            "mtime moves when Ray creates a checkpoint dir under a DEAD trial"
        )


def test_an_interrupt_during_the_ACK_WAIT_is_not_a_wrapper_failure(
    tmp_path: Path,
) -> None:
    """⚑ #1: THE LONGEST PHASE WAS THE ONE THAT REPORTED WRONG.

    `PHASE` only leaves `setup` when the JOB starts, so the ack wait — bounded
    by `CAE_PAUSE_ACK_TIMEOUT`, 1800s by default, and by far the likeliest
    moment for an operator to reach for Ctrl-C — was inside the remap. A human
    interrupt or a systemd timeout therefore exited 7, counted against
    `CAE_RATCHET_PAUSE_MAX_FAILS`, and two of them could trip the ledger's third
    KILL criterion without the mechanism having failed once.

    The existing interrupt test pins the same property only AFTER the job has
    started, which is the short phase and the unlikely one.
    """
    work, bin_dir, _ = _sandbox(tmp_path, workers=False)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAE_PAUSE_POLL_SECONDS"] = "1"
    env["CAE_PAUSE_ALLOW_NO_WORKERS"] = "1"
    env["CAE_PAUSE_ACK_TIMEOUT"] = "600"
    ran = tmp_path / "ran"
    p = subprocess.Popen(
        [str(SCRIPT), "--work-dir", str(work), "--trial-id", TRIAL_ID,
         "--", "touch", str(ran)],
        env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + ACK_WAIT
        while not (work / "tune" / "pause.txt").exists():
            assert p.poll() is None, "the script exited before parking"
            assert time.monotonic() < deadline, "marker never appeared"
            time.sleep(0.2)
        time.sleep(1.5)          # squarely inside the ack wait; no ack is ever written
        p.terminate()
        rc = p.wait(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()

    wrapper_rc = int(re.findall(r"^WRAPPER_FAILED_RC=(\d+)", SCRIPT.read_text(), re.M)[0])
    assert rc != wrapper_rc, (
        f"an interrupt during the ack wait reported {rc} = 'the wrapper failed'. "
        "Two operator Ctrl-Cs would disable the window for the day and could "
        "trip a pre-registered KILL criterion with nothing having gone wrong"
    )
    assert rc == 130, f"an interrupt must report 130, got {rc}"
    assert not ran.exists(), "the job ran despite the interrupt"
    assert not (work / "tune" / "pause.txt").exists(), "interrupted holding the marker"
