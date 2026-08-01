"""The daily ratchet must say which search each row was measured with.

``scripts/arena_standard.py`` now requires ``--search-shape``, and the ratchet
is its only shell caller: without the flag the arena exits 1, the log has no
``[arena] Elo:`` line, the parser's guard returns early and **no CSV row is
written at all** — a long-run progress instrument that fails silently by
producing nothing.

Fixing the invocation is not enough, because either shape changes what the
series measures: rows written before 2026-07-29 were measured on the play shape
at ``vloss_weight=0``, a ruler neither ``--search-shape`` choice can reproduce.
So the shape is recorded per row and ``ratchet_slope.py`` refuses to fit a
slope across two of them — the G16 lesson (a "new best model" that was really a
ruler change) applied to the ratchet.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pytest

from scripts.ratchet_slope import _one_ruler_only, row_search_shape

ROOT = Path(__file__).resolve().parent.parent
RATCHET_SH = ROOT / "scripts" / "daily_gate_ratchet.sh"
LOOP_SH = ROOT / "scripts" / "ratchet_loop.sh"
COMMON_SH = ROOT / "scripts" / "ratchet_common.sh"
LEGACY = "legacy_play_vloss0"


def _row(shape: str | None, elo: float = 0.0) -> dict[str, str]:
    row = {"series": "vs_boot512", "iter": "1", "elo": str(elo)}
    if shape is not None:
        row["search_shape"] = shape
    return row


def _args(*, search_shape: str | None = None, allow_mixed_rulers: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        search_shape=search_shape, allow_mixed_rulers=allow_mixed_rulers,
    )


# ---------------------------------------------------------------------------
# The shell caller
# ---------------------------------------------------------------------------


def _arena_invocations(text: str) -> list[str]:
    """Every `python3 scripts/arena_standard.py ...` command, continuations included.

    Scoped to the command itself rather than the whole file: a script that
    merely MENTIONS --search-shape in a comment while invoking the arena
    without it is the failure being guarded against.
    """
    return [
        m.group(0)
        for m in re.finditer(r"scripts/arena_standard\.py((?:[^\n]*\\\n)*[^\n]*)", text)
    ]


def _arena_invocation() -> str:
    """The arena command block from the ratchet script."""
    found = _arena_invocations(RATCHET_SH.read_text(encoding="utf-8"))
    assert len(found) == 1, (
        f"expected exactly one arena invocation in {RATCHET_SH.name}, found "
        f"{len(found)}; re-point this test rather than deleting it"
    )
    return found[0]


def test_the_ratchet_passes_a_search_shape_to_the_arena() -> None:
    """Without it the arena exits 1 and the ratchet writes NO row."""
    invocation = _arena_invocation()

    assert "--mode matched_sims" in invocation
    assert "--search-shape" in invocation, (
        "daily_gate_ratchet.sh calls the arena without --search-shape: the run "
        "exits 1 and the CSV silently stops growing"
    )


def test_the_shape_is_a_variable_not_a_literal() -> None:
    """It has to be the same value the CSV row records.

    A literal here and a different `SHAPE=` below is how a column comes to
    disagree with the number next to it.
    """
    assert '--search-shape "$SHAPE"' in _arena_invocation()


def test_every_ratchet_row_carries_the_shape() -> None:
    text = RATCHET_SH.read_text(encoding="utf-8")

    assert 'CSV_HEADER="date,iter,series,elo,ci_lo,ci_hi,score,games,search_shape"' in text
    assert '$score,$played,$SHAPE" >> "$LOG"' in text


def test_the_old_schema_is_migrated_rather_than_appended_to() -> None:
    """Appending a 9th field to an 8-column CSV is the silent-corruption path.

    ``csv.DictReader`` would put the shape in the restkey and every OLD row
    would inherit the new ruler by omission — the break would read as
    agreement. The migration stamps the old rows with what they measured.
    """
    text = RATCHET_SH.read_text(encoding="utf-8")

    assert f"LEGACY_SHAPE={LEGACY}" in text
    assert "date,iter,series,elo,ci_lo,ci_hi,score,games" in text  # the old header, matched
    assert "$LOG.tmp" in text  # written aside...
    assert 'mv "$LOG.tmp" "$LOG"' in text  # ...then swapped in atomically


def test_a_ratchet_that_measures_the_loop_uses_the_training_shape() -> None:
    """The pre-committed choice, so a later edit is a decision and not a drift.

    ratchet_slope.py fits Elo against cumulative OPTIMIZER STEPS: it is a claim
    about the RL loop, so it must rank nets under the search selfplay runs.
    """
    assert re.search(r"^SHAPE=training$", RATCHET_SH.read_text(encoding="utf-8"), re.M)


def test_no_repo_script_arenas_without_naming_a_shape() -> None:
    """The general form of the defect the reviewer found in the ratchet."""
    offenders = []
    scanned = 0
    for path in sorted((ROOT / "scripts").glob("*.sh")):
        for call in _arena_invocations(path.read_text(encoding="utf-8")):
            if "matched_sims" not in call:
                continue
            scanned += 1
            if "--search-shape" not in call:
                offenders.append(f"{path.name}: {call.splitlines()[0]}")

    assert scanned, "no matched_sims arena invocation found in scripts/*.sh at all"
    assert not offenders, (
        f"{offenders} run a matched_sims arena without --search-shape; the arena "
        "will exit 1 and the script will record nothing"
    )


# ---------------------------------------------------------------------------
# The reader: one ruler per fit
# ---------------------------------------------------------------------------


def test_a_row_without_the_column_is_labelled_as_the_old_ruler() -> None:
    """Pre-2026-07-29 rows are NOT 'play' — that shape now carries vloss 3."""
    assert row_search_shape(_row(None)) == LEGACY
    assert row_search_shape(_row("training")) == "training"


def test_a_slope_is_not_fitted_across_a_ruler_break() -> None:
    rows = [_row(None), _row(None), _row("training"), _row("training")]

    kept = _one_ruler_only(rows, _args())

    assert [row_search_shape(r) for r in kept] == ["training", "training"]


def test_a_single_ruler_series_is_untouched() -> None:
    """The guard must not eat rows when there is no break."""
    rows = [_row("training") for _ in range(3)]

    assert _one_ruler_only(rows, _args()) == rows


def test_an_explicit_shape_selects_that_ruler() -> None:
    rows = [_row(None), _row("training")]

    kept = _one_ruler_only(rows, _args(search_shape=LEGACY))

    assert [row_search_shape(r) for r in kept] == [LEGACY]


def test_an_explicit_shape_that_matches_nothing_is_an_error() -> None:
    with pytest.raises(SystemExit, match="no rows with search_shape"):
        _one_ruler_only([_row("training")], _args(search_shape="play"))


def test_mixing_rulers_is_possible_but_never_silent(capsys: pytest.CaptureFixture[str]) -> None:
    rows = [_row(None), _row("training")]

    kept = _one_ruler_only(rows, _args(allow_mixed_rulers=True))

    assert kept == rows
    assert "FITTING ACROSS 2 RULERS" in capsys.readouterr().out


# Every repo entry point that now hard-requires --search-shape. Keyed by script
# name so adding a third is one line rather than a new test.
_SHAPE_REQUIRING_SCRIPTS = ("arena_standard.py", "elo_vs_sims.py")

# Docs that RECORD commands already run, rather than instructing anyone to run
# them. The ledger's arena invocations were executed before --search-shape
# existed; rewriting them to satisfy this test would falsify the record of what
# was actually measured. Instruction docs (eval_protocol, operations) are in
# scope precisely because someone will copy-paste from them.
_HISTORICAL_DOCS = frozenset({"experiment_ledger.md", "rl_loop_audit.md"})


def _shape_requiring_invocations(text: str) -> list[tuple[str, str]]:
    """Every invocation of a script that hard-requires --search-shape."""
    out: list[tuple[str, str]] = []
    for name in _SHAPE_REQUIRING_SCRIPTS:
        pattern = rf"scripts/{re.escape(name)}((?:[^\n]*\\\n)*[^\n]*)"
        out.extend((name, m.group(0)) for m in re.finditer(pattern, text))
    return out


def test_no_doc_or_script_invocation_omits_a_required_search_shape() -> None:
    """Docs must not print a command that exits 1.

    ``test_no_repo_script_arenas_without_naming_a_shape`` scans only
    ``scripts/*.sh`` and only ``arena_standard.py``. The #286 review found
    ``docs/eval_protocol.md`` still printing an ``elo_vs_sims.py`` command
    without ``--search-shape`` -- a documented yardstick that cannot run, which
    is the same defect class as an arena measuring the wrong search: the
    instrument is inert and the page still reads authoritative.
    """
    offenders: list[str] = []
    scanned = 0
    targets = [
        p for p in sorted((ROOT / "docs").glob("*.md"))
        if p.name not in _HISTORICAL_DOCS and not p.name.startswith("AUDIT_")
    ] + sorted((ROOT / "scripts").glob("*.sh"))
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for name, call in _shape_requiring_invocations(text):
            # matched_time legitimately rejects --search-shape.
            if "matched_time" in call:
                continue
            # arena_standard only requires it for matched_sims; elo_vs_sims always.
            if name == "arena_standard.py" and "matched_sims" not in call:
                continue
            scanned += 1
            if "--search-shape" not in call:
                offenders.append(f"{path.relative_to(ROOT)}: {call.splitlines()[0].strip()}")

    assert scanned, "no shape-requiring invocation found in docs/*.md or scripts/*.sh at all"
    assert not offenders, (
        f"{offenders} omit --search-shape; these commands exit 1 as written"
    )


# ---------------------------------------------------------------------------
# The wall-clock cap must not destroy the reading it caps.
# ---------------------------------------------------------------------------

def test_the_ratchet_caps_the_arena_from_inside_not_only_with_timeout() -> None:
    """`timeout` alone SIGKILLs the arena and the reading dies with it.

    2026-07-30 and 07-31: the arena reached its cap, was killed, and the single
    RUNNING-Elo block it had computed never left the block-buffered stdout pipe
    (buffering loses exactly the LAST block, so only a run too slow to print a
    second one loses everything). Both logs end mid-report and the parser wrote
    no CSV row. `--max-seconds` makes the arena stop on its own clock and
    finalize instead.
    """
    invocation = _arena_invocation()
    assert "--max-seconds" in invocation, (
        "daily_gate_ratchet.sh relies on an external `timeout` alone: a "
        "SIGKILLed arena returns no reading at all"
    )


def test_the_internal_cap_fires_before_the_external_backstop() -> None:
    """An internal budget >= the external one is a cap that cannot take effect.

    The arena needs headroom after its deadline to score, print and append the
    record; if `timeout` fires first, --max-seconds is decorative.
    """
    text = RATCHET_SH.read_text(encoding="utf-8")
    match = re.search(r"local inner=\$\(\(\s*budget\s*-\s*(\d+)\s*\)\)", text)
    assert match is not None, (
        "expected the internal budget to be derived from the external one as "
        "`budget - <grace>`"
    )
    assert int(match.group(1)) > 0, "the internal cap must be strictly shorter than `timeout`"
    assert '--max-seconds "$inner"' in _arena_invocation()
    assert 'timeout -k 20 "${budget}s"' in text, (
        "keep the external backstop: --max-seconds is only checked between "
        "plies, so a hang inside the C search still needs killing"
    )


# ---------------------------------------------------------------------------
# The zero-row / retry-storm behaviour, tested by RUNNING the script.
#
# The first version of the zero-row guard was asserted by source text, and
# inserting `ROWS_WRITTEN=1` above the check left all 27 tests green -- the
# repo's signature defect (a gate that cannot fail) sitting inside the guard
# against it. Everything below drives the real script against a sandbox tree
# with a STUB arena, so the only thing asserted is what the script did.
# ---------------------------------------------------------------------------

# The stub stands in for scripts/arena_standard.py. It records every launch, so
# a test can ask the one question that matters for a retry storm: HOW MANY
# TIMES DID THIS SPEND GPU?
_ARENA_STUB = '''#!/usr/bin/env python3
import os, pathlib, sys

pathlib.Path(os.environ["ARENA_CALLS"]).open("a").write(" ".join(sys.argv[1:]) + "\\n")
mode = os.environ.get("ARENA_MODE", "good")
if mode == "nopairs":
    # Verbatim shape of arena_standard.py's own no-pairs exit (rc 3), which is
    # what arena_2026-07-31_vs_boot512.log would have produced had it not been
    # SIGKILLed first.
    print("[arena] NO COMPLETE PAIRS in 880s - nothing to score.")
    raise SystemExit(3)
if mode == "silent":       # SIGKILLed / block-buffered: nothing parseable
    raise SystemExit(137)
print("[arena] 64 games (32 opening pairs)")
print("[arena] Elo: +16.3  95% CI: [-70.3, +105.0]")
print("[arena] score: 0.523")
'''


def _sandbox(tmp_path: Path, *, checkpoint: str = "checkpoint_000478") -> Path:
    """A tree the real ratchet script can be pointed at, with a stub arena."""
    import shutil

    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(RATCHET_SH, root / "scripts" / RATCHET_SH.name)
    shutil.copy(LOOP_SH, root / "scripts" / LOOP_SH.name)
    shutil.copy(COMMON_SH, root / "scripts" / COMMON_SH.name)
    (root / "scripts" / "arena_standard.py").write_text(_ARENA_STUB)
    ck = root / "runs" / "pbt2_small" / "tune" / "train_trial_x" / checkpoint
    ck.mkdir(parents=True)
    (ck / "trainer.pt").write_text("candidate-net")
    anchor = root / "scratchpad" / "scaleup" / "gateread" / "boot_snap_recheck_0711_0404.pt"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("boot512-net")
    return root


def _run_ratchet(root: Path, *args: str, mode: str = "good", env: dict | None = None):
    """Run the real script against `root`. Returns (rc, stdout+stderr)."""
    import subprocess

    r = subprocess.run(
        ["bash", f"scripts/{RATCHET_SH.name}", *args],
        cwd=str(root), capture_output=True, text=True, check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "RATCHET_ROOT": str(root),
            "ARENA_MODE": mode,
            "ARENA_CALLS": str(root / "arena_calls.txt"),
            "PYTHONDONTWRITEBYTECODE": "1",
            **(env or {}),
        },
    )
    return r.returncode, r.stdout + r.stderr


def _arena_launches(root: Path) -> int:
    p = root / "arena_calls.txt"
    return len(p.read_text().splitlines()) if p.exists() else 0


def _csv_rows(root: Path) -> list[str]:
    p = root / "data" / "ratchet" / "ratchet.csv"
    return [ln for ln in p.read_text().splitlines()[1:] if ln.strip()] if p.exists() else []


def _attempt_rows(root: Path) -> list[str]:
    p = root / "data" / "ratchet" / "attempts.csv"
    return [ln for ln in p.read_text().splitlines()[1:] if ln.strip()] if p.exists() else []


def test_a_readable_arena_writes_a_row_and_exits_zero(tmp_path) -> None:
    """The happy path, so every failure assertion below means something."""
    root = _sandbox(tmp_path)
    rc, out = _run_ratchet(root)
    assert rc == 0, f"a run that wrote a row must exit 0, got {rc}:\n{out}"
    rows = _csv_rows(root)
    assert len(rows) == 1, f"expected one vs_boot512 row (day 1 has no vs_prev): {rows}"
    assert rows[0].endswith(",training"), rows[0]
    assert ",478," in rows[0], f"the iteration must reach the row: {rows[0]}"
    assert _arena_launches(root) == 1
    # ATTEMPTED is recorded separately from SUCCEEDED.
    assert _attempt_rows(root) == [f"{rows[0].split(',')[0]},478,1,0,1,vs_prev:nosnap|vs_boot512:row"]


def test_a_second_run_the_same_day_costs_no_gpu(tmp_path) -> None:
    """The already-done path must not re-arena, or the cap protects nothing."""
    root = _sandbox(tmp_path)
    assert _run_ratchet(root)[0] == 0
    rc, out = _run_ratchet(root)
    assert rc == 0, f"a day that already has its row must exit 0, got {rc}:\n{out}"
    assert _arena_launches(root) == 1, "the already-done run launched the arena again"
    assert len(_csv_rows(root)) == 1


def test_a_zero_row_day_is_a_failure_not_a_success(tmp_path) -> None:
    """No CSV row must be a non-zero exit, judged by RUNNING the script.

    ``ratchet_loop.sh`` stamps ``data/ratchet/last_run_date`` only on exit 0 and
    documents that a failure "retries on the next poll instead of silently
    skipping the whole day". It could never do that: every early ``return`` in
    ``run_arena`` left the script exiting 0, so 2026-07-30 and 07-31 each burned
    their one attempt and were recorded as successful days.

    Asserted by effect: the source-text version of this test stayed green with
    ``ROWS_WRITTEN=1`` inserted before the check.
    """
    root = _sandbox(tmp_path)
    rc, out = _run_ratchet(root, mode="nopairs")
    assert rc != 0, f"a run with no CSV row must not exit 0:\n{out}"
    assert _csv_rows(root) == [], "no row should have been written"
    assert "NO ROWS WRITTEN" in out, f"the failure must be loud:\n{out}"
    assert _arena_launches(root) == 1


def test_a_zero_row_day_cannot_become_an_all_day_retry_storm(tmp_path) -> None:
    """THE BOUND. A day that keeps producing nothing must stop spending GPU.

    ``ratchet_loop.sh`` polls every 600s and stamps the day only on exit 0, so
    "exit non-zero and let it retry" against a failure that REPRODUCES is a
    30-minute 16-concurrent arena every ~40 minutes: ~18 GPU-hours/day, spent by
    the observer against the training it observes, and self-reinforcing
    (contention -> no complete pairs -> no row -> retry -> more contention). The
    2026-07-31 log is the shape: 4 lines ending at ``SEARCH reference:`` after
    880s, which a retry replays identically.

    The bound is asserted the only way that means anything -- by counting how
    many times the arena was launched.
    """
    root = _sandbox(tmp_path)

    rc1, out1 = _run_ratchet(root, mode="nopairs")
    assert rc1 == 1, f"the first zero-row run is retryable (exit 1), got {rc1}:\n{out1}"

    rc2, out2 = _run_ratchet(root, mode="nopairs")
    assert rc2 == 5, (
        f"a second attempt failing IDENTICALLY reproduces, so it is not "
        f"retryable; expected exit 5, got {rc2}:\n{out2}"
    )
    assert "RATCHET DOWN" in out2, f"giving up must be loud:\n{out2}"
    assert "NOT a null reading" in out2, (
        f"giving up must be louder than failing, not quieter:\n{out2}"
    )
    assert _arena_launches(root) == 2

    # Every later poll of the day is refused BEFORE the arena starts.
    for _ in range(3):
        rc, out = _run_ratchet(root, mode="nopairs")
        assert rc == 5, f"an exhausted day must keep saying so, got {rc}:\n{out}"
    assert _arena_launches(root) == 2, (
        "the day kept launching arenas after giving up — this is the retry storm"
    )
    assert _csv_rows(root) == [], "a dead day must never gain a row"
    # ...and the day is legible afterwards: attempted twice, succeeded never.
    attempts = _attempt_rows(root)
    assert len(attempts) == 2, f"one ledger row per GPU-spending attempt: {attempts}"
    assert all(a.split(",")[4] == "0" for a in attempts), attempts


def test_the_operator_can_reopen_a_day_they_have_fixed(tmp_path) -> None:
    """The cap is a budget, not a lockout — but only the operator lifts it."""
    root = _sandbox(tmp_path)
    _run_ratchet(root, mode="nopairs")
    _run_ratchet(root, mode="nopairs")
    assert _arena_launches(root) == 2
    rc, out = _run_ratchet(root, "--max-attempts", "0")
    assert rc == 0, f"--max-attempts 0 must run again, got {rc}:\n{out}"
    assert _arena_launches(root) == 3
    assert len(_csv_rows(root)) == 1


def test_a_structural_failure_is_not_retried_even_once(tmp_path) -> None:
    """Nothing later today can make a missing reference file exist.

    A retry that will deterministically reproduce is not a retry, so this one
    must never reach the arena at all.
    """
    root = _sandbox(tmp_path)
    (root / "scratchpad/scaleup/gateread/boot_snap_recheck_0711_0404.pt").unlink()
    rc, out = _run_ratchet(root)
    assert rc == 5, f"an all-structural failure must not be retryable, got {rc}:\n{out}"
    assert "structurally" in out, out
    assert _arena_launches(root) == 0, "GPU was spent on a run that could not work"


def test_a_changed_failure_is_still_retried(tmp_path) -> None:
    """The give-up rule must not swallow genuinely transient failures.

    Without this the cap could be satisfied by never retrying anything, which
    is the opposite defect: a lost race or a burst of contention deserves the
    second attempt this instrument was given an exit status for.
    """
    root = _sandbox(tmp_path)
    rc1, _ = _run_ratchet(root, mode="silent")      # backstop / SIGKILL shape
    assert rc1 == 1
    rc2, out2 = _run_ratchet(root, mode="nopairs")  # a DIFFERENT failure
    assert rc2 == 1, f"a different failure is still retryable, got {rc2}:\n{out2}"
    assert _arena_launches(root) == 2
    # ...and the third is refused by the absolute cap rather than by the reason.
    rc3, out3 = _run_ratchet(root, mode="silent")
    assert rc3 == 5, f"the 3rd attempt must exhaust the day's budget, got {rc3}:\n{out3}"
    assert "budget" in out3, out3
    assert _arena_launches(root) == 3, "the capping run must not itself skip the arena"


def test_the_loop_stamps_the_day_only_for_a_real_reading(tmp_path) -> None:
    """ratchet_loop.sh's half of the contract, executed.

    Three outcomes, three different states: a reading stamps the day; a
    retryable failure stamps nothing; giving up stops the day WITHOUT ever
    claiming it succeeded. Collapsing the last two into one file is how a dead
    instrument would come to read as a null result.
    """
    import subprocess

    src = LOOP_SH.read_text(encoding="utf-8")
    preamble = "\n".join(
        ln for ln in (src + COMMON_SH.read_text(encoding="utf-8")).splitlines()
        if re.match(r"(STATE|GIVEUP_STATE|LOG|RATCHET_EXIT_NO_RETRY)=", ln)
    )
    assert preamble.count("\n") == 3, f"expected 4 settings, got:\n{preamble}"
    log_fn = next(ln for ln in src.splitlines() if ln.startswith("log() {"))
    body = (
        preamble + "\n" + log_fn + "\n"
        + _extract_shell_func("ratchet_outcome", LOOP_SH)
    )

    def call(rc: int) -> tuple[str, str]:
        (tmp_path / "data" / "ratchet").mkdir(parents=True, exist_ok=True)
        (tmp_path / "scratchpad").mkdir(exist_ok=True)
        subprocess.run(
            ["bash", "-c", f'set -u\n{body}\nratchet_outcome {rc} 2026-08-01'],
            cwd=str(tmp_path), capture_output=True, text=True, check=True,
        )
        done = tmp_path / "data" / "ratchet" / "last_run_date"
        gave_up = tmp_path / "data" / "ratchet" / "last_giveup_date"
        return (
            done.read_text().strip() if done.exists() else "",
            gave_up.read_text().strip() if gave_up.exists() else "",
        )

    assert call(1) == ("", ""), "a retryable failure must stamp nothing at all"
    assert call(5) == ("", "2026-08-01"), (
        "giving up must stop the day WITHOUT stamping it as done"
    )
    assert call(0) == ("2026-08-01", "2026-08-01"), "a real reading stamps the day"
    log = (tmp_path / "scratchpad" / "ratchet_loop.log").read_text()
    assert "GAVE UP" in log, log
    assert "NO strength measurement" in log, log


def _run_one_poll(root: Path, *, mode: str = "good", timeout: float = 120.0):
    """Drive ONE real poll of ratchet_loop.sh end to end. Returns (rc, output).

    The loop's own body, not an extracted function: the defect this whole
    change is about is a rule that is pinned while the wiring INTO it is not,
    and `ratchet_outcome "$?"` -> `ratchet_outcome 0` is exactly that mutation.
    """
    import subprocess

    pidfile = root / "trainer.pid"
    # A process that really is alive, so `trainer_running` passes for the same
    # reason it does in production rather than because a check was stubbed out.
    alive = subprocess.Popen(["sleep", "600"])
    try:
        pidfile.write_text(f"{alive.pid}\n")
        r = subprocess.run(
            ["bash", f"scripts/{LOOP_SH.name}", "--once"],
            cwd=str(root), capture_output=True, text=True, check=False,
            timeout=timeout,
            env={
                "PATH": os.environ.get("PATH", ""),
                "RATCHET_ROOT": str(root),
                "TRAIN_PIDFILE": str(pidfile),
                "RATCHET_POLL": "1",
                "ARENA_MODE": mode,
                "ARENA_CALLS": str(root / "arena_calls.txt"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    finally:
        alive.terminate()
        alive.wait()
    log = root / "scratchpad" / "ratchet_loop.log"
    return r.returncode, r.stdout + r.stderr + (log.read_text() if log.exists() else "")


def _loop_state(root: Path) -> tuple[str, str]:
    """(last_run_date, last_giveup_date) as recorded by the loop."""
    d = root / "data" / "ratchet"
    return (
        (d / "last_run_date").read_text().strip() if (d / "last_run_date").exists() else "",
        (d / "last_giveup_date").read_text().strip() if (d / "last_giveup_date").exists() else "",
    )


def test_one_poll_records_the_outcome_it_actually_got(tmp_path) -> None:
    """The loop BODY, executed: a whole poll, both scripts, a stub arena.

    ``ratchet_outcome`` is tested above, but nothing executed the call site that
    feeds it, and a call site is where this repo's defect lives: replacing
    ``ratchet_outcome "$rc"`` with ``ratchet_outcome 0`` — or slipping any
    command between the ratchet run and the status capture, which clobbers
    ``$?`` — restores the exact silent hole the exit status exists to prevent,
    with every rule below it still perfect. Both mutations were run against this
    test and both kill it.
    """
    import datetime

    today = datetime.date.today().isoformat()

    # 1. A real reading: the day is stamped DONE and nothing else is.
    good = _sandbox(tmp_path / "good")
    rc, out = _run_one_poll(good, mode="good")
    assert rc == 0, f"a poll that got a reading must exit 0, got {rc}:\n{out}"
    assert _arena_launches(good) == 1, f"the poll never ran the arena:\n{out}"
    assert len(_csv_rows(good)) == 1, f"no CSV row was written:\n{out}"
    assert _loop_state(good) == (today, ""), (
        f"expected only last_run_date stamped, got {_loop_state(good)}:\n{out}"
    )

    # 2. A retryable failure: NOTHING is stamped, so the next poll tries again.
    bad = _sandbox(tmp_path / "bad")
    rc, out = _run_one_poll(bad, mode="nopairs")
    assert rc == 1, f"a zero-row poll must report the retryable status, got {rc}:\n{out}"
    assert _loop_state(bad) == ("", ""), (
        f"a retryable failure stamped a day: {_loop_state(bad)}:\n{out}"
    )

    # 3. The same failure again: the day gives up. last_run_date STILL unset —
    #    the day has no reading and must never read as though it had one.
    rc, out = _run_one_poll(bad, mode="nopairs")
    assert rc == 5, f"the repeat failure must be non-retryable, got {rc}:\n{out}"
    assert _loop_state(bad) == ("", today), (
        f"give-up must stamp ONLY last_giveup_date, got {_loop_state(bad)}:\n{out}"
    )
    assert "GAVE UP" in out, f"the give-up must reach the loop log:\n{out}"
    assert _csv_rows(bad) == []

    # 4. ...and the next poll costs nothing at all.
    launches = _arena_launches(bad)
    rc, out = _run_one_poll(bad, mode="nopairs")
    assert _arena_launches(bad) == launches, (
        f"a day that gave up kept spending GPU:\n{out}"
    )


def test_neither_script_can_disagree_with_the_other() -> None:
    """The four shared facts have ONE definition, so drift is inexpressible.

    Keeping two copies EQUAL is not the same as having one, and the difference
    was not hypothetical: ratchet_loop.sh honoured $TRAIN_WORK_DIR while
    daily_gate_ratchet.sh hard-coded runs/pbt2_small, so under
    TRAIN_WORK_DIR=runs/other the loop chose the iteration from one tree and
    the ratchet snapshotted and recorded from the other. The same class left
    the checkpoint_000000 parse duplicated with only one copy covered.

    So this asserts ABSENCE in the two scripts, not agreement between them.
    """
    common = COMMON_SH.read_text(encoding="utf-8")
    m = re.search(r"^RATCHET_EXIT_NO_RETRY=(\d+)", common, re.M)
    assert m is not None, f"{COMMON_SH} defines no RATCHET_EXIT_NO_RETRY"
    status = int(m.group(1))
    # Must not collide with success, the retryable failure, bad usage, the
    # arena's own no-pairs status, or `timeout`'s.
    assert status not in (0, 1, 2, 3, 124, 137)
    assert re.search(r'^WORK_DIR="\$\{TRAIN_WORK_DIR:-', common, re.M), common

    for path in (RATCHET_SH, LOOP_SH):
        src = path.read_text(encoding="utf-8")
        assert 'ratchet_common.sh"' in src, f"{path.name} does not source the shared file"
        for pattern, what in (
            (r"^\s*RATCHET_EXIT_(NO_)?RETRY=", "its own copy of an exit status"),
            (r"^\s*WORK_DIR=", "its own copy of the run directory"),
            (r"^\s*cd \S*/home/josh", "its own hard-coded repo path"),
            (r"sed 's/checkpoint_0\*//'", "the old iteration parse"),
            (r"sed -E 's/\^checkpoint_//", "its own copy of the iteration parse"),
        ):
            assert not re.search(pattern, src, re.M), (
                f"{path.name} re-defines {what}; that is the duplication this "
                "shared file exists to remove, and it has drifted before"
            )
    # The hard-coded run dir specifically: the drift that was actually shipped.
    assert "runs/pbt2_small" not in RATCHET_SH.read_text(encoding="utf-8")


def test_the_two_scripts_pick_the_same_trial_tree(tmp_path) -> None:
    """$TRAIN_WORK_DIR must move BOTH, or a row names a foreign checkpoint.

    Reproduces the shipped divergence by effect: with the run directory
    redirected, the ratchet must measure the redirected tree — previously it
    kept reading runs/pbt2_small while the loop decided from the other one.
    """
    root = _sandbox(tmp_path)
    other = root / "runs" / "other" / "tune" / "train_trial_z" / "checkpoint_000900"
    other.mkdir(parents=True)
    (other / "trainer.pt").write_text("net-900")

    rc, out = _run_ratchet(root, env={"TRAIN_WORK_DIR": "runs/other"})
    assert rc == 0, out
    rows = _csv_rows(root)
    assert len(rows) == 1 and ",900," in rows[0], (
        f"the ratchet measured a different tree than TRAIN_WORK_DIR names: {rows}"
    )


def test_nothing_to_measure_yet_is_retryable_not_a_finished_day(tmp_path) -> None:
    """Ray creates checkpoint_NNNNNN/ before it writes trainer.pt.

    That race is a state every fresh restart passes through, and the three
    early-outs answered it with `exit 0`: no attempts.csv row (they return
    before the ledger is initialised), so ratchet_loop.sh stamped
    last_run_date, `loop_health.ratchet_gap_alerts` could not see the hole,
    result.json stayed green and the log said "daily ratchet done". The day was
    gone. It costs no GPU to say "not yet" instead — nothing here has run an
    arena — and the day then keeps its remaining polls.
    """
    import datetime

    # (a) the live race: the directory exists, the file does not.
    racing = _sandbox(tmp_path / "race")
    (racing / "runs/pbt2_small/tune/train_trial_x/checkpoint_000478/trainer.pt").unlink()
    rc, out = _run_ratchet(racing)
    assert rc == 1, f"a half-written checkpoint must be retryable, got {rc}:\n{out}"
    assert "YET" in out, out
    assert _arena_launches(racing) == 0, "no GPU may be spent to discover this"
    assert _attempt_rows(racing) == [], (
        "a startup race must not consume one of the day's 3 attempts"
    )
    # ...and the loop must not stamp the day.
    rc, out = _run_one_poll(racing)
    assert rc == 1, f"the loop must see the retryable status, got {rc}:\n{out}"
    assert _loop_state(racing) == ("", ""), (
        f"the day was stamped despite having no reading: {_loop_state(racing)}"
    )

    # (b) no checkpoint at all, and (c) no trial dir at all.
    for label, victim in (("nock", "runs/pbt2_small/tune/train_trial_x/checkpoint_000478"),
                          ("notrial", "runs/pbt2_small/tune/train_trial_x")):
        import shutil

        root = _sandbox(tmp_path / label)
        shutil.rmtree(root / victim)
        rc, out = _run_ratchet(root)
        assert rc == 1, f"{label}: expected retryable, got {rc}:\n{out}"
        assert _arena_launches(root) == 0

    # And the recovery: once trainer.pt lands, the same day proceeds normally.
    (racing / "runs/pbt2_small/tune/train_trial_x/checkpoint_000478/trainer.pt").write_text("net")
    rc, out = _run_one_poll(racing)
    assert rc == 0, f"the day must still be measurable once the file lands:\n{out}"
    assert _loop_state(racing) == (datetime.date.today().isoformat(), "")
    assert len(_csv_rows(racing)) == 1


def test_a_skipped_series_hands_its_budget_to_the_other(tmp_path) -> None:
    """A skipped vs_prev must not forfeit half the day's clock.

    The budget split is (time left / series still to come), and only
    ``run_arena`` decremented that count — so when the self-match guard skipped
    vs_prev, vs_boot512 still divided by 2 and ~900s of a 1800s budget went
    unused. That fired on exactly the training-down days when the guard exists
    to fire, i.e. when the one remaining series is the only chance of a row.

    Asserted through the --max-seconds the arena is actually launched with.
    """
    root = _sandbox(tmp_path)
    rc, out = _run_ratchet(root)   # day 1: vs_prev has no earlier snapshot
    assert rc == 0, out
    calls = (root / "arena_calls.txt").read_text().splitlines()
    assert len(calls) == 1, calls
    m = re.search(r"--max-seconds (\d+)", calls[0])
    assert m is not None, f"no --max-seconds in the arena invocation: {calls[0]}"
    inner = int(m.group(1))
    # 30 minutes total, one series running: it must get essentially all of it,
    # not the ~855s that the un-decremented divisor produced.
    assert inner > 1500, (
        f"vs_boot512 was launched with only {inner}s of an 1800s budget — the "
        "skipped series never gave its share back"
    )


def test_both_series_still_split_the_budget(tmp_path) -> None:
    """The other direction: two live series must still get half each."""
    root = _sandbox(tmp_path)
    snaps = root / "data" / "ratchet" / "snapshots"
    snaps.mkdir(parents=True)
    old = snaps / "ck_2026-07-30_iter409.pt"
    old.write_text("an older, genuinely different net")
    os.utime(old, (1, 1))

    rc, out = _run_ratchet(root)
    assert rc == 0, out
    calls = (root / "arena_calls.txt").read_text().splitlines()
    assert len(calls) == 2, f"both series should have run: {calls}"
    m = re.search(r"--max-seconds (\d+)", calls[0])
    assert m is not None, f"no --max-seconds in the arena invocation: {calls[0]}"
    first = int(m.group(1))
    assert 700 < first < 1000, (
        f"the first of two series took {first}s of 1800s — the split is wrong"
    )




def test_already_done_and_the_row_counter_are_keyed_on_the_CSV() -> None:
    """"Done" must mean "a CSV row exists", not "the log looks parseable".

    Both weaker keys were shipped and both were wrong:
      ``grep "Elo:"``            matched the RUNNING-block HEADER, i.e. exactly
                                 what a SIGKILLed run leaves behind.
      ``grep "^\\[arena\\] Elo:"``  means the LOG parsed, which is still not a
                                 row: every field guard below `return`s without
                                 writing, and the one-sided `n/a` CI case is
                                 common at small pair counts
                                 (arena_2026-07-29_vs_prev.log has one at 5
                                 pairs). With --max-seconds a capped run now
                                 ALWAYS prints a final summary, so that case got
                                 more likely, not less -- and the skip guard
                                 would then report "already done today", exit 0,
                                 and let ratchet_loop.sh stamp a day whose CSV
                                 gained nothing.
    """
    text = RATCHET_SH.read_text(encoding="utf-8")
    assert 'grep -q "^$today,$iter,$series," "$LOG"' in text, (
        "the already-done guard must ask the CSV, not the arena log"
    )
    assert 'grep -qE "^\\[arena\\] Elo:" "$out"' not in text
    assert 'grep -q "Elo:" "$out"' not in text
    # The counter that decides the exit status must be incremented only where a
    # row is actually appended (plus the CSV-keyed skip above).
    assert text.count("ROWS_WRITTEN=$(( ROWS_WRITTEN + 1 ))") == 2
    body = text.split("run_arena () {", 1)[1]
    append_idx = body.index('$score,$played,$SHAPE" >> "$LOG"')
    incr_idx = body.index("ROWS_WRITTEN=$(( ROWS_WRITTEN + 1 ))", append_idx)
    assert incr_idx > append_idx, (
        "the row counter must be incremented AFTER the append, so no early "
        "return can count a row that was never written"
    )


# ---------------------------------------------------------------------------
# --min-pairs must be tested by its EFFECT. The first version of this guard was
# pure source-text matching, and a reviewer killed the filter with `if False`
# while every assertion still passed -- the repo's signature defect sitting
# inside the guard against it.
# ---------------------------------------------------------------------------

def _ratchet_row(date: str, it: int, elo: float, lo: float, hi: float, games: int) -> dict[str, str]:
    return {
        "date": date, "iter": str(it), "series": "vs_boot512",
        "elo": str(elo), "ci_lo": str(lo), "ci_hi": str(hi),
        "score": "0.5", "games": str(games), "search_shape": "training",
    }


def test_min_pairs_actually_removes_the_row_from_the_fitted_set() -> None:
    """The short row must be ABSENT from the fitted x/y, not merely mentioned.

    Real values from data/ratchet/ratchet.csv (vs_boot512): the 13-pair
    2026-07-29 row carries 3.0% of the 1/se^2 weight but 93.5% of the chi2
    numerator (residual +98 Elo), so it sets `se_slope` for the whole fit. The
    exclusion is OUTLIER REMOVAL; a de-weighting story would be backwards,
    because a short run's CI is wide and its weight therefore small.
    """
    from scripts.ratchet_slope import select_fit_rows

    rows = [
        _ratchet_row("2026-07-27", 122, -11.1, -44.8, 22.5, 314),  # 157 pairs
        _ratchet_row("2026-07-28", 172, -7.6, -91.0, 75.0, 46),    # 23 pairs
        _ratchet_row("2026-07-29", 218, 81.6, -39.5, 227.5, 26),   # 13 pairs
    ]
    cum = {122: 6213.0, 172: 10943.0, 218: 14959.0}

    sel = select_fit_rows(rows, cum, min_pairs=26)
    fitted_iters = [m[1] for m in sel.meta]
    assert fitted_iters == [122], (
        f"--min-pairs 26 must fit only the 157-pair row; fitted {fitted_iters}"
    )
    assert len(sel.xs) == len(sel.ys) == len(sel.ses) == 1
    assert 14959.0 not in sel.xs, "the 13-pair row leaked into the fitted x"
    excluded = sorted((it, pairs) for _d, it, pairs, *_rest in sel.too_small)
    assert excluded == [(172, 23), (218, 13)]

    # Disabling the floor must put them back -- otherwise the flag is not what
    # is doing the work.
    off = select_fit_rows(rows, cum, min_pairs=0)
    assert sorted(m[1] for m in off.meta) == [122, 172, 218]
    assert off.too_small == []


def test_excluded_rows_carry_the_number_a_reader_needs() -> None:
    """Naming a dropped row without its Elo/CI asks for trust, not judgement."""
    from scripts.ratchet_slope import select_fit_rows

    rows = [_ratchet_row("2026-07-29", 218, 81.6, -39.5, 227.5, 26)]
    sel = select_fit_rows(rows, {218: 14959.0}, min_pairs=26)
    assert sel.too_small == [("2026-07-29", 218, 13, 81.6, -39.5, 227.5)]


def test_min_pairs_changes_the_fitted_slope(tmp_path) -> None:
    """End-to-end: the flag must move the number, not just the log.

    Guards the whole path -- argparse -> select_fit_rows -> weighted_slope --
    rather than any one link. A dead filter gives both invocations the same fit.
    """
    import subprocess
    import sys

    csv_path = tmp_path / "ratchet.csv"
    csv_path.write_text(
        "date,iter,series,elo,ci_lo,ci_hi,score,games,search_shape\n"
        "2026-07-26,25,vs_boot512,-12.2,-61.5,36.7,0.48,200,training\n"
        "2026-07-27,122,vs_boot512,-11.1,-44.8,22.5,0.48,314,training\n"
        "2026-07-29,218,vs_boot512,81.6,-39.5,227.5,0.62,26,training\n"
        "2026-07-30,300,vs_boot512,-20.0,-70.0,30.0,0.47,220,training\n"
        "2026-07-31,478,vs_boot512,-39.8,-91.4,10.1,0.44,114,training\n"
    )
    # FIVE rows so that FOUR survive the floor: at four-with-one-dropped the run
    # is MUTED (exit 4) and prints no slope, which would defeat this test.
    trial = tmp_path / "run" / "tune" / "train_trial_x"
    trial.mkdir(parents=True)
    (trial / "progress.csv").write_text(
        "training_iteration,trainer_steps_done\n"
        "25,961\n122,5252\n218,8746\n300,4000\n478,23103\n"
    )

    def run(*extra: str) -> str:
        return subprocess.run(
            [sys.executable, "scripts/ratchet_slope.py",
             "--ratchet-csv", str(csv_path), "--run-dir", str(trial.parent.parent),
             *extra],
            cwd=str(ROOT), capture_output=True, text=True, check=True,
            env={"PYTHONPATH": str(ROOT), "PATH": os.environ.get("PATH", ""),
                 "PYTHONDONTWRITEBYTECODE": "1"},
        ).stdout

    default = run()
    disabled = run("--min-pairs", "0")

    def slope_of(out: str) -> str:
        line = next(ln for ln in out.splitlines() if "WLS slope:" in ln)
        return line.split("WLS slope:")[1].strip()

    assert "rows usable: 4" in default
    assert "rows usable: 5" in disabled
    assert slope_of(default) != slope_of(disabled), (
        "--min-pairs did not change the fit -- the filter is dead:\n"
        f"  default : {slope_of(default)}\n  disabled: {slope_of(disabled)}"
    )
    assert "EXCLUDED below --min-pairs 26" in default
    assert "EXCLUDED" not in disabled


def test_the_verdict_line_carries_its_own_caveats(tmp_path) -> None:
    """Caveats printed 15 lines above the VERDICT are caveats nobody reads.

    The VERDICT block reads only `len(xs)` and the CI sign, so without this the
    tool prints a bare KILL/PIVOT while every fitted row is flagged UNVERIFIED.
    """
    import subprocess
    import sys

    csv_path = tmp_path / "ratchet.csv"
    csv_path.write_text(
        "date,iter,series,elo,ci_lo,ci_hi,score,games,search_shape\n"
        "2026-07-01,10,vs_boot512,10.0,0.0,20.0,0.51,60,training\n"
        "2026-07-02,20,vs_boot512,0.0,-10.0,10.0,0.50,60,training\n"
        "2026-07-03,30,vs_boot512,-10.0,-20.0,0.0,0.49,60,training\n"
        "2026-07-04,40,vs_boot512,-20.0,-30.0,-10.0,0.48,60,training\n"
    )
    trial = tmp_path / "run" / "tune" / "train_trial_x"
    trial.mkdir(parents=True)
    (trial / "progress.csv").write_text(
        "training_iteration,trainer_steps_done\n10,1000\n20,1000\n30,1000\n40,1000\n"
    )
    out = subprocess.run(
        [sys.executable, "scripts/ratchet_slope.py",
         "--ratchet-csv", str(csv_path), "--run-dir", str(trial.parent.parent)],
        cwd=str(ROOT), capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(ROOT), "PATH": os.environ.get("PATH", ""),
             "PYTHONDONTWRITEBYTECODE": "1"},
    ).stdout
    lines = out.splitlines()
    verdict_idx = next(i for i, ln in enumerate(lines) if "VERDICT:" in ln)
    after = "\n".join(lines[verdict_idx:])
    assert "30 pairs" in out or "below 100 pairs" in after
    assert "CAVEAT" in after, (
        "every fitted row is under 100 pairs and the verdict says nothing:\n" + after
    )
    assert "UNVERIFIED" in out

    # The EXCLUDED caveat is a SEPARATE branch and escaped its own mutation.
    # Add a 10-pair row: the fit drops it, and the verdict must say so where the
    # verdict is read, not 15 lines up.
    csv_path.write_text(
        csv_path.read_text()
        + "2026-07-05,50,vs_boot512,-99.0,-200.0,2.0,0.40,20,training\n"
    )
    (trial / "progress.csv").write_text(
        (trial / "progress.csv").read_text() + "50,1000\n"
    )
    out2 = subprocess.run(
        [sys.executable, "scripts/ratchet_slope.py",
         "--ratchet-csv", str(csv_path), "--run-dir", str(trial.parent.parent)],
        cwd=str(ROOT), capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(ROOT), "PATH": os.environ.get("PATH", ""),
             "PYTHONDONTWRITEBYTECODE": "1"},
    ).stdout
    lines2 = out2.splitlines()
    after2 = "\n".join(lines2[next(i for i, ln in enumerate(lines2) if "VERDICT:" in ln):])
    assert "excluded by --min-pairs" in after2, (
        "a row was dropped from the fit and the VERDICT block does not mention "
        "it:\n" + after2
    )


def test_pick_log_path_never_returns_an_existing_file(tmp_path) -> None:
    """A retry must not overwrite the attempt it is retrying.

    Executes the REAL ``pick_log_path`` out of daily_gate_ratchet.sh rather than
    grepping for it: the earlier version of this file's --min-pairs guard was
    pure source matching and could not fail when the logic was dead, so this one
    extracts the function and runs it.

    The failure it guards is on the record. The log name used to be
    ``arena_<date>_<series>.log`` opened with ``>``, and that is exactly how the
    2026-07-27 00:09 failure log was destroyed by a later run the same day --
    which then caused that day to be mis-attributed in the first version of this
    fix. The exit-1 retry path makes repeat attempts routine, so the collision
    went from rare to expected.
    """
    import subprocess

    src = RATCHET_SH.read_text(encoding="utf-8")
    start = src.index("pick_log_path () {")
    end = src.index("\n}\n", start) + len("\n}\n")
    func = src[start:end]
    assert "printf" in func, "extraction did not capture the whole function"

    (tmp_path / "data" / "ratchet").mkdir(parents=True)

    def call() -> str:
        out = subprocess.run(
            ["bash", "-c", f"{func}\npick_log_path 2026-07-31 478 vs_prev"],
            cwd=str(tmp_path), capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out

    # The function is only worth anything if run_arena USES it. A perfect,
    # tested, uncalled helper is the same defect one band down.
    assert 'out=$(pick_log_path "$today" "$iter" "$series")' in src, (
        "run_arena does not call pick_log_path; the helper is dead code and the "
        "retry is back to overwriting the previous attempt's log"
    )
    assert 'out="data/ratchet/arena_${today}_${series}.log"' not in src

    first = call()
    assert first.endswith("arena_2026-07-31_iter478_vs_prev.log")
    assert "iter478" in first, "the log name must be keyed on the iteration too"

    # Simulate three successive attempts; each must get a fresh path.
    seen = []
    for _ in range(3):
        path = call()
        assert not (tmp_path / path).exists(), (
            f"pick_log_path returned an EXISTING file ({path}); the retry would "
            "overwrite the previous attempt's evidence"
        )
        seen.append(path)
        (tmp_path / path).write_text("attempt log\n")
    assert len(set(seen)) == 3, f"collided: {seen}"


def _extract_shell_func(name: str, path: Path = RATCHET_SH) -> str:
    """Pull one function out of a real script so a test can RUN it."""
    src = path.read_text(encoding="utf-8")
    start = src.index(f"{name} () {{")
    end = src.index("\n}\n", start) + len("\n}\n")
    return src[start:end]


def test_vs_prev_refuses_to_arena_a_net_against_itself(tmp_path) -> None:
    """A same-iteration reference has true Elo 0 and must not become a row.

    `prev` was selected by FILENAME only. With training down the daily snapshot
    is byte-identical to yesterday's under a new date, so `vs_prev` played the
    net against ITSELF: sha256 of ck_2026-07-31_iter478.pt and
    ck_2026-08-01_iter478.pt are the same file, and the 2026-08-01 run recorded
    -43.7 Elo [-224.5, +114.9] for a comparison whose true value is exactly 0.
    The preserved attempt-1 log is the accidental null control -- +0.0 at 14
    pairs, -15.8 at 22, +5.8 at 30, converging to zero as a self-match must.

    Harmless while a short run wrote nothing; `--max-seconds` made every run
    write a row, and the ratchet is advised to run in a training-down window --
    exactly when this fires. Executed, not grepped: N7 showed a guard can be
    perfect and uncalled.

    The comparison is on CONTENT (`cmp -s`), not on the iteration parsed out of
    the filename: the name is a claim about the file and the file is the file.
    The iteration check survives as a second net, for a same-iteration
    checkpoint that was re-SAVED and so differs byte-wise while still being a
    self-match for Elo purposes.
    """
    import subprocess

    func = _extract_shell_func("pick_prev_snapshot")
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    today = snaps / "ck_2026-08-01_iter478.pt"

    def call(it: str) -> tuple[int, str]:
        r = subprocess.run(
            ["bash", "-c", f"{func}\npick_prev_snapshot '{snaps}' '{today}' '{it}'"],
            cwd=str(tmp_path), capture_output=True, text=True, check=False,
        )
        return r.returncode, r.stdout.strip()

    # 1. No earlier snapshot at all.
    today.write_text("net-478")
    rc, out = call("478")
    assert rc == 1, f"expected 'none available', got rc={rc} {out!r}"
    assert out == "", f"nothing should be printed when there is no reference: {out!r}"

    # 2. THE BUG: the newest earlier snapshot is byte-identical to today's. Its
    #    NAME says iter409, so a filename comparison waves it through; it is the
    #    same net, and the row would be a self-match with true Elo 0.
    twin = snaps / "ck_2026-07-31_iter409.pt"
    twin.write_text("net-478")
    os.utime(twin, (2, 2))  # older than today's snapshot, newer than the rest
    rc, out = call("478")
    assert rc == 3, (
        f"a byte-identical reference must be refused, got rc={rc} {out!r} — this "
        "is the case the iteration-from-the-filename check cannot see"
    )
    assert out == "", "a refused reference must not be handed back as a path"
    twin.unlink()

    # 3. The second net: a re-saved checkpoint of the SAME iteration differs
    #    byte-wise, so `cmp` alone would accept it.
    resaved = snaps / "ck_2026-07-31_iter478.pt"
    resaved.write_text("net-478 re-saved (different optimizer moments)")
    os.utime(resaved, (2, 2))
    rc, out = call("478")
    assert rc == 3, f"same-iteration reference must be refused, got rc={rc} {out!r}"
    assert out == "", "a refused reference must not be handed back as a path"

    # 4. A genuinely earlier, genuinely different net is still used.
    real = snaps / "ck_2026-07-30_iter409.pt"
    real.write_text("net-409")
    os.utime(real, (1, 1))  # oldest, so `ls -t` still ranks iter478 first
    rc, out = call("478")
    assert rc == 3, (
        "the newest earlier snapshot is still iter478; falling back to an older "
        "iteration would silently relabel a 2-day gap as 'vs_prev'"
    )
    resaved.unlink()
    rc, out = call("478")
    assert rc == 0, f"a different net must be usable, got rc={rc} {out!r}"
    assert out.endswith("ck_2026-07-30_iter409.pt"), (
        f"wrong reference returned: {out!r}"
    )


def test_the_self_match_guard_is_wired_into_the_series() -> None:
    """The helper has to be what actually selects the vs_prev reference."""
    src = RATCHET_SH.read_text(encoding="utf-8")
    assert 'prev=$(pick_prev_snapshot "$SNAP_DIR" "$snap" "$iter")' in src, (
        "the helper must be handed today's snapshot PATH — it compares content"
    )
    assert 'grep -v "$(basename "$snap")"' not in src, (
        "the filename-only selector is what produced the self-match row"
    )
    # rc 3 must be handled distinctly, or a self-match degrades to "no snapshot"
    # and the operator never learns why the series stopped.
    assert "3)" in src.split("series 1:")[1].split("series 2:")[0]


def test_the_iteration_parse_survives_checkpoint_000000(tmp_path) -> None:
    """`sed 's/checkpoint_0*//'` returns the EMPTY STRING for iteration 0.

    The zeros it strips ARE the number. The snapshot then lands at
    ``ck_<date>_iter.pt``, ``ratchet_loop.sh``'s ``[ "${iter:-0}" -ge ...``
    silently reads 0, and every CSV row for that day carries an empty iter
    column. Executes the real line out of the script rather than asserting on
    its text.

    BOTH scripts parse it, and the loop's copy was NOT covered when this test
    only read the ratchet's: reverting `ratchet_loop.sh` alone survived the
    whole battery. There is now exactly ONE parse, in ratchet_common.sh, which
    is what this executes — a mutation cannot hide in a second copy because
    there is no second copy (test_neither_script_can_disagree_with_the_other
    fails if one reappears).
    """
    import subprocess

    func = _extract_shell_func("ratchet_iter_from_checkpoint", COMMON_SH)
    for name, want in (
        ("checkpoint_000000", "0"),
        ("checkpoint_000042", "42"),
        ("checkpoint_000478", "478"),
        ("checkpoint_001000", "1000"),
        ("runs/x/tune/train_trial_a/checkpoint_000007/", "7"),
    ):
        out = subprocess.run(
            ["bash", "-c", f'{func}\nprintf "%s" "$(ratchet_iter_from_checkpoint "{name}")"'],
            cwd=str(tmp_path), capture_output=True, text=True, check=True,
        ).stdout
        assert out == want, f"{name} parsed as {out!r}, expected {want!r}"


def _slope_fixture(tmp_path, rows: list[tuple[str, int, float, int]]):
    """(csv_path, run_dir) for ratchet_slope, from (date, iter, elo, games)."""
    csv_path = tmp_path / "ratchet.csv"
    lines = ["date,iter,series,elo,ci_lo,ci_hi,score,games,search_shape"]
    prog = ["training_iteration,trainer_steps_done"]
    for date, it, elo, games in rows:
        lines.append(
            f"{date},{it},vs_boot512,{elo},{elo - 50.0},{elo + 50.0},0.5,{games},training"
        )
        prog.append(f"{it},1000")
    csv_path.write_text("\n".join(lines) + "\n")
    trial = tmp_path / "run" / "tune" / "train_trial_x"
    trial.mkdir(parents=True)
    (trial / "progress.csv").write_text("\n".join(prog) + "\n")
    return csv_path, trial.parent.parent


def _run_slope(csv_path, run_dir, *extra: str):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "scripts/ratchet_slope.py",
         "--ratchet-csv", str(csv_path), "--run-dir", str(run_dir), *extra],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(ROOT), "PATH": os.environ.get("PATH", ""),
             "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return r.returncode, r.stdout


def test_a_floor_that_silences_the_instrument_is_loud_and_exits_nonzero(tmp_path):
    """"No verdict because everything was filtered" must not look like "no trend".

    Under sustained GPU contention EVERY capped run lands under --min-pairs, so
    this state persists indefinitely while showing only a small `rows usable:`
    number. A strength instrument that has gone quiet must never be mistakable
    for a strength instrument reporting no trend -- so it says MUTED and exits
    with its own status.
    """
    from scripts.ratchet_slope import MUTED_EXIT

    # Four rows, all 20 pairs: enough rows for a verdict, none above the floor.
    csv_path, run_dir = _slope_fixture(tmp_path, [
        ("2026-07-01", 10, 10.0, 40), ("2026-07-02", 20, 0.0, 40),
        ("2026-07-03", 30, -10.0, 40), ("2026-07-04", 40, -20.0, 40),
    ])
    rc, out = _run_slope(csv_path, run_dir)
    # NON-ZERO is the load-bearing property, and it must be asserted as a
    # LITERAL. Comparing only against MUTED_EXIT is circular: setting the
    # constant to 0 would satisfy it while destroying the whole point, and a
    # mutation run proved exactly that.
    assert rc != 0, (
        "a muted run must be distinguishable from a real reading by exit status "
        f"alone; got {rc}:\n{out}"
    )
    assert rc == MUTED_EXIT, f"muted run must exit {MUTED_EXIT}, got {rc}:\n{out}"
    assert MUTED_EXIT != 0, "MUTED_EXIT must stay non-zero"
    assert "INSTRUMENT MUTED" in out
    assert "NOT 'no trend'" in out
    assert "--min-pairs 26 removed 4 row(s)" in out
    # The rows that caused it must be named with their numbers.
    assert "2026-07-01" in out
    assert "20 pairs" in out
    assert "VERDICT:" not in out, "a muted run must not also print a verdict"

    # Disabling the floor un-mutes it -- proving the floor is the cause.
    rc0, out0 = _run_slope(csv_path, run_dir, "--min-pairs", "0")
    assert rc0 == 0, f"with the floor off this must be a normal run, got {rc0}"
    assert "INSTRUMENT MUTED" not in out0
    assert "VERDICT:" in out0


def test_muted_does_not_fire_when_the_data_is_merely_thin(tmp_path):
    """Two good rows and nothing excluded is a QUIET instrument, not a muted one.

    The distinction is whether the excluded rows WOULD have been enough. Firing
    here would make the exit status meaningless -- every early-life series would
    look like a filtering failure.
    """
    csv_path, run_dir = _slope_fixture(tmp_path, [
        ("2026-07-01", 10, 10.0, 400), ("2026-07-02", 20, 0.0, 400),
    ])
    rc, out = _run_slope(csv_path, run_dir)
    assert rc == 0, f"thin-but-unfiltered data must exit 0, got {rc}:\n{out}"
    assert "INSTRUMENT MUTED" not in out
    assert "VERDICT: NONE" in out


def test_muted_does_not_fire_when_enough_rows_survive_the_floor(tmp_path):
    """Exclusions that still leave >= min_rows are a normal, verdict-bearing run."""
    csv_path, run_dir = _slope_fixture(tmp_path, [
        ("2026-07-01", 10, 10.0, 400), ("2026-07-02", 20, 0.0, 400),
        ("2026-07-03", 30, -10.0, 400), ("2026-07-04", 40, -20.0, 400),
        ("2026-07-05", 50, -99.0, 20),   # dropped, but 4 remain
    ])
    rc, out = _run_slope(csv_path, run_dir)
    assert rc == 0, f"expected a normal run, got {rc}:\n{out}"
    assert "INSTRUMENT MUTED" not in out
    assert "VERDICT:" in out
    assert "excluded by --min-pairs" in out, "the drop must still be announced"
