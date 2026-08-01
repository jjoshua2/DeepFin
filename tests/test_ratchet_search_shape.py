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


def test_a_zero_row_day_is_a_failure_not_a_success() -> None:
    """The script must exit non-zero when it writes no CSV row.

    ``ratchet_loop.sh`` stamps ``data/ratchet/last_run_date`` only on exit 0 and
    documents that a failure "retries on the next poll instead of silently
    skipping the whole day". It could never do that: every early ``return`` in
    ``run_arena`` left the script exiting 0, so 2026-07-30 and 07-31 each burned
    their one attempt and were recorded as successful days.
    """
    text = RATCHET_SH.read_text(encoding="utf-8")
    assert "ROWS_WRITTEN=0" in text
    assert re.search(
        r'if \[ "\$ROWS_WRITTEN" -eq 0 \]; then\n(?:.*\n)*?\s*exit 1\n', text,
    ), "no row written must exit non-zero so the loop retries"




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
        "2026-07-31,478,vs_boot512,-39.8,-91.4,10.1,0.44,114,training\n"
    )
    trial = tmp_path / "run" / "tune" / "train_trial_x"
    trial.mkdir(parents=True)
    (trial / "progress.csv").write_text(
        "training_iteration,trainer_steps_done\n"
        "25,961\n122,5252\n218,8746\n478,23103\n"
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

    assert "rows usable: 3" in default
    assert "rows usable: 4" in disabled
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


def _extract_shell_func(name: str) -> str:
    """Pull one function out of the real ratchet script so a test can RUN it."""
    src = RATCHET_SH.read_text(encoding="utf-8")
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
    """
    import subprocess

    func = _extract_shell_func("pick_prev_snapshot")
    snaps = tmp_path / "snaps"
    snaps.mkdir()

    def call(current: str, it: str) -> tuple[int, str]:
        r = subprocess.run(
            ["bash", "-c", f"{func}\npick_prev_snapshot '{snaps}' '{current}' '{it}'"],
            cwd=str(tmp_path), capture_output=True, text=True, check=False,
        )
        return r.returncode, r.stdout.strip()

    # 1. No earlier snapshot at all.
    (snaps / "ck_2026-08-01_iter478.pt").write_text("x")
    rc, out = call("ck_2026-08-01_iter478.pt", "478")
    assert rc == 1, f"expected 'none available', got rc={rc} {out!r}"
    assert out == "", f"nothing should be printed when there is no reference: {out!r}"

    # 2. THE BUG: the newest earlier snapshot is the same iteration.
    older = snaps / "ck_2026-07-31_iter478.pt"
    older.write_text("x")
    os.utime(older, (2, 2))  # older than today's snapshot, newer than iter409
    rc, out = call("ck_2026-08-01_iter478.pt", "478")
    assert rc == 3, f"same-iteration reference must be refused, got rc={rc} {out!r}"
    assert out == "", "a refused reference must not be handed back as a path"

    # 3. A genuinely earlier iteration is still used.
    real = snaps / "ck_2026-07-30_iter409.pt"
    real.write_text("x")
    os.utime(real, (1, 1))  # oldest, so `ls -t` still ranks iter478 first
    rc, out = call("ck_2026-08-01_iter478.pt", "478")
    assert rc == 3, (
        "the newest earlier snapshot is still iter478; falling back to an older "
        "iteration would silently relabel a 2-day gap as 'vs_prev'"
    )
    older.unlink()
    rc, out = call("ck_2026-08-01_iter478.pt", "478")
    assert rc == 0, f"a different iteration must be usable, got rc={rc} {out!r}"
    assert out.endswith("ck_2026-07-30_iter409.pt"), (
        f"wrong reference returned: {out!r}"
    )


def test_the_self_match_guard_is_wired_into_the_series() -> None:
    """The helper has to be what actually selects the vs_prev reference."""
    src = RATCHET_SH.read_text(encoding="utf-8")
    assert 'prev=$(pick_prev_snapshot "$SNAP_DIR" "$(basename "$snap")" "$iter")' in src
    assert 'grep -v "$(basename "$snap")"' not in src, (
        "the filename-only selector is what produced the self-match row"
    )
    # rc 3 must be handled distinctly, or a self-match degrades to "no snapshot"
    # and the operator never learns why the series stopped.
    assert "3)" in src.split("series 1:")[1].split("series 2:")[0]
