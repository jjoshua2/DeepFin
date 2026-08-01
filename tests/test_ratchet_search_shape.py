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
# The slope must not inverse-variance-weight a CI the audit measured as
# anti-conservative.
# ---------------------------------------------------------------------------

def test_tiny_rows_are_excluded_from_the_slope_not_silently_weighted() -> None:
    """`w = 1/se^2` gives the least trustworthy row the MOST weight.

    docs/rl_loop_audit.md L9 measured the 25-pair cell at 92.3-93.8% coverage
    (71.2% in a degenerate draw-draw regime) — there the se is too small, so
    the fit would lean hardest on exactly the rows it should distrust. Before
    `--max-seconds` a contended day usually recorded nothing; now it always
    records, so this regime is the common case rather than the exception.

    The row still belongs in the CSV — it is a true observation — so the floor
    lives in the fit, not in the writer.
    """
    from scripts.ratchet_slope import MIN_TRUSTED_PAIRS

    # 13-pair and 23-pair rows are real: data/ratchet/ratchet.csv 2026-07-29
    # (26 games) and 2026-07-28 (46 games).
    assert MIN_TRUSTED_PAIRS > 25, (
        "the floor must exclude the 25-pair cell L9 measured as undercovering"
    )
    src = (ROOT / "scripts" / "ratchet_slope.py").read_text(encoding="utf-8")
    assert "--min-pairs" in src
    assert "args.min_pairs" in src, "the flag must be read, not just declared"
    # Excluded rows have to be announced; a silently shorter fit is how a
    # verdict changes without anyone noticing.
    assert "EXCLUDED below --min-pairs" in src
