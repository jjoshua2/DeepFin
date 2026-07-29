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
