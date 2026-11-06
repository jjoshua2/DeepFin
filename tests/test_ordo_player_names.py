"""Preserve Ordo player identity through parsing and the reporting caller."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from scripts import ordo_pooled_fit as fit


@pytest.mark.parametrize("name", ["ArmA", "Model A", "Stockfish 16.1", "Model  A", "Model : v2", "123 Engine"])
@pytest.mark.parametrize("with_error", [False, True])
def test_parse_preserves_complete_name_and_numeric_columns(name: str, with_error: bool) -> None:
    columns = "RATING ERROR POINTS PLAYED (%)" if with_error else "RATING POINTS PLAYED (%)"
    values = "12.5 3.25 60.0 100 60" if with_error else "12.5 60.0 100 60"
    result = fit.parse_ordo(f"   # PLAYER : {columns}\n   1. {name:<24} : {values}\n")
    assert set(result) == {name}
    rating, error = result[name]
    assert rating == 12.5
    if with_error:
        assert error == 3.25
    else:
        assert math.isnan(error)


def test_similarly_named_players_do_not_collapse() -> None:
    result = fit.parse_ordo(
        "# PLAYER : RATING ERROR POINTS PLAYED (%)\n"
        "1 Model A : 15.0 ---- 60.0 100 60\n"
        "2 Model B : -5.0 2.5 40.0 100 40\n"
        "3 Model : -10.0 3.0 30.0 100 30\n"
    )
    assert set(result) == {"Model A", "Model B", "Model"}
    assert result["Model A"][0] == 15.0
    assert math.isnan(result["Model A"][1])
    assert result["Model B"] == (-5.0, 2.5)
    assert result["Model"] == (-10.0, 3.0)


def test_header_change_resets_error_column() -> None:
    result = fit.parse_ordo(
        "# PLAYER : RATING ERROR POINTS\n"
        "1 Model A : 15.0 2.5 60.0\n"
        "# PLAYER : RATING POINTS\n"
        "1 Model B : -5.0 40.0\n"
    )
    assert result["Model A"] == (15.0, 2.5)
    assert result["Model B"][0] == -5.0
    assert math.isnan(result["Model B"][1])


@pytest.mark.parametrize("line", ["", "1 Model A 15.0", "1 : 15.0", "note Model A : 15.0", "1 Model A : unread"])
def test_non_result_lines_still_do_not_create_players(line: str) -> None:
    assert fit.parse_ordo(line) == {}


def test_multiword_names_reach_ratings_bootstrap_and_contrast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    pgn = tmp_path / "paired.pgn"
    games = [
        fit.Game("Model A", "Model B", "1-0", "0", None, None, 0),
        fit.Game("Model B", "Model A", "0-1", "0", None, None, 1),
    ]
    fit.write_pgn(pgn, games)
    binary = tmp_path / "not-executed-ordo"
    binary.write_text("")
    calls = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        # Exercise the real PGN writer, reader, subprocess adapter, parser,
        # bootstrap and report. Only the external Ordo fit is a text fixture.
        rows = fit.read_games([Path(cmd[cmd.index("-p") + 1])])
        assert {g.white for g in rows} | {g.black for g in rows} == {"Model A", "Model B"}
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text(
            "# PLAYER : RATING ERROR POINTS PLAYED (%)\n"
            "1 Model A : 10.0 2.0 60.0 100 60\n"
            "2 Model B : 0.0 ---- 40.0 100 40\n"
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(fit.tempfile, "mkdtemp", lambda **_kwargs: str(tmp_path))
    monkeypatch.setattr("sys.argv", [
        "ordo_pooled_fit.py", str(pgn), "--ordo", str(binary),
        "--bootstrap", "2", "--ordo-sims", "0",
    ])
    assert fit.main() == 0
    assert len(calls) == 3
    out = capsys.readouterr().out
    ratings = out.split("=== ratings", 1)[1].split("=== pair-level", 1)[0]
    bootstrap = out.split("=== pair-level", 1)[1].split("=== pairwise", 1)[0]
    contrasts = out.split("=== pairwise", 1)[1]
    assert "Model A" in ratings
    assert "Model B" in ratings
    assert "Model A" in bootstrap
    assert "Model B" in bootstrap
    assert "Model A - Model B" in contrasts
    assert "+10.0" in contrasts
