from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import monitor_sf_refute_outcomes as monitor


def _result(trial: Path, stats: dict[str, int], iteration: int = 1) -> Path:
    trial.mkdir(parents=True, exist_ok=True)
    path = trial / "result.json"
    path.write_text(json.dumps({"training_iteration": iteration, "outcome_stats": stats}) + "\n")
    return path


@pytest.mark.parametrize(("stats", "observed"), [
    ({"selfplay_fenlist_games": 4}, 0),
    ({"selfplay_fenlist_games": 4, "selfplay_fenlist_stm_l": 2}, 2),
])
def test_missing_or_incomplete_outcomes_have_unknown_rates(stats: dict[str, int], observed: int) -> None:
    rates = monitor._stm_rates(stats, "fenlist")
    assert rates["n"] == observed
    assert rates["games"] == 4
    assert rates["lr"] is rates["wr"] is rates["dr"] is None
    if not observed:
        assert rates["w"] is rates["d"] is rates["l"] is None


def test_sparse_complete_outcomes_treat_absent_categories_as_zero() -> None:
    rates = monitor._stm_rates({"selfplay_fenlist_games": 4, "selfplay_fenlist_stm_l": 4}, "fenlist")
    assert rates["n"] == 4
    assert rates["lr"] == 1.0
    assert rates["wr"] == rates["dr"] == 0.0


def test_jsonl_ignores_partial_and_nonobject_rows_without_pooling_backed_seeds(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    path = _result(tmp_path / "trial", {
        "selfplay_fenlist_games": 2, "selfplay_fenlist_stm_l": 2,
        "selfplay_fenlist_backed_stm_w": 90,
        "selfplay_fenlist_sf_refute_games": 1, "selfplay_fenlist_sf_refute_stm_w": 1,
    })
    with path.open("a") as fh:
        fh.write('null\n{"training_iteration":')
    rows = monitor._iter_rows(path, last=10)
    assert len(rows) == 1
    assert rows[0]["trial"] == str(path.parent.resolve())
    assert rows[0]["pure_n"] == 2
    assert rows[0]["refute_n"] == 1
    assert rows[0]["gap_lr"] == -1.0
    assert "ignored 2 malformed/partial" in capsys.readouterr().err


def test_csv_identity_includes_trial_and_remains_idempotent(tmp_path: Path) -> None:
    rows = [monitor._iter_rows(_result(tmp_path / name / "trial", {
        "selfplay_fenlist_stm_l": 1,
    }), last=1)[0] for name in ("first", "second")]
    path = tmp_path / "out.csv"
    monitor._append_csv(path, rows + rows)
    monitor._append_csv(path, rows)
    with path.open() as fh:
        stored = list(csv.DictReader(fh))
    assert len(stored) == 2
    assert {row["trial"] for row in stored} == {row["trial"] for row in rows}
    assert {row["iter"] for row in stored} == {"1"}


@pytest.mark.parametrize("damage", ["legacy", "truncated", "short_row", "extra_field"])
def test_csv_refuses_unattributable_or_incomplete_records_without_writing(
    tmp_path: Path, damage: str,
) -> None:
    rows = monitor._iter_rows(_result(tmp_path / "trial", {"selfplay_fenlist_stm_l": 1}), last=1)
    path = tmp_path / "out.csv"
    monitor._append_csv(path, rows)
    original = path.read_text()
    if damage == "legacy":
        broken = "iter,pure_n\n1,1\n"
    elif damage == "truncated":
        broken = original.rstrip("\n")
    elif damage == "short_row":
        broken = original + "trial,2\n"
    else:
        broken = original.rstrip("\n") + ",unexpected\n"
    path.write_text(broken)
    with pytest.raises(ValueError, match="CSV"):
        monitor._append_csv(path, rows)
    assert path.read_text() == broken


def test_partial_outcomes_are_not_used_in_pooled_gap(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    path = _result(tmp_path / "trial", {
        "selfplay_fenlist_games": 10, "selfplay_fenlist_stm_l": 2,
        "selfplay_fenlist_sf_refute_games": 1, "selfplay_fenlist_sf_refute_stm_l": 1,
    })
    rows = monitor._iter_rows(path, last=1)
    assert rows[0]["gap_lr"] is None
    monitor._print_table(rows, trial=path.parent)
    output = capsys.readouterr().out
    assert "n/a" in output
    assert "gap_L (refute" not in output


def test_watch_pins_automatically_selected_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _result(tmp_path / "first", {"selfplay_fenlist_stm_l": 1}).parent
    second = _result(tmp_path / "second", {"selfplay_fenlist_stm_w": 1}).parent
    selections: list[Path] = []
    printed: list[Path] = []

    def select(_work_dir: Path) -> Path:
        selections.append(first if not selections else second)
        return selections[-1]

    def record(_rows: list[dict], *, trial: Path) -> None:
        printed.append(trial)

    class StopWatch(Exception):
        pass

    def sleep(_seconds: int) -> None:
        if len(printed) == 2:
            raise StopWatch

    monkeypatch.setattr(monitor, "_newest_trial", select)
    monkeypatch.setattr(monitor, "_print_table", record)
    monkeypatch.setattr(monitor.time, "sleep", sleep)
    monkeypatch.setattr("sys.argv", ["monitor", "--watch", "--work-dir", str(tmp_path)])
    with pytest.raises(StopWatch):
        monitor.main()
    assert selections == [first]
    assert printed == [first, first]
