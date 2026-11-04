"""EXP-02: absent comparison populations are not measured zero bias."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import dq_free_dataset_screen as screen
from scripts.tail_censor_screen import Row


def _rows(*, prior: bool = True) -> list[Row]:
    rows = [Row(np.array([1, 2]), {1: 0.0, 2: 0.5}, [1], [2], 0.2,
                game_id=i, ply=3) for i in range(2)]
    if prior:
        for row in rows:
            row.prior = np.array([0.8, 0.2])
    return rows


@pytest.mark.parametrize("kind", ["recovered", "dropped", "empty", "no_prior"])
def test_absent_attrition_population_has_no_comparison(kind: str) -> None:
    rows = [] if kind == "empty" else _rows(prior=kind != "no_prior")
    moves: list[int | None] = [None, None] if kind == "dropped" else [1, 2]
    assert screen.attrition_bias(rows, moves) == []


def test_mixed_attrition_means_are_preserved() -> None:
    rows = _rows()
    rows[1].ply = 7
    rows[1].r_k = 0.4
    bias = {name: (kept, dropped)
            for name, kept, dropped in screen.attrition_bias(rows, [1, None])}
    assert bias["ply"] == (3.0, 7.0)
    assert bias["r_k"] == (0.2, 0.4)
    assert bias["legal_moves"] == (2.0, 2.0)
    assert bias["prior_top1"] == (0.8, 0.8)


@pytest.mark.parametrize(("prior", "moves", "status", "counts"), [
    (True, [1, 2], "no_dropped_rows", (2, 2, 0, 0)),
    (True, [None, None], "no_recovered_rows", (2, 0, 2, 0)),
    (False, [1, None], "no_eligible_rows", (0, 0, 0, 2)),
    (True, [1, None], "comparison_available", (2, 1, 1, 0)),
])
def test_cli_reports_attrition_population_honestly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    prior: bool, moves: list[int | None], status: str, counts: tuple[int, ...],
) -> None:
    rows = _rows(prior=prior)
    # A zero dropped mean has an unavailable ratio, not a printed NaN.
    rows[1].r_k = 0.0
    output = tmp_path / "readout.json"
    monkeypatch.setattr(sys, "argv", ["dq_free_dataset_screen", "--replay-dir",
                        str(tmp_path), "--checkpoint", "fixture", "--json-out",
                        str(output)])
    monkeypatch.setattr(screen, "select_shards", lambda *_: [])
    monkeypatch.setattr(screen, "collect", lambda *_: (rows, []))
    monkeypatch.setattr(screen, "check_invariants", lambda *_: None)
    monkeypatch.setattr(screen, "attach_prior", lambda *_: "cpu")
    monkeypatch.setattr(screen, "recover_search_moves", lambda *_: (
        moves, {"recovered": sum(m is not None for m in moves),
                "ambiguous": 0, "no_child": sum(m is None for m in moves)}))
    assert screen.main() == 0
    saved = json.loads(output.read_text())
    assert saved["attrition_population"] == dict(zip(
        ["eligible", "recovered", "dropped", "prior_absent", "status"],
        [*counts, status], strict=True))
    if status == "comparison_available":
        assert len(saved["attrition_bias"]) == 6
    else:
        assert saved["attrition_bias"] == []
    console = capsys.readouterr().out
    if status != "comparison_available":
        assert f"SELECTION BIAS unavailable: {status}" in console
    else:
        assert "n/a" in console
    assert "nan" not in console.lower()
    # Reject Python's nonstandard NaN/Infinity JSON extension as a control.
    json.dumps(saved, allow_nan=False)
