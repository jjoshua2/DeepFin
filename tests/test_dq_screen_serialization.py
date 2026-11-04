"""Saved proxy-screen readouts retain the diagnostics printed by the CLI."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import dq_free_dataset_screen as screen
from scripts.tail_censor_screen import Row


@pytest.mark.parametrize("search_move", [2, 1, None])
def test_cli_json_retains_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, search_move: int | None,
) -> None:
    rows = [Row(np.array([1, 2]), {1: 0.0, 2: 0.5}, [1], [2], 0.2,
                game_id=i, ply=3) for i in range(2)]
    for row in rows:
        row.prior = np.array([0.8, 0.2])
    moves = [search_move, None]
    recovery = {"recovered": int(search_move is not None), "ambiguous": 0,
                "no_child": 1 + int(search_move is None)}
    output = tmp_path / "readout.json"
    monkeypatch.setattr(sys, "argv", ["dq_free_dataset_screen", "--replay-dir",
                        str(tmp_path), "--checkpoint", "fixture", "--json-out",
                        str(output)])
    monkeypatch.setattr(screen, "select_shards", lambda *_: [])
    monkeypatch.setattr(screen, "collect", lambda *_: (rows, []))
    monkeypatch.setattr(screen, "check_invariants", lambda *_: None)
    monkeypatch.setattr(screen, "attach_prior", lambda *_: "cpu")
    monkeypatch.setattr(screen, "recover_search_moves", lambda *_: (moves, recovery))

    assert screen.main() == 0
    saved = json.loads(output.read_text())
    observations = screen.build_pair_observations(rows, moves)
    assert saved["confidence_curves"] == json.loads(
        json.dumps(screen.confidence_curves(observations)))
    assert saved["attrition_bias"] == [
        {"observable": name, "recovered": kept, "dropped": dropped}
        for name, kept, dropped in screen.attrition_bias(rows, moves)
    ]
    assert saved["pairs"] == screen.headline_split(observations)
    assert saved["recovery"] == recovery
    assert "free_regret" not in saved
    assert "buyable_regret" not in saved
    assert saved["free_regret_summary"] == {"n": 0}
    assert saved["buyable_regret_summary"] == {"n": 0}
    assert saved["checkpoint"] == "fixture"
    assert saved["k"] == 6
    assert saved["shards"] == []
