"""Inert numeric/result-file fixtures for the local PBT poller."""
import json
from pathlib import Path

import pytest

from scripts import pbt_30m_poll as poll


@pytest.mark.parametrize(("value", "expected"), [
    (float("inf"), 9), (float("-inf"), 9), (float("nan"), 9),
    ("42", 42), (42, 42), (42.9, 42), (True, 1), (None, 9),
], ids=["infinity", "negative-infinity", "nan", "string", "integer", "finite-float", "boolean", "none"])
def test_safe_int_preserves_conversion_defaults(value: object, expected: int) -> None:
    assert poll._safe_int(value, 9) == expected


@pytest.mark.parametrize("field", ["iter", "training_iteration", "shared_trials_selected",
                                   "shared_trials_ingested", "shared_samples_ingested"])
@pytest.mark.parametrize("literal", ["1e309", "-1e309"], ids=["positive", "negative"])
def test_overflowing_json_counter_does_not_stop_poll(tmp_path: Path, capsys, field: str, literal: str) -> None:
    trial = tmp_path / "runs/pbt2_small/tune/train_trial_fixture_00001"
    trial.mkdir(parents=True)
    row = {"training_iteration": 2, "win": 1, "draw": 0, "loss": 0,
           "shared_trials_selected": 1, "shared_trials_ingested": 1,
           "shared_samples_ingested": 3, "config": {}}
    row.pop(field, None)
    encoded = json.dumps(row)[:-1] + ", " + json.dumps(field) + ": " + literal + "}\n"
    (trial / "result.json").write_text(encoded)
    assert poll.run_once(root=tmp_path, prefix="fixture") == 0
    output = capsys.readouterr().out
    label = {"iter": "iter", "training_iteration": "iter", "shared_trials_selected": "share_sel",
             "shared_trials_ingested": "share_ing", "shared_samples_ingested": "share_samp"}[field]
    fallback = -1 if label == "iter" else 0
    assert f"{label}={fallback}" in output
    assert "rows=1/1" in output
