"""Actual analyzer bindings for the Tactical300 provenance corrections."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import sf_policy_rewrite
from scripts import tactical300_calibration as calibration
from scripts import tactical300_registration as registration
from scripts.bt4_policy_dump import file_sha256
from tests.test_adaptive_sf_value import g10_row
from tests.test_tactical300_calibration import _moves, _policy_for


def test_analyzer_rejects_partly_mating_tied_top_set(monkeypatch: pytest.MonkeyPatch) -> None:
    row = g10_row(extended=True)
    mate, other = _moves(row)[:2]
    row["phases"][0]["per_depth"][0]["lines"][0][2] = 99900.0
    monkeypatch.setattr(
        calibration.adaptive, "select",
        lambda *_args: ({mate: 99900.0, other: 0.0}, 12, "synthetic_roster"),
    )
    result = calibration.analyze_row(row, _policy_for(row, {mate: 0.5, other: 0.5}))
    assert result["bt4_top"] == sorted([mate, other])
    assert result["final_has_winning_mate"] is True
    assert result["bt4_top_is_final_winning_mate"] is False


def test_producer_receipt_pins_mate_domain_and_registration() -> None:
    pins = calibration._producer_sha256()
    for module in (sf_policy_rewrite, registration):
        path = Path(module.__file__).resolve()
        assert pins[str(path)] == file_sha256(path)
