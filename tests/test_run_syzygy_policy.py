"""CLI and YAML must reach the canonical tablebase policy setting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.tune.trial_config import TrialConfig
from tests.test_run_relation_preflight import _startup


@pytest.mark.parametrize(
    ("config", "cli", "expected"),
    [
        ({}, [], False),
        ({"syzygy_rescore_policy": True}, [], True),
        ({"syzygy_rescore_policy": False}, [], False),
        ({}, ["--syzygy-policy"], True),
        ({}, ["--syzygy-rescore-policy"], True),
        ({"syzygy_rescore_policy": False}, ["--syzygy-policy"], True),
        ({"syzygy_rescore_policy": False}, ["--syzygy-rescore-policy"], True),
    ],
    ids=["default", "yaml-true", "yaml-false", "legacy", "canonical",
         "legacy-overrides-yaml", "canonical-overrides-yaml"],
)
def test_syzygy_policy_reaches_trial_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any], cli: list[str], expected: bool,
) -> None:
    # The shared startup fixture captures dispatch before any trial/engine starts
    # and rejects GPU availability probes while exercising the real main parser.
    calls = _startup(tmp_path, monkeypatch, config, cli)
    assert len(calls) == 1
    effective = calls[0]
    assert "syzygy_policy" not in effective
    assert effective["syzygy_rescore_policy"] is expected
    assert TrialConfig.from_dict(effective).syzygy_rescore_policy is expected
