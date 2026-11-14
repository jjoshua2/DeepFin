"""Search-type (`mcts`) values at the boundaries the entry-point tests do not reach.

* The ``--mcts`` CLI options of ``run.py`` and the worker accept any case and
  hand the consumer the normalised value.
* The relations-transport guard treats ``"Gumbel"`` as Gumbel (it must NOT
  refuse it).
* A live yaml reload with an invalid ``mcts`` keeps the previous value and the
  trial's next ``TrialConfig.from_dict`` still succeeds; at launch the same
  value is fatal. (``mcts: null`` is dropped by the flattener: it means unset.)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests.test_run_relation_preflight import _startup


def test_run_cli_mcts_is_case_insensitive_and_published_normalised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _startup(tmp_path, monkeypatch, {}, ["--mcts", "PUCT"])
    assert len(calls) == 1
    assert calls[0]["mcts"] == "puct"


def test_worker_cli_mcts_is_case_insensitive_and_normalised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parsed ``args.mcts`` is what ``_resolve_reco`` reads first."""
    from chess_anti_engine import worker as worker_mod

    class _Parsed(Exception):
        def __init__(self, args: Any) -> None:
            super().__init__("parsed")
            self.args_ns = args

    def _stop(args: Any) -> None:
        raise _Parsed(args)

    monkeypatch.setattr(worker_mod, "_setup_worker_logging", _stop)
    monkeypatch.setattr(sys, "argv", ["worker", "--mcts", "Gumbel"])
    with pytest.raises(_Parsed) as excinfo:
        worker_mod.main()
    assert excinfo.value.args_ns.mcts == "gumbel"


def test_relations_transport_guard_accepts_mixed_case_gumbel() -> None:
    from chess_anti_engine.tune.distributed_runtime import check_dynamic_relations_transport

    check_dynamic_relations_transport({"use_dynamic_relations": True, "mcts": "Gumbel"})
    with pytest.raises(ValueError, match="mcts != gumbel"):
        check_dynamic_relations_transport({"use_dynamic_relations": True, "mcts": "PUCT"})


# --------------------------------------------------------------------------
# Live reload vs launch
# --------------------------------------------------------------------------


def _write_yaml(path: Path, mcts_line: str) -> str:
    path.write_text(f"selfplay:\n  {mcts_line}\n  mcts_simulations: 77\n", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("bad", ["mcts: gumbell", "mcts: ''", "mcts: off"])
def test_live_reload_with_an_invalid_mcts_keeps_the_old_value_and_the_trial_continues(
    tmp_path: Path, bad: str, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
) -> None:
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
    from chess_anti_engine.tune.trial_config import TrialConfig

    config: dict[str, Any] = {"mcts": "gumbel", "mcts_simulations": 50}
    path = _write_yaml(tmp_path / "live.yaml", bad)

    with caplog.at_level("ERROR"):
        _reload_yaml_into_config(config, path, live_reload=True)

    assert config["mcts"] == "gumbel"
    # The unrelated edit in the same save still applied.
    assert config["mcts_simulations"] == 77
    # The iteration loop's next step succeeds on the kept value.
    tc = TrialConfig.from_dict(config)
    assert tc.mcts == "gumbel"
    out = capsys.readouterr().out
    assert "[trial] ERROR YAML reload: mcts=" in out
    assert "NOT applied" in out
    assert any("mcts=" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records)


def test_a_null_mcts_is_dropped_by_the_flattener_so_reload_keeps_the_old_value(tmp_path: Path) -> None:
    """``mcts: null`` never reaches ``config`` (the flattener drops nulls), so
    it is "unset", not invalid: reload keeps the old value, launch uses the default."""
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
    from chess_anti_engine.tune.trial_config import TrialConfig

    path = _write_yaml(tmp_path / "live.yaml", "mcts: null")
    config: dict[str, Any] = {"mcts": "puct"}
    _reload_yaml_into_config(config, path, live_reload=True)
    assert TrialConfig.from_dict(config).mcts == "puct"
    launch: dict[str, Any] = {}
    _reload_yaml_into_config(launch, path)
    assert TrialConfig.from_dict(launch).mcts == "gumbel"


def test_live_reload_with_a_mixed_case_mcts_applies_it(tmp_path: Path) -> None:
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
    from chess_anti_engine.tune.trial_config import TrialConfig

    config: dict[str, Any] = {"mcts": "gumbel"}
    _reload_yaml_into_config(config, _write_yaml(tmp_path / "live.yaml", "mcts: PUCT"), live_reload=True)
    assert TrialConfig.from_dict(config).mcts == "puct"


def test_launch_with_an_invalid_mcts_is_fatal(tmp_path: Path) -> None:
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
    from chess_anti_engine.tune.trial_config import TrialConfig

    config: dict[str, Any] = {}
    _reload_yaml_into_config(config, _write_yaml(tmp_path / "launch.yaml", "mcts: gumbell"))
    with pytest.raises(ValueError, match="mcts"):
        TrialConfig.from_dict(config)
