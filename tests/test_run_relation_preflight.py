from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from chess_anti_engine import run
from chess_anti_engine.model import ModelConfig
from chess_anti_engine.tune.distributed_runtime import build_recommended_worker


def _startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any],
             cli: list[str]) -> list[dict[str, Any]]:
    path = tmp_path / "startup.yaml"
    path.write_text(json.dumps(config))
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> None:
        calls.append(kwargs["base_config"])

    def forbid_device_probe() -> bool:
        raise AssertionError("CPU startup must not probe GPU availability")

    harness = ModuleType("chess_anti_engine.tune.harness")
    monkeypatch.setattr(harness, "run_tune", capture, raising=False)
    monkeypatch.setitem(sys.modules, "chess_anti_engine.tune.harness", harness)
    monkeypatch.setattr(sys, "argv", [
        "chess_anti_engine.run", "--config", str(path), "--device", "cpu",
        "--stockfish-path", "/inert/never-start-stockfish",
        "--work-dir", str(tmp_path / "unused"), *cli,
    ])
    monkeypatch.setattr(run.torch.cuda, "is_available", forbid_device_probe)
    # Restore the process-global backend flags changed by the real entry point.
    monkeypatch.setattr(run.torch.backends.cuda.matmul, "allow_tf32", run.torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(run.torch.backends.cudnn, "allow_tf32", run.torch.backends.cudnn.allow_tf32)
    run.main()
    return calls


@pytest.mark.parametrize("flag", ["use_dynamic_relations", "policy_dynamic_relations", "record_relations"])
@pytest.mark.parametrize("yaml_mcts", [None, "puct", "gumbel"])
def test_cli_gumbel_override_is_validated_and_published(tmp_path, monkeypatch, flag, yaml_mcts):
    config: dict[str, Any] = {flag: True}
    if yaml_mcts is not None:
        config["mcts"] = yaml_mcts
    calls = _startup(tmp_path, monkeypatch, config, ["--mcts", "gumbel"])
    assert len(calls) == 1
    effective = calls[0]
    assert effective["mcts"] == "gumbel"
    assert effective[flag] is True
    worker = build_recommended_worker(
        config=effective,
        model_cfg=ModelConfig(
            use_dynamic_relations=flag == "use_dynamic_relations",
            policy_dynamic_relations=flag == "policy_dynamic_relations",
        ),
        sf_nodes=1, mcts_simulations=1,
    )
    assert worker["mcts"] == "gumbel"
    assert worker["record_relations"] is True


@pytest.mark.parametrize("flag", ["use_dynamic_relations", "policy_dynamic_relations", "record_relations"])
def test_cli_puct_override_is_rejected_before_dispatch(tmp_path, monkeypatch, flag):
    with pytest.raises(ValueError, match="mcts != gumbel"):
        _startup(tmp_path, monkeypatch, {flag: True, "mcts": "gumbel"}, ["--mcts", "puct"])


def test_plain_puct_startup_is_preserved(tmp_path, monkeypatch):
    calls = _startup(tmp_path, monkeypatch, {}, ["--mcts", "puct"])
    assert len(calls) == 1
    assert calls[0]["mcts"] == "puct"


@pytest.mark.parametrize("unsupported", [
    {"distributed_inference_broker_enabled": True},
    {"distributed_worker_threaded": True},
    {"distributed_worker_aot_dir": "/inert/no-aot-package"},
])
def test_cli_gumbel_override_still_rejects_unsupported_transport(tmp_path, monkeypatch, unsupported):
    with pytest.raises(ValueError, match="dynamic relations"):
        _startup(tmp_path, monkeypatch, {"use_dynamic_relations": True, **unsupported}, ["--mcts", "gumbel"])


def test_help_precedes_transport_validation(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _startup(tmp_path, monkeypatch, {"use_dynamic_relations": True}, ["--help"])
    assert exc.value.code == 0
