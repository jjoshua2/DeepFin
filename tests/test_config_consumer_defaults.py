"""Configuration defaults and overrides must reach their actual consumers.

Bootstrap and the launcher resolve omitted settings differently from TrialConfig.
The required blend-key checks live in test_value_optimism.py, which exercises the
real resolver against the shipped config and rejects each missing required key.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest
import torch
import yaml

from chess_anti_engine.utils.config_yaml import load_yaml_file

_PRODUCTION_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pbt2_small.yaml"


class _CapturedSettings(Exception):
    """Stop at the consumer boundary before model training or trial launch."""


@pytest.mark.parametrize("source", ["omitted", "explicit", "template"])
def test_bootstrap_passes_resolved_schedule_to_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    from scripts import train_bootstrap

    if source == "template":
        config = load_yaml_file(str(_PRODUCTION_CONFIG))
        # These overrides must remain explicit: deleting them substitutes the
        # bootstrap defaults, independently of the active training schedule.
        expected = {key: config["train"][key] for key in ("lr_T0", "lr_T_mult")}
    elif source == "explicit":
        expected = {"lr_T0": 123, "lr_T_mult": 3}
        config = {"train": expected}
    else:
        config = {}
        expected = {"lr_T0": 2000, "lr_T_mult": 2}
    path = tmp_path / "bootstrap.yaml"
    path.write_text(yaml.safe_dump(config))
    captured: dict[str, object] = {}

    def capture_trainer(_model: object, **kwargs: object) -> NoReturn:
        captured.update(kwargs)
        raise _CapturedSettings

    monkeypatch.setattr(sys, "argv", ["train_bootstrap.py", "--config", str(path)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_bootstrap, "build_model", lambda _config: object())
    monkeypatch.setattr(train_bootstrap, "Trainer", capture_trainer)
    with pytest.raises(_CapturedSettings):
        train_bootstrap.main()
    assert {key: captured[key] for key in expected} == expected


@pytest.mark.parametrize("source", ["omitted", "yaml", "cli", "template"])
def test_launcher_resolves_starting_games_before_trial_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    from chess_anti_engine import run
    from chess_anti_engine.tune.trial_config import TrialConfig

    extra_args: list[str] = []
    if source == "template":
        config = load_yaml_file(str(_PRODUCTION_CONFIG))
        expected = config["selfplay"]["games_per_iter_start"]
    elif source in {"yaml", "cli"}:
        config = {"selfplay": {"games_per_iter": 29, "games_per_iter_start": 7}}
        expected = 7
        if source == "cli":
            extra_args = ["--games-per-iter-start", "11"]
            expected = 11
    else:
        config = {"selfplay": {"games_per_iter": 29}}
        expected = 0
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(config))
    captured: dict[str, Any] = {}
    build_config = run._build_tune_config_dict

    def capture_config(args: argparse.Namespace) -> NoReturn:
        captured.update(build_config(args))
        raise _CapturedSettings

    monkeypatch.setattr(sys, "argv", [
        "run.py", "--config", str(path), "--device", "cpu",
        "--stockfish-path", str(tmp_path / "unused-stockfish"), *extra_args,
    ])
    monkeypatch.setattr(run, "_build_tune_config_dict", capture_config)
    monkeypatch.setattr(run, "pin_nvml_cuda_check", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # main enables TF32 globally; restore these flags after inspecting config.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    with pytest.raises(_CapturedSettings):
        run.main()
    assert captured["games_per_iter_start"] == expected
    assert TrialConfig.from_dict(captured).games_per_iter_start == expected
    if source == "omitted":
        # Direct construction has a different fallback. Omitting the YAML key
        # does not omit it from the launcher's materialized configuration.
        assert TrialConfig.from_dict({"games_per_iter": 29}).games_per_iter_start == 29


def test_absent_blend_updates_preserve_the_trainers_existing_values() -> None:
    from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights

    existing = {
        "sf_search_dampen_sf_low": 0.2,
        "sf_search_dampen_sf_high": 0.3,
        "sf_wdl_temperature": 0.8,
    }
    trainer = SimpleNamespace(**existing)
    _apply_lr_gamma_weights(cast(Any, trainer), {}, rescale_current_lr=True)
    assert vars(trainer) == existing
    # Explicit updates still apply, without defaulting absent siblings.
    _apply_lr_gamma_weights(
        cast(Any, trainer), {"sf_search_dampen_sf_low": 0.4}, rescale_current_lr=True,
    )
    assert vars(trainer) == {**existing, "sf_search_dampen_sf_low": 0.4}
