"""simple_rl optimizer setup: reproduce the SL base's Aurora split.

The SL base's saved optimizer is an Aurora split whose polar knobs differ from
the Trainer's defaults, and its run used zclip. ``--optimizer-hparams-source
init-checkpoint`` fills every unset knob from the init checkpoint; the train
manifest records the REALISED param-group table (count, class, lr, betas,
eps, weight decay, Aurora knobs) read off the live optimizer, and these tests
assert it equals the checkpoint's own table. A stand-in checkpoint written by
the Trainer with the SL base's recipe plays the SL base (the real one is not in
the repo); the knob values are the ones read off the real SL base.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl.learn import (
    TrainSettings,
    build_trainer,
    param_group_table,
    param_group_table_of,
    resolve_optimizer_hparams,
    tables_match,
)
from tests.simple_rl_helpers import tiny_model_config

# Read off the SL base checkpoint's opt.param_groups (all four groups carry them).
SL_KNOBS: dict[str, Any] = {"aurora_pp_iterations": 3, "aurora_pp_beta": 0.25, "aurora_polar_steps": 8,
            "aurora_polar_method": "polar_express", "aurora_polar_dtype": "fp16"}
TRAINER_DEFAULTS: dict[str, Any] = {"aurora_pp_iterations": 2, "aurora_pp_beta": 0.5, "aurora_polar_steps": 12,
                    "aurora_polar_method": "simple", "aurora_polar_dtype": "auto"}
AURORA: dict[str, Any] = {"optimizer": "aurora", "matrix_optimizer_scope": "mlp_out", "lr": 3e-6, "device": "cpu"}


def _write_sl_like(path: Path) -> Path:
    """A Trainer checkpoint with the SL base's optimizer recipe, past zclip's 25-step
    warmup (so it carries optimizer moments and an initialised zclip)."""
    from chess_anti_engine.model import build_model
    from chess_anti_engine.simple_rl.learn import trainer_kwargs
    from chess_anti_engine.train import Trainer
    from tests.test_simple_rl_learn import _scheduled

    torch.manual_seed(3)
    kw = trainer_kwargs(TrainSettings(**AURORA, mirror_prob=0.0, **SL_KNOBS, grad_clip="zclip"))
    trainer = Trainer(build_model(tiny_model_config()), model_config=tiny_model_config(), log_dir=None, **kw)
    trainer.train_steps(_scheduled(52, [np.arange(2 * i, 2 * i + 2) for i in range(26)]), batch_size=2, steps=26)
    trainer.save(path)
    return path


@pytest.fixture(scope="module")
def sl_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_sl_like(tmp_path_factory.mktemp("sl") / "sl_like.pt")


def _ckpt_table(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return param_group_table_of(payload["opt"]["param_groups"])


def test_the_stand_in_checkpoint_carries_the_sl_knobs_not_the_trainer_defaults(sl_ckpt: Path) -> None:
    table = _ckpt_table(sl_ckpt)
    assert [r["class"] for r in table[:4]] == ["aurora", "adamw", "adamw", "adamw"]
    for row in table:
        assert {k: row[k] for k in SL_KNOBS} == SL_KNOBS
    assert table[0]["lr"] == pytest.approx(6e-5)
    assert table[2]["lr"] == pytest.approx(3e-6)
    assert torch.load(str(sl_ckpt), map_location="cpu", weights_only=False)["zclip"]["initialized"]


def test_resolving_from_the_checkpoint_realises_its_param_group_table(sl_ckpt: Path) -> None:
    settings = resolve_optimizer_hparams(TrainSettings(**AURORA, optimizer_hparams_source="init-checkpoint"),
                                         sl_ckpt)
    assert settings.optimizer_hparams_resolved is not None
    assert settings.optimizer_hparams_resolved["filled"] == sorted([*SL_KNOBS, "matrix_lr_multiplier",
                                                                     "matrix_weight_decay", "aux_weight_decay",
                                                                     "grad_clip"])
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    # Logged from the CONSUMER: the live optimizer's groups and the Trainer's clip mode.
    assert tables_match(param_group_table(trainer), _ckpt_table(sl_ckpt)) == []
    assert trainer.grad_clip_mode == "zclip"
    assert type(trainer.opt).__name__ == "AuroraWithAuxAdam"


def test_without_resolution_the_tables_differ() -> None:
    """Non-vacuous: the Trainer's defaults are a different setup."""
    trainer = build_trainer(tiny_model_config(), TrainSettings(**AURORA), log_dir=None)
    table = param_group_table(trainer)
    assert {k: table[0][k] for k in TRAINER_DEFAULTS} == TRAINER_DEFAULTS
    assert trainer.grad_clip_mode == "none"


def test_an_explicit_knob_wins_over_the_checkpoint_and_is_recorded(sl_ckpt: Path) -> None:
    settings = resolve_optimizer_hparams(
        TrainSettings(**AURORA, optimizer_hparams_source="init-checkpoint", aurora_polar_steps=5), sl_ckpt)
    assert settings.optimizer_hparams_resolved is not None
    assert settings.optimizer_hparams_resolved["explicit"] == ["aurora_polar_steps"]
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    diff = tables_match(param_group_table(trainer), _ckpt_table(sl_ckpt))
    assert diff
    assert all("aurora_polar_steps" in d for d in diff)
    assert {g["aurora_polar_steps"] for g in trainer.opt.param_groups} == {5}


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"aurora_pp_beta": 0.25}, "would be ignored"),
    ({"matrix_lr_multiplier": 20.0}, "would be ignored"),
    ({"optimizer_hparams_source": "init-checkpoint"}, "would be ignored"),
    ({"zclip_max_norm": 1.0}, "read only with grad_clip zclip"),
    ({"grad_clip": "clip"}, "grad_clip must be one of"),
])
def test_a_knob_that_would_be_ignored_is_refused(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TrainSettings(device="cpu", **kwargs).validate()  # optimizer nadamw (the default)


def test_a_checkpoint_without_the_aurora_layout_is_refused(tmp_path: Path) -> None:
    from tests.simple_rl_helpers import save_tiny_checkpoint

    ckpt = save_tiny_checkpoint(tmp_path / "nadamw.pt")  # NAdamW groups
    with pytest.raises(ValueError, match="not the Trainer's Aurora layout"):
        resolve_optimizer_hparams(TrainSettings(**AURORA, optimizer_hparams_source="init-checkpoint"), ckpt)


def test_continuing_the_checkpoint_state_needs_no_adopt_flag_and_restores_zclip(sl_ckpt: Path) -> None:
    from chess_anti_engine.simple_rl.learn import init_optimizer_from_checkpoint

    settings = resolve_optimizer_hparams(TrainSettings(**AURORA, optimizer_hparams_source="init-checkpoint",
                                                       init_optimizer_state="checkpoint"), sl_ckpt)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    record = init_optimizer_from_checkpoint(trainer, sl_ckpt, settings)
    assert record["group_resolution"] == "identical" or not record.get("group_diff")
    assert record["zclip_state_restored"] is True
    assert trainer.zclip.initialized


def test_driver_records_the_realised_table_equal_to_the_init_checkpoint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sl_ckpt: Path) -> None:
    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _setup
    from tests.test_simple_rl_snapshots_arms import _argv

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "sl"
    argv = _argv(run, openings, model_config, "--optimizer", "aurora", "--matrix-optimizer-scope", "mlp_out",
                 "--optimizer-hparams-source", "init-checkpoint", init=sl_ckpt)
    argv[argv.index("--lr") + 1] = "3e-6"
    assert simple_rl_loop.main(argv) == 0
    manifest = json.loads((run / "gen_0001" / "train" / "manifest.json").read_text())
    realised = manifest["config"]["realised"]["optimizer"]
    assert tables_match(realised["param_group_table"], _ckpt_table(sl_ckpt)) == []
    assert realised["grad_clip"]["mode"] == "zclip"
    resolved = manifest["config"]["settings"]["optimizer_hparams_resolved"]
    assert resolved["source"].endswith("gen_0000/model.pt")
    assert "grad_clip" in resolved["filled"]
