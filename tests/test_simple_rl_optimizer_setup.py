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
    assert settings.optimizer_hparams_resolved["explicit"] == ["aurora_polar_steps", "matrix_optimizer_scope"]
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
    # No --matrix-optimizer-scope: the init-checkpoint source derives the checkpoint's split (review of #1445).
    argv = _argv(run, openings, model_config, "--optimizer", "aurora",
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
    assert "matrix_optimizer_scope" in resolved["filled"]
    assert realised["built_split_matches_init_checkpoint"]["names_equal"] is True
    assert {"zclip_initialized", "zclip_buffer_len"} <= set(realised["grad_clip"])


def _lrs(settings: TrainSettings, phase_updates: int | None, steps: int) -> list[float]:
    """The matrix-free AdamW group's lr the optimizer applies at each step (pre-hook)."""
    from tests.test_simple_rl_learn import _scheduled

    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None, phase_updates=phase_updates)
    seen: list[float] = []
    trainer.opt.register_step_pre_hook(lambda o, _a, _k: seen.append(float(o.param_groups[-1]["lr"])))
    trainer.train_steps(_scheduled(4 * steps, [np.arange(4 * i, 4 * i + 4) for i in range(steps)]),
                        batch_size=4, steps=steps)
    return seen


def test_cosine_phase_decays_from_lr_to_the_floor_over_the_phase_and_flat_stays_flat() -> None:
    import math

    n, peak, floor = 10, 1e-3, 1e-4
    cos = _lrs(TrainSettings(device="cpu", mirror_prob=0.0, lr=peak, lr_floor=floor, lr_schedule="cosine-phase"),
               n, n)
    # The Trainer steps its scheduler after each update, from the update count, so
    # updates 0 and 1 both run at the peak and update k >= 1 at position k - 1.
    want = [floor + (peak - floor) * (1 + math.cos(math.pi * max(0, k - 1) / n)) / 2 for k in range(n)]
    np.testing.assert_allclose(cos, want, rtol=1e-6)
    flat = _lrs(TrainSettings(device="cpu", mirror_prob=0.0, lr=peak, lr_floor=floor), None, n)
    np.testing.assert_allclose(flat, [peak] * n, rtol=1e-6)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"lr_schedule": "cosine-phase"}, "give an lr_floor below lr"),
    ({"lr_schedule": "cosine-phase", "lr_floor": 1e-6, "optimizer_state": "continue"}, "use fresh state"),
    ({"lr_schedule": "cosine-phase", "lr_floor": 1e-6, "sample_with_replacement": True}, "exact-once"),
    ({"lr_schedule": "linear"}, "lr_schedule must be one of"),
])
def test_a_schedule_that_could_not_take_effect_is_refused(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TrainSettings(device="cpu", **kwargs).validate()


def test_train_generation_realises_the_cosine_over_its_planned_updates(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import planned_phase_updates, train_generation
    from tests.simple_rl_helpers import save_tiny_checkpoint
    from tests.test_simple_rl_learn import _write_generation

    run = tmp_path / "run"
    (run / "gen_0000").mkdir(parents=True)
    save_tiny_checkpoint(run / "gen_0000" / "model.pt")
    _write_generation(run, 1, 40)
    settings = TrainSettings(device="cpu", batch_size=8, holdout_frac=0.0, seed=1, lr=1e-3, lr_floor=1e-4,
                             lr_schedule="cosine-phase")
    planned = planned_phase_updates(run, 1, settings)
    summary = train_generation(run_dir=run, generation=1, init_checkpoint=run / "gen_0000" / "model.pt",
                               prev_trainer_checkpoint=None, settings=settings)
    sched = summary["realised_optimizer"]["lr_schedule"]
    assert sched["T_0"] == planned
    assert sched["eta_min"] == pytest.approx(1e-4)


def _group_lrs(settings: TrainSettings, phase_updates: int | None, steps: int) -> list[list[float]]:
    """Every group's lr the optimizer applies at each step (step pre-hook)."""
    from tests.test_simple_rl_learn import _scheduled

    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None, phase_updates=phase_updates)
    seen: list[list[float]] = []
    trainer.opt.register_step_pre_hook(lambda o, _a, _k: seen.append([float(g["lr"]) for g in o.param_groups]))
    trainer.train_steps(_scheduled(4 * steps, [np.arange(4 * i, 4 * i + 4) for i in range(steps)]),
                        batch_size=4, steps=steps)
    return seen


def test_the_default_scope_is_derived_from_the_checkpoint_split(sl_ckpt: Path) -> None:
    """Review of #1445: without --matrix-optimizer-scope the init-checkpoint source must
    still build the checkpoint's split (it derives mlp_out from opt_param_names)."""
    from chess_anti_engine.simple_rl.learn import check_built_split

    plain: dict[str, Any] = {k: v for k, v in AURORA.items() if k != "matrix_optimizer_scope"}
    unresolved = build_trainer(tiny_model_config(), TrainSettings(**plain), log_dir=None)
    with pytest.raises(ValueError, match=r"not .*split"):  # non-vacuous: the default scope is another split
        check_built_split(unresolved, sl_ckpt)
    settings = resolve_optimizer_hparams(TrainSettings(**plain, optimizer_hparams_source="init-checkpoint"),
                                         sl_ckpt)
    assert settings.matrix_optimizer_scope == "mlp_out"
    assert settings.optimizer_hparams_resolved is not None
    assert "matrix_optimizer_scope" in settings.optimizer_hparams_resolved["filled"]
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    assert check_built_split(trainer, sl_ckpt)["names_equal"] is True
    assert tables_match(param_group_table(trainer), _ckpt_table(sl_ckpt)) == []
    with pytest.raises(ValueError, match="differs from"):
        resolve_optimizer_hparams(TrainSettings(**plain, matrix_optimizer_scope="block_all",
                                                optimizer_hparams_source="init-checkpoint"), sl_ckpt)


def test_cosine_phase_keeps_every_groups_ratio_and_ends_each_at_its_own_floor() -> None:
    """Review of #1445: one scalar eta_min decayed the Aurora group to the AdamW floor.
    The floor is per group (lr_floor * group base / lr), so the ratio holds throughout."""
    import math

    n, peak, floor, mult = 10, 1e-3, 1e-4, 20.0
    kw: dict[str, Any] = {**AURORA, "lr": peak}
    seen = _group_lrs(TrainSettings(**kw, mirror_prob=0.0, lr_floor=floor,
                                    matrix_lr_multiplier=mult, lr_schedule="cosine-phase"), n, n)
    for lrs in seen:
        assert lrs[0] / lrs[2] == pytest.approx(mult, rel=1e-6)
    assert seen[0][0] == pytest.approx(peak * mult, rel=1e-6)
    # The phase's last update runs at cosine position n - 2 (updates 0 and 1 both at the peak).
    f = (1 + math.cos(math.pi * (n - 2) / n)) / 2
    assert seen[-1][2] == pytest.approx(floor + (peak - floor) * f, rel=1e-6)
    assert seen[-1][0] == pytest.approx(mult * (floor + (peak - floor) * f), rel=1e-6)  # its own floor, x20
    assert seen[-1][0] > 2 * peak  # a shared scalar floor would give ~5.9e-4 here


def test_an_explicit_matrix_lr_multiplier_reaches_the_aurora_group() -> None:
    lrs = _group_lrs(TrainSettings(**AURORA, mirror_prob=0.0, matrix_lr_multiplier=7.0), None, 2)
    assert lrs[0][0] == pytest.approx(7.0 * AURORA["lr"], rel=1e-6)
    assert lrs[0][2] == pytest.approx(AURORA["lr"], rel=1e-6)


def test_the_documented_reproduction_equals_the_checkpoint_state(sl_ckpt: Path) -> None:
    """What reproduces exactly: aurora + init-checkpoint + mlp_out at the checkpoint's lr
    gives its groups, names, order, lr, betas, wd, Aurora knobs and zclip mode."""
    from chess_anti_engine.simple_rl.learn import check_built_split

    settings = resolve_optimizer_hparams(TrainSettings(**AURORA, optimizer_hparams_source="init-checkpoint"),
                                         sl_ckpt)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    assert check_built_split(trainer, sl_ckpt)["names_equal"] is True
    assert tables_match(param_group_table(trainer), _ckpt_table(sl_ckpt)) == []
    assert trainer.grad_clip_mode == "zclip"


def test_accumulation_with_aurora_and_cosine_counts_optimizer_steps_and_keeps_the_ratio(tmp_path: Path) -> None:
    """Review of #1445: under --accum-steps the cosine's T_0, the planned updates and the
    realised optimizer steps are all OPTIMIZER steps (not micro-batches), and every group
    ends at its own floor position with the Aurora/AdamW ratio held at 20."""
    import math

    from chess_anti_engine.simple_rl.learn import planned_phase_updates, train_generation
    from tests.simple_rl_helpers import save_tiny_checkpoint
    from tests.test_simple_rl_learn import _write_generation

    run = tmp_path / "run"
    (run / "gen_0000").mkdir(parents=True)
    save_tiny_checkpoint(run / "gen_0000" / "model.pt")
    _write_generation(run, 1, 40)               # 40 rows: 5 micro-batches of 8, 3 optimizer steps of 16
    peak, floor = 1e-3, 1e-4
    settings = TrainSettings(**{**AURORA, "lr": peak}, mirror_prob=0.0, batch_size=8, accum_steps=2,
                             holdout_frac=0.0, seed=1, lr_floor=floor, lr_schedule="cosine-phase")
    planned = planned_phase_updates(run, 1, settings)
    assert planned == 3
    summary = train_generation(run_dir=run, generation=1, init_checkpoint=run / "gen_0000" / "model.pt",
                               prev_trainer_checkpoint=None, settings=settings)
    realised = summary["realised_optimizer"]
    assert summary["end_step"] - summary["start_step"] == planned == summary["updates"]
    assert summary["rows_per_optimizer_step"] == {"steps": 3, "min": 8, "max": 16, "rows": 40}
    assert realised["lr_schedule"]["T_0"] == planned     # optimizer steps, not the 5 micro-batches
    assert realised["lr_schedule"]["group_eta_min"] == pytest.approx([20 * floor, floor, floor, floor])
    lrs = realised["param_group_lr"]
    assert lrs[0] / lrs[2] == pytest.approx(20.0, rel=1e-9)
    # After the phase's 3rd step the scheduler sits at position 2 of T_0 = 3.
    want = floor + (peak - floor) * (1 + math.cos(math.pi * 2 / planned)) / 2
    assert lrs[2] == pytest.approx(want, rel=1e-9)
    assert lrs[0] == pytest.approx(20 * want, rel=1e-9)
