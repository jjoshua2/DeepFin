from __future__ import annotations

from chess_anti_engine.tune.trainable_config_ops import (
    _apply_lr_gamma_weights,
    _play_batch_kwargs,
    _reload_yaml_into_config,
)
from chess_anti_engine.tune.trial_config import TrialConfig


class _FakeOpt:
    def __init__(self) -> None:
        self.gamma = 0.0
        self.param_groups = [
            {"use_aurora": True},
            {"use_aurora": False},
        ]


class _FakeModel:
    def __init__(self) -> None:
        self.resid_channel_dropout = 0.0
        self.resid_channel_balance_weight = 0.0


class _FakeTrainer:
    def __init__(self) -> None:
        self.opt = _FakeOpt()
        self.model = _FakeModel()
        self.peak_lr_calls: list[tuple[float, bool]] = []
        self.w_policy = 0.0
        self.w_soft = 0.0
        self.w_future = 0.0
        self.w_wdl = 0.0
        self.w_sf_move = 0.0
        self.w_sf_eval = 0.0
        self.w_categorical = 0.0
        self.w_volatility = 0.0
        self.w_sf_volatility = 0.0
        self.w_moves_left = 0.0
        self.sf_wdl_frac = 0.0
        self.search_wdl_frac = 0.0
        self.sf_wdl_conf_power = 0.0
        self.sf_wdl_draw_scale = 0.0
        self.sf_wdl_temperature = 0.0
        self.sf_search_dampen_sf_low = 0.0
        self.sf_search_dampen_sf_high = 0.0
        self.resid_channel_dropout = 0.0
        self.resid_channel_balance_weight = 0.0

    def set_peak_lr(self, value: float, *, rescale_current: bool) -> None:
        self.peak_lr_calls.append((value, rescale_current))


def test_play_batch_kwargs_preserves_syzygy_adjudication_knobs() -> None:
    tc = TrialConfig.from_dict(
        {
            "syzygy_path": "/tmp/tb",
            "stockfish_syzygy_path": "/tmp/ssd-tb",
            "syzygy_rescore_policy": True,
            "syzygy_adjudicate": True,
            "syzygy_adjudicate_fraction": 0.5,
            "syzygy_in_search": True,
        }
    )

    game = _play_batch_kwargs(tc)["game"]

    assert game.syzygy_path == "/tmp/tb"
    assert game.stockfish_syzygy_path == "/tmp/ssd-tb"
    assert game.syzygy_rescore_policy is True
    assert game.syzygy_adjudicate is True
    assert game.syzygy_adjudicate_fraction == 0.5
    assert game.syzygy_in_search is True


def test_play_batch_kwargs_threads_temperature_and_gumbel_knobs() -> None:
    tc = TrialConfig.from_dict(
        {
            "temperature": 0.7,
            "temperature_after": 0.2,
            "temperature_decay_start_move": 15,
            "temperature_decay_moves": 20,
            "temperature_endgame": 0.1,
            "selfplay_temperature": 0.9,
            "selfplay_temperature_decay_start_move": 12,
            "selfplay_temperature_decay_moves": 18,
            "selfplay_temperature_endgame": 0.05,
            "gumbel_topk": 24,
            "gumbel_c_scale": 0.15,
            "gumbel_scale": 0.8,
            "gumbel_scale_after": 0.1,
            "gumbel_scale_decay_start_move": 10,
            "gumbel_scale_decay_moves": 16,
            "curriculum_gumbel_scale": 0.4,
            "curriculum_gumbel_scale_after": 0.05,
            "curriculum_gumbel_scale_decay_start_move": 8,
            "curriculum_gumbel_scale_decay_moves": 12,
        }
    )

    kwargs = _play_batch_kwargs(tc)
    temp = kwargs["temp"]
    search = kwargs["search"]

    assert temp.temperature == 0.7
    assert temp.after == 0.2
    assert temp.decay_start_move == 15
    assert temp.decay_moves == 20
    assert temp.endgame == 0.1
    assert temp.selfplay_temperature == 0.9
    assert temp.selfplay_decay_start_move == 12
    assert temp.selfplay_decay_moves == 18
    assert temp.selfplay_endgame == 0.05
    assert search.gumbel_topk == 24
    assert search.gumbel_c_scale == 0.15
    assert search.gumbel_scale == 0.8
    assert search.gumbel_scale_after == 0.1
    assert search.gumbel_scale_decay_start_move == 10
    assert search.gumbel_scale_decay_moves == 16
    assert search.curriculum_gumbel_scale == 0.4
    assert search.curriculum_gumbel_scale_after == 0.05
    assert search.curriculum_gumbel_scale_decay_start_move == 8
    assert search.curriculum_gumbel_scale_decay_moves == 12


def test_apply_lr_gamma_weights_syncs_all_trainer_loss_kwargs() -> None:
    trainer = _FakeTrainer()

    _apply_lr_gamma_weights(
        trainer,
        {
            "lr": 0.123,
            "cosmos_gamma": 0.456,
            "w_policy": 1.1,
            "w_soft": 1.2,
            "w_future": 1.3,
            "w_wdl": 1.4,
            "w_sf_move": 1.5,
            "w_sf_eval": 1.6,
            "w_categorical": 1.7,
            "w_volatility": 1.8,
            "w_sf_volatility": 1.9,
            "w_moves_left": 2.0,
            "sf_wdl_frac": 2.1,
            "search_wdl_frac": 2.2,
            "sf_wdl_conf_power": 2.3,
            "sf_wdl_draw_scale": 2.4,
            "sf_wdl_temperature": 2.5,
            "sf_search_dampen_sf_low": 2.6,
            "sf_search_dampen_sf_high": 2.7,
        },
        rescale_current_lr=True,
    )

    assert trainer.peak_lr_calls == [(0.123, True)]
    assert trainer.opt.gamma == 0.456
    assert trainer.w_policy == 1.1
    assert trainer.w_soft == 1.2
    assert trainer.w_future == 1.3
    assert trainer.w_wdl == 1.4
    assert trainer.w_sf_move == 1.5
    assert trainer.w_sf_eval == 1.6
    assert trainer.w_categorical == 1.7
    assert trainer.w_volatility == 1.8
    assert trainer.w_sf_volatility == 1.9
    assert trainer.w_moves_left == 2.0
    assert trainer.sf_wdl_frac == 2.1
    assert trainer.search_wdl_frac == 2.2
    assert trainer.sf_wdl_conf_power == 2.3
    assert trainer.sf_wdl_draw_scale == 2.4
    assert trainer.sf_wdl_temperature == 2.5
    assert trainer.sf_search_dampen_sf_low == 2.6
    assert trainer.sf_search_dampen_sf_high == 2.7


def test_apply_lr_gamma_weights_preserves_sf_volatility_fallback() -> None:
    trainer = _FakeTrainer()

    _apply_lr_gamma_weights(
        trainer,
        {"w_volatility": 0.33},
        rescale_current_lr=False,
    )

    assert trainer.w_volatility == 0.33
    assert trainer.w_sf_volatility == 0.33


def test_apply_lr_gamma_weights_syncs_aurora_and_channel_knobs() -> None:
    trainer = _FakeTrainer()

    _apply_lr_gamma_weights(
        trainer,
        {
            "aurora_pp_iterations": 3,
            "aurora_pp_beta": 0.25,
            "aurora_polar_steps": 8,
            "aurora_polar_method": "polar_express",
            "aurora_polar_dtype": "fp16",
            "aurora_polar_safety": 1.02,
            "resid_channel_dropout": 0.03,
            "resid_channel_balance_weight": 0.001,
        },
        rescale_current_lr=False,
    )

    aurora_group = trainer.opt.param_groups[0]
    adam_group = trainer.opt.param_groups[1]
    assert aurora_group["aurora_pp_iterations"] == 3
    assert aurora_group["aurora_pp_beta"] == 0.25
    assert aurora_group["aurora_polar_steps"] == 8
    assert aurora_group["aurora_polar_method"] == "polar_express"
    assert aurora_group["aurora_polar_dtype"] == "fp16"
    assert aurora_group["aurora_polar_safety"] == 1.02
    assert "aurora_pp_iterations" not in adam_group
    assert trainer.resid_channel_dropout == 0.03
    assert trainer.model.resid_channel_dropout == 0.03
    assert trainer.resid_channel_balance_weight == 0.001
    assert trainer.model.resid_channel_balance_weight == 0.001


def test_yaml_reload_does_not_add_missing_topology_keys(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
model:
  input_history_encoding: lc0_root
  qkv_projection: split
train:
  lr: 0.0007
""",
        encoding="utf-8",
    )
    config = {"lr": 0.0003}

    _reload_yaml_into_config(config, str(yaml_path))

    assert config["lr"] == 0.0007
    assert "input_history_encoding" not in config
    assert "qkv_projection" not in config
