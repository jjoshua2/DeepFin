from __future__ import annotations

from chess_anti_engine.tune.trainable_config_ops import (
    _apply_lr_gamma_weights,
    _play_batch_kwargs,
)
from chess_anti_engine.tune.trial_config import TrialConfig


class _FakeOpt:
    def __init__(self) -> None:
        self.gamma = 0.0


class _FakeTrainer:
    def __init__(self) -> None:
        self.opt = _FakeOpt()
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

    def set_peak_lr(self, value: float, *, rescale_current: bool) -> None:
        self.peak_lr_calls.append((value, rescale_current))


def test_play_batch_kwargs_preserves_syzygy_adjudication_knobs() -> None:
    tc = TrialConfig.from_dict(
        {
            "syzygy_path": "/tmp/tb",
            "syzygy_rescore_policy": True,
            "syzygy_adjudicate": True,
            "syzygy_adjudicate_fraction": 0.5,
            "syzygy_in_search": True,
        }
    )

    game = _play_batch_kwargs(tc)["game"]

    assert game.syzygy_path == "/tmp/tb"
    assert game.syzygy_rescore_policy is True
    assert game.syzygy_adjudicate is True
    assert game.syzygy_adjudicate_fraction == 0.5
    assert game.syzygy_in_search is True


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
