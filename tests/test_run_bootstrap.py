from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

import chess_anti_engine.run as run_module


class _FakeModel:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state) -> None:
        self.loaded = state


class _FakeTrainer:
    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.step = 0


class _FakeReplayBuffer:
    def __init__(self, *args, **kwargs) -> None:
        self.capacity = int(args[0])

    def __len__(self) -> int:
        return 0


class _FakeStockfish:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _single_args(tmp_path: Path, ckpt_path: Path) -> Namespace:
    return Namespace(
        device="cpu",
        iterations=0,
        replay_capacity=32,
        work_dir=str(tmp_path / "run"),
        seed=1,
        lr=1e-3,
        batch_size=4,
        train_steps=1,
        games_per_iter=1,
        selfplay_fraction=0.0,
        temperature=1.0,
        temperature_drop_plies=0,
        temperature_after=0.0,
        temperature_decay_start_move=0,
        temperature_decay_moves=0,
        temperature_endgame=0.0,
        max_plies=32,
        opening_book_path=None,
        opening_book_max_plies=4,
        opening_book_max_games=16,
        opening_book_prob=0.0,
        opening_book_path_2=None,
        opening_book_max_plies_2=16,
        opening_book_max_games_2=16,
        opening_book_mix_prob_2=0.0,
        random_start_plies=0,
        sf_policy_temp=1.0,
        sf_policy_label_smooth=0.0,
        timeout_adjudication_threshold=0.9,
        stockfish_path="/bin/true",
        sf_nodes=1,
        sf_multipv=1,
        sf_hash_mb=16,
        sf_pid_enabled=False,
        sf_pid_target_winrate=0.5,
        sf_pid_ema_alpha=0.1,
        sf_pid_deadzone=0.05,
        sf_pid_rate_limit=0.1,
        sf_pid_min_games_between_adjust=1,
        sf_pid_kp=0.0,
        sf_pid_ki=0.0,
        sf_pid_kd=0.0,
        sf_pid_integral_clamp=1.0,
        sf_pid_min_nodes=1,
        sf_pid_max_nodes=1,
        model="transformer",
        embed_dim=8,
        num_layers=1,
        num_heads=1,
        ffn_mult=1,
        no_smolgen=False,
        use_nla=False,
        gradient_checkpointing=False,
        grad_clip=1.0,
        no_amp=True,
        feature_dropout_p=0.0,
        w_volatility=0.0,
        accum_steps=1,
        warmup_steps=0,
        warmup_lr_start=None,
        lr_eta_min=0.0,
        lr_T0=1,
        lr_T_mult=1,
        use_compile=False,
        optimizer="adamw",
        cosmos_rank=1,
        cosmos_gamma=0.0,
        swa_start=0,
        swa_freq=1,
        w_policy=1.0,
        w_soft=0.0,
        w_future=0.0,
        w_wdl=1.0,
        w_sf_move=0.0,
        w_sf_eval=0.0,
        w_categorical=0.0,
        w_sf_volatility=0.0,
        w_moves_left=0.0,
        w_sf_wdl=0.0,
        sf_wdl_conf_power=0.0,
        sf_wdl_draw_scale=1.0,
        replay_window_start=32,
        replay_window_max=32,
        replay_window_growth=1,
        shuffle_buffer_size=32,
        shard_size=8,
        shuffle_refresh_interval=1,
        shuffle_refresh_shards=1,
        shuffle_draw_cap_frac=0.9,
        shuffle_wl_max_ratio=1.5,
        bootstrap_checkpoint=str(ckpt_path),
        bootstrap_zero_policy_heads=True,
        bootstrap_reinit_volatility_heads=True,
        sf_workers=1,
    )


def test_run_single_bootstrap_resume_uses_parsed_args(monkeypatch, tmp_path: Path) -> None:
    ckpt_path = tmp_path / "bootstrap.pt"
    torch.save({"model": {"weight": 1}}, ckpt_path)

    fake_model = _FakeModel()
    zeroed: list[str] = []
    reinit: list[str] = []

    monkeypatch.setattr(run_module, "build_model", lambda cfg: fake_model)
    monkeypatch.setattr(run_module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(run_module, "DiskReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(run_module, "StockfishUCI", _FakeStockfish)
    monkeypatch.setattr(run_module, "StockfishPool", _FakeStockfish)
    monkeypatch.setattr(
        run_module,
        "zero_policy_head_parameters_",
        lambda model: zeroed.append("zeroed") or ["policy_head"],
    )
    monkeypatch.setattr(
        run_module,
        "reinit_volatility_head_parameters_",
        lambda model: reinit.append("reinit") or ["volatility_head"],
    )

    run_module._run_single(_single_args(tmp_path, ckpt_path))

    assert fake_model.loaded == {"weight": 1}
    assert zeroed == ["zeroed"]
    assert reinit == ["reinit"]
