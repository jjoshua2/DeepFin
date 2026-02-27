from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import LC0_FULL
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.train import Trainer
from chess_anti_engine.train.trainer import TrainMetrics


class _CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(0.0))
        self.last_x: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        self.last_x = x.detach().clone()
        b = x.shape[0]
        z = self.p + torch.zeros((b, 1), device=x.device, dtype=x.dtype)
        return {
            "policy_own": z + torch.zeros((b, POLICY_SIZE), device=x.device, dtype=x.dtype),
            "wdl": z + torch.zeros((b, 3), device=x.device, dtype=x.dtype),
        }


class _NoPostMetricsTrainer(Trainer):
    @torch.no_grad()
    def _compute_metrics(self, *, buf: ReplayBuffer, batch_size: int, steps: int, tag: str) -> TrainMetrics:
        # Avoid a second forward pass that would overwrite _CaptureModel.last_x.
        return TrainMetrics(
            loss=0.0,
            policy_loss=0.0,
            soft_policy_loss=0.0,
            future_policy_loss=0.0,
            wdl_loss=0.0,
            sf_move_loss=0.0,
            sf_move_acc=0.0,
            sf_eval_loss=0.0,
            categorical_loss=0.0,
            volatility_loss=0.0,
            sf_volatility_loss=0.0,
            moves_left_loss=0.0,
        )


def _make_sample(*, base_val: float, feat_val: float) -> ReplaySample:
    base = int(LC0_FULL.num_planes)
    x = np.zeros((146, 8, 8), dtype=np.float32)
    x[:base, :, :] = base_val
    x[base:, :, :] = feat_val

    policy = np.full((POLICY_SIZE,), 1.0 / float(POLICY_SIZE), dtype=np.float32)
    return ReplaySample(
        x=x,
        policy_target=policy,
        wdl_target=1,
        is_network_turn=True,
        has_policy=True,
        has_volatility=False,
    )


def test_encode_position_has_146_planes():
    import chess

    b = chess.Board()
    x = encode_position(b, add_features=True)
    assert x.shape == (146, 8, 8)


def test_feature_dropout_only_zeroes_extra_planes_when_enabled(tmp_path):
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(10, rng=rng)
    buf.add(_make_sample(base_val=2.0, feat_val=3.0))

    model = _CaptureModel()
    trainer = _NoPostMetricsTrainer(
        model,
        device="cpu",
        lr=1e-3,
        log_dir=tmp_path,
        use_amp=False,
        feature_dropout_p=1.0,
    )

    trainer.train_steps(buf, batch_size=1, steps=1)
    assert model.last_x is not None

    x = model.last_x[0].cpu().numpy()
    base = int(LC0_FULL.num_planes)
    assert np.allclose(x[:base], 2.0)
    assert np.allclose(x[base:], 0.0)


def test_feature_dropout_can_be_disabled(tmp_path):
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(10, rng=rng)
    buf.add(_make_sample(base_val=2.0, feat_val=3.0))

    model = _CaptureModel()
    trainer = _NoPostMetricsTrainer(
        model,
        device="cpu",
        lr=1e-3,
        log_dir=tmp_path,
        use_amp=False,
        feature_dropout_p=0.0,
    )

    trainer.train_steps(buf, batch_size=1, steps=1)
    assert model.last_x is not None

    x = model.last_x[0].cpu().numpy()
    base = int(LC0_FULL.num_planes)
    assert np.allclose(x[:base], 2.0)
    assert np.allclose(x[base:], 3.0)
