from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from chess_anti_engine.moves import POLICY_SIZE


def _load_shrink_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "shrink_ffn_checkpoint.py"
    spec = importlib.util.spec_from_file_location("shrink_ffn_checkpoint_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_cfg(*, ffn_mult_by_layer: tuple[float, ...]) -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        embed_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_mult=sum(ffn_mult_by_layer) / len(ffn_mult_by_layer),
        ffn_mult_by_layer=ffn_mult_by_layer,
        use_smolgen=False,
        use_deepnorm=False,
    )


def test_shrink_model_state_dict_prunes_important_ffn_units() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_cfg = _small_cfg(ffn_mult_by_layer=(1.0, 1.0))
    source_state = build_model(source_cfg).state_dict()
    target_template = build_model(target_cfg).state_dict()

    with torch.no_grad():
        up_w = source_state["blocks.0.ffn.0.weight"]
        up_b = source_state["blocks.0.ffn.0.bias"]
        down_w = source_state["blocks.0.ffn.2.weight"]
        up_w.zero_()
        up_b.zero_()
        down_w.zero_()
        keep_units = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.long)
        for rank, unit in enumerate(keep_units.tolist(), start=1):
            up_w[unit, 0] = float(rank)

    out, copied = module.shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=(8, 8),
    )

    assert copied
    assert out["embed.weight"].shape == source_state["embed.weight"].shape
    torch.testing.assert_close(out["embed.weight"], source_state["embed.weight"])
    torch.testing.assert_close(out["blocks.0.ffn.0.weight"], source_state["blocks.0.ffn.0.weight"][keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.0.bias"], source_state["blocks.0.ffn.0.bias"][keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.2.weight"], source_state["blocks.0.ffn.2.weight"][:, keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.2.bias"], source_state["blocks.0.ffn.2.bias"])


def test_shrink_model_state_dict_can_use_external_unit_scores() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_cfg = _small_cfg(ffn_mult_by_layer=(1.0, 1.0))
    source_state = build_model(source_cfg).state_dict()
    target_template = build_model(target_cfg).state_dict()

    layer0_scores = torch.zeros(16)
    keep_units = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.long)
    layer0_scores[keep_units] = torch.arange(1, 9, dtype=torch.float32)

    out, _copied = module.shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=(8, 8),
        unit_scores_by_layer={0: layer0_scores},
    )

    torch.testing.assert_close(out["blocks.0.ffn.0.weight"], source_state["blocks.0.ffn.0.weight"][keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.2.weight"], source_state["blocks.0.ffn.2.weight"][:, keep_units])


def test_activation_unit_scores_from_x_shapes() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    ckpt = {
        "model": build_model(source_cfg).state_dict(),
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }
    x = torch.randn(4, 146, 8, 8)

    scores, stats = module.activation_unit_scores_from_x(
        ckpt,
        ckpt_path=Path("trainer.pt"),
        x=x,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert stats["activation_score_positions"] == 4
    assert stats["activation_score_layers"] == 2
    assert set(scores) == {0, 1}
    assert scores[0].shape == (16,)
    assert scores[1].shape == (12,)
    assert torch.isfinite(scores[0]).all()
    assert torch.isfinite(scores[1]).all()
    assert torch.all(scores[0] >= 0)
    assert torch.all(scores[1] >= 0)


def test_taylor_unit_scores_from_arrays_shapes() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    ckpt = {
        "model": build_model(source_cfg).state_dict(),
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }
    n = 3
    arrs = {
        "x": torch.randn(n, 146, 8, 8).numpy(),
        "policy_target": torch.nn.functional.one_hot(
            torch.zeros(n, dtype=torch.long),
            num_classes=POLICY_SIZE,
        ).float().numpy(),
        "wdl_target": torch.zeros(n, dtype=torch.long).numpy(),
    }

    scores, stats = module.taylor_unit_scores_from_arrays(
        ckpt,
        ckpt_path=Path("trainer.pt"),
        arrs=arrs,
        batch_size=2,
        device=torch.device("cpu"),
        loss_kwargs={},
    )

    assert stats["taylor_score_positions"] == n
    assert stats["taylor_score_layers"] == 2
    assert set(scores) == {0, 1}
    assert scores[0].shape == (16,)
    assert scores[1].shape == (12,)
    assert torch.isfinite(scores[0]).all()
    assert torch.isfinite(scores[1]).all()
    assert torch.all(scores[0] >= 0)
    assert torch.all(scores[1] >= 0)


def test_shrink_checkpoint_updates_arch_and_drops_optimizer(tmp_path: Path) -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_schedule = (1.0, 1.0)
    ckpt = {
        "model": build_model(source_cfg).state_dict(),
        "swa_model": build_model(source_cfg).state_dict(),
        "opt": {"state": {"stale": True}},
        "scheduler": {"stale": True},
        "step": 123,
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }

    new_ckpt, target_cfg, stats = module.shrink_checkpoint(
        ckpt,
        ckpt_path=tmp_path / "trainer.pt",
        target_schedule=target_schedule,
    )

    assert target_cfg.ffn_mult_by_layer == target_schedule
    assert tuple(new_ckpt["arch"]["ffn_mult_by_layer"]) == target_schedule
    assert new_ckpt["step"] == 123
    assert "opt" not in new_ckpt
    assert "scheduler" not in new_ckpt
    assert stats["dropped_optimizer"] is True
    assert stats["dropped_scheduler"] is True

    target_model = build_model(target_cfg)
    target_model.load_state_dict(new_ckpt["model"])
    target_model.load_state_dict(new_ckpt["swa_model"])
