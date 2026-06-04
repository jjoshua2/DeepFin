from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model


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
