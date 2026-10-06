from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding.features import RELATION_COUNT
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.onnx import export_onnx, export_onnx_int8


def _tiny(*, trunk: bool, policy: bool) -> torch.nn.Module:
    model = build_model(
        ModelConfig(
            kind="transformer",
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            ffn_mult=2,
            use_smolgen=False,
            use_nla=False,
            use_dynamic_relations=trunk,
            policy_dynamic_relations=policy,
        )
    ).cpu()
    model.eval()
    return model


def _planes_and_relations(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    planes = input_plane_count(getattr(model, "input_extra_features", None))
    x = torch.randn(2, planes, 8, 8)
    relations = torch.zeros(2, RELATION_COUNT, 64, 64, dtype=torch.uint8)
    relations[:, 0, :8, :8] = 1
    return x, relations


def _fill(model: torch.nn.Module, name: str, value: float) -> None:
    param = getattr(model, name)
    assert isinstance(param, torch.nn.Parameter)
    with torch.no_grad():
        param.fill_(value)


def _policy_moves(model: torch.nn.Module, x: torch.Tensor, relations: torch.Tensor) -> bool:
    with torch.no_grad():
        bare = model(x)["policy_own"]
        with_relations = model(x, relations=relations)["policy_own"]
    return not torch.allclose(bare, with_relations, rtol=1e-4, atol=1e-4)


def test_relation_weights_change_the_torch_forward_and_onnx_export_refuses(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny(trunk=True, policy=True)
    _fill(model, "dynamic_relation_weight", 0.75)
    _fill(model, "policy_relation_weight", 0.75)
    x, relations = _planes_and_relations(model)
    assert _policy_moves(model, x, relations)

    with pytest.raises(ValueError, match="input_planes"):
        export_onnx(model, out_path=tmp_path / "relation.onnx", device="cpu")
    with pytest.raises(ValueError, match="input_planes"):
        export_onnx_int8(model, out_path=tmp_path / "relation.int8.onnx", device="cpu")
    assert not (tmp_path / "relation.onnx").exists()
    assert not (tmp_path / "relation.int8.onnx").exists()
    assert not (tmp_path / "relation.int8.fp32.onnx").exists()


def test_relation_weights_nested_under_module_are_refused(tmp_path: Path) -> None:
    inner = _tiny(trunk=True, policy=False)
    _fill(inner, "dynamic_relation_weight", 0.75)

    class _Holds(torch.nn.Module):
        def __init__(self, child: torch.nn.Module) -> None:
            super().__init__()
            self.module = child

        def forward(self, x: torch.Tensor, relations: torch.Tensor | None = None):
            if relations is None:
                return self.module(x)
            return self.module(x, relations=relations)

    with pytest.raises(ValueError, match="dynamic_relation_weight"):
        export_onnx(_Holds(inner), out_path=tmp_path / "wrapped.onnx", device="cpu")
    assert not (tmp_path / "wrapped.onnx").exists()


def test_policy_relation_weights_alone_are_enough_to_refuse_export(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny(trunk=False, policy=True)
    assert getattr(model, "dynamic_relation_weight", None) is None
    _fill(model, "policy_relation_weight", 0.75)
    x, relations = _planes_and_relations(model)
    assert _policy_moves(model, x, relations)
    with pytest.raises(ValueError, match="policy_relation_weight"):
        export_onnx(model, out_path=tmp_path / "policy-only.onnx", device="cpu")
