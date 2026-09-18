"""Bounded CPU architecture smoke; seeded untrained ChessNet, NOT a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils import _pytree

from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from native.bend_engine.neural_probe.adapter import mapping_hash, sha256


class PolicyWDL(torch.nn.Module):
    def __init__(self, planes: int):
        super().__init__()
        self.network = ChessNet(
            TransformerConfig(
                in_planes=planes,
                embed_dim=32,
                num_layers=1,
                num_heads=2,
                ffn_mult=1.0,
                use_smolgen=False,
            )
        ).eval()
        self.network._inference_only = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.network(x)
        return result["policy_own"], result["wdl"]


def make_model(planes: int, dtype: str = "float32") -> PolicyWDL:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1909)
        return PolicyWDL(planes).eval().to(dtype=getattr(torch, dtype))


def export(package: Path, planes: int = 146, dtype: str = "float32") -> PolicyWDL:
    if planes not in (146, 175, 179) or dtype not in ("float32", "bfloat16"):
        raise ValueError("unsupported smoke model input")
    torch.set_num_threads(2)
    model = make_model(planes, dtype)
    example = torch.zeros((1, planes, 8, 8), dtype=getattr(torch, dtype))
    package.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        graph = torch.export.export(model, (example,))
        torch._inductor.aoti_compile_and_package(graph, package_path=str(package))
    output_spec = _pytree.treespec_dumps(
        _pytree.tree_flatten((torch.empty(0), torch.empty(0)))[1]
    )
    package.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "deepfin-bend-cpu-tuple-v1",
                "device": "cpu",
                "batch": 1,
                "planes": planes,
                "dtype": dtype,
                "history_mode": 1,
                "history_rep_fix": True,
                "policy_map_sha256": mapping_hash(),
                "package_sha256": sha256(package),
                "output_spec": output_spec,
                "torch": torch.__version__,
                "model": "seed-1909 untrained production ChessNet class, width32/layer1/head2, no smolgen",
                "outputs": ["compact_policy_logits", "wdl_logits"],
            },
            indent=2,
        )
        + "\n"
    )
    return model
