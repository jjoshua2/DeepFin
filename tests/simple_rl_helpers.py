"""Shared fixtures for the simple_rl tests: a tiny net and a quiet fake engine."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tests.selfplay_helpers import FakeStockfish

TINY_ARCH: dict[str, Any] = {
    "embed_dim": 32, "num_layers": 1, "num_heads": 2, "ffn_mult": 1.0,
    "smolgen_hidden_channels": 8, "smolgen_hidden_sz": 32, "smolgen_gen_sz": 32,
    "enable_policy_sf_head": False,
}


def tiny_model_config() -> Any:
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
    from chess_anti_engine.uci.model_loader import model_config_from_arch

    return model_config_from_arch({"_schema_version": ARCH_SCHEMA_VERSION, **asdict(ModelConfig()), **TINY_ARCH})


def tiny_model(seed: int = 0) -> torch.nn.Module:
    from chess_anti_engine.model import build_model

    torch.manual_seed(seed)
    return build_model(tiny_model_config()).eval()


def save_tiny_checkpoint(path: Path, seed: int = 0) -> Path:
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer

    torch.manual_seed(seed)
    trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu"), log_dir=None)
    trainer.save(Path(path))
    return Path(path)


class QuietStockfish(FakeStockfish):
    """Fixed-answer engine with the ``close`` the collector calls."""

    def close(self) -> None:
        return None
