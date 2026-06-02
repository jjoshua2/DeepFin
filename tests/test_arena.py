from __future__ import annotations

import argparse

import pytest

from chess_anti_engine.arena import _model_config_from_cli_args, _parse_ffn_mult_by_layer
from chess_anti_engine.model import model_config_from_manifest_dict


def test_arena_cli_model_config_preserves_ffn_schedule() -> None:
    args = argparse.Namespace(
        model="transformer",
        embed_dim=384,
        num_layers=3,
        num_heads=12,
        ffn_mult=1.0,
        ffn_mult_by_layer=(1.0, 1.25, 1.5),
        use_smolgen=True,
        use_nla=False,
        gradient_checkpointing=False,
    )

    cfg = model_config_from_manifest_dict(_model_config_from_cli_args(args))

    assert cfg.ffn_mult_by_layer == (1.0, 1.25, 1.5)


def test_arena_ffn_schedule_parser_rejects_nonpositive_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive finite"):
        _parse_ffn_mult_by_layer("1.0,0.0")
