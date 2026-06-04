from __future__ import annotations

import argparse

import pytest

from chess_anti_engine.arena import (
    _model_config_from_cli_args,
    _parse_ffn_mult_by_layer,
    _parse_phase_piece_thresholds,
)
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
        smolgen_pooling="mean",
        phase_output_adapter=True,
        phase_output_adapter_dim=48,
        phase_smolgen=True,
        phase_piece_thresholds=(12, 21),
        use_nla=False,
        gradient_checkpointing=False,
    )

    cfg = model_config_from_manifest_dict(_model_config_from_cli_args(args))

    assert cfg.ffn_mult_by_layer == (1.0, 1.25, 1.5)
    assert cfg.smolgen_pooling == "mean"
    assert cfg.phase_output_adapter is True
    assert cfg.phase_output_adapter_dim == 48
    assert cfg.phase_smolgen is True
    assert cfg.phase_piece_thresholds == (12, 21)


def test_arena_ffn_schedule_parser_rejects_nonpositive_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive finite"):
        _parse_ffn_mult_by_layer("1.0,0.0")


def test_arena_phase_threshold_parser_rejects_invalid_values() -> None:
    assert _parse_phase_piece_thresholds("13,22") == (13, 22)
    with pytest.raises(argparse.ArgumentTypeError, match="ordered"):
        _parse_phase_piece_thresholds("22,13")
