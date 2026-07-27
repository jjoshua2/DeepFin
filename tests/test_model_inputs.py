"""Encoding for a model must come from the model, never from a default.

docs/rl_loop_audit.md M11: 98 of 175 planes differ between the encoder's
``legacy`` default and the production ``lc0_root_legacy_meta`` layout, and a
model fed the wrong one produces confidently wrong numbers with no error.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import (
    encode_position,
    encode_position_for_model,
    encode_positions_batch_for_model,
    model_encoding_kwargs,
    model_input_plane_count,
)


class _FakeModel:
    """Stands in for a built ChessNet: only the runtime metadata matters."""

    def __init__(self, history: str, extra: str) -> None:
        self.input_history_encoding = history
        self.input_extra_features = extra


def _production_model() -> _FakeModel:
    return _FakeModel("lc0_root_legacy_meta", "v2_threats")


def _played_board() -> chess.Board:
    board = chess.Board()
    for uci in ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"):
        board.push(chess.Move.from_uci(uci))
    return board


def test_model_encoding_kwargs_reads_the_model() -> None:
    assert model_encoding_kwargs(_production_model()) == {
        "input_history_encoding": "lc0_root_legacy_meta",
        "input_extra_features": "v2_threats",
    }


def test_model_encoding_kwargs_refuses_to_guess() -> None:
    class _Bare:
        pass

    with pytest.raises(ValueError, match="does not declare"):
        model_encoding_kwargs(_Bare())

    partial = _production_model()
    partial.input_history_encoding = None  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="input_history_encoding"):
        model_encoding_kwargs(partial)


def test_encoding_for_model_differs_from_the_legacy_default() -> None:
    """The whole point: the silent default is NOT what production wants."""
    board = _played_board()
    model = _production_model()

    for_model = encode_position_for_model(model, board)
    defaulted = encode_position(board, add_features=True)

    # The default gets BOTH halves wrong: 146 v1 planes instead of 175, and
    # legacy history instead of lc0_root_legacy_meta.
    assert for_model.shape == (175, 8, 8)
    assert defaulted.shape == (146, 8, 8)

    # Holding the extra-feature version fixed, the history mode alone changes
    # most of the tensor — a mismatch no model can detect.
    history_only = encode_position(
        board,
        add_features=True,
        input_history_encoding="legacy",
        input_extra_features="v2_threats",
    )
    differing = sum(
        1
        for plane in range(175)
        if not np.array_equal(for_model[plane], history_only[plane])
    )
    assert differing > 50, f"only {differing} planes differ; is the default still legacy?"


def test_encoding_for_model_matches_the_explicit_call() -> None:
    board = _played_board()
    model = _production_model()
    explicit = encode_position(
        board,
        add_features=True,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    np.testing.assert_array_equal(encode_position_for_model(model, board), explicit)


def test_batch_encoding_for_model_matches_per_board() -> None:
    model = _production_model()
    boards = [chess.Board(), _played_board()]
    batch = encode_positions_batch_for_model(model, boards)
    assert batch.shape == (2, 175, 8, 8)
    for i, board in enumerate(boards):
        np.testing.assert_array_equal(batch[i], encode_position_for_model(model, board))


def test_plane_count_comes_from_the_model() -> None:
    assert model_input_plane_count(_production_model()) == 175
    assert model_input_plane_count(_FakeModel("legacy", "v1")) == 146


def test_a_real_built_model_declares_its_encoding() -> None:
    """`build_model` must keep attaching what this module depends on."""
    from chess_anti_engine.model import ModelConfig, build_model

    cfg = ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    model = build_model(cfg)
    assert model_encoding_kwargs(model) == {
        "input_history_encoding": "lc0_root_legacy_meta",
        "input_extra_features": "v2_threats",
    }
    assert model_input_plane_count(model) == 175
