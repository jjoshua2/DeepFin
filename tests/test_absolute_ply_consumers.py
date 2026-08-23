"""Regression coverage for consumers of absolute selfplay ``ply_index``."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

import chess_anti_engine.selfplay.finalize as finalize_mod
from chess_anti_engine.selfplay.blindspot_harvest import pre_move_boards
from chess_anti_engine.selfplay.resume import RESUME_FORMAT_VERSION, should_resume_game


_MIDGAME_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 69"


def test_v2_python_fallback_resume_is_rejected_after_ply_convention_change() -> None:
    """v2 fallback records used move-stack-relative plies; v3 uses absolute."""
    assert RESUME_FORMAT_VERSION >= 3
    assert should_resume_game(
        {
            "format_version": 2,
            "compat_fingerprint": "same",
            "has_c_ply": False,
        },
        compat_fingerprint="same",
        trial_id="trial-a",
    ) == "version_mismatch"


def test_blindspot_reconstruction_matches_absolute_fen_ply() -> None:
    """A FEN root has empty local history but a nonzero absolute game ply."""
    starting = chess.Board(_MIDGAME_FEN)
    assert len(starting.move_stack) == 0
    assert starting.ply() == 136

    move = chess.Move.from_uci("e2e4")
    final = starting.copy()
    final.push(move)

    boards, played = pre_move_boards(
        starting,
        list(final.move_stack),
        [136],
        opening_len=0,
    )

    assert boards[0] is not None
    assert boards[0].ply() == 136
    assert boards[0].fen() == starting.fen()
    assert played[0] == move


def test_syzygy_policy_rescore_matches_absolute_fen_ply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy rescoring must look up a FEN-root record by absolute game ply."""
    starting = chess.Board(_MIDGAME_FEN)
    assert len(starting.move_stack) == 0
    assert starting.ply() == 136

    move = chess.Move.from_uci("e2e4")
    final = starting.copy()
    final.push(move)

    state: Any = SimpleNamespace(
        game=SimpleNamespace(
            syzygy_path="test-tablebase",
            syzygy_rescore_policy=True,
            policy_encoding="test-policy",
        ),
        starting_boards=[starting],
        has_c_ply=False,
    )
    records: list[Any] = [SimpleNamespace(ply_index=136)]

    monkeypatch.setattr(finalize_mod, "rescore_game_samples", lambda *_args: None)
    monkeypatch.setattr(finalize_mod, "is_tb_eligible", lambda _board: True)
    monkeypatch.setattr(finalize_mod, "probe_best_move", lambda *_args: move)
    monkeypatch.setattr(
        finalize_mod,
        "move_to_index_for_encoding",
        lambda *_args, **_kwargs: 7,
    )
    monkeypatch.setattr(finalize_mod, "policy_size_for_encoding", lambda _encoding: 16)

    result, overrides = finalize_mod._rescore_with_syzygy(
        state,
        0,
        final,
        records,
        "1/2-1/2",
    )

    assert result == "1/2-1/2"
    assert set(overrides) == {0}
    assert overrides[0].shape == (16,)
    assert overrides[0].dtype == np.float32
    assert overrides[0][7] == 1.0
    assert float(overrides[0].sum()) == 1.0
