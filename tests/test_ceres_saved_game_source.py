"""A complete natural game exercises tracked Ceres ZIP publication/readback."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from hashlib import blake2b
from pathlib import Path
from typing import Any, cast

import chess
import chess.syzygy
import numpy as np
import pytest

from chess_anti_engine.encoding.ceres_tpg import stored_x_to_ceres_tpg_bytes
from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.leela_index import compact_index_for_move, leela_index_for_move
from chess_anti_engine.source.ceres_saved_game import (
    SavedCeresGame,
    read_saved_game_archive,
    write_saved_game,
)


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _game() -> SavedCeresGame:
    board = chess.Board()
    start = board.fen()
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    history: list[str] = []
    xs: list[np.ndarray] = []
    rows = []
    compact: list[int] = []
    leela: list[int] = []
    offsets = [0]
    for ply, played in enumerate(moves):
        x = encode_position(
            board, input_history_encoding="lc0_root_legacy_meta",
            input_extra_features="v2_threats",
        ).astype(np.float16)
        xs.append(x)
        feed = stored_x_to_ceres_tpg_bytes(
            x, input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True)
        legal = sorted(board.legal_moves, key=lambda move: compact_index_for_move(board, move))
        names = [move.uci() for move in legal]
        compact.extend(compact_index_for_move(board, move) for move in legal)
        leela.extend(leela_index_for_move(board, move) for move in legal)
        offsets.append(len(compact))
        rows.append({
            "source_namespace": "1" * 64, "source_shard": "synthetic_natural",
            "game_id": 7, "row_index": ply, "ply_index": ply,
            "root_fen": board.fen(), "pov_white": bool(board.turn),
            "history_stack_sha256": _sha(json.dumps(history, separators=(",", ":")).encode()),
            "x_stored_sha256": _sha(x.tobytes()),
            "stored_input_key": blake2b(np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                                         digest_size=16).hexdigest(),
            "feed_sha256": _sha(feed.tobytes()),
            "legal_moves_uci_sorted": names,
            "played_move_uci": played, "played_sorted_index": names.index(played),
        })
        board.push_uci(played)
        history.append(played)
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None
    assert outcome.result() == "0-1"
    n = len(xs)
    arrays = {
        "x": np.stack(xs),
        "value": np.zeros((n, 3), dtype=np.float16),
        "value2": np.zeros((n, 3), dtype=np.float16),
        "legal_offsets": np.asarray(offsets, dtype=np.uint32),
        "legal_compact": np.asarray(compact, dtype=np.uint16),
        "legal_leela": np.asarray(leela, dtype=np.uint16),
        "legal_logits": np.zeros(len(compact), dtype=np.float16),
    }
    meta = {
        "game_id": 7, "row_start": 0, "row_end": n,
        "source_namespace": "1" * 64, "source_shard": "synthetic_natural",
        "initial_replay_root_fen": start, "initial_history_uci": [],
        "initial_fen": start, "terminal_fen": board.fen(),
        "termination": "natural", "result": "0-1",
        "outcome_provenance": {"mode": "rule50_match_v1"},
    }
    return SavedCeresGame(arrays, rows, meta, "2" * 64, "3" * 64)


def test_complete_natural_game_roundtrip_and_collision(tmp_path: Path) -> None:
    original = _game()
    root = tmp_path / "fixture"
    receipt = write_saved_game(original, root)
    reopened = read_saved_game_archive(root)
    assert receipt["rows"] == 4
    assert reopened.rows == original.rows
    assert reopened.game == original.game
    assert all(np.array_equal(reopened.arrays[key], value)
               for key, value in original.arrays.items())
    with pytest.raises(ValueError, match="fresh absolute output"):
        write_saved_game(original, root)
    with (root / "game.zarr.zip").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="archive SHA"):
        read_saved_game_archive(root)


def test_reject_altered_saved_feature_and_unfinished_game(tmp_path: Path) -> None:
    original = _game()
    arrays = {key: value.copy() for key, value in original.arrays.items()}
    arrays["x"][0, 0, 1, 0] = 0
    with pytest.raises(ValueError, match="row/input/history/feed"):
        write_saved_game(replace(original, arrays=arrays), tmp_path / "bad_x")
    assert not (tmp_path / "bad_x").exists()
    game = {**original.game, "terminal_fen": original.game["initial_fen"]}
    with pytest.raises(ValueError, match="saved game terminal FEN"):
        write_saved_game(replace(original, game=game), tmp_path / "bad_terminal")


class FakeStrictTablebase:
    def __init__(self, wdl: int = -2, dtz: int = -1) -> None:
        self.wdl = {"KQRvKNP": object()}
        self.dtz = {"KQRvKNP": object()}
        self.wdl_value = wdl
        self.dtz_value = dtz
        self.calls: list[str] = []

    def probe_wdl(self, _board: chess.Board) -> int:
        self.calls.append("wdl")
        return self.wdl_value

    def probe_dtz(self, _board: chess.Board) -> int:
        self.calls.append("dtz")
        return self.dtz_value


def _strict_handle(fake: FakeStrictTablebase) -> chess.syzygy.Tablebase:
    return cast(chess.syzygy.Tablebase, cast(Any, fake))


def _syzygy_game(
    *, start_fen: str = "6nk/1p5p/8/8/8/8/8/KQR5 w - - 0 1",
    played: str = "b1b7",
) -> SavedCeresGame:
    board = chess.Board(start_fen)
    start = board.fen()
    x = encode_position(
        board, input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    ).astype(np.float16)
    feed = stored_x_to_ceres_tpg_bytes(
        x, input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True)
    legal = sorted(board.legal_moves, key=lambda move: compact_index_for_move(board, move))
    moves = [move.uci() for move in legal]
    row = {
        "source_namespace": "1" * 64, "source_shard": "synthetic_syzygy",
        "game_id": 9, "row_index": 0, "ply_index": 0,
        "root_fen": start, "pov_white": bool(board.turn),
        "history_stack_sha256": _sha(b"[]"),
        "x_stored_sha256": _sha(x.tobytes()),
        "stored_input_key": blake2b(np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                                     digest_size=16).hexdigest(),
        "feed_sha256": _sha(feed.tobytes()),
        "legal_moves_uci_sorted": moves,
        "played_move_uci": played, "played_sorted_index": moves.index(played),
    }
    board.push_uci(played)
    assert chess.popcount(board.occupied) == 6
    meta = {
        "game_id": 9, "row_start": 0, "row_end": 1,
        "source_namespace": "1" * 64, "source_shard": "synthetic_syzygy",
        "initial_replay_root_fen": start, "initial_history_uci": [],
        "initial_fen": start, "terminal_fen": board.fen(),
        "termination": "syzygy", "detail": "rule50_match_wdl_dtz", "result": "1-0",
        "outcome_provenance": {
            "mode": "rule50_match_v1", "syzygy_path": "fake:strict",
            "max_pieces": 6, "handle_contract": "caller_owned_capacity_checked",
            "wdl_table_count": 1, "dtz_table_count": 1,
        },
    }
    arrays = {
        "x": x[None],
        "value": np.zeros((1, 3), dtype=np.float16),
        "value2": np.zeros((1, 3), dtype=np.float16),
        "legal_offsets": np.array([0, len(moves)], dtype=np.uint32),
        "legal_compact": np.asarray([compact_index_for_move(chess.Board(start), move)
                                     for move in legal], dtype=np.uint16),
        "legal_leela": np.asarray([leela_index_for_move(chess.Board(start), move)
                                   for move in legal], dtype=np.uint16),
        "legal_logits": np.zeros(len(moves), dtype=np.float16),
    }
    return SavedCeresGame(arrays, [row], meta, "2" * 64, "3" * 64)


def test_strict_syzygy_roundtrip_reopens_and_probes(tmp_path: Path) -> None:
    original = _syzygy_game()
    fake = FakeStrictTablebase()
    root = tmp_path / "syzygy"
    receipt = write_saved_game(original, root, syzygy_path="fake:strict",
                               match_tablebase=_strict_handle(fake))
    assert receipt["rows"] == 1
    with pytest.raises(ValueError, match="strict six-man"):
        read_saved_game_archive(root)
    reopened = read_saved_game_archive(root, syzygy_path="fake:strict",
                                       match_tablebase=_strict_handle(fake))
    assert reopened.game == original.game
    assert fake.calls == ["wdl", "dtz"] * 4


def test_strict_syzygy_rejects_false_adjudication(tmp_path: Path) -> None:
    original = _syzygy_game()
    fake = FakeStrictTablebase()
    handle = _strict_handle(fake)
    with pytest.raises(ValueError, match="strict Syzygy terminal"):
        write_saved_game(replace(original, game={**original.game, "result": "0-1"}),
                         tmp_path / "wrong_winner", syzygy_path="fake:strict",
                         match_tablebase=handle)
    with pytest.raises(ValueError, match="strict Syzygy terminal"):
        write_saved_game(original, tmp_path / "cursed_draw", syzygy_path="fake:strict",
                         match_tablebase=_strict_handle(FakeStrictTablebase(-1, -101)))
    class MissingDtz(FakeStrictTablebase):
        def probe_dtz(self, _board: chess.Board) -> int:
            raise chess.syzygy.MissingTableError("absent DTZ")
    with pytest.raises(RuntimeError, match="missing eligible match Syzygy probe"):
        write_saved_game(original, tmp_path / "missing_dtz", syzygy_path="fake:strict",
                         match_tablebase=_strict_handle(MissingDtz()))
    positive_clock = _syzygy_game(
        start_fen="1n6/1p6/7K/2b5/8/1k5P/8/8 b - - 1 58", played="c5d4")
    with pytest.raises(ValueError, match="adjudicated before row 0"):
        write_saved_game(positive_clock, tmp_path / "positive_clock",
                         syzygy_path="fake:strict",
                         match_tablebase=_strict_handle(FakeStrictTablebase(-2, -1)))
    with pytest.raises(ValueError, match="exact strict six-man"):
        write_saved_game(original, tmp_path / "wrong_path", syzygy_path="other:strict",
                         match_tablebase=handle)
