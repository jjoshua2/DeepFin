"""Blind-spot FEN-list opening seeding (selfplay/opening.py)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.selfplay.opening import (
    OpeningConfig,
    _load_fen_list,
    sample_starting_board,
)

FEN_BLACK = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FEN_WHITE = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"


def _write_list(tmp_path: Path, lines: list[str]) -> str:
    p = tmp_path / "fens_test.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_load_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = _write_list(tmp_path, ["# header", "", FEN_BLACK, "  ", FEN_WHITE])
    assert _load_fen_list(path) == (FEN_BLACK, FEN_WHITE)


def test_load_rejects_invalid_fen(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_BLACK, "not a fen"])
    with pytest.raises(ValueError, match=r"invalid FEN.*:2"):
        _load_fen_list(path)


def test_load_rejects_terminal_position(tmp_path: Path) -> None:
    # Fool's mate final position: game over, useless as a game start.
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    path = _write_list(tmp_path, [mate])
    with pytest.raises(ValueError, match="terminal position"):
        _load_fen_list(path)


def test_prob_one_always_samples_fenlist(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_BLACK, FEN_WHITE])
    cfg = OpeningConfig(opening_fen_list_path=path, opening_fen_prob=1.0)
    rng = np.random.default_rng(0)
    fens = {FEN_BLACK, FEN_WHITE}
    for _ in range(20):
        start = sample_starting_board(rng=rng, cfg=cfg)
        assert start.source == "fenlist"
        assert start.board.fen() in fens


def test_prob_zero_or_unset_never_samples_fenlist(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_BLACK])
    rng = np.random.default_rng(0)
    for cfg in (
        OpeningConfig(opening_fen_list_path=path, opening_fen_prob=0.0),
        OpeningConfig(),  # default: feature off
    ):
        for _ in range(20):
            assert sample_starting_board(rng=rng, cfg=cfg).source != "fenlist"


def test_fenlist_preserves_side_to_move_and_castling(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_BLACK])
    cfg = OpeningConfig(opening_fen_list_path=path, opening_fen_prob=1.0)
    board = sample_starting_board(rng=np.random.default_rng(0), cfg=cfg).board
    assert board.turn is chess.BLACK
    assert board.has_kingside_castling_rights(chess.WHITE)
    assert board.fullmove_number == 3


def test_fenlist_takes_priority_over_book(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_WHITE])
    cfg = OpeningConfig(
        opening_fen_list_path=path,
        opening_fen_prob=1.0,
        # book path unset is the common case; random_start_plies exercises the
        # fallback branch that fenlist must preempt
        random_start_plies=2,
    )
    rng = np.random.default_rng(1)
    for _ in range(10):
        assert sample_starting_board(rng=rng, cfg=cfg).source == "fenlist"


def test_production_seed_asset_loads() -> None:
    asset = Path(__file__).resolve().parents[1] / "data" / "blindspot_fens_v1.txt"
    if not asset.exists():
        pytest.skip("seed asset not present in this checkout")
    fens = _load_fen_list(str(asset))
    assert len(fens) == len(set(fens)) == 113
