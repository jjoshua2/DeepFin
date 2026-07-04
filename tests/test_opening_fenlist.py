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


def test_load_rejects_illegal_position(tmp_path: Path) -> None:
    # Parseable but semantically invalid: black has no king.
    no_king = "8/8/8/8/8/8/8/4K3 w - - 0 1"
    path = _write_list(tmp_path, [no_king])
    with pytest.raises(ValueError, match="illegal position"):
        _load_fen_list(path)


def test_load_rejects_forced_single_move(tmp_path: Path) -> None:
    # Exactly one legal move -> _apply_forced_moves skips NN/MCTS, no record.
    forced = "r5k1/p1P3pp/3QN1q1/3P1pB1/8/5P2/P6P/2n1q1K1 w - - 0 31"
    assert chess.Board(forced).legal_moves.count() == 1
    path = _write_list(tmp_path, [forced])
    with pytest.raises(ValueError, match="forced"):
        _load_fen_list(path)


def test_load_rejects_claim_draw_terminal(tmp_path: Path) -> None:
    # halfmove_clock >= 100: claimable 50-move draw; CBoard treats it as over,
    # so it must be rejected even though is_game_over() (no claim) returns False.
    fifty = "4k3/8/4K3/8/8/8/8/5R2 w - - 100 120"
    b = chess.Board(fifty)
    assert not b.is_game_over()
    assert b.is_game_over(claim_draw=True)
    path = _write_list(tmp_path, [fifty])
    with pytest.raises(ValueError, match="terminal position"):
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


def test_tb_adjudication_defers_virgin_fenlist_slot(monkeypatch) -> None:
    """A TB-eligible FEN seed must survive adjudication until a ply is played."""
    from types import SimpleNamespace
    from typing import Any, cast

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.selfplay import manager as mgr
    from chess_anti_engine.selfplay.state import SelfplayState

    fen = "8/8/8/4k3/8/3K4/4P3/8 w - - 0 1"  # KPvK: TB-eligible at ply 0
    board = chess.Board(fen)
    moved = chess.Board(fen)
    moved.push_uci("e2e4")

    st = cast(Any, object.__new__(SelfplayState))
    st.batch_size = 3
    st.done_arr = np.zeros(3, dtype=np.int8)
    st.finalized_arr = np.zeros(3, dtype=np.int8)
    st.tb_result_arr = [None, None, None]
    st.tb_adj_roll_arr = np.ones(3, dtype=np.int8)
    st.pending_sf_moves = {}
    # slot 0: virgin fenlist start -> deferred; slot 1: same position from a
    # book -> adjudicated; slot 2: fenlist AFTER one ply -> adjudicated.
    st.cboards = [
        CBoard.from_board(board), CBoard.from_board(board), CBoard.from_board(moved),
    ]
    st.opening_source_arr = ["fenlist", "book1", "fenlist"]
    st.starting_boards = [board.copy(), board.copy(), board.copy()]
    st.tb_probe = SimpleNamespace(max_pieces=6)
    st.game = SimpleNamespace(syzygy_path="/nonexistent-tb-path")
    monkeypatch.setattr(mgr, "tb_adjudicate_result", lambda _b, _p: "1/2-1/2")

    assert mgr._tb_adjudicate_active_games(st) == 2
    assert list(st.done_arr) == [0, 1, 1]


def test_production_seed_asset_loads() -> None:
    # 76 = panel v2 minus the 35 held-out v1 rows, minus 2 forced-move seeds
    # curated out. v1 stays a pure generalization yardstick.
    asset = Path(__file__).resolve().parents[1] / "data" / "blindspot_fens_v1.txt"
    if not asset.exists():
        pytest.skip("seed asset not present in this checkout")
    fens = _load_fen_list(str(asset))
    assert len(fens) == len(set(fens)) == 76
