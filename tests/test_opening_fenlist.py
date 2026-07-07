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


# Unusable seed lines are SKIPPED (with a warning), not raised per-line, so one
# bad line in a hand-edited file can't crash the worker fleet (Codex review). We
# raise only when NOTHING usable remains. Each bad type is paired with a good FEN
# to assert it is dropped but the good one survives.
_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # fool's mate, terminal
_NO_KING = "8/8/8/8/8/8/8/4K3 w - - 0 1"  # illegal: black has no king
_FORCED = "r5k1/p1P3pp/3QN1q1/3P1pB1/8/5P2/P6P/2n1q1K1 w - - 0 31"  # 1 legal move
_FIFTY = "4k3/8/4K3/8/8/8/8/5R2 w - - 100 120"  # claimable 50-move draw


@pytest.mark.parametrize("bad", ["not a fen", _NO_KING, _FORCED, _FIFTY, _MATE])
def test_load_skips_unusable_lines_keeps_good(tmp_path: Path, bad: str) -> None:
    path = _write_list(tmp_path, [FEN_BLACK, bad, FEN_WHITE])
    assert _load_fen_list(path) == (FEN_BLACK, FEN_WHITE)


def test_load_tolerates_utf8_bom(tmp_path: Path) -> None:
    # A BOM-prefixed file (common from Windows editors) must not corrupt the
    # first line — the utf-8-sig read strips it.
    p = tmp_path / "fens_bom.txt"
    p.write_text("\n".join([FEN_BLACK, FEN_WHITE]) + "\n", encoding="utf-8-sig")
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")
    assert _load_fen_list(str(p)) == (FEN_BLACK, FEN_WHITE)


def test_claim_draw_fen_is_the_reason_case() -> None:
    # Guards the specific CBoard-vs-python-chess mismatch: is_game_over() alone
    # (no claim) misses halfmove>=100, which CBoard treats as over.
    b = chess.Board(_FIFTY)
    assert not b.is_game_over()
    assert b.is_game_over(claim_draw=True)


def test_load_raises_when_nothing_usable(tmp_path: Path) -> None:
    # All-bad / all-comment / empty file must fail fast at load (warm-up), not
    # mid-run: _sample_fen_list has no empty guard, so a silent empty tuple would
    # crash the selfplay thread on first sample.
    for lines in ([_NO_KING, _FORCED], ["# only comments", ""], []):
        path = _write_list(tmp_path, lines) if lines else _write_list(tmp_path, ["# x"])
        with pytest.raises(ValueError, match="no usable seeds"):
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
    moved.push_uci("e2e4")  # net (white) moves
    moved.push_uci("e5e6")  # opponent replies -> 2 plies played, net has decided

    st = cast(Any, object.__new__(SelfplayState))
    st.batch_size = 3
    st.done_arr = np.zeros(3, dtype=np.int8)
    st.finalized_arr = np.zeros(3, dtype=np.int8)
    st.tb_result_arr = [None, None, None]
    st.tb_adj_roll_arr = np.ones(3, dtype=np.int8)
    st.pending_sf_moves = {}
    # slot 0: virgin fenlist (0 plies) -> deferred; slot 1: same position from a
    # book -> adjudicated (not protected); slot 2: fenlist after 2 plies (net has
    # moved) -> adjudicated. The guard defers while <2 plies are played so it
    # covers the opponent-replies-first seating too.
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


def test_fenlist_curriculum_excluded_from_pid_winrate() -> None:
    """A blind-spot seed is a systematic loss; it must not enter the PID winrate
    sample (state.stats.w/d/l) or it eases SF across the whole batch — but its
    outcome must still be visible in per-source telemetry (Codex review [1])."""
    from types import SimpleNamespace
    from typing import Any, cast

    from chess_anti_engine.selfplay.finalize import _update_aggregate_stats
    from chess_anti_engine.selfplay.state import SelfplayState

    def mk(source: str) -> Any:
        st = cast(Any, object.__new__(SelfplayState))
        st.selfplay_arr = np.array([0], dtype=np.int8)   # curriculum slot
        st.net_color_arr = np.array([1], dtype=np.int8)  # net plays white
        st.opening_source_arr = [source]
        st.stats = SimpleNamespace(
            w=0, d=0, l=0, plies_win=0, plies_draw=0, plies_loss=0,
            selfplay_games=0, curriculum_games=0,
            selfplay_adjudicated_games=0, curriculum_adjudicated_games=0,
            total_draw_games=0, selfplay_draw_games=0, curriculum_draw_games=0,
            outcome_stats={},
        )
        return st

    # net (white) loses -> "0-1"
    book = mk("book1")
    _update_aggregate_stats(book, 0, result="0-1", was_adjudicated=False, game_plies=40)
    assert book.stats.l == 1  # a normal curriculum loss DOES feed the PID sample

    fen = mk("fenlist")
    _update_aggregate_stats(fen, 0, result="0-1", was_adjudicated=False, game_plies=40)
    assert fen.stats.l == 0            # seed loss EXCLUDED from the PID sample
    assert fen.stats.plies_loss == 0   # and from plies accounting
    assert fen.stats.curriculum_games == 1  # still counted as a game played
    assert fen.stats.outcome_stats.get("curriculum_fenlist_l") == 1  # telemetry kept


def test_production_seed_asset_loads() -> None:
    # 76 = panel v2 minus the 35 held-out v1 rows, minus 2 forced-move seeds
    # curated out. v1 stays a pure generalization yardstick.
    asset = Path(__file__).resolve().parents[1] / "data" / "blindspot_fens_v1.txt"
    if not asset.exists():
        pytest.skip("seed asset not present in this checkout")
    fens = _load_fen_list(str(asset))
    assert len(fens) == len(set(fens)) == 76
