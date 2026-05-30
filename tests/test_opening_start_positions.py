import zipfile

import numpy as np
import pytest

from chess_anti_engine.selfplay.opening import (
    OpeningConfig,
    _load_pgn_opening_sequences,
    make_starting_board,
    sample_starting_board,
    warm_opening_book_cache,
)


def _write_test_pgn(path):
    # Two short games with distinct first moves.
    pgn = """[Event \"Test1\"]
[Site \"?\"]
[Date \"2026.02.21\"]
[Round \"-\"]
[White \"W\"]
[Black \"B\"]
[Result \"*\"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *

[Event \"Test2\"]
[Site \"?\"]
[Date \"2026.02.21\"]
[Round \"-\"]
[White \"W\"]
[Black \"B\"]
[Result \"*\"]

1. d4 d5 2. c4 e6 *
"""
    path.write_text(pgn, encoding="utf-8")


def test_make_starting_board_random_plies():
    rng = np.random.default_rng(0)
    cfg = OpeningConfig(opening_book_path=None, random_start_plies=3)
    b = make_starting_board(rng=rng, cfg=cfg)
    assert len(b.move_stack) == 3
    assert b.is_valid()


def test_sample_starting_board_reports_random_source():
    rng = np.random.default_rng(0)
    cfg = OpeningConfig(opening_book_path=None, random_start_plies=3)
    start = sample_starting_board(rng=rng, cfg=cfg)
    assert start.source == "random"
    assert len(start.board.move_stack) == 3
    assert start.board.is_valid()


def test_make_starting_board_pgn(tmp_path):
    pgn_path = tmp_path / "book.pgn"
    _write_test_pgn(pgn_path)

    rng = np.random.default_rng(123)
    cfg = OpeningConfig(
        opening_book_path=str(pgn_path),
        opening_book_prob=1.0,
        opening_book_max_plies=4,
        opening_book_max_games=10,
        random_start_plies=0,
    )
    b = make_starting_board(rng=rng, cfg=cfg)
    assert 1 <= len(b.move_stack) <= 4
    assert b.move_stack[0].uci() in {"e2e4", "d2d4"}


def test_sample_starting_board_reports_book_source(tmp_path):
    pgn_path = tmp_path / "book.pgn"
    _write_test_pgn(pgn_path)

    rng = np.random.default_rng(123)
    cfg = OpeningConfig(
        opening_book_path=str(pgn_path),
        opening_book_prob=1.0,
        opening_book_max_plies=4,
        opening_book_max_games=10,
        random_start_plies=0,
    )
    start = sample_starting_board(rng=rng, cfg=cfg)
    assert start.source == "book1"
    assert 1 <= len(start.board.move_stack) <= 4


def test_make_starting_board_pgn_zip(tmp_path):
    pgn_path = tmp_path / "book.pgn"
    _write_test_pgn(pgn_path)

    zip_path = tmp_path / "book.pgn.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("book.pgn", pgn_path.read_text(encoding="utf-8"))

    rng = np.random.default_rng(7)
    cfg = OpeningConfig(
        opening_book_path=str(zip_path),
        opening_book_prob=1.0,
        opening_book_max_plies=2,
        opening_book_max_games=10,
        random_start_plies=0,
    )
    b = make_starting_board(rng=rng, cfg=cfg)
    assert 1 <= len(b.move_stack) <= 2
    assert b.move_stack[0].uci() in {"e2e4", "d2d4"}


def test_warm_opening_book_cache_preloads_pgn_books(tmp_path):
    pgn_path = tmp_path / "book.pgn"
    _write_test_pgn(pgn_path)

    _load_pgn_opening_sequences.cache_clear()
    cfg = OpeningConfig(
        opening_book_path=str(pgn_path),
        opening_book_prob=1.0,
        opening_book_max_plies=4,
        opening_book_max_games=10,
    )

    warm_opening_book_cache(cfg)

    info = _load_pgn_opening_sequences.cache_info()
    assert info.currsize == 1
    assert info.misses == 1


def test_opening_book_prob_zero_falls_back_to_random(tmp_path):
    pgn_path = tmp_path / "book.pgn"
    _write_test_pgn(pgn_path)

    rng = np.random.default_rng(0)
    cfg = OpeningConfig(
        opening_book_path=str(pgn_path),
        opening_book_prob=0.0,
        random_start_plies=1,
    )
    b = make_starting_board(rng=rng, cfg=cfg)
    assert len(b.move_stack) == 1
    assert b.is_valid()


def test_selected_empty_opening_book_raises(tmp_path):
    empty_pgn = tmp_path / "empty.pgn"
    empty_pgn.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="produced no usable opening moves"):
        make_starting_board(
            rng=np.random.default_rng(0),
            cfg=OpeningConfig(
                opening_book_path=str(empty_pgn),
                opening_book_prob=1.0,
                random_start_plies=2,
            ),
        )


def test_selected_secondary_empty_opening_book_raises(tmp_path):
    pgn_path = tmp_path / "book1.pgn"
    _write_test_pgn(pgn_path)
    empty_pgn = tmp_path / "book2.pgn"
    empty_pgn.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="produced no usable opening moves"):
        make_starting_board(
            rng=np.random.default_rng(0),
            cfg=OpeningConfig(
                opening_book_path=str(pgn_path),
                opening_book_prob=1.0,
                opening_book_path_2=str(empty_pgn),
                opening_book_mix_prob_2=1.0,
                random_start_plies=2,
            ),
        )
