"""FEN-list opening loader for the paired arena (scripts/arena_standard.py).

The loader reuses selfplay's _load_fen_list, so illegal/terminal/forced FENs
are SKIPPED with a warning (not played as phantom draws, not a hard abort) and
only a zero-usable-seed file fails fast — these tests pin that behavior plus
the arena-specific dedup/subsample/provenance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.selfplay.opening import _load_fen_list
from scripts.arena_standard import load_fen_openings, load_fen_seed_count

FEN_A = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FEN_B = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
FEN_C = "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
FEN_D = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5"
FEN_E = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
KINGLESS = "8/8/8/8/8/8/8/8 w - - 0 1"


@pytest.fixture(autouse=True)
def _clear_fen_cache() -> None:
    # _load_fen_list is lru_cached by path; clear so a reused tmp path never
    # returns a previous test's parse.
    _load_fen_list.cache_clear()


def _write(tmp_path: Path, lines: list[str], *, name: str = "fens.txt",
           encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


def test_loads_skips_comments_and_dedupes(tmp_path: Path) -> None:
    path = _write(tmp_path, ["# header", "", FEN_A, FEN_B, FEN_A])
    boards = load_fen_openings(path, n_pairs=10, rng=np.random.default_rng(0))
    assert [b.fen() for b in boards] == [FEN_A, FEN_B]


def test_illegal_fen_is_skipped_not_fatal(tmp_path: Path) -> None:
    # python-chess parses a kingless board without raising; the shared validator
    # rejects it (is_valid()=False), so it must be skipped, not played.
    path = _write(tmp_path, [FEN_A, KINGLESS, FEN_B])
    boards = load_fen_openings(path, n_pairs=10, rng=np.random.default_rng(0))
    assert [b.fen() for b in boards] == [FEN_A, FEN_B]


def test_halfmove_distinct_rows_both_kept(tmp_path: Path) -> None:
    # Same placement/side/castling/ep, different halfmove clock => distinct test
    # cases; dedup on the full (normalized) FEN must keep both.
    near_draw = FEN_A.replace(" 3 3", " 98 60")
    path = _write(tmp_path, [FEN_A, near_draw])
    boards = load_fen_openings(path, n_pairs=10, rng=np.random.default_rng(0))
    assert len(boards) == 2
    assert {b.fen() for b in boards} == {FEN_A, near_draw}


def test_ep_normalization_equivalents_collapse(tmp_path: Path) -> None:
    # Two spellings that python-chess normalizes to the SAME board (a redundant
    # en-passant square that drops to '-') must dedup to one board, else the
    # position is played twice and double-counted in the pentanomial.
    import chess

    with_ep = FEN_E  # "... w KQkq c6 0 2" — no legal ep capture
    normalized = chess.Board(with_ep).fen()  # "... w KQkq - 0 2"
    assert normalized != with_ep
    path = _write(tmp_path, [with_ep, normalized])
    boards = load_fen_openings(path, n_pairs=10, rng=np.random.default_rng(0))
    assert len(boards) == 1
    assert load_fen_seed_count(path) == 1


def test_seed_selects_which_subset(tmp_path: Path) -> None:
    import chess

    path = _write(tmp_path, [FEN_A, FEN_B, FEN_C, FEN_D, FEN_E])
    picks = {
        seed: tuple(b.fen() for b in load_fen_openings(
            path, n_pairs=2, rng=np.random.default_rng(seed)))
        for seed in range(6)
    }
    # python-chess normalizes some inputs (e.g. a redundant ep square), so the
    # membership set is built from normalized fens, not the raw input strings.
    all_fens = {chess.Board(f).fen() for f in (FEN_A, FEN_B, FEN_C, FEN_D, FEN_E)}
    for chosen in picks.values():
        assert len(chosen) == 2
        assert set(chosen) <= all_fens
    # Same seed reproduces; different seeds do not all pick the same subset.
    again = tuple(b.fen() for b in load_fen_openings(
        path, n_pairs=2, rng=np.random.default_rng(3)))
    assert again == picks[3]
    assert len(set(picks.values())) > 1


def test_uses_all_rows_when_fewer_than_pairs(tmp_path: Path) -> None:
    path = _write(tmp_path, [FEN_A])
    boards = load_fen_openings(path, n_pairs=5, rng=np.random.default_rng(0))
    assert len(boards) == 1


def test_bom_first_line_is_tolerated(tmp_path: Path) -> None:
    path = _write(tmp_path, [FEN_A, FEN_B], encoding="utf-8-sig")  # writes a BOM
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    boards = load_fen_openings(path, n_pairs=10, rng=np.random.default_rng(0))
    assert [b.fen() for b in boards] == [FEN_A, FEN_B]


def test_zero_usable_seeds_fails_fast(tmp_path: Path) -> None:
    path = _write(tmp_path, ["# only comments", KINGLESS])
    with pytest.raises(SystemExit, match="no usable seeds"):
        load_fen_openings(path, n_pairs=2, rng=np.random.default_rng(0))


def test_seed_count_matches_loaded(tmp_path: Path) -> None:
    path = _write(tmp_path, [FEN_A, KINGLESS, FEN_B, FEN_C])
    assert load_fen_seed_count(path) == 3  # KINGLESS skipped
