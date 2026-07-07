"""FEN-list opening loader for the paired arena (scripts/arena_standard.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.arena_standard import load_fen_openings

FEN_A = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FEN_B = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
FEN_C = "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "fens.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_loads_skips_comments_and_dedupes(tmp_path: Path) -> None:
    path = _write(tmp_path, ["# header", "", FEN_A, FEN_B, FEN_A])
    boards = load_fen_openings(path, n_pairs=10, rng=np.random.default_rng(0))
    assert [b.fen() for b in boards] == [FEN_A, FEN_B]


def test_seeded_subsample_when_more_rows_than_pairs(tmp_path: Path) -> None:
    path = _write(tmp_path, [FEN_A, FEN_B, FEN_C])
    first = load_fen_openings(path, n_pairs=2, rng=np.random.default_rng(7))
    again = load_fen_openings(path, n_pairs=2, rng=np.random.default_rng(7))
    assert len(first) == 2
    assert [b.fen() for b in first] == [b.fen() for b in again]


def test_uses_all_rows_when_fewer_than_pairs(tmp_path: Path) -> None:
    path = _write(tmp_path, [FEN_A])
    boards = load_fen_openings(path, n_pairs=5, rng=np.random.default_rng(0))
    assert len(boards) == 1


def test_invalid_fen_fails_fast_with_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path, [FEN_A, "not a fen"])
    with pytest.raises(SystemExit, match=r":2: invalid FEN"):
        load_fen_openings(path, n_pairs=2, rng=np.random.default_rng(0))


def test_empty_file_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, ["# only comments"])
    with pytest.raises(SystemExit, match="no FEN rows"):
        load_fen_openings(path, n_pairs=2, rng=np.random.default_rng(0))
