"""Tests for arena PGN output and its use in a pooled rating fit.

The standing bias for this file: the defect to catch is a value that is
accepted and then silently ignored — a tag that never reaches the file, a pair
id that does not survive a round trip, a "default off" that is not off.
"""

from __future__ import annotations

import io
from pathlib import Path

import chess
import chess.pgn
import pytest

from chess_anti_engine.eval.arena_pgn import (
    ArenaGame,
    ArenaPgnWriter,
    engine_name_from_checkpoint,
    read_pair_blocks,
    sanitize_engine_name,
)

E4E5 = ("e2e4", "e7e5", "g1f3")


def _moves(board: chess.Board, ucis: tuple[str, ...]) -> tuple[chess.Move, ...]:
    out = []
    b = board.copy()
    for u in ucis:
        m = chess.Move.from_uci(u)
        out.append(m)
        b.push(m)
    return tuple(out)


def test_sanitize_engine_name_strips_ordo_breaking_chars() -> None:
    # Ordo selects the anchor by exact name and prints whitespace columns, so a
    # space or quote in a name breaks the parse AND the -A match.
    assert sanitize_engine_name("arm A/ckpt 99") == "arm_A_ckpt_99".replace(" ", "_")
    assert " " not in sanitize_engine_name('we"ird name')
    assert sanitize_engine_name("!!!") == "unknown"
    assert sanitize_engine_name("arm_A-1.2+x") == "arm_A-1.2+x"


def test_engine_name_uses_two_components_and_leaks_no_path() -> None:
    # Deliberately a synthetic absolute path: this repo is public, so a real
    # one must not be committed even in a fixture.
    name = engine_name_from_checkpoint(
        "/abs/path/to/repo/scratchpad/tier13/banked/arm_A_iter100/"
        "checkpoint_000099",
        fallback="candidate",
    )
    assert name == "arm_A_iter100_checkpoint_000099"
    assert "repo" not in name
    assert "/" not in name


def test_engine_name_distinguishes_arms_with_identical_basenames() -> None:
    # Every arm has a checkpoint_000099. Collapsing them into one player is the
    # contamination this default exists to prevent.
    a = engine_name_from_checkpoint("banked/arm_A_iter100/checkpoint_000099",
                                    fallback="c")
    b = engine_name_from_checkpoint("banked/arm_B_iter100/checkpoint_000099",
                                    fallback="c")
    assert a != b


def test_write_game_roundtrips_through_python_chess(tmp_path: Path) -> None:
    board = chess.Board()
    game = ArenaGame(
        white="armA", black="armB", result="1-0",
        moves=_moves(board, E4E5), pair_id=3, pair_half=0,
        extra={"WhiteSearch": "shape=training", "Termination": "rules"},
    )
    out = tmp_path / "a.pgn"
    with ArenaPgnWriter(out, event="tier13", base_tags={"ConfigHash": "abc123"}) as w:
        w.write_game(game)

    parsed = chess.pgn.read_game(io.StringIO(out.read_text()))
    assert parsed is not None
    assert parsed.headers["White"] == "armA"
    assert parsed.headers["Black"] == "armB"
    assert parsed.headers["Result"] == "1-0"
    assert parsed.headers["Event"] == "tier13"
    assert parsed.headers["ConfigHash"] == "abc123"
    assert parsed.headers["PairId"] == "3"
    assert parsed.headers["PairHalf"] == "0"
    assert parsed.headers["WhiteSearch"] == "shape=training"
    assert parsed.headers["Round"] == "4.1"  # 1-based pair.half
    assert [m.uci() for m in parsed.mainline_moves()] == list(E4E5)


def test_book_opening_writes_setup_and_fen(tmp_path: Path) -> None:
    # A game that starts from a book position must say so, or it cannot be
    # replayed and the position is silently attributed to the standard start.
    fen = "rnbqkbnr/pp2pppp/2p5/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    board = chess.Board(fen)
    game = ArenaGame(white="a", black="b", result="1/2-1/2",
                     moves=_moves(board, ("e4d5",)), start_fen=fen,
                     pair_id=0, pair_half=1)
    out = tmp_path / "b.pgn"
    with ArenaPgnWriter(out) as w:
        w.write_game(game)
    text = out.read_text()
    assert '[SetUp "1"]' in text
    assert f'[FEN "{fen}"]' in text
    parsed = chess.pgn.read_game(io.StringIO(text))
    assert parsed is not None
    assert parsed.board().fen() == fen


def test_standard_start_omits_fen_tag(tmp_path: Path) -> None:
    out = tmp_path / "c.pgn"
    with ArenaPgnWriter(out) as w:
        w.write_game(ArenaGame(white="a", black="b", result="0-1",
                               start_fen=chess.STARTING_FEN))
    assert "SetUp" not in out.read_text()


def test_each_game_is_flushed_so_a_partial_file_parses(tmp_path: Path) -> None:
    # This is what makes incremental use possible: a killed run must still leave
    # a valid PGN of the games that finished.
    out = tmp_path / "d.pgn"
    w = ArenaPgnWriter(out)
    try:
        for i in range(3):
            w.write_game(ArenaGame(white="a", black="b", result="1-0",
                                   pair_id=i, pair_half=0))
            # Read the file WITHOUT closing the writer.
            parsed = list(_iter_games(out.read_text()))
            assert len(parsed) == i + 1, "game not visible before close"
    finally:
        w.close()


def _iter_games(text: str):
    fh = io.StringIO(text)
    while True:
        g = chess.pgn.read_game(fh)
        if g is None:
            return
        yield g


def test_appends_rather_than_truncates(tmp_path: Path) -> None:
    out = tmp_path / "e.pgn"
    with ArenaPgnWriter(out) as w:
        w.write_game(ArenaGame(white="a", black="b", result="1-0"))
    with ArenaPgnWriter(out) as w:
        w.write_game(ArenaGame(white="a", black="b", result="0-1"))
    assert len(list(_iter_games(out.read_text()))) == 2


def test_rejects_bad_result() -> None:
    with pytest.raises(ValueError, match="result must be one of"):
        ArenaGame(white="a", black="b", result="win")


def test_read_pair_blocks_groups_mirrored_games(tmp_path: Path) -> None:
    out = tmp_path / "f.pgn"
    with ArenaPgnWriter(out) as w:
        for pid in range(2):
            w.write_game(ArenaGame(white="a", black="b", result="1-0",
                                   pair_id=pid, pair_half=0))
            w.write_game(ArenaGame(white="b", black="a", result="1-0",
                                   pair_id=pid, pair_half=1))
    blocks = dict(read_pair_blocks(out))
    assert len(blocks) == 2
    assert all(len(v) == 2 for v in blocks.values())


def test_read_pair_blocks_does_not_merge_across_matchups(tmp_path: Path) -> None:
    # Two matchups both numbering pairs from 0 must NOT collapse into one block.
    out = tmp_path / "g.pgn"
    with ArenaPgnWriter(out) as w:
        w.write_game(ArenaGame(white="a", black="b", result="1-0",
                               pair_id=0, pair_half=0))
        w.write_game(ArenaGame(white="c", black="d", result="1-0",
                               pair_id=0, pair_half=0))
    blocks = dict(read_pair_blocks(out))
    assert len(blocks) == 2
