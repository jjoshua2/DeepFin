"""``scripts/arena_openings_minus_screen.py``: confirmation openings = EVAL lines minus a screen's.

The load-bearing property is that the script's screen draw IS the arena's: the
first test runs ``arena_standard.run_arena`` itself (play stubbed) on a real
split's ``eval.pgn`` and compares the openings it would have played.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import pytest

from chess_anti_engine.simple_rl.openings import split_openings
from scripts import arena_openings_minus_screen as mod

SCREEN_SEED = 2026112003


def _book(path: Path, n: int = 60, plies: int = 8) -> Path:
    """n distinct random legal lines of ``plies`` plies, as PGN games."""
    rng = random.Random(5)
    seen: set[tuple[str, ...]] = set()
    games = []
    while len(games) < n:
        board = chess.Board()
        for _ in range(plies):
            board.push(rng.choice(list(board.legal_moves)))
        moves = tuple(m.uci() for m in board.move_stack)
        if board.is_game_over() or moves in seen:
            continue
        seen.add(moves)
        games.append(str(chess.pgn.Game.from_board(board)))
    path.write_text("\n\n".join(games) + "\n")
    return path


@pytest.fixture
def split(tmp_path: Path) -> Path:
    out = tmp_path / "openings"
    split_openings(books=[(_book(tmp_path / "book.pgn"), 8)], out_dir=out, eval_lines=30, salt="t")
    return out


def _arena_openings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, split: Path, *,
                    games: int, seed: int) -> list[chess.Board]:
    """The openings run_arena hands to its play loop, for an --openings eval.pgn screen."""
    import scripts.arena_standard as arena

    played: list[chess.Board] = []

    def _play(_cand: str, _ref: str, openings: list[chess.Board], **_kw: Any) -> list[float]:
        played.extend(b.copy() for b in openings)
        return [1.0] * len(openings)

    monkeypatch.setattr(arena, "play_paired_games_matched_time", _play)
    plies = int(json.loads((split / "summary.json").read_text())["eval_max_plies"])
    arena.run_arena(
        candidate="cand.pt", reference="ref.pt", games=games,
        openings_path=split / "eval.pgn", openings_fen=None, opening_plies=plies, mode="matched_time",
        sims_candidate=32, sims_reference=32, ms_per_move=100, max_plies=300, temperature=0.1,
        gumbel_add_noise=True, device="cpu", seed=seed, out_path=None,
        game_log_path=tmp_path / "screen.games.jsonl", resume=False, pgn_out=None,
    )
    return played


def test_the_screen_draw_is_the_arenas_own(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, split: Path) -> None:
    played = _arena_openings(monkeypatch, tmp_path, split, games=20, seed=SCREEN_SEED)
    plies = int(json.loads((split / "summary.json").read_text())["eval_max_plies"])
    ours = mod.screen_openings(split / "eval.pgn", games=20, seed=SCREEN_SEED, opening_plies=plies)
    assert len(played) == 10
    assert [mod.opening_key(b) for b in ours] == [mod.opening_key(b) for b in played]


def test_the_output_is_eval_minus_screen_with_full_history(tmp_path: Path, split: Path) -> None:
    out = tmp_path / "conf.fen"
    report = mod.build(split, out, screen_games=20, screen_seed=SCREEN_SEED, check_games=20, check_seed=5)
    full = mod.eval_lines(split / "eval.jsonl")
    plies = report["opening_plies"]
    screen = {mod.opening_key(b) for b in mod.screen_openings(split / "eval.pgn", games=20, seed=SCREEN_SEED,
                                                              opening_plies=plies)}
    rows = out.read_text().splitlines()
    assert report["eval_lines"] == len(full) == 30
    assert report["confirmation_rows"] == len(rows) == len(full) - len(screen)
    assert set(rows) == {line for key, line in full.items() if key not in screen}
    assert all(" | " in row for row in rows), "full-history rows, never bare FENs"
    assert report["check_pairs"] == report["check_with_history"] == 10
    assert report["check_overlap_with_screen"] == 0
    import hashlib
    assert report["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_screen_openings_cut_short_of_the_eval_lines_are_refused(tmp_path: Path, split: Path) -> None:
    """--opening-plies below the lines' length: the screen boards match no EVAL line."""
    with pytest.raises(mod.Refusal) as exc:
        mod.build(split, tmp_path / "conf.fen", screen_games=20, screen_seed=SCREEN_SEED, opening_plies=4)
    assert "match no EVAL line" in exc.value.message
    assert not (tmp_path / "conf.fen").exists()


def test_a_confirmation_larger_than_the_remaining_lines_is_refused_and_removed(
    tmp_path: Path, split: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "conf.fen"
    rc = mod.main(["--openings-dir", str(split), "--screen-games", "20", "--screen-seed", str(SCREEN_SEED),
                   "--check-games", "60", "--check-seed", "5", "--out", str(out)])
    assert rc == 2
    assert "pairs loadable, 30 needed" in capsys.readouterr().err
    assert not out.exists(), "a file that fails the check must not be left behind"
