"""External matches must play the declared history and engine configuration."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import chess
import chess.engine
import chess.pgn
import pytest

from chess_anti_engine.utils.game_log import GameLogWriter, read_game_log
from scripts import match_vs_uci as match


def _as_engine(value: object) -> chess.engine.SimpleEngine:
    return cast(chess.engine.SimpleEngine, value)


def _opening(moves: str) -> chess.Board:
    board = match._parse_opening_line(f"{chess.STARTING_FEN} | {moves}")
    assert board is not None
    return board


def test_opening_file_replays_history_and_preserves_comment_and_offset(tmp_path: Path) -> None:
    path = tmp_path / "book.fen"
    path.write_text(f"# book\n{chess.STARTING_FEN}\n{chess.STARTING_FEN} | f2f3 e7e5 # pair\n")
    boards = match._load_openings(path, offset=1, limit=1)
    assert len(boards) == 1
    assert [m.uci() for m in boards[0].move_stack] == ["f2f3", "e7e5"]
    assert boards[0].root().fen() == chess.STARTING_FEN
    assert boards[0].fen() != chess.STARTING_FEN


@pytest.mark.parametrize("moves", ["e2e5", "0000", "e2e4 e7e5 e4e5", "not-a-move"])
def test_bad_history_refuses_the_panel_instead_of_playing_root(tmp_path: Path, moves: str) -> None:
    path = tmp_path / "book.fen"
    path.write_text(f"# book\n{chess.STARTING_FEN} | {moves}\n")
    with pytest.raises(ValueError, match=r"book.fen:2: invalid opening history"):
        match._load_openings(path)


def test_bad_plain_row_is_not_silently_omitted(tmp_path: Path) -> None:
    path = tmp_path / "book.fen"
    path.write_text(f"{chess.STARTING_FEN}\nnot a position\n")
    with pytest.raises(ValueError, match=r"book.fen:2: invalid opening"):
        match._load_openings(path)


def test_panel_identity_distinguishes_equal_fens_with_different_histories() -> None:
    a = _opening("g1f3 g8f6 f3g1 f6g8")
    b = _opening("b1c3 b8c6 c3b1 c6b8")
    assert a.fen() == b.fen()
    assert match._opening_panel_digest([a]) != match._opening_panel_digest([b])
    assert match._opening_panel_digest([a, b]) != match._opening_panel_digest([b, a])
    assert match._opening_panel_digest([a]) == match._opening_panel_digest([a.copy()])


def test_opening_repetition_remains_visible_before_any_engine_move(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _opening("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8")
    monkeypatch.setattr(match, "_play_move", lambda *_a, **_k: pytest.fail("lost repetition history"))
    record = match.play_one_game(
        _as_engine(None), _as_engine(None),
        limit_w=chess.engine.Limit(time=0.01), limit_b=chess.engine.Limit(time=0.01),
        enforce_nodes_w=False, enforce_nodes_b=False, max_plies=4, start_board=board,
        preserve_opening_history=True,
    )
    assert record.result == "1/2-1/2"
    assert record.termination == "rules"
    assert record.plies == 0
    assert len(record.start_board.move_stack) == len(board.move_stack) == 8


def test_play_and_pgn_preserve_book_while_budget_counts_additional_plies(monkeypatch: pytest.MonkeyPatch) -> None:
    opening = _opening("f2f3 e7e5")
    observed: list[list[str]] = []
    moves = iter(["g2g4", "d8h4"])

    def play(_engine: object, board: chess.Board, *_a: object, **_k: object) -> match.MoveResult:
        observed.append([m.uci() for m in board.move_stack])
        return match.MoveResult(chess.Move.from_uci(next(moves)), {})

    monkeypatch.setattr(match, "_play_move", play)
    record = match.play_one_game(
        _as_engine(None), _as_engine(None),
        limit_w=chess.engine.Limit(time=0.01), limit_b=chess.engine.Limit(time=0.01),
        enforce_nodes_w=False, enforce_nodes_b=False, max_plies=2, start_board=opening,
        preserve_opening_history=True,
    )
    assert record.result == "0-1"
    assert record.plies == 2
    assert [m.uci() for m in record.moves] == ["g2g4", "d8h4"]
    assert observed == [["f2f3", "e7e5"], ["f2f3", "e7e5", "g2g4"]]
    assert len(opening.move_stack) == 2
    text = io.StringIO()
    match._write_pgn_game(text, record=record, event="test", white_name="A", black_name="B", round_name="1")
    game = chess.pgn.read_game(io.StringIO(text.getvalue()))
    assert game is not None
    assert not game.errors
    assert game.headers["OpeningPlyCount"] == "2"
    assert [m.uci() for m in game.mainline_moves()] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert game.end().board().is_checkmate()


@pytest.mark.parametrize("stored_history", [None, ["b1c3", "b8c6", "c3b1", "c6b8"]])
def test_resume_refuses_missing_or_different_history_at_identical_fen(
    tmp_path: Path, stored_history: list[str] | None,
) -> None:
    opening = _opening("g1f3 g8f6 f3g1 f6g8")
    path = tmp_path / "games.jsonl"
    settings = {"games": 2}
    row: dict[str, Any] = {"game_index": 0, "a_is_white": True, "opening_index": 0,
           "opening_fen": opening.fen(), "result": "1/2-1/2", "score_a": 0.5}
    if stored_history is not None:
        row.update(opening_root_fen=chess.STARTING_FEN, opening_uci=stored_history)
    with GameLogWriter(path, driver="match_vs_uci", settings=settings) as writer:
        writer.write_game(row)
    with pytest.raises(SystemExit, match="opening history differs"):
        match.load_match_resume(path, settings=settings, openings=[opening])


def test_resume_of_pre_identity_log_refuses_with_restart_message(tmp_path: Path) -> None:
    """A log written before opening-history identity is still refused, but the
    message must not say "re-run without the changes" (impossible for keys the
    old driver never wrote)."""
    opening = _opening("g1f3 g8f6 f3g1 f6g8")
    current = {"games": 2, "opening_history_version": 1,
               "opening_panel_sha256": match._opening_panel_digest([opening])}
    legacy = {"games": 2}
    path = tmp_path / "games.jsonl"
    with GameLogWriter(path, driver="match_vs_uci", settings=legacy) as writer:
        writer.write_game({"game_index": 0, "a_is_white": True, "opening_index": 0,
                           "opening_fen": opening.fen(), "result": "1/2-1/2", "score_a": 0.5})
    with pytest.raises(SystemExit) as exc:
        match.load_match_resume(path, settings=current, openings=[opening])
    msg = str(exc.value)
    assert "predates opening-history identity" in msg
    assert "restart the match in a new log" in msg
    assert "opening_panel_sha256" in msg
    assert "Re-run without the changes" not in msg
    # A log that HAS the identity keys but differs keeps the generic refusal.
    other = tmp_path / "other.jsonl"
    with GameLogWriter(other, driver="match_vs_uci",
                       settings={**current, "opening_panel_sha256": "0" * 64}) as writer:
        writer.write_game({"game_index": 0, "a_is_white": True, "opening_index": 0,
                           "opening_fen": opening.fen(), "result": "1/2-1/2", "score_a": 0.5})
    with pytest.raises(SystemExit, match="DIFFERENT settings"):
        match.load_match_resume(other, settings=current, openings=[opening])


class _Engine:
    def __init__(self, options: dict[str, chess.engine.Option]) -> None:
        self.options = options
        self.configured: list[dict[str, chess.engine.ConfigValue]] = []
        self.id = {"name": "test"}
        self.timeout = 10.0
        self.closed = False
        self.quit_called = False

    def configure(self, values: dict[str, chess.engine.ConfigValue]) -> None:
        parsed = {key: self.options[key].parse(value) for key, value in values.items()}
        self.configured.append(parsed)

    def quit(self) -> None:
        self.quit_called = True

    def close(self) -> None:
        self.closed = True


def test_requested_threads_cannot_be_silently_ignored() -> None:
    engine = _Engine({"Hash": chess.engine.Option("Hash", "spin", 16, 1, 1024, None)})
    with pytest.raises(ValueError, match="Threads"):
        match._set_options(_as_engine(engine), {"Hash": "128", "Threads": "8"})
    assert engine.configured == []


@pytest.mark.parametrize("value", ["0", "17", "not-an-int"])
def test_invalid_threads_refuses_configuration(value: str) -> None:
    engine = _Engine({"Threads": chess.engine.Option("Threads", "spin", 1, 1, 16, None)})
    with pytest.raises((ValueError, chess.engine.EngineError)):
        match._set_options(_as_engine(engine), {"Threads": value})
    assert engine.configured == []


def test_typed_options_disable_strength_limit_explicitly() -> None:
    engine = _Engine({
        "Threads": chess.engine.Option("Threads", "spin", 1, 1, 16, None),
        "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", True, None, None, None),
    })
    match._set_options(_as_engine(engine), {"Threads": "8", "UCI_LimitStrength": "false"})
    assert engine.configured == [{"Threads": 8, "UCI_LimitStrength": False}]
    with pytest.raises(ValueError, match="invalid boolean"):
        match._set_options(_as_engine(engine), {"UCI_LimitStrength": "fales"})


def test_option_failure_closes_startup_owner_before_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _Engine({})
    monkeypatch.setattr(match, "_open_engine", lambda *_a, **_k: engine)
    monkeypatch.setattr(match, "_warmup_engine", lambda *_a, **_k: pytest.fail("unsupported options reached warmup"))
    with pytest.raises(ValueError, match="Threads"):
        match._open_warm_engine("test", {"Threads": "8"}, nodes=1, label="test", warmup_timeout_s=20)
    assert engine.closed
    assert engine.quit_called
    assert engine.timeout == match._ENGINE_COMMAND_TIMEOUT_S


@pytest.mark.parametrize("strip_history", [False, True])
def test_actual_match_log_requires_the_history_returned_by_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strip_history: bool,
) -> None:
    moves = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"
    book = tmp_path / "book.fen"
    book.write_text(f"{chess.STARTING_FEN} | {moves}\n")
    log = tmp_path / "games.jsonl"
    monkeypatch.setattr(match, "_open_warm_engine", lambda *_a, **_k: _Engine({}))

    def play(*, start_board: chess.Board, **_kwargs: object) -> match.GameRecord:
        return match.GameRecord("1/2-1/2", 0, start_board.copy(stack=not strip_history), (), ())

    monkeypatch.setattr(match, "_play_one_pairing", play)
    monkeypatch.setattr("sys.argv", ["match_vs_uci.py", "--engine-a", "fake-a", "--engine-b", "fake-b",
                                    "--games", "2", "--openings", str(book), "--games-out", str(log)])
    if strip_history:
        with pytest.raises(ValueError, match="did not preserve"):
            match.main()
        assert read_game_log(log).games == []
    else:
        match.main()
        saved = read_game_log(log)
        assert saved.settings["opening_panel_sha256"] == match._opening_panel_digest([_opening(moves)])
        assert len(saved.games) == 2
        for row in saved.games:
            assert row["opening_root_fen"] == chess.STARTING_FEN
            assert row["opening_uci"] == moves.split()
