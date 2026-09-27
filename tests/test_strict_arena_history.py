"""Tiny synthetic games only; no engine, GPU, book, or tablebase files."""

from __future__ import annotations

import json
import io
import copy
from pathlib import Path

import chess
import chess.pgn
import pytest

from chess_anti_engine.eval.arena_pgn import ArenaGame, ArenaPgnWriter
from chess_anti_engine.eval.strict_arena_history import (
    SYZYGY_PROTOCOL,
    HistoryRefusal,
    verify_strict_arena_history,
)
from chess_anti_engine.utils.game_log import GameLogWriter


SETTINGS = {
    "mode": "matched_sims", "seed": 7, "games": 2,
    "syzygy": "/injected/strict-six-man",
    "syzygy_max_pieces": 6, "syzygy_protocol": SYZYGY_PROTOCOL,
}
TAGS = {
    "ConfigHash": "synthetic-config", "GitSha": "synthetic-source",
    "ArenaMode": "matched_sims", "SyzygyProtocol": SYZYGY_PROTOCOL,
    "SyzygyMaxPieces": "6",
}


class FakeWdlDtz:
    def __init__(self, wdl: int = 2, dtz: int = 7, *, fail_dtz: bool = False) -> None:
        self.wdl = wdl
        self.dtz = dtz
        self.fail_dtz = fail_dtz
        self.wdl_calls = 0
        self.dtz_calls = 0

    def probe_wdl(self, board: chess.Board) -> int:
        assert isinstance(board, chess.Board)
        self.wdl_calls += 1
        return self.wdl

    def probe_dtz(self, board: chess.Board) -> int:
        assert isinstance(board, chess.Board)
        self.dtz_calls += 1
        if self.fail_dtz:
            raise KeyError("missing DTZ")
        return self.dtz


def _knight_opening() -> chess.Board:
    board = chess.Board()
    for uci in (
        "a2a3", "a7a6", "b2b3", "b7b6", "c2c3", "c7c6",
        "d2d3", "d7d6", "e2e3", "e7e6", "h2h3", "h7h6",
        "g1f3", "g8f6", "f3g1", "f6g8",
    ):
        board.push_uci(uci)
    assert len(board.move_stack) == 16
    return board


def _capture(
    tmp_path: Path, opening: chess.Board, moves: tuple[str, ...],
    result: str, termination: str, *,
    mutate_row: dict | None = None, mutate_pgn: dict | None = None,
    settings: dict | None = None,
) -> tuple[bytes, bytes]:
    log_path = tmp_path / "games.jsonl"
    pgn_path = tmp_path / "games.pgn"
    log_path.unlink(missing_ok=True)
    pgn_path.unlink(missing_ok=True)
    config = SETTINGS if settings is None else settings
    played = chess.Board(opening.fen())
    move_objects = tuple(chess.Move.from_uci(uci) for uci in moves)
    for move in move_objects:
        assert move in played.legal_moves
        played.push(move)
    root_fen = opening.root().fen()
    stack = [move.uci() for move in opening.move_stack]
    with (
        GameLogWriter(log_path, driver="arena_standard", settings=config) as log,
        ArenaPgnWriter(pgn_path, base_tags=TAGS) as pgn,
    ):
        for half in (0, 1):
            candidate_white = half == 0
            score = 0.5 if result == "1/2-1/2" else float(
                (result == "1-0") == candidate_white
            )
            row = {
                "pair_id": 0, "half": half, "opening_index": 0,
                "a_is_white": candidate_white, "opening_fen": opening.fen(),
                "start_fen": opening.fen(), "opening_root_fen": root_fen,
                "opening_uci": stack, "seed": 7, "loop": "chunked",
                "result": result, "score_candidate": score,
                "termination": termination, "plies": len(moves),
                "compile": "off", "eval_hoist": "off",
            }
            if mutate_row:
                row.update(mutate_row)
            extra = {
                "OpeningRootFEN": root_fen,
                "OpeningUCI": json.dumps(stack, separators=(",", ":")),
                "Termination": termination, "Plies": str(len(moves)),
                "WhiteSearch": "cand-search" if candidate_white else "ref-search",
                "BlackSearch": "ref-search" if candidate_white else "cand-search",
                "EvaluatorHoist": "off",
            }
            if mutate_pgn:
                extra.update(mutate_pgn)
            pgn.write_game(ArenaGame(
                white="candidate" if candidate_white else "reference",
                black="reference" if candidate_white else "candidate",
                result=result, moves=move_objects, start_fen=opening.fen(),
                pair_id=0, pair_half=half, extra=extra,
            ))
            log.write_game(row)
    return log_path.read_bytes(), pgn_path.read_bytes()


def _verify(log: bytes, pgn: bytes, opening: chess.Board, tablebase: FakeWdlDtz):
    return verify_strict_arena_history(
        log, pgn, expected_settings=SETTINGS, expected_openings={0: opening},
        expected_pgn_tags=TAGS, candidate_name="candidate",
        reference_name="reference", candidate_search="cand-search",
        reference_search="ref-search", tablebase=tablebase,
        expected_opening_root_fen=opening.root().fen(),
        expected_opening_plies=len(opening.move_stack),
    )


def test_knight_cycle_claim_replays_preplay_history(tmp_path: Path) -> None:
    opening = _knight_opening()
    moves = ("g1f3", "g8f6", "f3g1")
    full = opening.copy()
    for uci in moves:
        full.push_uci(uci)
    assert full.result(claim_draw=True) == "1/2-1/2"
    stripped = chess.Board(opening.fen())
    for uci in moves:
        stripped.push_uci(uci)
    assert stripped.outcome(claim_draw=True) is None
    log, pgn = _capture(tmp_path, opening, moves, "1/2-1/2", "rules")
    receipt = _verify(log, pgn, opening, FakeWdlDtz())
    assert (receipt.games, receipt.rules_games, receipt.syzygy_games) == (2, 2, 0)
    assert len(receipt.jsonl_sha256) == len(receipt.pgn_sha256) == 64


def test_same_opening_and_result_cannot_hide_wrong_pgn_moves(tmp_path: Path) -> None:
    opening = _knight_opening()
    log, pgn = _capture(
        tmp_path, opening, ("g1f3", "g8f6", "f3d4"), "1/2-1/2", "rules",
    )
    with pytest.raises(HistoryRefusal, match="natural result differs"):
        _verify(log, pgn, opening, FakeWdlDtz())


def test_rule50_claim_and_early_stop(tmp_path: Path) -> None:
    opening = chess.Board("8/8/8/8/8/8/4k3/6KR w - - 98 1")
    assert opening.outcome(claim_draw=True) is None
    log, pgn = _capture(tmp_path, opening, ("h1h2",), "1/2-1/2", "rules")
    receipt = _verify(log, pgn, opening, FakeWdlDtz())
    assert receipt.rules_games == 2
    bad_log, bad_pgn = _capture(tmp_path, opening, ("h1h2", "e2e3"), "1/2-1/2", "rules")
    with pytest.raises(HistoryRefusal, match="played after an adjudicable state"):
        _verify(bad_log, bad_pgn, opening, FakeWdlDtz())


def test_syzygy_uses_both_wdl_and_dtz_and_rule50(tmp_path: Path) -> None:
    opening = chess.Board("8/8/8/8/8/8/4k3/6KR w - - 0 1")
    assert opening.outcome(claim_draw=True) is None
    log, pgn = _capture(tmp_path, opening, (), "1-0", "syzygy")
    probe = FakeWdlDtz()
    receipt = _verify(log, pgn, opening, probe)
    assert (receipt.rules_games, receipt.syzygy_games) == (0, 2)
    assert probe.wdl_calls == probe.dtz_calls == 2
    with pytest.raises(HistoryRefusal, match="missing eligible WDL/DTZ"):
        _verify(log, pgn, opening, FakeWdlDtz(fail_dtz=True))
    with pytest.raises(HistoryRefusal, match="magnitude mismatch"):
        _verify(log, pgn, opening, FakeWdlDtz(dtz=101))


@pytest.mark.parametrize("kind", ["matched_time", "missing_json_history", "wrong_pgn_history", "censored"])
def test_refuses_missing_or_mismatched_authority(tmp_path: Path, kind: str) -> None:
    opening = _knight_opening()
    moves = ("g1f3", "g8f6", "f3g1")
    row = {"opening_uci": None} if kind == "missing_json_history" else None
    pgn_tags = {"OpeningUCI": "[]"} if kind == "wrong_pgn_history" else None
    termination = "max_plies" if kind == "censored" else "rules"
    log, pgn = _capture(
        tmp_path, opening, moves, "1/2-1/2", termination,
        mutate_row=row, mutate_pgn=pgn_tags,
    )
    expected = dict(SETTINGS)
    if kind == "matched_time":
        expected["mode"] = "matched_time"
    with pytest.raises(HistoryRefusal):
        verify_strict_arena_history(
            log, pgn, expected_settings=expected, expected_openings={0: opening},
            expected_pgn_tags=TAGS, candidate_name="candidate",
            reference_name="reference", candidate_search="cand-search",
            reference_search="ref-search", tablebase=FakeWdlDtz(),
        )


def test_refuses_unmatched_and_ambiguous_resume_attempts(tmp_path: Path) -> None:
    opening = _knight_opening()
    log, pgn = _capture(
        tmp_path, opening, ("g1f3", "g8f6", "f3g1"), "1/2-1/2", "rules",
    )
    lines = log.splitlines(keepends=True)
    duplicate = b"".join([lines[0], lines[1], lines[1], lines[2]])
    with pytest.raises(HistoryRefusal, match="unmatched JSONL/PGN attempts"):
        _verify(duplicate, pgn, opening, FakeWdlDtz())
    # A PGN-only attempt lacks the JSONL commit record.
    with pytest.raises(HistoryRefusal, match="unmatched JSONL/PGN attempts"):
        _verify(log, pgn + pgn, opening, FakeWdlDtz())


def test_selects_only_unambiguous_marked_resume_pair(tmp_path: Path) -> None:
    opening = _knight_opening()
    log, pgn = _capture(
        tmp_path, opening, ("g1f3", "g8f6", "f3g1"), "1/2-1/2", "rules",
    )
    lines = [json.loads(line) for line in log.splitlines()]
    stale_row = dict(lines[1], termination="max_plies", plies=0)
    resumed_log = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n"
        for row in (lines[0], stale_row, lines[1], lines[2])
    )
    stream = io.StringIO(pgn.decode())
    first = chess.pgn.read_game(stream)
    second = chess.pgn.read_game(stream)
    assert first is not None
    assert second is not None
    stale_pgn = copy.deepcopy(first)
    stale_pgn.variations.clear()
    stale_pgn.headers["Termination"] = "max_plies"
    stale_pgn.headers["Plies"] = "0"
    first.headers["ResumeReplay"] = "1"
    second.headers["ResumeReplay"] = "1"
    resumed_pgn = ("\n\n".join(
        game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
        for game in (stale_pgn, first, second)
    )
                   + "\n\n").encode()
    receipt = _verify(resumed_log, resumed_pgn, opening, FakeWdlDtz())
    assert receipt.replayed_pairs == (0,)
    # Identical JSONL attempts cannot bind their PGN movetext to the selected
    # attempt: this schema has no per-attempt move digest.
    ambiguous_log = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n"
        for row in (lines[0], lines[1], lines[1], lines[2])
    )
    repeated_first = copy.deepcopy(first)
    repeated_first.headers.pop("ResumeReplay")
    ambiguous_pgn = ("\n\n".join(
        game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
        for game in (repeated_first, first, second)
    ) + "\n\n").encode()
    with pytest.raises(HistoryRefusal, match="indistinguishable JSONL attempts"):
        _verify(ambiguous_log, ambiguous_pgn, opening, FakeWdlDtz())
