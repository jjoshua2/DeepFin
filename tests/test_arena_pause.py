"""Cooperative pause of a running arena (``--pause-marker``).

While a marker exists the rolling loop plays no ply and holds its in-flight
games; it continues exactly where it stopped. Paused time is excluded from the
``--max-seconds`` deadline (``_arena_clock``) and recorded.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

import scripts.arena_standard as arena
from chess_anti_engine.moves import move_to_index
from scripts.arena_standard import resolve_search_shape


def _first_legal_actions(boards: list[chess.Board]) -> list[int]:
    return [int(move_to_index(next(iter(b.legal_moves)), b)) for b in boards]


def _openings() -> list[chess.Board]:
    out = []
    for uci in ("e2e4", "d2d4"):
        b = chess.Board()
        b.push_uci(uci)
        out.append(b)
    return out


def _play(deadline: float | None, games_out: list[dict[str, Any]], *, rolling: bool = True) -> list[float]:
    side = resolve_search_shape("training")

    def sink(**kw: Any) -> None:
        games_out.append({k: kw[k] for k in ("pair_id", "half", "moves", "result", "duration_s")})

    common: dict[str, Any] = {
        "device": "cpu", "rng": np.random.default_rng(0), "sims_candidate": 2, "sims_reference": 2,
        "max_plies": 40, "temperature": 1.0, "gumbel_add_noise": False, "search_candidate": side,
        "search_reference": side, "pgn_sink": sink}
    if rolling:
        return arena.play_paired_games_matched_sims_rolling(
            None, None, _openings(), pool_size=4, report_every=1000, deadline=deadline, **common)
    return arena.play_paired_games_matched_sims(None, None, _openings(), **common)


@pytest.fixture(autouse=True)
def _fresh_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arena, "_PAUSE", arena._ArenaPause())


def _release_after(marker: Path, hold_s: float, window: dict[str, float]) -> threading.Thread:
    def go() -> None:
        while not marker.exists():
            time.sleep(0.001)
        time.sleep(hold_s)
        window["released"] = time.time()
        marker.unlink()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


@pytest.mark.parametrize("rolling", [True, False], ids=["rolling", "lockstep"])
def test_a_pause_holds_every_ply_and_resumes_to_identical_games(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rolling: bool) -> None:
    calls: list[float] = []
    marker = tmp_path / "pause"
    pause_at: list[int] = []  # the call that creates the marker (none for the plain run)

    def stub(_model: Any, boards: list[chess.Board], **_kw: Any) -> list[int]:
        calls.append(time.time())
        if pause_at and len(calls) == pause_at[0]:
            marker.write_text("")  # deterministically mid-run, from inside the loop
        return _first_legal_actions(boards)

    monkeypatch.setattr("chess_anti_engine.selfplay.match.pick_moves_for_boards", stub)
    plain_games: list[dict[str, Any]] = []
    plain = _play(None, plain_games, rolling=rolling)
    assert arena._PAUSE.total == 0.0

    arena._PAUSE.configure([marker])
    window: dict[str, float] = {}
    pause_at.append(6)
    calls.clear()
    t = _release_after(marker, 1.5, window)
    paused_games: list[dict[str, Any]] = []
    paused = _play(None, paused_games, rolling=rolling)
    t.join(5)
    assert paused == plain
    keys = ("pair_id", "half", "moves", "result")
    assert [{k: g[k] for k in keys} for g in paused_games] == [{k: g[k] for k in keys} for g in plain_games]
    assert len(arena._PAUSE.intervals) == 1
    assert arena._PAUSE.total == pytest.approx(1.5, abs=0.6)
    start, _end = arena._PAUSE.intervals[0]
    assert not [c for c in calls if start < c < window["released"]], "a ply was played while paused"
    assert arena._pause_record()["timing_independent"] is False
    assert arena._pause_record()["paused_seconds"] == arena._PAUSE.total


def test_a_matched_time_pause_holds_before_the_next_game(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.match_vs_uci as mvu

    class _Engine:
        def quit(self) -> None:
            return None

    marker = tmp_path / "pause"
    games: list[float] = []

    def one_game(_w: Any, _b: Any, **kw: Any) -> Any:
        games.append(time.time())
        if len(games) == 1:
            marker.write_text("")  # the pause lands during game 1
        board = kw["start_board"]
        return mvu.GameRecord(result="1/2-1/2", plies=0, start_board=board, moves=(), move_records=())

    monkeypatch.setattr(mvu, "_open_engine", lambda *_a, **_k: _Engine())
    monkeypatch.setattr(mvu, "play_one_game", one_game)
    arena._PAUSE.configure([marker])
    window: dict[str, float] = {}
    t = _release_after(marker, 1.0, window)
    scores = arena.play_paired_games_matched_time(
        "a.pt", "b.pt", _openings(), device="cpu", ms_per_move=10, max_plies=10, uci_args="")
    t.join(5)
    assert scores == [1.0, 1.0]
    assert len(games) == 4
    assert games[1] >= window["released"], "game 2 started while the marker was held"
    assert len(arena._PAUSE.intervals) == 1


def test_paused_time_does_not_spend_the_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chess_anti_engine.selfplay.match.pick_moves_for_boards",
                        lambda _m, boards, **_k: _first_legal_actions(boards))
    marker = tmp_path / "pause"
    marker.write_text("")
    arena._PAUSE.configure([marker])

    def release() -> None:
        time.sleep(2.0)
        marker.unlink()

    threading.Thread(target=release, daemon=True).start()
    # 1.5 s of playing time; a pause of 2 s at the start would exhaust a wall-clock budget.
    pairs = _play(arena._arena_clock() + 1.5, [])
    assert len(pairs) == 2, "the pause was charged to --max-seconds"
    assert arena._PAUSE.total >= 1.9


def test_no_marker_configured_is_a_no_op() -> None:
    before = arena._arena_clock()
    arena._PAUSE.hold()
    assert arena._PAUSE.total == 0.0
    assert arena._arena_clock() - before < 0.5
    assert arena._pause_record() == {"paused_seconds": 0.0, "pause_intervals": [], "timing_independent": True}


class _LoadedUnderPause(Exception):
    pass


def test_an_arena_launched_under_a_marker_loads_no_model_until_it_is_gone(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hold comes BEFORE either checkpoint is loaded (nothing put on the GPU)."""
    from chess_anti_engine.selfplay import match as match_helpers
    from chess_anti_engine.uci import model_loader

    monkeypatch.setattr(match_helpers, "_HAS_GUMBEL_C", True)
    monkeypatch.setattr(arena, "load_fen_openings", lambda *_a, **_k: [chess.Board() for _ in range(4)])
    marker = tmp_path / "pause"
    marker.write_text("")
    arena._PAUSE.configure([marker])
    window: dict[str, float] = {}
    _release_after(marker, 1.0, window)
    loads: list[float] = []

    def load(*_a: Any, **_k: Any) -> Any:
        loads.append(time.time())
        raise _LoadedUnderPause  # stop the arena here; only the ordering matters

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", load)
    side = arena.SideSearch(shape="training", source="fixture", gumbel={}, vloss_weight=1, target_batch=1)
    with pytest.raises(_LoadedUnderPause):
        arena.run_arena(
            candidate="candidate.pt", reference="reference.pt", games=4, openings_path=None,
            openings_fen=tmp_path / "openings.fen", opening_plies=16, mode="matched_sims",
            sims_candidate=1, sims_reference=1, ms_per_move=0, max_plies=1, temperature=0.1,
            gumbel_add_noise=False, device="cpu", seed=1, out_path=None, compile_models=False,
            eval_max_batch=0, max_concurrent_games=2, search_candidate=side, search_reference=side,
            game_log_path=tmp_path / "arena.games.jsonl")
    assert loads
    assert loads[0] >= window["released"], "a checkpoint was loaded while the marker was held"
