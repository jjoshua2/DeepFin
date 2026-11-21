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


def _play(deadline: float | None, games_out: list[dict[str, Any]]) -> list[float]:
    side = resolve_search_shape("training")

    def sink(**kw: Any) -> None:
        games_out.append({k: kw[k] for k in ("pair_id", "half", "moves", "result", "duration_s")})

    return arena.play_paired_games_matched_sims_rolling(
        None, None, _openings(), device="cpu", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2, max_plies=40, temperature=1.0, gumbel_add_noise=False,
        search_candidate=side, search_reference=side, pool_size=4, report_every=1000,
        deadline=deadline, pgn_sink=sink,
    )


@pytest.fixture(autouse=True)
def _fresh_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arena, "_PAUSE", arena._ArenaPause())


def test_a_pause_holds_every_ply_and_resumes_to_identical_games(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def stub(_model: Any, boards: list[chess.Board], **_kw: Any) -> list[int]:
        calls.append(time.time())
        return _first_legal_actions(boards)

    monkeypatch.setattr("chess_anti_engine.selfplay.match.pick_moves_for_boards", stub)
    plain_games: list[dict[str, Any]] = []
    plain = _play(None, plain_games)

    marker = tmp_path / "pause"
    arena._PAUSE.configure([marker])
    window: dict[str, float] = {}

    def pauser() -> None:
        while len(calls) < 10:  # some plies into the replay
            time.sleep(0.001)
        marker.write_text("")
        window["set"] = time.time()
        time.sleep(1.5)
        window["released"] = time.time()
        marker.unlink()

    calls.clear()
    t = threading.Thread(target=pauser, daemon=True)
    t.start()
    paused_games: list[dict[str, Any]] = []
    paused = _play(None, paused_games)
    t.join(5)
    assert paused == plain
    strip = [{k: g[k] for k in ("pair_id", "half", "moves", "result")} for g in paused_games]
    assert strip == [{k: g[k] for k in ("pair_id", "half", "moves", "result")} for g in plain_games]
    assert arena._PAUSE.total >= 1.4
    assert len(arena._PAUSE.intervals) == 1
    start, _end = arena._PAUSE.intervals[0]
    # At most the ply already in flight when the marker appeared; none while held.
    assert not [c for c in calls if start + 0.01 < c < window["released"]], "a ply was played while paused"


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
