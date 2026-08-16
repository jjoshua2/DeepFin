"""Tests for play_batch continuous mode, recycling, and on_game_complete callback.

These tests lock down behavior prior to the Phase 5 refactor of play_batch.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any, cast

import chess
import numpy as np

from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import CompletedGameBatch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.selfplay.stockfish_turn import finish_pending_curriculum_moves
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult
from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel


class _DelayedStockfishPool(StockfishPool):
    """StockfishPool-shaped fake whose result futures complete after a delay."""

    def __init__(self, delay_s: float):  # pyright: ignore[reportMissingSuperCall]
        self.nodes = 1
        self.delay_s = float(delay_s)
        self._timers: list[threading.Timer] = []

    def submit(
        self, fen: str, *, nodes: int | None = None,
        syzygy_path: str | None = None, fresh: bool = False,
        searchmoves: Sequence[str] | None = None,
    ) -> Future[StockfishResult]:
        del nodes, syzygy_path, fresh
        board = chess.Board(fen)
        # Honour the restriction rather than discarding it: this fake picks the
        # first legal move, which would mask a caller that lost its searchmoves.
        if searchmoves:
            move = chess.Move.from_uci(str(next(iter(searchmoves))))
        else:
            move = next(iter(board.legal_moves), chess.Move.null())
        result = StockfishResult(
            bestmove_uci=move.uci(),
            wdl=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            pvs=[],
        )
        future: Future[StockfishResult] = Future()
        timer = threading.Timer(self.delay_s, future.set_result, args=(result,))
        timer.daemon = True
        self._timers.append(timer)
        timer.start()
        return future

    def close(self) -> None:
        for timer in self._timers:
            timer.join(timeout=max(0.1, self.delay_s * 2.0))


def test_continuous_mode_stops_on_stop_fn_and_delivers_via_callback():
    """Continuous mode: stop_fn terminates the loop; samples flow via callback."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(0)

    completed: list[CompletedGameBatch] = []

    # Stop once we've seen at least 2 completed games.
    def _stop() -> bool:
        return len(completed) >= 2

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,  # continuous mode
        stop_fn=_stop,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    # Continuous mode must NOT accumulate samples in the returned list
    # (they are delivered via on_game_complete to bound memory growth).
    assert samples == []
    # But stats should still reflect games played.
    assert stats.games >= 2
    # Callback should have received each completed game.
    assert len(completed) >= 2
    # Each CompletedGameBatch reports its own positions count.
    for cg in completed:
        assert isinstance(cg, CompletedGameBatch)
        assert cg.positions == len(cg.samples)


def test_continuous_mode_recycles_slots_beyond_initial_batch():
    """Continuous mode with small batch_size must recycle slots indefinitely."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(1)

    completed: list[CompletedGameBatch] = []

    def _stop() -> bool:
        return len(completed) >= 4  # more than batch_size

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2,  # only 2 parallel slots
        target_games=0,
        stop_fn=_stop,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    # Slot recycling must produce more completed games than the batch size.
    assert stats.games >= 4
    assert len(completed) >= 4
    assert samples == []


def test_continuous_slot_oversubscribe_multiplies_slots_and_completes():
    """Oversubscription multiplies concurrent slots and games still complete.

    Asserts the concurrency property only (state.batch_size scales with the
    factor; at least as many games finish). Per-game outcome identity across
    factors is NOT claimed — scheduling interleaving differs by construction,
    and the FakeStockfish + max_plies=2 setup would make such a claim vacuous
    (nearly every game adjudicates to the same draw tuple).
    """

    def _run(factor: float) -> tuple[int, list[tuple[int, int, int, int]]]:
        completed: list[CompletedGameBatch] = []
        batch_sizes: list[int] = []
        _samples, _stats = play_batch(
            UniformPolicyValueModel().eval(), device="cpu",
            rng=np.random.default_rng(12),
            stockfish=FakeStockfish([0.0, 1.0, 0.0]),
            games=2, target_games=0,
            stop_fn=lambda: len(completed) >= 2,
            on_state_ready=lambda state: batch_sizes.append(state.batch_size),
            on_game_complete=completed.append,
            slot_oversubscribe=factor,
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(
                simulations=1, playout_cap_fraction=1.0, fast_simulations=1,
            ),
            opening=OpeningConfig(random_start_plies=0),
            diff_focus=DiffFocusConfig(enabled=False),
            game=GameConfig(max_plies=2, selfplay_fraction=1.0),
        )
        results = [(game.w, game.d, game.l, game.total_game_plies) for game in completed]
        return batch_sizes[0], results

    baseline_slots, baseline = _run(1.0)
    extra_slots, oversubscribed = _run(2.0)

    assert baseline_slots == 2
    assert extra_slots == 4
    assert len(oversubscribed) >= len(baseline)


def test_finite_mode_ignores_slot_oversubscribe_and_keeps_exact_target():
    completed: list[CompletedGameBatch] = []
    batch_sizes: list[int] = []
    _samples, stats = play_batch(
        UniformPolicyValueModel().eval(), device="cpu",
        rng=np.random.default_rng(13),
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=3,
        on_state_ready=lambda state: batch_sizes.append(state.batch_size),
        on_game_complete=completed.append,
        slot_oversubscribe=3.0,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    assert batch_sizes == [2]
    assert stats.games == 3
    assert len(completed) == 3


def test_pending_curriculum_slot_does_not_block_runnable_selfplay_slot():
    """A pending SF opponent move must overlap runnable work in another slot."""
    pool = _DelayedStockfishPool(0.2)
    completed: list[CompletedGameBatch] = []
    timings: dict[str, float] = {}

    def _set_mixed_slots(state: SelfplayState) -> None:
        state.selfplay_arr[:] = np.asarray([False, True], dtype=np.bool_)

    try:
        play_batch(
            UniformPolicyValueModel().eval(), device="cpu",
            rng=np.random.default_rng(14), stockfish=pool,
            games=2, target_games=0,
            stop_fn=lambda: len(completed) >= 2,
            on_state_ready=_set_mixed_slots,
            on_game_complete=completed.append,
            on_timing=lambda key, value: timings.__setitem__(
                key, timings.get(key, 0.0) + value,
            ),
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
            opening=OpeningConfig(random_start_plies=0),
            diff_focus=DiffFocusConfig(enabled=False),
            game=GameConfig(max_plies=4, selfplay_fraction=0.5),
        )
    finally:
        pool.close()

    assert timings.get("sf_pending_with_runnable_steps", 0.0) > 0.0
    assert len(completed) >= 2


def test_starved_block_time_attributed_to_sf_block_starved_not_finish_cur():
    """The ledger yardstick reads `sf_block_starved` as the true starved-wait
    share; a refactor that re-lumps the blocked wait into `sf_finish_curriculum`
    would silently break that mechanism check. Curriculum-only slots with a
    slow SF pool must accumulate the wait under sf_block_starved."""
    pool = _DelayedStockfishPool(0.1)
    completed: list[CompletedGameBatch] = []
    timings: dict[str, float] = {}

    try:
        play_batch(
            UniformPolicyValueModel().eval(), device="cpu",
            rng=np.random.default_rng(16), stockfish=pool,
            games=1, target_games=0,
            stop_fn=lambda: len(completed) >= 1,
            on_game_complete=completed.append,
            on_timing=lambda key, value: timings.__setitem__(
                key, timings.get(key, 0.0) + value,
            ),
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
            opening=OpeningConfig(random_start_plies=0),
            diff_focus=DiffFocusConfig(enabled=False),
            game=GameConfig(max_plies=4, selfplay_fraction=0.0),
        )
    finally:
        pool.close()

    starved = timings.get("sf_block_starved", 0.0)
    finish_cur = timings.get("sf_finish_curriculum", 0.0)
    assert starved > 0.05  # the 0.1s SF delays must land here
    assert finish_cur < starved  # non-blocking polls only, never the wait


def test_pause_with_pending_curriculum_future_holds_then_resumes():
    pool = _DelayedStockfishPool(0.05)
    completed: list[CompletedGameBatch] = []
    timings: dict[str, float] = {}
    live_state: list[SelfplayState] = []
    pause_polls = 0

    def _pause() -> bool:
        nonlocal pause_polls
        if not live_state or not live_state[0].pending_sf_moves:
            return False
        pause_polls += 1
        return pause_polls < 3

    try:
        play_batch(
            UniformPolicyValueModel().eval(), device="cpu",
            rng=np.random.default_rng(15), stockfish=pool,
            games=2, target_games=0,
            stop_fn=lambda: len(completed) >= 2,
            pause_fn=_pause,
            on_state_ready=live_state.append,
            on_game_complete=completed.append,
            on_timing=lambda key, value: timings.__setitem__(
                key, timings.get(key, 0.0) + value,
            ),
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
            opening=OpeningConfig(random_start_plies=0),
            diff_focus=DiffFocusConfig(enabled=False),
            game=GameConfig(max_plies=4, selfplay_fraction=0.0),
        )
    finally:
        pool.close()

    assert pause_polls >= 3
    assert timings.get("pause_hold", 0.0) > 0.0
    assert len(completed) >= 2


def test_blocking_pending_move_wait_is_bounded():
    future: Future[StockfishResult] = Future()
    state: Any = type("PendingState", (), {"pending_sf_moves": {0: future}})()
    late_result = StockfishResult(
        bestmove_uci="e2e4",
        wdl=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        pvs=[],
    )
    timer = threading.Timer(0.2, future.set_result, args=(late_result,))
    timer.daemon = True
    timer.start()
    started = time.perf_counter()
    finished = finish_pending_curriculum_moves(
        state, block=True, block_timeout_s=0.02,
    )
    elapsed = time.perf_counter() - started

    assert finished == 0
    assert 0.015 <= elapsed < 0.15
    timer.join(timeout=0.5)
    assert future.done()


def test_continuous_mode_stats_sum_matches_callback_sum():
    """Aggregated BatchStats must match the sum of individual CompletedGameBatch
    deliveries, with no double counting after slot recycling."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(2)

    completed: list[CompletedGameBatch] = []

    def _stop() -> bool:
        return len(completed) >= 3

    _samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,
        stop_fn=_stop,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        # All curriculum games with the fake SF's draw-only WDL.
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )

    # Continuous mode clears all_samples to bound memory, so stats.positions
    # is always 0; all position accounting is delivered via the callback.
    assert stats.positions == 0
    assert sum(cg.games for cg in completed) == stats.games
    assert sum(cg.w for cg in completed) == stats.w
    assert sum(cg.d for cg in completed) == stats.d
    assert sum(cg.l for cg in completed) == stats.l
    assert sum(cg.total_game_plies for cg in completed) == stats.total_game_plies
    assert sum(cg.adjudicated_games for cg in completed) == stats.adjudicated_games
    assert sum(cg.curriculum_games for cg in completed) == stats.curriculum_games
    assert sum(cg.selfplay_games for cg in completed) == stats.selfplay_games


def test_finite_mode_does_not_deliver_via_callback_when_none_given():
    """Finite mode without callback: samples are returned directly, no crash."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(3)

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=2,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )

    assert stats.games == 2
    assert len(samples) == stats.positions


def test_finite_mode_with_callback_also_delivers_via_callback():
    """When both finite target_games and callback are set, both paths deliver."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(4)

    completed: list[CompletedGameBatch] = []

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=3,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )

    # Finite mode returns samples AND calls callback.
    assert stats.games == 3
    assert len(completed) == 3
    # Sample counts match.
    assert sum(cg.positions for cg in completed) == len(samples)


def test_continuous_mode_respects_on_step_callback():
    """on_step should be invoked during the main loop in continuous mode."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(5)

    step_calls: list[int] = []
    completed: list[CompletedGameBatch] = []

    def _on_step() -> None:
        step_calls.append(1)

    def _stop() -> bool:
        return len(completed) >= 2

    play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,
        stop_fn=_stop,
        on_step=_on_step,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    # on_step should fire at least once per main-loop iteration.
    assert len(step_calls) > 0


def test_pause_fn_holds_without_abandoning_and_resumes():
    """pause_fn=True holds the loop (no exit, state preserved); games keep
    completing after release. on_timing must record the hold duration."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(2)

    completed: list[CompletedGameBatch] = []
    timings: dict[str, float] = {}
    pause_state = {"paused": False, "pause_polls": 0}

    def _pause() -> bool:
        # Engage the pause once the first game lands; release after a few polls.
        if completed and not pause_state["paused"] and pause_state["pause_polls"] == 0:
            pause_state["paused"] = True
        if pause_state["paused"]:
            pause_state["pause_polls"] += 1
            if pause_state["pause_polls"] >= 3:
                pause_state["paused"] = False
        return bool(pause_state["paused"])

    def _stop() -> bool:
        return len(completed) >= 3

    def _on_timing(key: str, dt: float) -> None:
        timings[key] = timings.get(key, 0.0) + dt

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,
        stop_fn=_stop,
        pause_fn=_pause,
        on_game_complete=completed.append,
        on_timing=_on_timing,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    assert samples == []
    # The pause engaged (was polled through its release sequence) ...
    assert pause_state["pause_polls"] >= 3
    assert "pause_hold" in timings
    # ... and games kept completing after the hold released.
    assert len(completed) >= 3
    assert stats.games >= 3


def test_stop_during_pause_exits_promptly():
    """A stop that fires while the loop is held must exit the session
    (teardown always wins over hold; no deadlock)."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(3)

    calls = {"on_step": 0}
    stop = {"flag": False}

    def _on_step() -> None:
        calls["on_step"] += 1
        if calls["on_step"] >= 5:
            stop["flag"] = True

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,
        stop_fn=lambda: stop["flag"],
        pause_fn=lambda: True,  # never released
        on_step=_on_step,
        on_game_complete=lambda _cg: None,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )
    assert samples == []
    assert stats.games == 0  # held from the start; nothing completed, no hang


def test_tb_adjudicate_gate_reads_state_game(monkeypatch):
    """play_batch's TB gate must follow state.game, not the closed-over session game.

    Session starts with syzygy_adjudicate=False (closed-over ``game`` stays False).
    on_state_ready installs a non-None tb_probe and flips state.game to True.
    If the gate still closed over session-start ``game``, adjudication never runs.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from chess_anti_engine.selfplay import manager as mgr

    calls: list[int] = []

    def _spy(_state: Any) -> int:
        calls.append(1)
        return 0

    monkeypatch.setattr(mgr, "_tb_adjudicate_active_games", _spy)

    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(4)
    steps = {"n": 0}

    def _on_ready(state: SelfplayState) -> None:
        state.tb_probe = cast(Any, SimpleNamespace(max_pieces=6))
        state.game = replace(state.game, syzygy_adjudicate=True)

    def _stop() -> bool:
        steps["n"] += 1
        return bool(calls) or steps["n"] >= 30

    play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=1,
        target_games=0,
        stop_fn=_stop,
        on_state_ready=_on_ready,
        on_game_complete=lambda _cg: None,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0, syzygy_adjudicate=False),
    )
    assert calls, "TB gate must call adjudicate when state.game.syzygy_adjudicate is True"


def _parse_exit_line(message: str) -> dict[str, int]:
    """The three counters off a ``play_batch exit:`` log line."""
    return {
        k: int(v)
        for k, v in re.findall(r"(\w+)=(\d+)", message.split("play_batch exit:", 1)[1])
    }


def test_play_batch_exit_line_reports_in_flight_without_claiming_abandonment(caplog):
    """The teardown counter must not assert a fate ``play_batch`` cannot see.

    ``play_batch`` returns BEFORE the caller suspends anything, so at this line
    the in-flight games are candidates for ``selfplay/resume.py`` to persist,
    not losses. The old name ``in_flight_abandoned`` said otherwise and was
    logged beside ``suspended games=20`` for the same games on 2026-07-30.

    MUTATION A: rename the token back to ``in_flight_abandoned`` -- the
    "abandoned" assertion below fires.
    MUTATION B: drop the pointer to the resume line -- a reader is left with a
    number and no way to complete it.
    MUTATION C: report ``0`` (or ``games_completed``) instead of
    ``started - completed`` -- the identity and the ``> 0`` assertion fire.
    """
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(11)
    steps: list[int] = []
    completed: list[CompletedGameBatch] = []

    def _on_step() -> None:
        steps.append(1)

    # Stop while every slot still has a long game running, so the teardown
    # genuinely has in-flight games to report. A test that tore down an idle
    # loop would read 0 and could not tell the three mutations apart.
    def _stop() -> bool:
        return len(steps) >= 2

    with caplog.at_level(logging.INFO, logger="chess_anti_engine.worker"):
        play_batch(
            model, device="cpu", rng=rng,
            stockfish=FakeStockfish([0.0, 1.0, 0.0]),
            games=3,
            target_games=0,
            stop_fn=_stop,
            on_step=_on_step,
            on_game_complete=completed.append,
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
            opening=OpeningConfig(random_start_plies=0),
            diff_focus=DiffFocusConfig(enabled=False),
            game=GameConfig(max_plies=400, selfplay_fraction=1.0),
        )

    lines = [r.getMessage() for r in caplog.records if "play_batch exit:" in r.getMessage()]
    assert len(lines) == 1, lines
    line = lines[0]

    counts = _parse_exit_line(line)
    assert set(counts) == {"completed", "started", "in_flight_at_exit"}, counts
    assert counts["in_flight_at_exit"] == counts["started"] - counts["completed"]
    assert counts["in_flight_at_exit"] > 0, "test must exercise a live teardown"

    # The name may not claim the games were lost, and the line must say where
    # the other half of the answer lives.
    assert "abandoned=" not in line
    assert "selfplay resume: suspended games=" in line
