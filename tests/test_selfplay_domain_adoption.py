"""Actual manager/domain calls with inert game collaborators, never engines."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.generation_runtime import GenerationSessionRunner
from chess_anti_engine.selfplay import manager


def _state(*, continuous=True, done=True):
    state = SimpleNamespace(
        continuous=continuous, done_arr=[int(done)], finalized_arr=[0],
        games_started=1, games_completed=0, samples_per_game=[[]],
        pending_sf_labels=[], pending_sf_moves={}, tb_probe=None, mcts_tree=None,
        batch_size=1, fen_dole_queue=["retained-seed"],
        rng=np.random.default_rng(25), game=SimpleNamespace(syzygy_adjudicate=False),
    )

    def recycle(slot):
        state.fen_dole_queue.pop(0)
        state.rng.random()
        state.games_started += 1
        state.done_arr[slot] = 0
        state.finalized_arr[slot] = 0

    state.recycle_slot = recycle
    state.classify_active_slots = lambda: ([0], [], [], False)
    state.stats = SimpleNamespace(to_batch_stats=lambda **kwargs: kwargs)
    return state


def _finalizer(events):
    def finalize(state, slot, samples, callback):
        events.append(("finalize", slot, state.continuous))
        samples.append("completed-sample")
        if callback is not None:
            callback("completed-game")
        if state.continuous:
            samples.clear()
    return finalize


@pytest.mark.parametrize("continuous", [True, False])
def test_actual_stop_domain_finalizes_once_without_seed_rng_or_start(continuous, monkeypatch):
    state = _state(continuous=continuous)
    events = []
    callback = []
    samples = []
    initial_rng = copy.deepcopy(state.rng.bit_generator.state)
    monkeypatch.setattr(manager, "finalize_game", _finalizer(events))
    domain = manager.SelfplayDomain(
        state, stockfish=SimpleNamespace(), all_samples=samples,
        batch_size=1, continuous=continuous, target=5, max_steps=10,
        stop_fn=lambda: True, on_game_complete=callback.append,
        on_suspend=lambda current: events.append(("suspend", current.games_completed)),
    )
    GenerationSessionRunner().run(domain.advance)
    assert domain.advance().terminal  # repeated terminal observation is inert
    assert events == [("finalize", 0, continuous), ("suspend", 1)]
    assert callback == ["completed-game"]
    assert samples == ([] if continuous else ["completed-sample"])
    assert state.fen_dole_queue == ["retained-seed"]
    assert state.games_started == 1 and state.games_completed == 1
    assert state.rng.bit_generator.state == initial_rng
    assert state.finalized_arr == [1]


def test_stop_poll_completion_is_finalized_before_suspend(monkeypatch):
    state = _state(done=False)
    state.pending_sf_labels = [object()]
    events = []

    def complete_labels(current):
        events.append("poll")
        current.pending_sf_labels.clear()
        current.done_arr[0] = 1

    monkeypatch.setattr(manager, "poll_async_sf_labels", complete_labels)
    monkeypatch.setattr(manager, "finalize_game", _finalizer(events))
    domain = manager.SelfplayDomain(
        state, stockfish=SimpleNamespace(), all_samples=[], batch_size=1,
        continuous=True, target=1, max_steps=3, stop_fn=lambda: True,
        on_suspend=lambda current: events.append(("suspend", current.finalized_arr[0])),
    )
    GenerationSessionRunner().run(domain.advance)
    assert events == ["poll", ("finalize", 0, True), ("suspend", 1)]
    assert state.fen_dole_queue == ["retained-seed"]


@pytest.mark.parametrize("continuous,target,recycled", [(True, 1, True), (False, 2, True), (False, 1, False)])
def test_normal_finalization_keeps_recycling_contract(continuous, target, recycled, monkeypatch):
    state = _state(continuous=continuous)
    monkeypatch.setattr(manager, "finalize_game", _finalizer([]))
    manager._finalize_completed_slots(
        state, all_samples=[], on_game_complete=None, batch_size=1,
        continuous=continuous, target=target,
    )
    assert state.games_started == 1 + int(recycled)
    assert state.fen_dole_queue == ([] if recycled else ["retained-seed"])


def test_starved_continuation_still_finalizes_and_recycles(monkeypatch):
    state = _state(done=False)
    state.pending_sf_labels = [object()]
    state.classify_active_slots = lambda: ([], [], [], False)
    waited = []

    def poll(current):
        if waited:
            current.pending_sf_labels.clear()
            current.done_arr[0] = 1

    monkeypatch.setattr(manager, "poll_async_sf_labels", poll)
    monkeypatch.setattr(manager, "_wait_for_starved_sf", lambda *_args, **_kwargs: waited.append(True))
    monkeypatch.setattr(manager, "finalize_game", _finalizer([]))
    domain = manager.SelfplayDomain(
        state, stockfish=SimpleNamespace(), all_samples=[], batch_size=1,
        continuous=True, target=1, max_steps=1,
    )
    GenerationSessionRunner().run(domain.advance)
    assert waited == [True]
    assert state.games_completed == 1 and state.games_started == 2
    assert state.fen_dole_queue == []


def test_play_batch_calls_common_runner_and_domain_on_caller_thread(monkeypatch):
    state = _state(continuous=False, done=False)
    seen = []
    monkeypatch.setattr(manager.rep_fix, "apply", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(manager.SelfplayState, "create", lambda **_kwargs: state)
    monkeypatch.setattr(manager, "finalize_game", _finalizer(seen))

    def step(current, **_kwargs):
        current.done_arr[0] = 1
        return 1.0, 2.0

    state.classify_active_slots = lambda: ([0], [], [], bool(state.finalized_arr[0]))
    monkeypatch.setattr(manager, "_run_step", step)
    actual_runner = manager.GenerationSessionRunner

    class RecordingRunner(actual_runner):
        def run(self, advance):
            assert isinstance(advance.__self__, manager.SelfplayDomain)
            seen.append("runner")
            return super().run(advance)

    monkeypatch.setattr(manager, "GenerationSessionRunner", RecordingRunner)
    samples, stats = manager.play_batch(
        None, device="cpu", rng=state.rng, stockfish=SimpleNamespace(nodes=0),
        games=1,
    )
    assert seen == ["runner", ("finalize", 0, False)]
    assert samples == ["completed-sample"] and stats["games"] == 1


def test_pause_controls_can_stop_without_scheduling_or_recycling(monkeypatch):
    state = _state(done=False)
    polls = []
    suspended = []
    monkeypatch.setattr(manager.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(manager, "_run_step", lambda *_args, **_kwargs: pytest.fail("scheduled while paused"))
    domain = manager.SelfplayDomain(
        state, stockfish=SimpleNamespace(), all_samples=[], batch_size=1,
        continuous=True, target=1, max_steps=10,
        on_step=lambda: polls.append(True), pause_fn=lambda: True,
        stop_fn=lambda: len(polls) >= 3,
        on_suspend=lambda current: suspended.append(current),
    )
    GenerationSessionRunner().run(domain.advance)
    assert len(suspended) == 1 and suspended[0] is state
    assert state.games_started == 1 and state.games_completed == 0
    assert state.fen_dole_queue == ["retained-seed"]
