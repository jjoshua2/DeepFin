from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np

from chess_anti_engine.selfplay import manager


def _state_with_pending_label() -> Any:
    record = object()
    return SimpleNamespace(
        done_arr=np.array([1, 0], dtype=np.int8),
        finalized_arr=np.zeros(2, dtype=np.int8),
        samples_per_game=[[record], []],
        pending_sf_labels=[SimpleNamespace(record=record, future=object())],
        pending_sf_moves={},
        games_completed=0,
        games_started=2,
        recycle_slot=Mock(),
    )


def test_finalize_defers_only_the_game_with_an_outstanding_label(monkeypatch) -> None:
    state = _state_with_pending_label()
    finalize = Mock()
    monkeypatch.setattr(manager, "finalize_game", finalize)

    manager._finalize_completed_slots(
        state,
        all_samples=[],
        on_game_complete=None,
        batch_size=2,
        continuous=True,
        target=2,
        defer_pending_labels=True,
    )
    finalize.assert_not_called()
    assert state.games_completed == 0
    state.recycle_slot.assert_not_called()

    # The regular label poll removes an attached future from this list. The
    # next scheduler pass must then finalize and recycle the exact same game.
    state.pending_sf_labels.clear()
    manager._finalize_completed_slots(
        state,
        all_samples=[],
        on_game_complete=None,
        batch_size=2,
        continuous=True,
        target=2,
        defer_pending_labels=True,
    )
    finalize.assert_called_once_with(state, 0, [], None)
    assert state.finalized_arr.tolist() == [1, 0]
    assert state.games_completed == 1
    state.recycle_slot.assert_called_once_with(0)


def test_label_only_starvation_uses_bounded_future_wait(monkeypatch) -> None:
    state = _state_with_pending_label()
    wait = Mock()
    monkeypatch.setattr(manager, "wait", wait)

    manager._wait_for_starved_sf(state)

    wait.assert_called_once_with(
        (state.pending_sf_labels[0].future,),
        timeout=0.05,
        return_when=manager.FIRST_COMPLETED,
    )


def test_finite_batch_keeps_blocking_finalization_semantics(monkeypatch) -> None:
    state = _state_with_pending_label()
    finalize = Mock()
    monkeypatch.setattr(manager, "finalize_game", finalize)

    manager._finalize_completed_slots(
        state,
        all_samples=[],
        on_game_complete=None,
        batch_size=2,
        continuous=False,
        target=2,
    )

    finalize.assert_called_once_with(state, 0, [], None)
    assert state.games_completed == 1
