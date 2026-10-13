from __future__ import annotations

import logging
from typing import Any, cast

import pytest

from chess_anti_engine.worker import WorkerSession


@pytest.mark.parametrize("result", ["failure", "exception", "success"])
def test_reassignment_stops_session_even_if_manifest_fetch_fails(result: str) -> None:
    session = cast(Any, object.__new__(WorkerSession))
    session.leased_trial_id = "trial_b"
    session.pause_selfplay_active = False
    session._stop_selfplay = False
    session.log = logging.getLogger(__name__)

    def poll():
        # Lease negotiation commits the assignment before manifest fetching.
        session.leased_trial_id = "trial_a"
        if result == "exception":
            raise ConnectionError("manifest fetch failed after assignment")
        return {"task": {"type": "selfplay"}} if result == "success" else None

    session._poll_manifest = poll
    WorkerSession._periodic_manifest_poll(session)
    assert session.leased_trial_id == "trial_a"
    assert session._stop_selfplay is True


@pytest.mark.parametrize("paused", [False, True])
def test_same_trial_failed_manifest_preserves_pause_behavior(paused: bool) -> None:
    session = cast(Any, object.__new__(WorkerSession))
    session.leased_trial_id = "trial_b"
    session.pause_selfplay_active = paused
    session._stop_selfplay = False
    session._hold_selfplay = False
    session._hold_on_pause = True
    session.log = logging.getLogger(__name__)
    session._poll_manifest = lambda: None
    WorkerSession._periodic_manifest_poll(session)
    assert session._stop_selfplay is False
    assert session._hold_selfplay is paused
