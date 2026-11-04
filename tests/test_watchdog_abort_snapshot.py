"""H6 interleavings, with no thread start and only an inert exit recorder."""
from collections.abc import Callable

import pytest

from chess_anti_engine.broker_hang import BrokerHangWatchdog, HANG_ABORT_EXIT_CODE


@pytest.mark.parametrize('transition', ['complete', 'replace', 'arm'])
def test_snapshot_transition_cannot_commit_stale_abort(transition):
    now = [0.]
    callback: list[Callable[[], None] | None] = [None]
    exits = []

    def clock():
        action = callback[0]
        callback[0] = None
        if action is not None:
            action()
        return now[0]

    wd = BrokerHangWatchdog(threshold_s=5., boot_threshold_s=10.,
                           clock=clock, exit_fn=exits.append)
    old = wd.mark_forward_start(1)
    other = wd.mark_forward_start(2)
    now[0] = 20.

    def transition_snapshot():
        if transition == 'arm':
            wd.mark_forward_done(token=other, success=True)
        else:
            wd.mark_forward_done(token=old, success=False)
            if transition == 'replace':
                # Same timestamp/label/batch content is not the same work token.
                now[0] = 0.
                wd.mark_forward_start(1)
                now[0] = 20.

    callback[0] = transition_snapshot
    assert not wd.check_once()
    assert exits == []
    assert not wd._aborted
    assert wd._thread is None
    # A fresh check still aborts genuine old work (the other or replacement).
    assert wd.check_once()
    assert exits == [HANG_ABORT_EXIT_CODE]


def test_completion_of_last_item_makes_stale_snapshot_idle():
    now = [0.]
    callback: list[Callable[[], None] | None] = [None]
    exits = []

    def clock():
        action = callback[0]
        callback[0] = None
        if action is not None:
            action()
        return now[0]

    wd = BrokerHangWatchdog(threshold_s=5., clock=clock, exit_fn=exits.append)
    token = wd.mark_forward_start(1)
    now[0] = 20.
    callback[0] = lambda: wd.mark_forward_done(token=token, success=False)
    assert not wd.check_once()
    assert not wd.check_once()
    assert exits == []


def test_unchanged_hang_commits_only_one_abort():
    now = [0.]
    exits = []
    wd = BrokerHangWatchdog(threshold_s=5., clock=lambda: now[0], exit_fn=exits.append)
    wd.mark_forward_start(1)
    now[0] = 20.
    assert wd.check_once()
    assert wd.check_once()
    assert exits == [HANG_ABORT_EXIT_CODE]
