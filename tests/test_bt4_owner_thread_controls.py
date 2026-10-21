"""Actual BT4 controller calls with synthetic sessions and owner-affine guards."""
from __future__ import annotations

from pathlib import Path
import threading
import time
from types import SimpleNamespace

import chess
import numpy as np
import pytest

from chess_anti_engine.generation_runtime import AdvanceResult
from chess_anti_engine.staged_execution import OwnerThreadCallbacks
from scripts import bt4_root_policy_worker as worker
from tests.test_cross_unit_teacher_generation import small_units


def test_actual_owned_roster_guard_and_checkpoint_lifecycle_remain_controller_affine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.shared_teacher_owner import run_owned_roster
    from tests.test_owned_teacher_roster import setup

    descriptors, options, _seen, _evaluator, loaded, _closure = setup(tmp_path, monkeypatch)
    owner = threading.get_ident()
    guard = options["guard"]
    original_check, original_charge = guard.check, guard.charge
    checked = []
    checkpoint_calls = []

    def check() -> None:
        assert threading.get_ident() == owner
        checked.append(owner)
        original_check()

    def charge(amount) -> None:
        assert threading.get_ident() == owner
        original_charge(amount)

    monkeypatch.setattr(guard, "check", check)
    monkeypatch.setattr(guard, "charge", charge)
    checkpoint_type = loaded["bt4_longblock_checkpoint_v2"].Checkpoint
    for name in ("recover", "begin", "completed", "record_game", "finish", "close"):
        original = getattr(checkpoint_type, name)

        def bound(checkpoint, *args, _name=name, _original=original, **kwargs):
            assert threading.get_ident() == owner
            checkpoint_calls.append(_name)
            return _original(checkpoint, *args, **kwargs)

        monkeypatch.setattr(checkpoint_type, name, bound)
    options["writer_threads"] = 2  # Actual concurrent writer callbacks share one controller guard.
    result = run_owned_roster(descriptors, **options)
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert checked
    assert set(checkpoint_calls) == {"recover", "begin", "completed", "record_game", "finish", "close"}
    assert sum(map(len, result["unit_receipts"].values())) == 256


def test_actual_worker_marshals_guards_but_keeps_raw_ack_on_controller(tmp_path: Path) -> None:
    fixture, out, units, args = small_units(tmp_path)
    owner = threading.get_ident()
    checks = []
    acknowledged = []

    def guard() -> None:
        assert threading.get_ident() == owner
        checks.append(owner)

    bridge = OwnerThreadCallbacks(guard, capacity=4, timeout=2, deadline=time.monotonic() + 30)

    def control() -> None:
        if threading.get_ident() == owner:
            bridge.service()
        bridge()

    def acknowledge(unit, game, receipt):
        assert threading.get_ident() == owner
        acknowledged.append((unit, game))
        return receipt

    previous = worker.rep_fix.current()
    worker.rep_fix.apply(True, boards_discarded=True)
    try:
        result = worker.run_pooled_units(
            units, fixture.FakeEvaluator(), out, **args, control=control,
            service_control=bridge.service, acknowledge=acknowledge,
        )
        assert result["status"] == "COMPLETE_NOT_ADMISSION"
        assert len(acknowledged) == 4
        assert len(set(acknowledged)) == 4
        assert checks
        assert sum(map(len, result["unit_receipts"].values())) == 4
    finally:
        bridge.close()
        worker.rep_fix.apply(previous is True, boards_discarded=True)


def test_actual_worker_services_guard_during_producer_dispatcher_and_writer_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, out, units, args = small_units(tmp_path)
    owner = threading.get_ident()
    reached = {name: threading.Event() for name in ("GameProducer", "TeacherDispatcher", "TeacherWriter")}
    guarded_on = []
    futures = []

    def guard() -> None:
        assert threading.get_ident() == owner
        guarded_on.append(owner)

    bridge = OwnerThreadCallbacks(guard, capacity=4, timeout=5, deadline=time.monotonic() + 30)

    def control() -> None:
        if threading.get_ident() == owner:
            bridge.service()
        else:
            name = threading.current_thread().name
            for prefix, event in reached.items():
                if name.startswith(prefix):
                    event.set()
        bridge()

    # Replace only chess-domain admission. All three actual execution resources,
    # their callbacks, the worker's cleanup order and finite joins remain real.
    class TerminalDriver:
        def __init__(self, pools, *, producers, **_options):
            self.pool = next(iter(pools.values()))
            self.producers = producers
            self.paused = False
            self.peak_live = 0
            self.credits = SimpleNamespace(peak=0)
            self.cpu_microbatches = {}

        def advance(self):
            futures.append(self.producers.submit(control))
            futures.append(self.pool._writer.submit(control))
            rows = ((chess.Board(), np.zeros((175, 8, 8), dtype=np.float32)),)
            futures.append(self.pool.dispatcher.submit("shutdown-guard", rows))
            for event in reached.values():
                assert event.wait(2), "background caller did not reach owner handoff"
            assert all(not future.done() for future in futures)
            return AdvanceResult(terminal=True)

    monkeypatch.setattr(worker, "BT4GenerationDriver", TerminalDriver)
    previous = worker.rep_fix.current()
    worker.rep_fix.apply(True, boards_discarded=True)
    try:
        worker.run_pooled_units(units, fixture.FakeEvaluator(), out, **args,
                                control=control, service_control=bridge.service)
        assert all(future.done() for future in futures)
        for future in futures:
            future.result(0)
        assert guarded_on
    finally:
        bridge.close()
        worker.rep_fix.apply(previous is True, boards_discarded=True)
