"""Tiny deterministic STOP/result interleavings; no model or native encoder."""
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts.shared_teacher_generation import BT4GamePool, StopSignal


class Stepper:
    def __init__(self, slot, rng, on_apply=lambda: None):
        self.slot = slot
        self.rng = rng
        self.on_apply = on_apply
        self.moves = []
        self.prepares = 0

    def apply_root_outputs(self, batch, outputs, *, temperatures):
        assert batch is not None
        assert temperatures == {self.slot: 1}
        self.moves.append((outputs, self.rng.random()))
        self.on_apply()

    def prepare_roots(self) -> Any:
        self.prepares += 1
        return None, [SimpleNamespace(slot_id=self.slot)]


class HeldWriter:
    def __init__(self):
        self.pending = []

    def submit(self, operation):
        future = Future()
        self.pending.append((operation, future))
        return future

    def finish(self):
        operation, future = self.pending.pop(0)
        future.set_result(operation())


class Dispatcher:
    def __init__(self):
        self.submissions = []

    def submit(self, key, rows):
        self.submissions.append((key, rows))
        return Future()


def pool(*, on_apply=lambda: None, capacity=2) -> tuple[Any, HeldWriter, Dispatcher, list[int]]:
    writer = HeldWriter()
    dispatcher = Dispatcher()
    committed = []

    def commit(game):
        committed.append(game.slot_id)
        return {"game_id": game.slot_id}

    # These deliberately tiny fixtures implement only the exercised protocol.
    value: Any = BT4GamePool(
        games=capacity, capacity=capacity, seed=14, temperature=1,
        make_stepper=cast(Any, lambda slot, rng: Stepper(slot, rng, on_apply)),
        dispatcher=cast(Any, dispatcher), commit=commit, max_writes=1, writer=cast(Any, writer),
    )
    value.advance()
    for game in value.active.values():
        game.batch = SimpleNamespace(generation=1)
        game.result = Future()
        game.result.set_result(("value",))
    return value, writer, dispatcher, committed


def test_stop_after_sweep_sample_before_later_unit_consumption():
    observed = [False]
    signal = StopSignal(lambda: observed[0])
    first, _, _, _ = pool(capacity=1)
    later, writer, dispatcher, committed = pool(capacity=1)
    assert signal.refresh() is False  # coordinator's old sweep snapshot
    observed[0] = True
    assert signal.refresh() is True  # backend, before publishing its future
    original = later.active[0].result
    state = repr(later.active[0].stepper.rng.bit_generator.state)
    for value in (first, later):
        value.advance(stopped=False, stop=signal.is_set)
    game = later.active[0]
    assert game.result is original
    assert game.batch is not None
    assert not game.stepper.moves
    assert game.stepper.prepares == 0
    assert repr(game.stepper.rng.bit_generator.state) == state
    assert not later.receipts
    assert not writer.pending
    assert not dispatcher.submissions
    assert not committed
    observed[0] = False
    assert signal.refresh() is True  # removing a marker cannot clear this run


def test_stop_during_one_pool_advance_preserves_other_results_and_rng():
    observed = [False]
    signal = StopSignal(lambda: observed[0])

    def stop_after_first_move():
        observed[0] = True
        signal.refresh()

    value, writer, dispatcher, committed = pool(on_apply=stop_after_first_move)
    second = value.active[1]
    state = repr(second.stepper.rng.bit_generator.state)
    original = second.result
    value.advance(stop=signal.is_set)
    assert len(value.active[0].stepper.moves) == 1  # preceding action is retained
    assert value.active[0].stepper.prepares == 0
    assert not second.stepper.moves
    assert second.stepper.prepares == 0
    assert second.result is original
    assert repr(second.stepper.rng.bit_generator.state) == state
    assert not value.receipts
    assert not writer.pending
    assert not dispatcher.submissions
    assert not committed


def test_stop_during_future_read_is_checked_before_application():
    signal = StopSignal(lambda: True)
    value, _, _, _ = pool(capacity=1)

    class CompletingFuture(Future):
        def result(self, timeout=None):
            signal.refresh()
            return super().result(timeout)

    future = CompletingFuture()
    future.set_result(("value",))
    value.active[0].result = future
    value.advance(stop=signal.is_set)
    assert not value.active[0].stepper.moves
    assert value.active[0].result is future


def test_stop_blocks_ready_submission_and_actor_refill():
    value, _, dispatcher, _ = pool(capacity=1)
    signal = StopSignal(lambda: True)

    class Batch:
        generation = 1

        def inference_inputs(self):
            signal.refresh()
            return [object()], [np.zeros(1)]

    value.active[0].result = None
    value.active[0].batch = Batch()
    value._ready.append(0)
    value.advance(stop=signal.is_set)
    assert not dispatcher.submissions
    assert list(value._ready) == [0]
    value.active.clear()
    value._ready.clear()
    value.next_id = 0
    value.advance(stop=signal.is_set)
    assert not value.active
    assert value.next_id == 0


def test_admitted_write_drains_at_stop_without_new_companion_ack():
    value, writer, _, committed = pool(capacity=1)
    game = value.active[0]
    game.finalized = SimpleNamespace(slot_id=0)
    game.result = None
    game.batch = None
    value.advance()
    assert len(writer.pending) == 1
    acknowledgments = []
    value.acknowledge = lambda slot, receipt: acknowledgments.append(slot) or receipt
    writer.finish()
    value.advance(stopped=True)
    assert committed == [0]
    assert not acknowledgments
    assert not value.receipts
    assert game.raw_receipt == {"game_id": 0}
    value.advance()  # resume acknowledges the exact existing raw receipt
    assert acknowledgments == [0]
    assert value.receipts == {0: {"game_id": 0}}


def test_previously_finalized_outcome_can_publish_at_stop():
    value, writer, _, committed = pool(capacity=1)
    value.active[0].finalized = SimpleNamespace(slot_id=0)
    value.advance(stopped=True)
    assert len(writer.pending) == 1
    writer.finish()
    value.advance(stopped=True)
    assert committed == [0]
    assert value.receipts == {0: {"game_id": 0}}


@pytest.mark.parametrize("pause", [False, True])
def test_resume_preserves_exact_result_rng_and_commit(pause):
    value, writer, _, committed = pool(capacity=1)
    if pause:
        signal = StopSignal(lambda: True)
        signal.refresh()
        value.advance(stop=signal.is_set)
        assert not value.active[0].stepper.moves
    # A resumed invocation has a fresh signal; pending result is not resubmitted.
    value.advance(stop=StopSignal(lambda: False).is_set)
    assert value.active[0].stepper.moves == [(("value",), np.random.default_rng(np.random.SeedSequence([14, 0])).random())]
    value.advance()
    writer.finish()
    value.advance()
    assert committed == [0]
    assert value.receipts == {0: {"game_id": 0}}


def test_real_coordinator_latches_backend_stop_before_later_unit(tmp_path, monkeypatch):
    """Force a result after the sweep sample, before a later unit advances."""
    import threading
    from scripts import bt4_root_policy_worker as worker

    class Spec:
        model_sha256 = "0" * 64
        requested_provider = "cpu"
        providers = ("CPUExecutionProvider",)
        syzygy_path = "unused-fixture"
        parallel_games = 1
        games = 1

        def __init__(self, out):
            self.out = out
            self.provider_options = {}

        def validate(self):
            pass

    class Batch:
        generation = 1

        def inference_inputs(self):
            return [object()], [np.zeros(1)]

    class RootStepper(Stepper):
        def prepare_roots(self) -> Any:
            self.prepares += 1
            return Batch(), []

    out = tmp_path / "coordinator"
    pools = []
    completed = threading.Event()
    admissions = []

    def evaluate(_evaluator, rows, _specs):
        (out / "STOP").write_text("stop before result publication", encoding="utf-8")
        completed.set()
        return ["value"] * len(rows)

    def make_pool(_spec, _handle, _games_dir, dispatcher, **kwargs):
        value: Any = BT4GamePool(
            games=1, capacity=1, seed=14, temperature=1,
            make_stepper=cast(Any, RootStepper),
            dispatcher=dispatcher, commit=lambda game: admissions.append(game) or {},
            max_writes=1, writer=kwargs["writer"],
        )
        value.advance()  # build actors before the deliberately stale sweep
        value.active[0].batch = Batch()
        value._ready.append(0)
        advance = value.advance
        index = len(pools)

        def interleaved_advance(**options):
            if index == 1:
                # Unit0 submitted below; wait until the real dispatcher has
                # published that result, after STOP and the backend latch.
                assert completed.wait(2)
                pools[0].active[0].result.result(timeout=2)
                future = Future()
                future.set_result(("later",))
                value.active[0].result = future
                value._ready.clear()
            return advance(**options)

        value.advance = interleaved_advance
        pools.append(value)
        return value

    monkeypatch.setattr(worker.rep_fix, "current", lambda: True)
    monkeypatch.setattr(worker.tablebase, "SyzygyProbe", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "table_file_inventory", lambda path: {})
    monkeypatch.setattr(worker, "_worker_manifest", lambda *args: {})
    monkeypatch.setattr(worker, "fsync_directory", lambda path: None)
    monkeypatch.setattr(worker, "_make_pooled_unit", make_pool)
    monkeypatch.setattr(worker, "_evaluate_shared_roots", evaluate)
    result = worker.run_pooled_units(
        cast(Any, {"a": (Spec(out / "a"), None), "b": (Spec(out / "b"), None)}), cast(Any, object()), out,
        max_units=2, max_live_games=2, target_rows=1, max_rows=2,
        batch_wait_ms=0, max_writes=1, deadline_seconds=5,
    )
    assert result["status"] == "PAUSED_NOT_ADMISSION"
    assert all(not receipts for receipts in result["unit_receipts"].values())
    assert not admissions
    assert not list(out.glob("*/games/*.npz"))
    for value in pools:
        assert not value.active[0].stepper.moves
        assert value.active[0].stepper.prepares == 0
