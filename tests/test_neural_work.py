from __future__ import annotations

import pytest

from chess_anti_engine.neural_work import NeuralBudgetExceeded, NeuralWorkLedger, WorkCounters


def test_physical_work_and_search_acceptance_are_separate() -> None:
    work = NeuralWorkLedger(neural_budget=3)
    for key in range(3):
        work.submit(key)
    work.dispatch(0, (0, 1, 2), physical_rows=8)
    assert work.resolve(1, "cancelled")
    assert work.resolve(2, "stale")
    assert work.complete(0)
    assert work.resolve(0, "accepted")
    work.simulations_completed(5)  # terminal/cache hits do not consume NN rows
    report = work.snapshot(2.0)
    assert report["completed_simulations"] == 5
    assert report["forward_calls"] == 1
    assert report["executed_real_rows"] == 3
    assert report["accepted_neural_rows"] == 1
    assert report["padded_rows"] == 5
    assert report["useful_eps"] == 0.5
    assert report["executed_eps"] == 1.5
    assert report["executed_wasted_rows"] == 2
    assert not work.complete(0)
    assert not work.resolve(0, "accepted")
    assert not work.resolve(1, "accepted")
    assert not work.resolve(2, "accepted")
    assert work.snapshot(2.0) == report


def test_budget_counts_real_rows_not_calls_or_padding_and_never_refunds() -> None:
    work = NeuralWorkLedger(neural_budget=3)
    work.submit(0, 2)
    work.submit(1)
    work.submit(2)
    work.dispatch(0, (0,), physical_rows=8)
    assert work.complete(0, success=False)
    assert not work.complete(0)  # failure followed by a late success
    with pytest.raises(NeuralBudgetExceeded):
        work.dispatch(1, (1, 2), physical_rows=2)
    # Failed admission is atomic; the batch ID and requests remain usable.
    work.dispatch(1, (1,), physical_rows=1)
    work.complete(1)
    work.resolve(1, "accepted")
    report = work.snapshot(1.0)
    assert report["forward_calls"] == 2
    assert report["dispatched_real_rows"] == 3
    assert report["executed_real_rows"] == 1
    assert report["failed_rows"] == 2
    assert report["failed_forward_rows"] == 2
    assert report["padded_rows"] == 6
    assert report["useful_eps"] == 1.0


def test_rejected_before_dispatch_and_bad_output_are_not_useful() -> None:
    work = NeuralWorkLedger()
    work.submit(0, 4)
    work.resolve(0, "rejected")
    with pytest.raises(ValueError, match="disposed"):
        work.dispatch(0, (0,), physical_rows=4)
    work.submit(1)
    with pytest.raises(ValueError, match="before confirmed"):
        work.resolve(1, "accepted")
    work.dispatch(0, (1,), physical_rows=1)
    work.complete(0)
    work.resolve(1, "rejected")
    report = work.snapshot(1.0)
    assert report["rejected_rows"] == 5
    assert report["executed_real_rows"] == 1
    assert report["accepted_neural_rows"] == 0
    assert report["executed_wasted_rows"] == 1


def test_cancel_before_dispatch_cannot_be_resubmitted() -> None:
    work = NeuralWorkLedger()
    work.submit(0)
    work.resolve(0, "cancelled")
    with pytest.raises(ValueError, match="disposed"):
        work.dispatch(0, (0,), physical_rows=1)
    assert work.snapshot(0)["executed_real_rows"] == 0
    assert work.snapshot(0)["useful_eps"] is None


@pytest.mark.parametrize("rows", [-1, 0, True, 1.5])
def test_invalid_counts_do_not_mutate_counters(rows) -> None:
    counters = WorkCounters()
    with pytest.raises(ValueError, match="real_rows"):
        counters.dispatch(rows, 8)
    assert counters.forward_calls == 0


def test_backend_cannot_claim_search_acceptance_or_gpu_timing() -> None:
    counters = WorkCounters()
    counters.dispatch(3, 8)
    counters.executed_real_rows += 3
    report = counters.snapshot(2)
    assert report["accepted_neural_rows"] is None
    assert report["completed_simulations"] is None
    assert report["useful_eps"] is None
    assert report["phase_seconds"]["gpu"] is None
    assert report["real_batch_histogram"] == {3: 1}
    assert report["physical_batch_histogram"] == {8: 1}


def test_phase_validation_and_duplicate_ids() -> None:
    work = NeuralWorkLedger()
    work.submit(0)
    with pytest.raises(ValueError, match="already used"):
        work.submit(0)
    with pytest.raises(ValueError, match="duplicated"):
        work.dispatch(0, (0, 0), physical_rows=2)
    with pytest.raises(ValueError, match="physical_rows"):
        work.dispatch(0, (0,), physical_rows=0)
    work.record_phase("selection", 0.1)
    work.record_phase("selection", 0.2)
    assert work.snapshot(1)["phase_seconds"]["selection"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="phase"):
        work.record_phase("gpu", float("nan"))
    with pytest.raises(ValueError, match="wall_seconds"):
        work.snapshot(float("inf"))
