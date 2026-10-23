"""Defect 7: malformed shared-slot row counts never reach dispatch."""
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine import inference as inf


class Watchdog:
    def mark_forward_start(self, batch: int) -> int:
        assert batch == 2
        return 1

    def mark_forward_done(self, *, success: bool, token: int) -> None:
        assert success
        assert token == 1


def slot(rows: int, *, mode: int = inf._MODE_DENSE_F32) -> Any:
    return SimpleNamespace(
        batch_size=rows, request_id=19, request_mode=mode,
        state=inf._STATE_REQUEST,
        input=np.full((2, 4), 3, dtype=np.float32),
        policy=np.full((2, 8), -99, dtype=np.float32),
        wdl=np.full((2, 3), -99, dtype=np.float32),
    )


def broker() -> Any:
    b = object.__new__(inf.SharedSlotBroker)
    b.device = "cpu"
    b._trial_models = cast(Any, {"healthy": object(), "bad": object()})
    b._trial_streams = {}
    b._trial_no_model_warned_at = {}
    b._trial_bad_mode_warned_at = {}
    b._trial_bad_size_warned_at = {}
    b._first_inference_done = False
    b._hang_watchdog = cast(Any, Watchdog())
    b.unsupported_mode_requests = 0
    return b


@pytest.mark.parametrize("rows", [-2, 0, 3, 1000000])
@pytest.mark.parametrize("same_trial", [False, True])
def test_invalid_rows_do_not_poison_healthy_neighbor(monkeypatch, rows, same_trial):
    b = broker()
    bad, good = slot(rows), slot(2)
    calls = []

    def forward(_model, x, *, device):
        del device
        calls.append(x.clone())
        return {"policy": torch.full((2, 8), 7.0), "wdl": torch.full((2, 3), 5.0)}

    monkeypatch.setattr(inf, "_forward_no_grad", forward)
    monkeypatch.setattr(inf, "_policy_output_full", lambda out: out["policy"])
    ready = {"healthy": [bad, good]} if same_trial else {"bad": [bad], "healthy": [good]}
    assert b._process_parallel(ready) == 0
    assert len(calls) == 1
    assert tuple(calls[0].shape) == (2, 4)
    assert bad.state == inf._STATE_IDLE
    assert np.all(bad.policy == -99)
    assert good.state == inf._STATE_RESPONSE
    assert good.request_id == 19
    assert np.all(good.policy == 7)
    assert b._last_served_rows == 2


def test_invalid_counts_with_no_model_or_unsupported_mode_are_not_throughput():
    b = broker()
    b._trial_models = {}
    bad = slot(-2)
    assert b._process_parallel({"bad": [bad]}) == 0
    assert b._last_served_rows == 0
    assert bad.state == inf._STATE_IDLE
    b._trial_models = {"bad": object()}
    unsupported = slot(2, mode=inf._MODE_LEGAL_BF16)
    assert b._process_parallel({"bad": [unsupported]}) == 2
    assert b._last_served_rows == 0
    assert unsupported.state == inf._STATE_IDLE
    assert b.unsupported_mode_requests == 1


def test_serve_loop_reports_only_successfully_scattered_rows(monkeypatch, capsys):
    b = broker()
    bad, good = slot(-2), slot(2)
    b._stop = False
    b._all_slots = [("healthy", bad), ("healthy", good)]
    b._trial_slots = {"healthy": []}
    b._idle_since = None
    b._hang_watchdog.start = lambda: None
    ticks = iter([1.0, 2.0, 3.0, 13.0])
    monkeypatch.setattr(inf.time, "monotonic", lambda: next(ticks))

    def forward(_model, _x, *, device):
        del device
        b._stop = True
        return {"policy": torch.ones((2, 8)), "wdl": torch.ones((2, 3))}

    monkeypatch.setattr(inf, "_forward_no_grad", forward)
    monkeypatch.setattr(inf, "_policy_output_full", lambda out: out["policy"])
    b.serve_forever()
    report = capsys.readouterr().out
    assert "avg 2.0 pos/batch" in report
    assert "-" not in report.split("avg ", 1)[1].split(" pos/s", 1)[0]
    assert bad.state == inf._STATE_IDLE
    assert good.state == inf._STATE_RESPONSE


def test_repeated_bad_size_rejection_throttles_warnings(monkeypatch, caplog):
    b = broker()
    ticks = iter([100.0, 100.01, 129.99, 130.0])
    monkeypatch.setattr(inf.time, "monotonic", lambda: next(ticks))
    bad = slot(-2)
    for _ in range(4):
        bad.state = inf._STATE_REQUEST
        assert b._process_parallel({"bad": [bad]}) == 0
        assert bad.state == inf._STATE_IDLE
        assert b._last_served_rows == 0
    warnings = [record for record in caplog.records if "invalid batch_size" in record.message]
    assert len(warnings) == 2
