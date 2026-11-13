"""Numeric admission uses real UCI entry points without constructing a worker."""
from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from chess_anti_engine.mcts.search_options import OPTIONS_BY_NAME
from chess_anti_engine.uci.__main__ import _refuse_out_of_range_startup_value
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command


@pytest.fixture
def admission_engine(monkeypatch: pytest.MonkeyPatch) -> Engine:
    # Only admission/dispatch state is needed. Never construct an evaluator,
    # model, search worker, thread, or engine runtime.
    engine = object.__new__(Engine)
    engine._options = EngineOptions()
    engine._closed = False
    engine._quit_requested = False
    engine._recovery_lock = threading.Lock()  # setoption retires a staged rebuild
    engine._recovery_epoch = 0
    engine._recovery_thread = None
    engine._staged_evaluator = None
    engine._deferred_setoptions = []
    engine._worker = Mock()  # setoption snapshots its search state; no real worker
    monkeypatch.setattr(engine, "_wait_for_search", Mock())
    monkeypatch.setattr(
        engine, "_apply_search_option",
        Mock(side_effect=AssertionError("Rejected option reached the worker")),
    )
    return engine


@pytest.mark.parametrize("raw", ["9" * 400, "-" + "9" * 400])
def test_huge_spin_rejected_and_dispatch_remains_usable(
    admission_engine: Engine, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], raw: str,
) -> None:
    engine = admission_engine
    before = engine._options.search_value("topk")
    ready = Mock()
    monkeypatch.setattr(engine, "_handle_isready", ready)
    engine.dispatch(parse_command(f"setoption name Topk value {raw}"))
    assert "out of range" in capsys.readouterr().out
    assert engine._options.search_value("topk") == before
    engine.dispatch(parse_command("isready"))
    ready.assert_called_once_with()
    with pytest.raises(SystemExit, match="accepts"):
        _refuse_out_of_range_startup_value(OPTIONS_BY_NAME["topk"], "topk", int(raw))


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e400"])
def test_float_nonfinite_values_remain_rejected(
    admission_engine: Engine, raw: str,
) -> None:
    opt = OPTIONS_BY_NAME["policytemperature"]
    assert admission_engine._parse_search_value(opt, raw) is None
    with pytest.raises(SystemExit, match="finite"):
        _refuse_out_of_range_startup_value(opt, "policy_temp", float(raw))


@pytest.mark.parametrize(
    ("name", "raw", "expected", "expected_type"),
    [("topk", "2", 2, int), ("topk", "256", 256, int),
     ("policytemperature", "0.5", 0.5, float),
     ("policytemperature", "5", 5.0, float)],
)
def test_numeric_endpoints_keep_native_type(
    admission_engine: Engine, name: str, raw: str,
    expected: int | float, expected_type: type,
) -> None:
    opt = OPTIONS_BY_NAME[name]
    parsed = admission_engine._parse_search_value(opt, raw)
    assert parsed == expected
    assert type(parsed) is expected_type
    _refuse_out_of_range_startup_value(opt, opt.field, expected)


@pytest.mark.parametrize(
    ("name", "raw"),
    [("topk", "1"), ("topk", "257"), ("topk", "-1"),
     ("policytemperature", "0.4"), ("policytemperature", "5.1")],
)
def test_numeric_values_outside_bounds_still_rejected(
    admission_engine: Engine, name: str, raw: str,
) -> None:
    opt = OPTIONS_BY_NAME[name]
    assert admission_engine._parse_search_value(opt, raw) is None
    value = int(raw) if opt.kind == "spin" else float(raw)
    with pytest.raises(SystemExit, match="accepts"):
        _refuse_out_of_range_startup_value(opt, opt.field, value)
