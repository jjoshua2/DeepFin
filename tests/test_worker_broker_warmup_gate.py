"""Broker-readiness gate at selfplay session start (`_await_broker_ready`).

A freshly launched broker torch.compiles the model on its FIRST forward
(~1-2 min under reduce-overhead), longer than the client's 30s request
timeout. Selfplay threads spawned into that window die on their first
``evaluate_legal_bf16`` and are never respawned — measured 4/32 threads per
worker on live trial 379f6 (-12.5% selfplay capacity for the session). The
gate sends one production-shaped probe through the EXISTING client (the
probe itself triggers the compile) before any selfplay thread is spawned.

Clocks are injected (worker_mod.time is swapped for a fake) — no
wall-clock-sensitive assertions; the suite runs under nice 19.
"""
from __future__ import annotations

import inspect
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

from chess_anti_engine import worker as worker_mod
from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.worker import WorkerSession


class _FakeTime:
    """Deterministic stand-in for the ``time`` module inside worker.py."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(float(s))
        self.now += float(s)


class _FakeClient:
    """Times out ``timeouts`` probe attempts, then succeeds.

    Each attempt advances the injected clock by ``attempt_s`` — the client's
    per-attempt request timeout — the way a real 30s timeout burns wall time.
    """

    def __init__(
        self, clock: _FakeTime, *, timeouts: int, attempt_s: float = 30.0,
        exc: BaseException | None = None,
    ) -> None:
        self._clock = clock
        self._timeouts = int(timeouts)
        self._attempt_s = float(attempt_s)
        self._exc = exc
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((x, legal_flat, legal_counts))
        if len(self.calls) <= self._timeouts:
            self._clock.now += self._attempt_s
            raise self._exc if self._exc is not None else TimeoutError(
                "inference broker timed out after 30.000s"
            )
        return (
            np.zeros((int(legal_flat.shape[0]),), dtype=np.uint16),
            np.zeros((int(x.shape[0]), 3), dtype=np.float32),
        )


def _session(
    client: _FakeClient, clock: _FakeTime, monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkerSession, Mock, Mock]:
    monkeypatch.setattr(worker_mod, "time", clock)
    session = object.__new__(WorkerSession)
    s = cast("Any", session)
    s.inference_client = client
    s.log = Mock()
    s._stop_selfplay = False
    s._shutdown_requested = False
    progress = Mock()
    s._note_selfplay_progress = progress
    return session, s.log, progress


def _cfgs() -> dict:
    return {"game": GameConfig()}


def test_gate_retries_through_timeouts_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K timeouts then success => K+1 probe attempts and the ready INFO line."""
    monkeypatch.delenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", raising=False)
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=3)
    session, log, _ = _session(client, clock, monkeypatch)

    session._await_broker_ready(_cfgs())

    assert len(client.calls) == 4
    log.error.assert_not_called()
    ready_lines = [
        c for c in log.info.call_args_list if "broker ready" in c.args[0]
    ]
    assert len(ready_lines) == 1
    # "broker ready after %.1fs (%d probe attempt(s))"
    assert ready_lines[0].args[2] == 4


def test_gate_probe_is_production_shaped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe must be the transport the session will use: one starting
    position in bf16 bits at the session's plane count, with full-4672 legal
    ids whose count matches — the exact `evaluate_legal_bf16` contract."""
    monkeypatch.delenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", raising=False)
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=0)
    session, _, _ = _session(client, clock, monkeypatch)
    cfgs = _cfgs()

    session._await_broker_ready(cfgs)

    (x, legal_flat, legal_counts), = client.calls
    planes = input_plane_count(cfgs["game"].input_extra_features)
    assert x.shape == (1, planes, 8, 8)
    assert x.dtype == np.uint16  # bf16 bits — _MODE_LEGAL_BF16's input dtype
    assert legal_counts.tolist() == [20]  # the starting position's legal moves
    assert legal_flat.shape == (20,)
    assert legal_flat.dtype == np.int32
    assert int(legal_flat.min()) >= 0
    assert int(legal_flat.max()) < POLICY_SIZE


def test_gate_is_a_single_probe_when_broker_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steady state: exactly one probe, no sleeps, no warnings/errors."""
    monkeypatch.delenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", raising=False)
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=0)
    session, log, progress = _session(client, clock, monkeypatch)

    session._await_broker_ready(_cfgs())

    assert len(client.calls) == 1
    assert clock.sleeps == []  # the gate itself adds no pacing delay
    log.error.assert_not_called()
    log.warning.assert_not_called()
    progress.assert_not_called()


def test_gate_gives_up_at_the_overall_deadline_and_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Always-timing-out broker: the gate logs ONE loud ERROR at the overall
    deadline and returns (no exception) — session start proceeds with today's
    behavior instead of turning a slow broker into a worker outage."""
    monkeypatch.delenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", raising=False)
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=10**9)
    session, log, progress = _session(client, clock, monkeypatch)

    session._await_broker_ready(_cfgs())  # must return, not raise

    # Default budget 240s at 30s per attempt: attempts at t=30..240, the
    # 8th crosses the deadline.
    assert len(client.calls) == 8
    log.error.assert_called_once()
    assert "broker not ready" in log.error.call_args.args[0]
    # The watchdog stayed fed on every timed-out attempt.
    assert progress.call_count == 8


def test_gate_deadline_env_override_is_in_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAE_WORKER_BROKER_WARMUP_TIMEOUT_S changes the realized budget."""
    monkeypatch.setenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", "45")
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=10**9)
    session, log, _ = _session(client, clock, monkeypatch)

    session._await_broker_ready(_cfgs())

    # 45s budget at 30s per attempt: t=30 < 45 retries, t=60 >= 45 gives up.
    assert len(client.calls) == 2
    log.error.assert_called_once()
    # The realized env value is named in the ERROR line (dead-knob rule).
    assert "45" in repr(log.error.call_args.args)


def test_gate_zero_timeout_disables_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", "0")
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=0)
    session, log, _ = _session(client, clock, monkeypatch)

    session._await_broker_ready(_cfgs())

    assert client.calls == []
    disabled = [c for c in log.info.call_args_list if "DISABLED" in c.args[0]]
    assert len(disabled) == 1


def test_gate_bad_env_value_warns_and_uses_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric override must not be silently in effect (dead-knob rule)."""
    monkeypatch.setenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", "fast")
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=0)
    session, log, _ = _session(client, clock, monkeypatch)

    session._await_broker_ready(_cfgs())

    assert len(client.calls) == 1  # gate still runs, on the 240s default
    log.warning.assert_called_once()
    assert "NOT in effect" in log.warning.call_args.args[0]


def test_gate_passes_a_non_timeout_error_to_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-timeout probe failure is not "still compiling": the gate proceeds
    after one attempt so the session-level reset/retry machinery sees the same
    failure unchanged, instead of burning the whole warmup budget on it."""
    monkeypatch.delenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", raising=False)
    clock = _FakeTime()
    client = _FakeClient(
        clock, timeouts=10**9,
        exc=RuntimeError("inference broker shut down while request was in flight"),
    )
    session, log, _ = _session(client, clock, monkeypatch)

    session._await_broker_ready(_cfgs())  # must return, not raise

    assert len(client.calls) == 1
    log.warning.assert_called_once()
    log.error.assert_not_called()


def test_gate_skips_loudly_on_unexpected_cfgs_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cfgs dict without the production "game" key (a test stub today, a
    changed cfg shape tomorrow) must not crash session start: the gate skips
    with a loud warning and dispatch is still reached (the gate returns
    normally, and tests/test_selfplay_resume.py drives the real _run_selfplay
    through dispatch with exactly such a stub)."""
    monkeypatch.delenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", raising=False)
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=0)
    session, log, _ = _session(client, clock, monkeypatch)

    session._await_broker_ready({"opening": object()})  # no "game": must not raise

    assert client.calls == []  # probe construction failed before any request
    log.warning.assert_called_once()
    assert "broker warmup gate SKIPPED" in log.warning.call_args.args[0]
    log.error.assert_not_called()


def test_gate_bails_on_stop_before_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shutdown requested mid-warmup must not be held up by probe attempts."""
    monkeypatch.delenv("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", raising=False)
    clock = _FakeTime()
    client = _FakeClient(clock, timeouts=0)
    session, _, _ = _session(client, clock, monkeypatch)
    cast("Any", session)._shutdown_requested = True

    session._await_broker_ready(_cfgs())

    assert client.calls == []


def test_run_selfplay_gates_before_threads_spawn_and_before_session_active() -> None:
    """Wiring pin (same style as test_run_selfplay_starts_the_model_watch_thread:
    _run_selfplay is too heavy to drive end-to-end here). The gate must run
    BEFORE any selfplay thread can be spawned (_dispatch_selfplay_one_shard,
    which covers the threaded AND single broker paths) and BEFORE the session
    is marked active, so the stall watchdog cannot see a legitimate warmup."""
    src = inspect.getsource(WorkerSession._run_selfplay)
    gate = src.index("self._await_broker_ready(cfgs)")
    session_active = src.index("self._selfplay_session_active = True")
    dispatch = src.index("self._dispatch_selfplay_one_shard(")
    assert gate < session_active < dispatch


def test_gate_only_runs_on_the_broker_path() -> None:
    """Local-evaluator sessions must not pay a probe: the call site is guarded
    on inference_client, and the method itself refuses a clientless session."""
    src = inspect.getsource(WorkerSession._run_selfplay)
    gate = src.index("self._await_broker_ready(cfgs)")
    guard = src.index("if self.inference_client is not None:")
    assert guard < gate

    session = object.__new__(WorkerSession)
    cast("Any", session).inference_client = None
    cast("Any", session).log = Mock()
    session._await_broker_ready({"game": GameConfig()})  # no probe, no raise
