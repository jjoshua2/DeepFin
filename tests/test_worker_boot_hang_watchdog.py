"""Audit R1/R3: the worker's lease-held init span must be bounded, and the
CUDA availability probe must not be a hang point.

R1 — the worker took a server lease in `_poll_manifest`, then ran
`model.to(self.device)` and `_build_evaluator`, both of which block forever on a
wedged WSL2 dxg bridge. Its only watchdog (`_start_selfplay_stall_watchdog`) is
armed after those calls and is gated on `_selfplay_session_active`, so the whole
span was uncovered: a wedged worker sat silently holding a 1 h lease, its shm
slots and its Stockfish children, producing no log line and never exiting.

R3 — `torch.cuda.is_available()` is itself a driver-init call unless
`PYTORCH_NVML_BASED_CUDA_CHECK=1`. It was set in production only because it was
inherited from the operator's shell; it appeared in no tracked file.

⚑ NOTHING HERE TOUCHES CUDA. Every test drives the watchdog's injected clock and
exit function, or reads source/env — the device this was written for is wedged
and a `torch.cuda` call would hang the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.broker_hang import (
    DEFAULT_WORKER_BOOT_HANG_ABORT_S,
    HANG_ABORT_EXIT_CODE,
    NVML_CUDA_CHECK_ENV,
    WORKER_HANG_ABORT_ENV,
    WORKER_HANG_ABORT_EXIT_CODE,
    BrokerHangWatchdog,
    pin_nvml_cuda_check,
    resolve_worker_boot_hang_abort_seconds,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _watchdog(clock: _Clock, exits: list[int], *, threshold_s: float = 100.0):
    return BrokerHangWatchdog(
        threshold_s=threshold_s,
        component="worker",
        exit_code=WORKER_HANG_ABORT_EXIT_CODE,
        exit_fn=exits.append,
        clock=clock,
    )


# ── the detector fires on a stuck stage, and only on a stuck stage ───────────


def test_a_stage_that_never_returns_aborts(caplog: pytest.LogCaptureFixture) -> None:
    clock, exits = _Clock(), []
    wd = _watchdog(clock, exits)

    with wd.stage("model_to_device"):
        assert wd.check_once() is False, "fired with the stage barely open"
        clock.t += 99.0
        assert wd.check_once() is False, "fired one second under the threshold"
        clock.t += 1.0
        with caplog.at_level("CRITICAL"):
            assert wd.check_once() is True

    assert exits == [WORKER_HANG_ABORT_EXIT_CODE]
    # The operator must be told WHICH stage wedged; "the worker died" alone
    # cannot distinguish a dead GPU from a bad checkpoint.
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "model_to_device" in msg, msg
    assert "worker hang abort" in msg, msg


def test_a_stage_that_completes_never_aborts() -> None:
    """Negative control. A long-but-finite model load must not be killed --
    otherwise this watchdog turns a slow cold cache into a crash loop."""
    clock, exits = _Clock(), []
    wd = _watchdog(clock, exits)

    with wd.stage("compile_inference_model"):
        clock.t += 99.0
    clock.t += 10_000.0  # long idle afterwards

    assert wd.check_once() is False
    assert exits == []


def test_the_oldest_stage_decides() -> None:
    """Nested/overlapping stages must not let a newer one mask an older one --
    completing the inner stage used to clear the outer one's start time."""
    clock, exits = _Clock(), []
    wd = _watchdog(clock, exits)

    with wd.stage("build_evaluator"):
        clock.t += 60.0
        with wd.stage("model_to_device"):
            clock.t += 1.0
        clock.t += 40.0  # outer is now 101s old, inner completed
        assert wd.check_once() is True
    assert exits == [WORKER_HANG_ABORT_EXIT_CODE]


def test_worker_and_broker_exit_codes_differ() -> None:
    """A supervisor should be able to tell which process's context died from the
    exit status alone."""
    assert WORKER_HANG_ABORT_EXIT_CODE != HANG_ABORT_EXIT_CODE


def test_disabled_threshold_never_fires() -> None:
    """`CAE_WORKER_BOOT_HANG_ABORT_S=0` is the escape hatch, and it is what an
    operator reaches for when the watchdog is misfiring. It must actually work."""
    clock, exits = _Clock(), []
    wd = _watchdog(clock, exits, threshold_s=0.0)
    with wd.stage("model_to_device"):
        clock.t += 10_000.0
        assert wd.check_once() is False
    assert exits == []
    wd.start()  # must not even spawn a thread
    assert wd._thread is None


# ── threshold resolution ─────────────────────────────────────────────────────


def test_threshold_env_override_and_default() -> None:
    assert resolve_worker_boot_hang_abort_seconds(env={}) == DEFAULT_WORKER_BOOT_HANG_ABORT_S
    assert resolve_worker_boot_hang_abort_seconds(
        env={WORKER_HANG_ABORT_ENV: "42.5"},
    ) == 42.5
    assert resolve_worker_boot_hang_abort_seconds(env={WORKER_HANG_ABORT_ENV: "0"}) == 0.0


def test_a_malformed_threshold_falls_back_to_the_default_not_to_disabled() -> None:
    """`float("banana")` raising would crash the worker at startup; treating it
    as 0 would silently DISABLE the watchdog, which is the one reading an
    operator never intends. Neither: fall back to the default and warn."""
    assert resolve_worker_boot_hang_abort_seconds(
        env={WORKER_HANG_ABORT_ENV: "banana"},
    ) == DEFAULT_WORKER_BOOT_HANG_ABORT_S


# ── the exit path: what it kills, and what it deliberately leaves ────────────


class _FakeProc:
    def __init__(self, pid: int, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive

    def poll(self) -> int | None:
        return None if self._alive else 0


class _FakeEngine:
    def __init__(self, pid: int, alive: bool = True) -> None:
        self.proc = _FakeProc(pid, alive)


class _FakePool:
    def __init__(self, pids: list[int]) -> None:
        self._engines = [_FakeEngine(p) for p in pids]


def _session(sf: Any) -> Any:
    """A `WorkerSession` reduced to what the hang-exit path reads."""
    import logging

    from chess_anti_engine.worker import WorkerSession

    session: Any = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_boot_hang")
    session.sf = sf
    return session


def test_the_hang_exit_kills_every_stockfish_child(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """`os._exit` skips every `finally`, so `_cleanup` -- and with it
    `sf.close()` -- never runs. Audit R2: an orphaned Stockfish is unmatchable
    by the repo's only reaper and survives until reboot holding ~2.6 GB. The
    watchdog therefore kills them itself before exiting.
    """
    from chess_anti_engine.worker import WorkerSession

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    session = _session(_FakePool([111, 222, 333]))

    with caplog.at_level("CRITICAL"):
        n = WorkerSession._kill_stockfish_children(session)

    assert n == 3
    assert [pid for pid, _ in killed] == [111, 222, 333]
    assert {sig for _, sig in killed} == {__import__("signal").SIGKILL}
    assert "SIGKILLed 3 Stockfish" in " ".join(r.getMessage() for r in caplog.records)


def test_already_dead_children_are_not_signalled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: without the `poll()` check this would report kills it
    did not make, and the count is the only evidence the step ran."""
    from chess_anti_engine.worker import WorkerSession

    killed: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, _sig: killed.append(pid))
    pool = _FakePool([111, 222])
    pool._engines[0].proc._alive = False

    assert WorkerSession._kill_stockfish_children(_session(pool)) == 1
    assert killed == [222]


def test_single_engine_and_no_engine_are_both_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`self.sf` is a `StockfishPool` with `--sf-workers > 1` and a bare
    `StockfishUCI` otherwise; both reach this path."""
    from chess_anti_engine.worker import WorkerSession

    killed: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, _sig: killed.append(pid))

    assert WorkerSession._kill_stockfish_children(_session(None)) == 0
    assert WorkerSession._kill_stockfish_children(_session(_FakeEngine(777))) == 1
    assert killed == [777]


def test_a_failing_kill_cannot_stop_the_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watchdog's whole job is to end the process. A child that vanished
    between poll and kill must not turn the abort into a traceback."""
    from chess_anti_engine.worker import WorkerSession

    def _boom(pid: int, _sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", _boom)
    assert WorkerSession._kill_stockfish_children(_session(_FakePool([1, 2]))) == 0

    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    session = _session(_FakePool([1, 2]))
    WorkerSession._boot_hang_exit(session, WORKER_HANG_ABORT_EXIT_CODE)
    assert exits == [WORKER_HANG_ABORT_EXIT_CODE]


def test_the_hang_exit_uses_os_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned because it is a deliberate, documented choice with a cost. The
    main thread is blocked in a driver call: `sys.exit` from the watchdog thread
    would unwind only that thread, and a self-SIGTERM lands in a handler that
    only sets flags the blocked main thread will never read."""
    from chess_anti_engine.worker import WorkerSession

    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)
    WorkerSession._boot_hang_exit(_session(_FakePool([9])), 43)
    assert exits == [43]


# ── R3: the NVML pin ─────────────────────────────────────────────────────────


def test_pin_sets_the_variable_when_absent() -> None:
    env: dict[str, str] = {}
    assert pin_nvml_cuda_check(env) is True
    assert env[NVML_CUDA_CHECK_ENV] == "1"


def test_pin_does_not_override_an_explicit_operator_choice() -> None:
    """setdefault, not set: someone who exports `0` to force the driver-based
    check -- e.g. to debug NVML itself -- keeps it."""
    env = {NVML_CUDA_CHECK_ENV: "0"}
    assert pin_nvml_cuda_check(env) is False
    assert env[NVML_CUDA_CHECK_ENV] == "0"


def test_the_worker_entry_point_pins_it_before_torch_is_probed() -> None:
    """The regression that matters: torch reads this with `os.getenv` at CALL
    time, so it is enough to set it before the first `is_available()` -- but
    only if something actually sets it. Driven as a subprocess with the variable
    scrubbed, so it cannot pass on an inherited value.
    """
    code = (
        "import os; os.environ.pop('PYTORCH_NVML_BASED_CUDA_CHECK', None);\n"
        "from chess_anti_engine.worker import _configure_worker_torch_env;\n"
        "_configure_worker_torch_env();\n"
        "print(os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK'))"
    )
    env = {k: v for k, v in os.environ.items() if k != NVML_CUDA_CHECK_ENV}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        timeout=300, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1", (out.stdout, out.stderr)


def test_train_sh_exports_it() -> None:
    """The script path is what production uses, and it covers the broker and
    ray as well as the worker -- the code-side pins only cover a process
    launched directly."""
    text = (Path(__file__).resolve().parents[1] / "scripts" / "train.sh").read_text()
    assert f'export {NVML_CUDA_CHECK_ENV}="${{{NVML_CUDA_CHECK_ENV}:-1}}"' in text, (
        "train.sh must export the NVML check, defaulting to 1 without clobbering "
        "an explicit operator value"
    )


# ── the stages are actually wired to the calls that wedge ────────────────────


@pytest.mark.parametrize(
    ("stage", "guarded_call"),
    [
        ("model_to_device", "model.to(self.device)"),
        ("compile_inference_model", "_maybe_compile_inference_model("),
        ("build_evaluator", "self._build_evaluator(self.model)"),
    ],
)
def test_each_blocking_init_call_is_inside_its_stage(stage: str, guarded_call: str) -> None:
    """A watchdog wired to nothing is the "gate that cannot fail" shape: every
    unit test above would still pass with no stage open anywhere in production.

    Checks the guarded call appears within a few lines AFTER its `stage(...)`
    line -- structural rather than exact-text, so reformatting does not break it
    but deleting the stage does.
    """
    from chess_anti_engine import worker as worker_mod

    lines = Path(worker_mod.__file__).read_text().splitlines()
    opens = [i for i, ln in enumerate(lines) if f'stage("{stage}")' in ln]
    assert len(opens) == 1, f"expected exactly one stage({stage!r}); found {len(opens)}"
    window = "\n".join(lines[opens[0]: opens[0] + 8])
    assert guarded_call in window, (
        f"stage({stage!r}) does not wrap {guarded_call!r}; the watchdog would "
        f"cover nothing on this path\n{window}"
    )


def test_the_watchdog_is_live_before_the_lease_is_taken() -> None:
    """R1 is about the span STARTING at lease acquisition, so the detector has
    to exist earlier than that -- i.e. be constructed in `__init__`, not lazily
    at session start like the stall watchdog it complements."""
    from chess_anti_engine import worker as worker_mod

    src = Path(worker_mod.__file__).read_text()
    init_at = src.index("self._boot_hang_watchdog = BrokerHangWatchdog(")
    start_at = src.index("self._boot_hang_watchdog.start()")
    poll_at = src.index("def _poll_manifest(")
    assert init_at < start_at < poll_at, (
        "the boot watchdog must be constructed and started in __init__, before "
        "any code path that can take a server lease"
    )
