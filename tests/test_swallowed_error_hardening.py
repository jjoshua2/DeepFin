"""Corrupt input must produce a DIFFERENT observation than absent input.

Tranche 8/10 of `scratchpad/audit_coverage_map_20260804.md`. Every finding here
is the same shape: an `except` that turns a corrupt value into the same result
as a legitimately missing one, so nothing downstream — and nobody reading the
logs — can tell the two apart. None of them alters behaviour on healthy input;
each adds an observation that did not exist.

⚑ SCOPE CORRECTION ON S1, recorded because the audit rated it HIGH on a premise
it flagged as unverified ("I did not read the call site"). I read it. The premise
does not hold:

* `_active_difficulty` has two production callers and NEITHER sets the
  difficulty: `_on_completed_game` stamps the pair onto the SHARD'S METADATA,
  and the session-start log line this tranche added reads it to print what the
  session adopted;
* the difficulty games are actually PLAYED at comes from
  `_build_selfplay_configs`, which reads the same two reco keys independently
  and casts them UNGUARDED — a corrupt value raises there and takes the session
  down loudly.

So a corrupt reco never produced well-formed training data at the wrong
curriculum point. It produced shards recording "difficulty unknown" while the
server had in fact published a value: a provenance defect, not a training-data
one. It still matters — the promotion gate's anchored current-vs-previous delta
is checked against that field — but it is MEDIUM, not HIGH.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from chess_anti_engine.worker import WorkerSession


class _RecoStub:
    """Only the surface `_active_difficulty` touches.

    Constructing a real `WorkerSession` needs a server, an argparse namespace
    and a GPU broker, so the shipped method is borrowed onto a stub — the same
    harness `test_promotion_gate.py` already uses for this method.
    """

    def __init__(self, reco: Any) -> None:
        self._active_reco = reco
        self._reco_corrupt_count = 0
        self._reco_corrupt_last: tuple[Any, Any] | None = None
        self.log = logging.getLogger("test_reco_stub")


def _difficulty_of(stub: object) -> tuple[float | None, int | None]:
  # Via `object`: a stub self is the honest harness here, and basedpyright
  # rejects a direct cast between two non-overlapping types.
    return WorkerSession._active_difficulty(cast("WorkerSession", stub))


# --------------------------------------------------------------------------
# S1 — a corrupt reco must be distinguishable from an absent one
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reco",
    [
        {"opponent_wdl_regret_limit": "not-a-number", "sf_nodes": 698_000},
        {"opponent_wdl_regret_limit": 0.0875, "sf_nodes": "lots"},
        {"opponent_wdl_regret_limit": {"nested": 1}, "sf_nodes": 698_000},
        {"opponent_wdl_regret_limit": 0.0875, "sf_nodes": [1, 2]},
        {"opponent_wdl_regret_limit": "None", "sf_nodes": 698_000},
    ],
)
def test_a_corrupt_reco_warns_and_counts(
    reco: dict[str, Any], caplog: pytest.LogCaptureFixture,
) -> None:
    """RED on origin/main: the old code returned (None, None) silently."""
    stub = _RecoStub(reco)

    with caplog.at_level(logging.WARNING, logger="test_reco_stub"):
        assert _difficulty_of(stub) == (None, None)

    assert stub._reco_corrupt_count == 1, "the corrupt path must be counted"
    assert "CORRUPT, not absent" in caplog.text, (
        "a corrupt reco must say so; it used to be byte-identical to a server "
        "that published no reco at all"
    )


@pytest.mark.parametrize(
    "reco",
    [
        {},
        {"opponent_wdl_regret_limit": None, "sf_nodes": None},
        {"sf_nodes": 698_000, "opponent_wdl_regret_limit": None},
        {"opponent_wdl_regret_limit": 0.0875, "sf_nodes": None},
    ],
)
def test_a_genuinely_ABSENT_reco_is_silent(
    reco: dict[str, Any], caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ NEGATIVE CONTROL — the brief's explicit constraint.

    Absent keys are the LEGITIMATE case the `is not None` guards exist for: a
    worker that has not yet been handed a reco, or a server that publishes only
    one of the pair. Warning on those would make the new line meaningless
    within a poll or two, and the counter useless. Nothing may change here.
    """
    stub = _RecoStub(reco)

    with caplog.at_level(logging.WARNING, logger="test_reco_stub"):
        result = _difficulty_of(stub)

    assert stub._reco_corrupt_count == 0, "absent is not corrupt"
    assert caplog.text == "", "an absent reco key must stay silent"
    assert result[0] in (None, 0.0875)
    assert result[1] in (None, 698_000)


def test_a_valid_reco_is_unchanged_and_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The healthy path must not move at all."""
    stub = _RecoStub({"opponent_wdl_regret_limit": 0.0875, "sf_nodes": 698_000})

    with caplog.at_level(logging.WARNING, logger="test_reco_stub"):
        assert _difficulty_of(stub) == (0.0875, 698_000)

    assert stub._reco_corrupt_count == 0
    assert caplog.text == ""


def test_no_reco_at_all_returns_unknown_without_touching_the_counter() -> None:
    """`_active_reco` unset entirely: the pre-first-poll state, still silent."""

    class _Bare:
        pass

    bare: object = _Bare()
    result = WorkerSession._active_difficulty(cast("WorkerSession", cast(Any, bare)))
    assert result == (None, None)


def test_the_warning_is_rate_limited_but_the_counter_is_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ `_active_difficulty` runs once per COMPLETED GAME.

    An unconditional warning would be a per-game flood on a stuck-corrupt
    manifest, which buries the line it exists to make greppable. The count must
    still be exact, or "how bad is it" becomes unanswerable.
    """
    stub = _RecoStub({"opponent_wdl_regret_limit": "bad", "sf_nodes": 1})

    with caplog.at_level(logging.WARNING, logger="test_reco_stub"):
        for _ in range(5):
            _difficulty_of(stub)

    assert stub._reco_corrupt_count == 5, "every occurrence counts"
    assert caplog.text.count("CORRUPT, not absent") == 1, "one line per value"

    with caplog.at_level(logging.WARNING, logger="test_reco_stub"):
        stub._active_reco = {"opponent_wdl_regret_limit": "different", "sf_nodes": 1}
        _difficulty_of(stub)

    assert caplog.text.count("CORRUPT, not absent") == 2, (
        "a NEW offending value must warn again -- rate-limiting per value, not "
        "once forever, or a second distinct corruption is invisible"
    )


def test_the_offending_value_appears_in_the_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warning that does not name the value cannot be acted on."""
    stub = _RecoStub({"opponent_wdl_regret_limit": "sixty-percent", "sf_nodes": 42})

    with caplog.at_level(logging.WARNING, logger="test_reco_stub"):
        _difficulty_of(stub)

    assert "sixty-percent" in caplog.text


# --------------------------------------------------------------------------
# S2 — corrupt arena results are quarantined, not filed as "uploaded"
# --------------------------------------------------------------------------


class _ArenaStub:
    """The surface `_upload_pending_arena_results` touches for a bad file."""

    def __init__(self, root: Path) -> None:
        self.arena_pending_dir = root / "pending"
        self.arena_uploaded_dir = root / "uploaded"
        self.arena_rejected_dir = root / "rejected"
        for d in (
            self.arena_pending_dir, self.arena_uploaded_dir, self.arena_rejected_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self.leased_trial_id = "t0"
        self.fixed_trial_id = ""
        self.trial_api_prefix = "/v1"
        self._arena_rejected_count = 0
        self.log = logging.getLogger("test_arena_stub")
        self._auth = ("u", "p")

    def _server_url_for(self, path: str) -> str:
        return "http://unused" + path


def _drain(stub: object) -> None:
    WorkerSession._upload_pending_arena_results(cast("WorkerSession", stub))


def test_a_corrupt_arena_result_goes_to_rejected_not_uploaded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """RED on origin/main, which moved it into `uploaded`.

    `uploaded` means "this reached the server". A file that reached nobody
    sitting in it is indistinguishable from a genuine upload, so any tooling
    counting uploads by listing the directory over-reports -- and arena results
    feed Elo readings and promotion decisions.
    """
    stub = _ArenaStub(tmp_path)
    bad = stub.arena_pending_dir / "result_0001.json"
    bad.write_text("{not json at all", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="test_arena_stub"):
        _drain(stub)

    assert (stub.arena_rejected_dir / "result_0001.json").exists()
    assert not (stub.arena_uploaded_dir / "result_0001.json").exists(), (
        "a file that reached nobody must not be filed as uploaded"
    )
    assert stub._arena_rejected_count == 1
    assert "was NOT uploaded" in caplog.text


def test_the_retry_storm_protection_is_preserved(tmp_path: Path) -> None:
    """⚑ The motivation for the original code was sound; keep it.

    The point of the move was that a permanently unreadable file must leave
    `pending` in ONE rename, or every drain re-reads it forever. Only the
    destination changed.
    """
    stub = _ArenaStub(tmp_path)
    (stub.arena_pending_dir / "bad.json").write_text("<<<", encoding="utf-8")

    _drain(stub)
    assert list(stub.arena_pending_dir.glob("*.json")) == [], (
        "the bad file must be out of pending after one drain"
    )

    _drain(stub)
    assert stub._arena_rejected_count == 1, (
        "a second drain must not re-process it -- that is the retry storm"
    )


def test_a_missing_quarantine_directory_fails_LOUDLY(tmp_path: Path) -> None:
    """Why `__init__` must `mkdir` the new directory, stated as a consequence.

    ⚑ This replaced a test that asserted `stub.arena_rejected_dir.is_dir()` --
    which only checked that the test's own stub had made its own directory, and
    would have passed on origin/main where the production attribute does not
    exist at all. It proved nothing about production.

    The real risk of forgetting the `mkdir` is that `jp.replace()` raises. That
    is the acceptable failure (loud, and the file stays in `pending`); the
    unacceptable one would be losing the result quietly. Pin the loud one.
    """
    stub = _ArenaStub(tmp_path)
    stub.arena_rejected_dir.rmdir()
    bad = stub.arena_pending_dir / "bad.json"
    bad.write_text("<<<", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        _drain(stub)

    assert bad.exists(), "the result must survive in pending, not vanish"


def test_a_valid_pending_result_is_not_quarantined(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: parseable files must never reach the reject path.

    Given a trial_id that does not match, the drain skips before any upload, so
    this exercises "parsed fine" without needing a server.
    """
    stub = _ArenaStub(tmp_path)
    good = stub.arena_pending_dir / "good.json"
    good.write_text(json.dumps({"trial_id": "some-other-trial"}), encoding="utf-8")

    _drain(stub)

    assert good.exists(), "a readable result must stay pending, not be rejected"
    assert stub._arena_rejected_count == 0
    assert list(stub.arena_rejected_dir.glob("*.json")) == []


# --------------------------------------------------------------------------
# S3 — malformed watchdog env vars announce the fallback and the realized value
# --------------------------------------------------------------------------


class _WatchdogStub:
    def __init__(self) -> None:
        self._stall_watchdog_started = False
        self._model_watch_started = False
        self.log = logging.getLogger("test_watchdog_stub")


@pytest.fixture
def no_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the watchdogs' daemon threads from actually running.

    Both methods end by spawning a loop thread. Against a stub self that loop
    raises on the first attribute it wants, and because it is a daemon thread
    the failure surfaces as an unrelated `PytestUnhandledThreadExceptionWarning`
    in whichever test happens to be running when it fires. The subject here is
    the parsing and the log line, both of which happen before the spawn.
    """
    import threading

    class _NoStartThread:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(threading, "Thread", _NoStartThread)


@pytest.mark.parametrize("raw", ["600s", "1_000x", "abc", "5 seconds", ""])
@pytest.mark.usefixtures("no_threads")
def test_a_malformed_stall_timeout_announces_the_fallback(
    raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """The stall watchdog is a SAFETY device.

    Running it at 300s when the operator exported 600s means it fires on
    healthy work, and before this there was no observation anywhere that
    distinguished "600 in force" from "600 rejected, using 300".
    """
    monkeypatch.setenv("CAE_WORKER_STALL_TIMEOUT_S", raw)
    stub = _WatchdogStub()

    with caplog.at_level(logging.INFO, logger="test_watchdog_stub"):
        WorkerSession._start_selfplay_stall_watchdog(cast("WorkerSession", cast(object, stub)))

    assert "is not a number" in caplog.text
    assert "timeout_s=300.0" in caplog.text, (
        "the REALIZED value must be printed, not just the rejection"
    )


@pytest.mark.usefixtures("no_threads")
def test_a_valid_stall_timeout_prints_the_realized_value_without_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """NEGATIVE CONTROL: the realized line is unconditional, the warning is not.

    Printing only on the failure path would leave the healthy case exactly as
    unobservable as before -- an operator still could not confirm their value
    took effect, which is the actual complaint.
    """
    monkeypatch.setenv("CAE_WORKER_STALL_TIMEOUT_S", "600")
    stub = _WatchdogStub()

    with caplog.at_level(logging.INFO, logger="test_watchdog_stub"):
        WorkerSession._start_selfplay_stall_watchdog(cast("WorkerSession", cast(object, stub)))

    assert "timeout_s=600.0" in caplog.text
    assert "is not a number" not in caplog.text


@pytest.mark.parametrize("raw", ["fast", "5s", "0.5x"])
@pytest.mark.usefixtures("no_threads")
def test_a_malformed_model_watch_interval_announces_the_fallback(
    raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CAE_WORKER_MODEL_WATCH_S", raw)
    stub = _WatchdogStub()

    with caplog.at_level(logging.INFO, logger="test_watchdog_stub"):
        WorkerSession._start_model_watch_thread(cast("WorkerSession", cast(object, stub)))

    assert "is not a number" in caplog.text
    assert "poll_s=5.0" in caplog.text


@pytest.mark.usefixtures("no_threads")
def test_a_disabled_watchdog_still_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """`<= 0` disables it. Silence there is the same defect in another coat:
    an operator who typo'd a negative gets no watchdog and no line."""
    monkeypatch.setenv("CAE_WORKER_STALL_TIMEOUT_S", "-1")
    stub = _WatchdogStub()

    with caplog.at_level(logging.INFO, logger="test_watchdog_stub"):
        WorkerSession._start_selfplay_stall_watchdog(cast("WorkerSession", cast(object, stub)))

    assert "DISABLED" in caplog.text
    assert stub._stall_watchdog_started is False


# --------------------------------------------------------------------------
# L1 — the shuffle-pool seed's threading invariant is enforced, not documented
# --------------------------------------------------------------------------


def test_seeding_the_shuffle_pool_refuses_once_the_prefetch_thread_exists() -> None:
    """⚑ RED on origin/main, which only had a comment.

    `_seed_shuffle_pool` draws from `_prefetch_rng` -- the prefetch thread's
    generator -- on the MAIN thread. That is safe only because `__init__` runs
    `_scan_existing_shards()` strictly before `_ensure_prefetch_thread()`. The
    refactor that breaks it looks harmless ("start the prefetch thread earlier
    so the pool is warm sooner"), and `np.random.Generator` has no internal
    lock, so the two threads would advance one bit-generator state
    non-atomically. What that corrupts is WHICH SHARDS enter the shuffle pool,
    i.e. the training-data composition -- surfacing as an unreproducible
    sampling distribution, never as a crash. Assert, do not document.
    """
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    class _Stub:
        _prefetch_thread = object()  # a thread already running

    with pytest.raises(AssertionError, match="not thread-safe"):
        DiskReplayBuffer._seed_shuffle_pool(
            cast("DiskReplayBuffer", cast(object, _Stub())), [],
        )


def test_the_assert_names_the_ordering_that_must_be_restored() -> None:
    """The message is the deliverable: whoever trips this is mid-refactor and
    needs to know which ordering they broke, not just that something is wrong.
    """
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    class _Stub:
        _prefetch_thread = object()

    with pytest.raises(AssertionError) as excinfo:
        DiskReplayBuffer._seed_shuffle_pool(
            cast("DiskReplayBuffer", cast(object, _Stub())), [],
        )

    message = str(excinfo.value)
    assert "_scan_existing_shards()" in message
    assert "_ensure_prefetch_thread()" in message


def test_the_production_construction_order_still_holds() -> None:
    """The assert is only useful while the real `__init__` satisfies it.

    If someone reorders `__init__` and "fixes" the failure by deleting the
    assert, this fails too -- it reads the source order of the two calls rather
    than the assert.
    """
    import inspect

    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    src = inspect.getsource(DiskReplayBuffer.__init__)
    assert src.index("_scan_existing_shards()") < src.index("_ensure_prefetch_thread()"), (
        "__init__ must scan (which seeds the shuffle pool on the main thread) "
        "BEFORE starting the prefetch thread that owns _prefetch_rng"
    )


# --------------------------------------------------------------------------
# S5 — the SummaryWriter import guard catches ImportError ONLY
# --------------------------------------------------------------------------


def _load_trainer_with_fake_tensorboard(exc: Exception | None) -> Any:
    """Execute `trainer.py` as a FRESH module with tensorboard sabotaged.

    A fresh module object rather than `importlib.reload`, so the canonical
    `chess_anti_engine.train.trainer` in `sys.modules` is never left in a
    half-executed state for the rest of the suite. `exc=None` means "import
    works normally".
    """
    import importlib.util
    import sys
    import types

    class _Sabotaged(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
  # Narrow on purpose: raising for EVERY attribute makes the import
  # machinery's own `__file__`/`__path__` lookups raise instead, and the
  # test would then pass without the guard under test being involved.
            if name == "SummaryWriter" and exc is not None:
                raise exc
            raise AttributeError(name)

    saved = sys.modules.get("torch.utils.tensorboard")
    sys.modules["torch.utils.tensorboard"] = _Sabotaged("torch.utils.tensorboard")
    try:
  # ⚑ Named INSIDE the real package. `trainer.py` uses relative imports
  # (`from .aurora import ...`), so a bare module name leaves __package__
  # empty and the load dies at line 89 with "attempted relative import with
  # no known parent package" -- which would make the ImportError case fail
  # for a reason that has nothing to do with the guard.
        spec = importlib.util.spec_from_file_location(
            "chess_anti_engine.train._probe_trainer_module",
            "chess_anti_engine/train/trainer.py",
        )
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
  # Registered before exec because `trainer.py` defines dataclasses, and
  # `dataclasses` resolves annotations via `sys.modules[cls.__module__]`.
  # Popped afterwards so the probe never shadows anything.
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(spec.name, None)
        return mod
    finally:
        if saved is not None:
            sys.modules["torch.utils.tensorboard"] = saved
        else:
            sys.modules.pop("torch.utils.tensorboard", None)


def test_a_broken_tensorboard_is_NOT_swallowed_as_not_installed() -> None:
    """RED on origin/main, whose `except Exception` absorbed everything.

    The fallback writer's methods are `pass`, so a genuine breakage inside an
    INSTALLED tensorboard -- a version incompatibility, a protobuf mismatch, a
    broken transitive dep -- silently produced a run with no event files and no
    message, turning every `add_scalar` in the trainer into a no-op. This repo
    has a documented incident where flat live progress signals let a real
    degradation go unread; a metrics writer that can quietly become a no-op is
    the same family.
    """
    with pytest.raises(RuntimeError, match="simulated tensorboard breakage"):
        _load_trainer_with_fake_tensorboard(
            RuntimeError("simulated tensorboard breakage"),
        )


def test_a_genuinely_absent_tensorboard_still_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ NEGATIVE CONTROL: the intended case must keep working.

    Narrowing the catch is only correct if "tensorboard is not installed" still
    degrades gracefully -- the `worker` extra ships without it. It must also now
    SAY so, since a run with no metrics should not be silent either.
    """
    with caplog.at_level(logging.WARNING):
        mod = _load_trainer_with_fake_tensorboard(
            ImportError("No module named 'tensorboard'"),
        )

    writer = mod._SummaryWriter(log_dir="unused")
    writer.add_scalar("x", 1.0, 0)
    writer.close()

    assert mod._SummaryWriter.__name__ == "_FallbackSummaryWriter"
    assert "tensorboard is not installed" in caplog.text


# --------------------------------------------------------------------------
# S4 — the SOAP param-group fallback announces that the split was discarded
# --------------------------------------------------------------------------


def test_the_soap_param_group_fallback_logs_that_groups_were_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """RED on origin/main: `except TypeError` rebuilt the optimizer silently.

    On the fallback branch every parameter gets the same flat `lr` and no
    group-specific weight decay -- the configured split is DISCARDED. Nothing
    recorded that, so a later analysis would cite a decay/grouping setting that
    never applied. Same shape as the `matrix_weight_decay is decorative`
    finding.

    ⚑ The fallback is KEPT rather than made fatal: production runs `aurora`,
    not `soap`, so raising would convert a dormant path into a hard failure for
    a research config nobody runs, for no benefit today. The call is stated in
    the PR body.
    """
    import sys
    import types

    import torch

    class _FakeSOAP(torch.optim.AdamW):
        def __init__(self, params: Any, lr: float = 1e-3) -> None:
            materialized = list(params)
            if materialized and isinstance(materialized[0], dict):
                raise TypeError("SOAP() does not accept param_groups")
            super().__init__(materialized, lr=lr)

    fake = types.ModuleType("soap")
    fake.SOAP = _FakeSOAP  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "soap", fake)

    from chess_anti_engine.train.trainer import Trainer

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            del x
            return {"policy": self.lin.weight[:1]}

    with caplog.at_level(logging.WARNING):
        trainer = Trainer(
            _Tiny(), device="cpu", lr=1e-3, optimizer="soap",
            matrix_optimizer_scope="default", warmup_steps=10,
            warmup_lr_start=1e-5, use_amp=False, log_dir=tmp_path,
            tb_log_interval=1000, prefetch_batches=False,
        )

    assert isinstance(trainer.opt, _FakeSOAP), "the fallback must still build one"
    assert "rejected param_groups" in caplog.text
    assert "NOT in effect" in caplog.text, (
        "the message must say the per-group split is gone, not merely that a "
        "fallback happened"
    )


def test_a_soap_that_accepts_param_groups_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """NEGATIVE CONTROL: no warning when the groups actually took."""
    import sys
    import types

    import torch

    fake = types.ModuleType("soap")
    fake.SOAP = torch.optim.AdamW  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "soap", fake)

    from chess_anti_engine.train.trainer import Trainer

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            del x
            return {"policy": self.lin.weight[:1]}

    with caplog.at_level(logging.WARNING):
        Trainer(
            _Tiny(), device="cpu", lr=1e-3, optimizer="soap",
            matrix_optimizer_scope="default", warmup_steps=10,
            warmup_lr_start=1e-5, use_amp=False, log_dir=tmp_path,
            tb_log_interval=1000, prefetch_batches=False,
        )

    assert "rejected param_groups" not in caplog.text


def test_the_real_worker_init_starts_the_corrupt_counter_at_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Pins `self._reco_corrupt_count = 0` in the PRODUCTION `__init__`.

    ⚑ EVERY OTHER TEST OF THIS COUNTER USES A STUB THAT SUPPLIES THE
    ATTRIBUTE ITSELF, so deleting the production initialiser leaves the whole
    suite green while the shipped worker raises `AttributeError` inside the
    corrupt branch — an exception on the error path, reached only once a
    corrupt reco actually arrives, which is the worst possible place for it.
    A stub that hands the class its own preconditions cannot see that.

    This runs the shipped `__init__` and then the shipped
    `_active_difficulty`, so the counter is exercised from the value
    production gives it.

    Two things are stubbed and neither is under test: `_collect_worker_info`
    (it calls `torch.cuda.is_available()`, and this repo's tests must not
    touch the GPU a live run owns) and `requests`. `device="cpu"` is passed so
    the one other CUDA probe in `__init__` short-circuits. Everything between
    those, including the initialiser this test exists for, is the real thing.
    """
    import argparse

    from chess_anti_engine import worker as worker_mod

    monkeypatch.setattr(
        worker_mod, "_collect_worker_info", lambda **_kw: {"hostname": "box0", "device": "cpu"}
    )
    args = argparse.Namespace(
        username="u", password="p", server_url="http://localhost:0", trial_id="",
        work_dir=str(tmp_path), shared_cache_dir=None, device="cpu", seed=0,
        games_per_batch=1, allow_overrides=False, save_config=False,
        inference_slot_name="",
    )
    session = WorkerSession(
        args,
        cfg={},
        cfg_path=tmp_path / "worker.yaml",
        log=logging.getLogger("test_real_worker_init"),
        pinned_games_per_batch_cli=False,
        requests_mod=None,
    )

    assert session._reco_corrupt_count == 0
    assert session._reco_corrupt_last is None

    session._active_reco = {"opponent_wdl_regret_limit": "not-a-number", "sf_nodes": 5000}
    with caplog.at_level(logging.WARNING):
        assert session._active_difficulty() == (None, None)

    assert session._reco_corrupt_count == 1, (
        "the shipped __init__ must supply the counter the corrupt path increments"
    )
    assert "reco difficulty is CORRUPT" in caplog.text
