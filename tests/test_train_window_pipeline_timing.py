"""Where a training window's wall clock went, decomposed and always on.

`_PipelinePhaseTimer` splits one `train_steps` window into five regions plus a
residual, publishes them on `TrainMetrics`, and prints one
`[trainer] window_timing ...` line per window. Two clocks per region:

* the phase key is the CPU WALL CLOCK. Those five plus `pipeline_other_s`
  PARTITION `train_time_s`, which is the property that makes the numbers
  readable as shares.
* `gpu_*_s` are CUDA event spans (`batch_prefetch_wait_s` deliberately has
  none -- see `_PIPELINE_PHASE_GPU_KEY`). They overlap, and on a CPU-only
  run they are 0.0 -- which is why `gpu_events=on|off` is printed beside them.

What this file pins, and why each one is a test rather than a comment:

* every key reaches `TrainMetrics`, so nothing here is computed and dropped;
* the CPU phases really do add up to the window;
* time spent in a phase is attributed to THAT phase -- injected delays are
  found where they were injected, which is what a dropped or mis-wired phase
  would break;
* the CUDA-event branch is exercised WITHOUT a GPU, through a fake event
  class, including that `drain` synchronizes once and the hot path never does.
"""

from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
import torch.nn as nn

from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.train.trainer import (
    _PIPELINE_PHASE_GPU_KEY,
    _PIPELINE_RESIDUAL_KEY,
    TrainMetrics,
    Trainer,
    _PipelinePhaseTimer,
)

_ZERO_LOSS_KEYS = tuple(trainer_mod._LOSS_KEY_TO_METRIC_FIELD)
_CPU_PHASE_KEYS = (*_PIPELINE_PHASE_GPU_KEY, _PIPELINE_RESIDUAL_KEY)
_GPU_PHASE_KEYS = tuple(twin for twin in _PIPELINE_PHASE_GPU_KEY.values() if twin is not None)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


class _RecordingWriter:
    """Captures `(tag, value, step)` instead of writing an event file."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: Any, step: Any = None) -> None:
        self.scalars.append((str(tag), float(value), int(step)))

    def close(self) -> None: ...


def _make_trainer(tmp_path: Path, **kwargs: Any) -> Trainer:
    trainer_kwargs: dict[str, Any] = {
        "device": "cpu",
        "lr": 1e-3,
        "optimizer": "aurora",
        "use_amp": False,
        "log_dir": tmp_path,
        "tb_log_interval": 1000,
        "prefetch_batches": False,
    }
    trainer_kwargs.update(kwargs)
    return Trainer(_TinyModel(), **trainer_kwargs)


def _install_fake_window(
    trainer: Trainer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    loss_delay_s: float = 0.0,
    batch_delay_s: float = 0.0,
) -> None:
    """A window that runs, with optional delays parked in known phases."""
    param = next(trainer.model.parameters())

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del out, batch, kwargs
        if loss_delay_s:
            time.sleep(loss_delay_s)
        return {
            "total": (param * param).sum(),
            **dict.fromkeys(_ZERO_LOSS_KEYS, torch.zeros(())),
        }

    def slow_batches(*_args: Any, **_kwargs: Any) -> Any:
        while True:
            if batch_delay_s:
                time.sleep(batch_delay_s)
            yield {"x": torch.zeros((1, 4, 8, 8))}

    monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})
    monkeypatch.setattr(trainer, "_iter_prefetched_batches", slow_batches)


# --- the keys exist, and they mean the window ------------------------------


def test_every_phase_key_is_a_train_metrics_field() -> None:
    """A timing computed into a dict nobody publishes is this repo's signature
    defect. The decomposition is only a decomposition if it lands somewhere."""
    fields = {f.name for f in trainer_mod.dataclasses.fields(TrainMetrics)}
    missing = sorted(set(_CPU_PHASE_KEYS + _GPU_PHASE_KEYS) - fields)

    assert not missing, f"phase keys with no TrainMetrics field: {missing}"


def test_the_cpu_phases_partition_the_window_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    _install_fake_window(trainer, monkeypatch)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=5)

    parts = {key: float(getattr(metrics, key)) for key in _CPU_PHASE_KEYS}
    assert all(value >= 0.0 for value in parts.values()), parts
    assert sum(parts.values()) == pytest.approx(metrics.train_time_s, abs=1e-9)
    assert metrics.train_time_s > 0.0
  # ⚑ EVERY named phase, not the sum. A phase dropped from the loop leaves its
  # time in `pipeline_other_s`, so the partition above still holds and says
  # nothing -- this line is what notices. Each region runs real work on every
  # step, so a zero here means the span is gone, not that it was fast.
    empty = sorted(k for k in _PIPELINE_PHASE_GPU_KEY if parts[k] <= 0.0)
    assert not empty, f"phases that recorded no time at all (span dropped?): {empty}"


@pytest.mark.parametrize(
    ("kwarg", "phase"),
    [("loss_delay_s", "fwd_loss_s"), ("batch_delay_s", "batch_prefetch_wait_s")],
)
def test_an_injected_delay_lands_in_the_phase_it_was_injected_into(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwarg: str, phase: str,
) -> None:
    """The attribution check. A decomposition whose buckets are all populated
    still tells you nothing if the time goes into the wrong one -- and a phase
    quietly dropped from the loop shows up here as its delay reappearing in
    `pipeline_other_s`."""
    steps, delay = 4, 0.02
    trainer = _make_trainer(tmp_path)
    _install_fake_window(trainer, monkeypatch, **{kwarg: delay})

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=steps)

    injected = steps * delay
    assert float(getattr(metrics, phase)) >= injected * 0.8
    others = [k for k in _CPU_PHASE_KEYS if k != phase]
    assert float(getattr(metrics, phase)) > max(float(getattr(metrics, k)) for k in others)


def test_the_window_prints_one_timing_line_carrying_every_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ The operator-facing half. `print`, not `logging.info`: the trial actor
    installs no handler, so an INFO record reaches nothing at all."""
    trainer = _make_trainer(tmp_path)
    _install_fake_window(trainer, monkeypatch)
    capsys.readouterr()

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=3)

    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if "window_timing" in line
    ]
    assert len(lines) == 1, lines
    fields = dict(token.split("=", 1) for token in lines[0].split() if "=" in token)
    for key in (*_CPU_PHASE_KEYS, *_GPU_PHASE_KEYS):
        assert key in fields, f"{key} missing from the window_timing line"
        assert float(fields[key]) == pytest.approx(float(getattr(metrics, key)), abs=5e-4)
    assert fields["steps"] == "3"
  # ⚑ The reading that stops all-zero `gpu_*` columns being misread as "the GPU
  # did nothing". There is no CUDA here, so it must say so.
    assert fields["gpu_events"] == "off"


def test_the_phase_keys_reach_tensorboard_under_train_avg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE CONSUMER'S OWN READING, not the producer's. A field on a dataclass
    is not an observable; `_log_metrics` walking `dataclasses.asdict` is what
    makes it one, and "computed, published nowhere" is precisely the defect
    this instrument was added to find in the training loop."""
    trainer = _make_trainer(tmp_path)
    writer = _RecordingWriter()
    monkeypatch.setattr(trainer, "writer", writer)
    _install_fake_window(trainer, monkeypatch)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=3)

    written = {tag: value for tag, value, _ in writer.scalars}
    for key in (*_CPU_PHASE_KEYS, *_GPU_PHASE_KEYS):
        tag = f"train_avg/{key}"
        assert tag in written, f"{key} never reached TensorBoard"
        assert written[tag] == pytest.approx(float(getattr(metrics, key)))


# --- the CUDA-event branch, exercised without a GPU ------------------------


class _FakeEvent:
    """Stands in for `torch.cuda.Event(enable_timing=True)`.

    `elapsed_time` returns MILLISECONDS, like the real one, so a test that
    passed against a fake returning seconds would be measuring the fake.
    """

    created: ClassVar[list[_FakeEvent]] = []

    def __init__(self, *, enable_timing: bool = False) -> None:
        assert enable_timing, "a timing span needs enable_timing=True"
        self.recorded_at: float | None = None
        self.stream: Any = None
        self.synchronized = 0
        _FakeEvent.created.append(self)

    def record(self, stream: Any = None) -> None:
        self.recorded_at = float(len(_FakeEvent.created))
        self.stream = stream

    def synchronize(self) -> None:
        self.synchronized += 1

    def elapsed_time(self, other: _FakeEvent) -> float:
        assert self.recorded_at is not None
        assert other.recorded_at is not None
        return other.recorded_at - self.recorded_at


class _FakeStream:
    """What `torch.cuda.current_stream(device)` hands back under the fake."""

    def __init__(self, device: Any) -> None:
        self.device = device


@pytest.fixture
def fake_cuda_events(monkeypatch: pytest.MonkeyPatch) -> type[_FakeEvent]:
    _FakeEvent.created = []
    monkeypatch.setattr(trainer_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_mod.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(trainer_mod.torch.cuda, "current_stream", lambda device=None: _FakeStream(device))
    return _FakeEvent


def test_the_fetch_phase_records_no_gpu_events_and_publishes_no_h2d_span(
    fake_cuda_events: type[_FakeEvent],
) -> None:
    """⚑ A span around `next(batches)` brackets the HOST wait: the device idles
    between the two records whenever the sampler is late, so it would report
    starvation as a slow copy. The phase keeps its wall clock and has no twin."""
    timer = _PipelinePhaseTimer(device="cuda:0")

    with timer.phase("batch_prefetch_wait_s"):
        time.sleep(0.005)

    out = timer.drain(window_wall_s=1.0)

    assert fake_cuda_events.created == [], "the fetch phase must not record events"
    assert out["batch_prefetch_wait_s"] >= 0.004
    assert "h2d_s" not in out
    assert "h2d_s" not in {f.name for f in trainer_mod.dataclasses.fields(TrainMetrics)}


def test_events_are_recorded_on_the_configured_devices_stream(
    fake_cuda_events: type[_FakeEvent],
) -> None:
    """A bare `event.record()` uses the process's CURRENT device (`cuda:0`
    unless `set_device` was called), so a trainer on `cuda:1` would time an
    idle GPU. Every event must be recorded on the configured device's stream."""
    timer = _PipelinePhaseTimer(device="cuda:1")

    with timer.phase("fwd_loss_s"):
        pass
    with timer.phase("opt_step_s"):
        pass

    assert len(fake_cuda_events.created) == 4
    for event in fake_cuda_events.created:
        assert isinstance(event.stream, _FakeStream), "recorded without a stream"
        assert event.stream.device == torch.device("cuda:1")


def test_the_gpu_spans_come_off_cuda_events_and_are_reported_in_seconds(
    fake_cuda_events: type[_FakeEvent],
) -> None:
    timer = _PipelinePhaseTimer(device="cuda:0")
    assert timer.cuda is True

    with timer.phase("fwd_loss_s"):
        pass
    with timer.phase("bwd_s"):
        pass

    out = timer.drain(window_wall_s=10.0)

  # Four events, recorded at ordinals 1..4 -> each span is 1.0 "ms" -> 0.001 s.
    assert len(fake_cuda_events.created) == 4
    assert out["gpu_fwd_loss_s"] == pytest.approx(0.001)
    assert out["gpu_bwd_s"] == pytest.approx(0.001)
    assert out["gpu_opt_step_s"] == 0.0


def test_drain_synchronizes_once_on_the_last_event_and_the_hot_path_never_does(
    fake_cuda_events: type[_FakeEvent],
) -> None:
    """⚑ The overhead claim, as a test. An instrument that serialized the
    pipeline to measure the pipeline would be measuring itself, so the spans
    only RECORD and every `elapsed_time` read is deferred to `drain`."""
    timer = _PipelinePhaseTimer(device="cuda:0")
    for _ in range(3):
        with timer.phase("opt_step_s"):
            pass

    assert [e.synchronized for e in fake_cuda_events.created] == [0] * 6

    timer.drain(window_wall_s=1.0)

    synchronized = [e.synchronized for e in fake_cuda_events.created]
    assert sum(synchronized) == 1
    assert synchronized[-1] == 1, "the LAST event is the one waited on"


def test_no_device_wide_synchronize_or_elapsed_read_outside_drain() -> None:
    """The same claim one level down, read off the source. A `synchronize()`
    added to a span would be invisible to the fake-event test above -- it would
    simply make it slower -- so pin where the calls are allowed to live."""

    def calls(fn: Any) -> list[str]:
        tree = ast.parse(inspect.getsource(fn).lstrip())
        return [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]

  # ⚑ `_record_event` is the ONLY method that touches CUDA on the record path,
  # so it is the one a stray `torch.cuda.synchronize()` would land in; and the
  # loss accumulator's methods are where a `.item()`/`.tolist()` would put the
  # per-microbatch stall back while the by-name spy test still passed (Grok
  # review, PR #496). Both are pinned here, read off the source.
    for method in (
        _PipelinePhaseTimer.begin_gpu,
        _PipelinePhaseTimer.end_gpu,
        _PipelinePhaseTimer.record,
        _PipelinePhaseTimer._record_event,
        trainer_mod._PipelinePhaseSpan.__enter__,
        trainer_mod._PipelinePhaseSpan.__exit__,
        trainer_mod._DeviceLossSums.add_losses,
        trainer_mod._DeviceLossSums.merge,
        trainer_mod._DeviceLossSums._accumulate,
        trainer_mod._DeviceLossSums.tensor,
        trainer_mod._DeviceLossSums.items,
        trainer_mod._drift_positions,
    ):
        named = calls(method)
        for forbidden in ("synchronize", "elapsed_time", "item", "tolist", "cpu", "numpy"):
            assert forbidden not in named, f"{method.__qualname__} calls .{forbidden}() on the hot path"

    drained = calls(_PipelinePhaseTimer.drain)
    assert drained.count("synchronize") == 1
    assert drained.count("elapsed_time") == 1


def test_a_repeated_drift_layout_builds_no_new_index_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The AST pin above forbids `.item()`-class reads, but the drift path's
    stall is `torch.tensor(list, device=...)` -- a pageable host->device copy
    on every flickering microbatch if `_drift_positions` ever loses its cache.
    Spy the constructor across two accumulations with the SAME drifted layout:
    the first may build the index once, the second must build nothing
    (Grok review, PR #496)."""
    real_tensor = torch.tensor
    built: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        built.append(args[0] if args else kwargs.get("data"))
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", spy)
    trainer_mod._DRIFT_POSITIONS_CACHE.clear()

    def drifted_window() -> None:
        sums = trainer_mod._DeviceLossSums()
        sums.add_losses({"total": real_tensor(1.0), "policy_ce": real_tensor(2.0)})
        sums.add_losses({"total": real_tensor(1.0), "channel_balance": real_tensor(0.5)})

    drifted_window()
    first = len(built)
    drifted_window()

    assert first >= 1, "the drifted layout never built an index tensor -- the spy saw nothing"
    assert len(built) == first, f"a repeated drift layout rebuilt its index tensor: {built[first:]}"


def test_without_cuda_the_spans_fall_back_to_wall_clock_and_never_touch_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only must not crash, and must not silently report GPU numbers."""

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("torch.cuda.Event must not be constructed without CUDA")

    monkeypatch.setattr(trainer_mod.torch.cuda, "Event", explode)
    timer = _PipelinePhaseTimer(device="cpu")
    assert timer.cuda is False

    with timer.phase("bwd_s"):
        time.sleep(0.01)

    out = timer.drain(window_wall_s=1.0)

    assert out["bwd_s"] >= 0.008
    assert all(out[key] == 0.0 for key in _GPU_PHASE_KEYS)
    assert out[_PIPELINE_RESIDUAL_KEY] == pytest.approx(1.0 - out["bwd_s"])


def test_an_unknown_phase_name_is_a_hard_error_not_a_silent_bucket() -> None:
    """A timing accumulated under a key with no metric field would be measured
    and then thrown away -- exactly the failure this instrument exists to find
    in the training loop, so it must not be possible inside the instrument."""
    timer = _PipelinePhaseTimer(device="cpu")

    with pytest.raises(KeyError), timer.phase("not_a_phase_s"):
        pass
