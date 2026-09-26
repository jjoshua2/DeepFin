from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from chess_anti_engine.train.trainer import Trainer, _SfRebuildCoverageAccumulator
from tests.test_game_aware_epoch_replay import _write


def _buffer(path: Path, *, overlap: bool, budget: int = 8 * 1024**3) -> GameAwareEpochBuffer:
    return GameAwareEpochBuffer(
        shard_dir=path, batch_size=4, seed=7, input_planes=146,
        input_history_encoding="legacy", history_rep_fix=False,
        mirror_augmentation=True, plan_workers=2, load_workers=2,
        max_working_set_bytes=budget, host_batch_overlap=overlap,
    )


def _trainer() -> Any:
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer._sf_rebuild_coverage = _SfRebuildCoverageAccumulator()
    trainer.rebuild_sf_targets = False
    trainer.rebuild_categorical_target = False
    trainer.sf_policy_sparse_ce = False
    trainer._input_history_encoding = "legacy"
    return trainer


def _iterate(trainer: Any, buf: Any, count: int) -> Any:
    return trainer._iter_training_batches(buf, batch_size=4, mirror_prob=0.5, count=count)


def test_real_tensor_and_rng_parity_across_windows(tmp_path: Path) -> None:
    path = _write(tmp_path / "data", [
        [(game, game * 10 + row) for row in range(5) for game in range(8)],
        [(game, game * 10 + row) for row in range(5, 8) for game in range(8)],
    ])
    serial, overlap = _buffer(path, overlap=False), _buffer(path, overlap=True)
    left, right = _trainer(), _trainer()
    assert "host_batch_overlap" not in serial.plan.as_dict()
    assert overlap.plan.as_dict()["host_batch_overlap"] is True
    assert serial.plan.plan_sha256 == overlap.plan.plan_sha256
    assert overlap.plan.peak_working_set_bytes > serial.plan.peak_working_set_bytes
    consumed = 0
    while consumed < serial.num_batches:
        count = min(3, serial.num_batches - consumed)
        a = list(_iterate(left, serial, count))
        b = list(_iterate(right, overlap, count))
        assert len(a) == len(b) == count
        for aa, bb in zip(a, b, strict=True):
            assert aa.keys() == bb.keys()
            for key in aa:
                assert aa[key].dtype == bb[key].dtype
                assert torch.equal(aa[key], bb[key]), key
        consumed += count
        assert overlap._batch_index == consumed  # no cross-window lookahead
        for name in ("rng", "_choice_rng", "_row_rng"):
            assert getattr(serial, name).bit_generator.state == getattr(overlap, name).bit_generator.state
    assert serial.receipt()["complete"]
    assert overlap.receipt()["complete"]
    assert int(overlap.receipt()["peak_working_set_bytes"]) <= overlap.plan.peak_working_set_bytes
    serial.close()
    overlap.close()


def test_reserved_peak_is_enforced_by_plan_and_runtime(tmp_path: Path) -> None:
    path = _write(tmp_path / "data", [[(i, i) for i in range(8)]])
    buf = _buffer(path, overlap=True)
    peak = buf.plan.peak_working_set_bytes
    reserve = buf.plan.host_overlap_reserve_bytes
    assert reserve > 0
    buf.close()
    with pytest.raises(ValueError, match="working-set"):
        _buffer(path, overlap=True, budget=peak - 1)
    exact = _buffer(path, overlap=True, budget=peak)
    list(_iterate(_trainer(), exact, exact.num_batches))
    assert int(exact.receipt()["peak_working_set_bytes"]) <= peak
    with pytest.raises(RuntimeError, match="working-set"):
        exact._observe_working_set(peak - reserve + 1, phase="retained batch")
    exact.close()


def test_actual_optional_collation_fits_prepared_retention_bound(tmp_path: Path) -> None:
    path = _write(tmp_path / "data", [[(i, i) for i in range(4)]])
    buf = _buffer(path, overlap=True)
    trainer = _trainer()
    host = trainer._sample_batch_host(buf, batch_size=4, mirror_prob=0.5)
    bound = sum(a.nbytes + 8 * a.size for a in host.values())
    tensors = trainer._host_batch_to_tensors(host)
    actual = sum(a.nbytes for a in host.values()) + sum(t.numel() * t.element_size() for t in tensors.values())
    assert actual <= bound <= buf.plan.host_overlap_reserve_bytes
    assert tensors["wdl_t"].dtype == torch.int64
    assert tensors["x"].dtype == torch.float32
    buf.close()


@pytest.mark.parametrize("fail_second", [False, True])
def test_close_joins_one_running_future_and_never_schedules_third(fail_second: bool) -> None:
    trainer = _trainer()
    buf = SimpleNamespace(exact_without_replacement=True, host_batch_overlap=True,
                          plan=SimpleNamespace(host_overlap_reserve_bytes=1000))
    started, release, joined = threading.Event(), threading.Event(), threading.Event()
    calls: list[int] = []
    def prepare(*_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        calls.append(threading.get_ident())
        if len(calls) == 2:
            started.set()
            assert release.wait(3)
            if fail_second:
                raise RuntimeError("second prepare failed")
        return {"x": np.ones(1, dtype=np.float16)}
    trainer._sample_batch_host = prepare
    trainer._host_batch_to_tensors = lambda batch: {"x": torch.from_numpy(batch["x"])}
    it = _iterate(trainer, buf, 5)
    next(it)
    assert started.wait(3)
    def close() -> None:
        it.close()
        joined.set()
    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert not joined.wait(0.05)
        release.set()
        closer.join(3)
        assert joined.is_set()
        assert len(calls) == 2
        assert len(set(calls)) == 1
        assert not any(t.name.startswith("exact-host") for t in threading.enumerate())
    finally:
        release.set()
        closer.join(3)


def test_producer_exception_propagates_and_does_not_convert_extra_batch() -> None:
    trainer = _trainer()
    buf = SimpleNamespace(exact_without_replacement=True, host_batch_overlap=True,
                          plan=SimpleNamespace(host_overlap_reserve_bytes=1000))
    calls = 0
    def prepare(*_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("malformed next batch")
        return {"x": np.ones(1, dtype=np.float16)}
    trainer._sample_batch_host = prepare
    trainer._host_batch_to_tensors = lambda batch: {"x": torch.from_numpy(batch["x"])}
    it = _iterate(trainer, buf, 4)
    next(it)
    with pytest.raises(ValueError, match="malformed next"):
        next(it)
    assert calls == 2
    assert not any(t.name.startswith("exact-host") for t in threading.enumerate())


def test_missing_reserve_and_derived_field_overflow_refuse_before_conversion() -> None:
    trainer = _trainer()
    buf = SimpleNamespace(exact_without_replacement=True, host_batch_overlap=True,
                          plan=SimpleNamespace(host_overlap_reserve_bytes=0))
    with pytest.raises(RuntimeError, match="lacks a planned"):
        next(_iterate(trainer, buf, 1))
    buf.plan.host_overlap_reserve_bytes = 10
    trainer._sample_batch_host = lambda *a, **k: {"x": np.ones(5, dtype=np.float16)}
    trainer._host_batch_to_tensors = lambda _: pytest.fail("must refuse before collation")
    with pytest.raises(RuntimeError, match="exceeds overlap"):
        next(_iterate(trainer, buf, 1))


def test_transfer_is_retired_before_next_caller_collation(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer()
    trainer.device = "cuda"
    buf = SimpleNamespace(exact_without_replacement=True, host_batch_overlap=True,
                          plan=SimpleNamespace(host_overlap_reserve_bytes=1000))
    events: list[str] = []
    caller = threading.get_ident()
    trainer._sample_batch_host = lambda *a, **k: {"x": np.ones(1, dtype=np.float16)}
    def collate(batch: Any) -> dict[str, torch.Tensor]:
        assert threading.get_ident() == caller
        events.append("collate")
        return {"x": torch.from_numpy(batch["x"])}
    class Event:
        def record(self, _stream: Any) -> None:
            assert threading.get_ident() == caller
            events.append("record")
        def synchronize(self) -> None:
            events.append("retire")
    trainer._host_batch_to_tensors = collate
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    list(_iterate(trainer, buf, 2))
    assert events == ["collate", "record", "retire", "collate", "record", "retire"]


def test_real_cli_two_epoch_weights_match_default(tmp_path: Path) -> None:
    import json
    from scripts import lc0_control_train as driver
    from tests.test_offline_game_epochs import arguments

    serial, overlap = tmp_path / "serial", tmp_path / "overlap"
    assert driver.main(arguments(tmp_path, serial)) == 0
    assert driver.main([*arguments(tmp_path, overlap), "--epoch-host-batch-overlap"]) == 0
    a = torch.load(serial / "checkpoint.pt", map_location="cpu", weights_only=False)
    b = torch.load(overlap / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] == 10
    assert a["model"].keys() == b["model"].keys()
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    result = json.loads((overlap / "summary.json").read_text())
    epochs = result["sampling"]["epochs"]
    assert len(epochs) == 2
    assert result["sampling"]["complete"]
    for epoch in epochs:
        sampling = epoch["sampling"]
        assert sampling["host_batch_overlap"] is True
        assert sampling["host_overlap_reserve_bytes"] > 0
        assert sampling["peak_working_set_bytes"] <= sampling["max_working_set_bytes"]
        assert sampling["plan_sha256"] == sampling["realized_sha256"]
    assert not any(t.name.startswith("exact-host") for t in threading.enumerate())


def test_cli_rejects_overlap_outside_exact_mode_before_reading_inputs(tmp_path: Path) -> None:
    from scripts import lc0_control_train as driver
    with pytest.raises(SystemExit, match="requires --sampling-mode game_epoch"):
        driver.main(["--shards", str(tmp_path / "absent"), "--out-dir", str(tmp_path / "out"),
                     "--steps", "1", "--epoch-host-batch-overlap"])


def test_ragged_batch_preserves_real_rows_and_bytes(tmp_path: Path) -> None:
    path = _write(tmp_path / "data", [[(i, i) for i in range(199)]])
    def open_buffer(overlap: bool) -> GameAwareEpochBuffer:
        return GameAwareEpochBuffer(
            shard_dir=path, batch_size=100, seed=7, input_planes=146,
            input_history_encoding="legacy", history_rep_fix=False,
            mirror_augmentation=True, plan_workers=2, load_workers=2,
            host_batch_overlap=overlap,
        )
    serial, overlap = open_buffer(False), open_buffer(True)
    def batches(buf: Any) -> list[Any]:
        return list(_trainer()._iter_training_batches(buf, batch_size=100, mirror_prob=0.5, count=2))
    aa, bb = batches(serial), batches(overlap)
    assert sorted(int(batch["x"].shape[0]) for batch in bb) == [99, 100]
    for a, b in zip(aa, bb, strict=True):
        assert all(torch.equal(value, b[key]) for key, value in a.items())
    assert overlap.receipt()["rows_realized"] == 199
    serial.close()
    overlap.close()


@pytest.mark.parametrize("cleanup_failure", [None, "record", "retire"])
def test_partial_collation_records_and_retires_without_masking_primary_error(
    monkeypatch: pytest.MonkeyPatch, cleanup_failure: str | None,
) -> None:
    trainer = _trainer()
    trainer.device = "cuda"
    buf = SimpleNamespace(exact_without_replacement=True, host_batch_overlap=True,
                          plan=SimpleNamespace(host_overlap_reserve_bytes=1000))
    events: list[str] = []
    def prepare(*_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        events.append("prepare")
        return {"x": np.ones(1, dtype=np.float16)}
    primary = ValueError("second field conversion failed after enqueue")
    def collate(_batch: Any) -> dict[str, torch.Tensor]:
        events.append("partial_enqueue")
        raise primary
    class Event:
        def record(self, _stream: Any) -> None:
            events.append("record")
            if cleanup_failure == "record":
                raise RuntimeError("unusable CUDA context")
        def synchronize(self) -> None:
            events.append("retire")
            if cleanup_failure == "retire":
                raise RuntimeError("retirement failed")
    trainer._sample_batch_host = prepare
    trainer._host_batch_to_tensors = collate
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    with pytest.raises(ValueError, match="second field conversion failed") as caught:
        next(_iterate(trainer, buf, 2))
    assert caught.value is primary
    assert events == ["prepare", "partial_enqueue", "record", *([] if cleanup_failure == "record" else ["retire"])]
    assert not any(t.name.startswith("exact-host") for t in threading.enumerate())


def test_last_yield_close_propagates_retirement_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer()
    trainer.device = "cuda"
    buf = SimpleNamespace(exact_without_replacement=True, host_batch_overlap=True,
                          plan=SimpleNamespace(host_overlap_reserve_bytes=1000))
    trainer._sample_batch_host = lambda *a, **k: {"x": np.ones(1, dtype=np.float16)}
    trainer._host_batch_to_tensors = lambda batch: {"x": torch.from_numpy(batch["x"])}
    class Event:
        def record(self, _stream: Any) -> None:
            pass
        def synchronize(self) -> None:
            raise RuntimeError("last transfer retirement failed")
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    it = _iterate(trainer, buf, 1)
    next(it)  # train_steps closes here, without requesting StopIteration
    with pytest.raises(RuntimeError, match="last transfer retirement failed"):
        it.close()
    assert not any(t.name.startswith("exact-host") for t in threading.enumerate())
