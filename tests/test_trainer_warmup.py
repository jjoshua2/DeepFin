from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import INPUT_HISTORY_ENCODING_ARRAY_KEY, samples_to_arrays
from chess_anti_engine.train.aurora import AuroraWithAuxAdam
from chess_anti_engine.train.soda import SODAWeightDecayWrapper
from chess_anti_engine.train.trainer import (
    Trainer,
    TrainMetrics,
    _ChainedOptimizer,
    _SqrtReleaseLRScheduler,
    _TrainBatchIterator,
    select_input_history_arrays,
    select_input_history_samples,
    trainer_kwargs_from_config,
)


class _TinyMuonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def test_extract_loss_scalars_materializes_once(monkeypatch) -> None:
    class _FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def detach(self):
            return self

        def item(self):
            raise AssertionError("per-scalar materialization is forbidden")

    class _FakeStack:
        def __init__(self, values: list[_FakeScalar]) -> None:
            self.values = values

        def tolist(self) -> list[float]:
            calls["tolist"] += 1
            return [value.value for value in self.values]

    calls = {"stack": 0, "tolist": 0}

    def _stack(values: list[_FakeScalar]) -> _FakeStack:
        calls["stack"] += 1
        return _FakeStack(values)

    monkeypatch.setattr(trainer_mod.torch, "stack", _stack)
    scalars = Trainer._extract_loss_scalars(
        cast(
            dict[str, torch.Tensor],
            {"policy": _FakeScalar(1.25), "total": _FakeScalar(4.0)},
        ),
        total_override=cast(torch.Tensor, cast(object, _FakeScalar(2.0))),
        total_scale=2.0,
    )

    assert scalars == {"policy": 1.25, "loss": 4.0}
    assert calls == {"stack": 1, "tolist": 1}


def test_train_batch_iterator_prefetches_across_optimizer_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path, prefetch_batches=True)
    sample_index = 0
    worker_threads: list[threading.Thread] = []
    second_started = threading.Event()

    def fake_sample_batch_host(
        _buf: Any,
        *,
        batch_size: int,
        mirror_prob: float,
        **_kw: Any,
    ) -> dict[str, np.ndarray]:
        nonlocal sample_index
        del batch_size, mirror_prob
        index = sample_index
        sample_index += 1
        worker_threads.append(threading.current_thread())
        if index == 1:
            second_started.set()
        return {"x": np.asarray([index], dtype=np.float32)}

    def fake_host_batch_to_tensors(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {key: torch.from_numpy(value) for key, value in batch.items()}

    monkeypatch.setattr(trainer, "_sample_batch_host", fake_sample_batch_host)
    monkeypatch.setattr(trainer, "_host_batch_to_tensors", fake_host_batch_to_tensors)
    batch_iter = _TrainBatchIterator(
        lambda count: trainer._iter_prefetched_batches(
            cast(Any, None),
            batch_size=1,
            mirror_prob=0.0,
            count=count,
        ),
        2,
    )

    assert float(next(batch_iter)["x"].item()) == 0.0
    assert second_started.wait(timeout=1.0)
    assert float(next(batch_iter)["x"].item()) == 1.0
    batch_iter.close()

    assert sample_index == 2
    assert worker_threads
    assert all(thread is not threading.current_thread() for thread in worker_threads)
    assert all(not thread.is_alive() for thread in worker_threads)


def test_train_steps_extends_prefetch_exactly_for_cuda_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    next_index = 0
    factory_counts: list[int] = []
    closed_counts: list[int] = []
    seen: list[int] = []

    def fake_iter_prefetched_batches(
        _buf: Any,
        *,
        batch_size: int,
        mirror_prob: float,
        count: int,
    ):
        nonlocal next_index
        del batch_size, mirror_prob
        factory_counts.append(count)
        try:
            for _ in range(count):
                index = next_index
                next_index += 1
                yield {"x": torch.asarray(index)}
        finally:
            closed_counts.append(count)

    def fake_run_optimizer_step(
        *,
        step_sums: dict[str, float],
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: Any,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Any = None,
    ) -> tuple[int, float]:
        del step_opt_stats, step_acc_sums, buf, batch_size, update_lr, collect_optimizer_stats
        seen.append(int(next(batch_iter)["x"].item()))
        if len(seen) == 1:
            raise RuntimeError("CUDA transient test failure")
        for key in (
            "loss",
            "policy_loss",
            "soft_policy_loss",
            "future_policy_loss",
            "wdl_loss",
            "sf_move_loss",
            "sf_eval_loss",
            "categorical_loss",
            "volatility_loss",
            "sf_volatility_loss",
            "moves_left_loss",
        ):
            step_sums[key] = 0.0
        return 1, 0.0

    monkeypatch.setattr(trainer, "_iter_prefetched_batches", fake_iter_prefetched_batches)
    monkeypatch.setattr(trainer, "_run_optimizer_step", fake_run_optimizer_step)
    monkeypatch.setattr(trainer_mod.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(trainer_mod.time, "sleep", lambda _seconds: None)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=2)

    assert metrics.train_steps_done == 2
    assert seen == [0, 1, 2]
    assert factory_counts == [2, 1]
    assert closed_counts == [2, 1]


def test_train_steps_closes_prefetch_after_terminal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    closed = False

    def fake_iter_prefetched_batches(
        _buf: Any,
        *,
        batch_size: int,
        mirror_prob: float,
        count: int,
    ):
        nonlocal closed
        del batch_size, mirror_prob
        try:
            for index in range(count):
                yield {"x": torch.asarray(index)}
        finally:
            closed = True

    def fake_run_optimizer_step(
        *,
        step_sums: dict[str, float],
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: Any,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Any = None,
    ) -> tuple[int, float]:
        del step_opt_stats, step_sums, step_acc_sums, buf, batch_size, update_lr, collect_optimizer_stats
        next(batch_iter)
        raise RuntimeError("terminal host failure")

    monkeypatch.setattr(trainer, "_iter_prefetched_batches", fake_iter_prefetched_batches)
    monkeypatch.setattr(trainer, "_run_optimizer_step", fake_run_optimizer_step)

    with pytest.raises(RuntimeError, match="terminal host failure"):
        trainer.train_steps(cast(Any, None), batch_size=1, steps=2)

    assert closed is True


def test_select_input_history_arrays_uses_recorded_lc0_root_and_legacy_meta() -> None:
    legacy = np.zeros((2, 146, 8, 8), dtype=np.float32)
    legacy[:, 100, :, :] = 0.5
    legacy[:, 102, :, :] = 0.37
    recorded = np.zeros_like(legacy)
    recorded[0, 3, :, :] = 7.0

    out = select_input_history_arrays(
        {
            "x": legacy,
            "x_lc0_root": recorded,
            "has_x_lc0_root": np.array([1, 0], dtype=np.uint8),
        },
        input_history_encoding="lc0_root_legacy_meta",
        # Row 1 has no recorded root tensor, so it goes through the POV-lossy
        # synthetic remap (M12) — this test is about the remap machinery.
        allow_lossy_legacy_remap=True,
    )

    assert np.all(out["x"][0, 3] == 7.0)
    assert np.all(out["x"][:, 109] == 0.37)
    assert np.all(out["x"][:, 110] == 0.5)


def test_select_input_history_arrays_is_idempotent() -> None:
    legacy = np.zeros((1, 146, 8, 8), dtype=np.float32)
    legacy[:, 0, :, :] = 1.0

    once = select_input_history_arrays(
        {"x": legacy},
        input_history_encoding="lc0_root",
        allow_lossy_legacy_remap=True,
    )
    twice = select_input_history_arrays(
        once,
        input_history_encoding="lc0_root",
        allow_lossy_legacy_remap=True,
    )

    assert twice is once
    np.testing.assert_array_equal(twice["x"], once["x"])


def test_select_input_history_arrays_respects_stored_lc0_root_metadata() -> None:
    root = np.zeros((1, 146, 8, 8), dtype=np.float32)
    root[:, 13, :, :] = 9.0

    out = select_input_history_arrays(
        {
            "x": root,
            "_input_history_encoding": np.asarray("lc0_root"),
        },
        input_history_encoding="lc0_root",
    )

    np.testing.assert_array_equal(out["x"], root)
    assert str(np.asarray(out["_input_history_encoding_selected"]).item()) == "lc0_root"


def test_select_input_history_arrays_handles_mixed_legacy_and_root_rows() -> None:
    legacy = np.zeros((2, 146, 8, 8), dtype=np.float32)
    legacy[0, 0, :, :] = 1.0
    root = np.zeros_like(legacy)
    root[1, 13, :, :] = 9.0
    mixed = legacy.copy()
    mixed[1] = root[1]

    out = select_input_history_arrays(
        {
            "x": mixed,
            "_input_history_encoding": np.asarray(["", "lc0_root"], dtype=object),
        },
        input_history_encoding="lc0_root",
        allow_lossy_legacy_remap=True,
    )

    assert np.all(out["x"][0, 0] == 1.0)
    assert np.all(out["x"][1, 13] == 9.0)
    assert np.all(np.asarray(out["_input_history_encoding"]) == "lc0_root")


def test_select_input_history_arrays_rejects_root_reselect_mismatch() -> None:
    root = np.zeros((1, 146, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="incompatible stored history encodings"):
        select_input_history_arrays(
            {
                "x": root,
                "_input_history_encoding": np.asarray("lc0_root"),
            },
            input_history_encoding="lc0_root_legacy_meta",
        )


def test_select_input_history_arrays_rejects_root_storage_for_legacy_model() -> None:
    root = np.zeros((1, 146, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="cannot train"):
        select_input_history_arrays(
            {
                "x": root,
                "_input_history_encoding": np.asarray("lc0_root"),
            },
            input_history_encoding="legacy",
        )


def test_select_input_history_arrays_fast_path_handles_uniform_metadata() -> None:
    root = np.zeros((2, 146, 8, 8), dtype=np.float32)
    root[:, 13, :, :] = 4.0

    out = select_input_history_arrays(
        {
            "x": root,
            "_input_history_encoding": np.asarray(["lc0_root", "lc0_root"], dtype=object),
        },
        input_history_encoding="lc0_root",
    )

    np.testing.assert_array_equal(out["x"], root)
    assert str(np.asarray(out["_input_history_encoding_selected"]).item()) == "lc0_root"


def test_select_input_history_samples_uses_recorded_lc0_root() -> None:
    legacy = np.zeros((1, 146, 8, 8), dtype=np.float32)
    root = np.zeros_like(legacy)
    root[:, 7, :, :] = 3.0
    sample = ReplaySample(
        x=legacy[0],
        x_lc0_root=root[0],
        policy_target=np.zeros((4672,), dtype=np.float32),
        wdl_target=1,
    )

    out = select_input_history_samples(
        [sample],
        input_history_encoding="lc0_root",
    )

    assert out[0] is not sample
    assert out[0].input_history_encoding == "lc0_root"
    np.testing.assert_array_equal(out[0].x, root[0])
    np.testing.assert_array_equal(sample.x, legacy[0])


def test_select_input_history_samples_preserves_already_root_samples() -> None:
    root = np.zeros((146, 8, 8), dtype=np.float32)
    root[104, :, :] = 5.0
    sample = ReplaySample(
        x=root,
        policy_target=np.zeros((4672,), dtype=np.float32),
        wdl_target=1,
        input_history_encoding="lc0_root",
    )

    out = select_input_history_samples(
        [sample],
        input_history_encoding="lc0_root",
    )

    assert out[0].input_history_encoding == "lc0_root"
    np.testing.assert_array_equal(out[0].x, root)


def test_select_input_history_samples_rejects_root_samples_for_legacy_model() -> None:
    sample = ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=np.zeros((4672,), dtype=np.float32),
        wdl_target=1,
        input_history_encoding="lc0_root",
    )

    with pytest.raises(ValueError, match="cannot train"):
        select_input_history_samples([sample], input_history_encoding="legacy")


def test_selected_input_history_samples_serialize_history_metadata() -> None:
    legacy = np.zeros((1, 146, 8, 8), dtype=np.float32)
    root = np.zeros_like(legacy)
    root[:, 7, :, :] = 3.0
    sample = ReplaySample(
        x=legacy[0],
        x_lc0_root=root[0],
        policy_target=np.eye(1, 4672, 0, dtype=np.float32)[0],
        wdl_target=1,
    )

    selected = select_input_history_samples([sample], input_history_encoding="lc0_root")
    arrs = samples_to_arrays(selected)

    assert str(np.asarray(arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY]).item()) == "lc0_root"


class _TinyScopedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "ffn": torch.nn.Linear(4, 4),
                        "q_proj": torch.nn.Linear(4, 4),
                        "k_proj": torch.nn.Linear(4, 4),
                        "v_proj": torch.nn.Linear(4, 4),
                        "out_proj": torch.nn.Linear(4, 4),
                    }
                )
            ]
        )
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def _make_trainer(tmp_path: Path, **kwargs: Any) -> Trainer:
    trainer_kwargs: dict[str, Any] = {
        "device": "cpu",
        "lr": 1e-3,
        "optimizer": "muon",
        "warmup_steps": 10,
        "warmup_lr_start": 1e-5,
        "use_amp": False,
        "log_dir": tmp_path,
        "tb_log_interval": 1000,
        "prefetch_batches": False,
    }
    trainer_kwargs.update(kwargs)
    return Trainer(
        _TinyMuonModel(),
        **trainer_kwargs,
    )


def _make_aurora_trainer(tmp_path: Path) -> Trainer:
    return Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def _make_custom_aurora_trainer(tmp_path: Path) -> Trainer:
    return Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        matrix_lr_multiplier=12.0,
        matrix_weight_decay=3e-5,
        aux_weight_decay=7e-5,
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def _make_muon_scope_trainer(tmp_path: Path, scope: str) -> Trainer:
    return Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="muon",
        matrix_optimizer_scope=scope,
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def _make_scoped_trainer(tmp_path: Path, optimizer: str, scope: str) -> Trainer:
    return Trainer(
        _TinyScopedModel(),
        device="cpu",
        lr=1e-3,
        optimizer=optimizer,
        matrix_optimizer_scope=scope,
        matrix_weight_decay=3e-5,
        aux_weight_decay=7e-5,
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def test_muon_warmup_preserves_group_lr_ratio_from_step_zero(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trunk_lr = float(trainer.opt.param_groups[0]["lr"])
    aux_lr = float(trainer.opt.param_groups[2]["lr"])
    assert trunk_lr == aux_lr * 20.0


def test_muon_warmup_handoff_reaches_group_base_lr_without_ratio_jump(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.step = trainer._warmup_steps - 1
    trainer._update_lr()

    base_lrs = trainer._base_lrs()
    assert float(trainer.opt.param_groups[0]["lr"]) == float(base_lrs[0])
    assert float(trainer.opt.param_groups[2]["lr"]) == float(base_lrs[2])
    assert float(trainer.opt.param_groups[0]["lr"]) == float(trainer.opt.param_groups[2]["lr"]) * 20.0


def test_muon_set_peak_lr_rebases_from_search_lr_not_trunk_lr(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.step = trainer._warmup_steps
    old_base_lrs = trainer._base_lrs()

    trainer.set_peak_lr(2e-3, rescale_current=False)

    new_base_lrs = trainer._base_lrs()
    assert new_base_lrs[0] == old_base_lrs[0] * 2.0
    assert new_base_lrs[2] == old_base_lrs[2] * 2.0
    assert new_base_lrs[0] == new_base_lrs[2] * 20.0


def test_muon_load_restores_peak_lr_from_search_lr(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path / "src")
    ckpt = tmp_path / "trainer.pt"
    trainer.save(ckpt)

    loaded = _make_trainer(tmp_path / "dst")
    loaded.load(ckpt)

    assert loaded._peak_lr == 1e-3
    assert loaded._base_lrs()[0] == loaded._base_lrs()[2] * 20.0


def test_sqrt_release_lr_schedule_shape(tmp_path: Path) -> None:
    trainer = _make_trainer(
        tmp_path,
        warmup_steps=0,
        lr_schedule="sqrt_release",
        lr_release_cycle_steps=1000,
        lr_release_start_frac=0.85,
        lr_release_min_scale=0.1,
    )
    base_lrs = trainer._base_lrs()

    def lrs_at(step: int) -> list[float]:
        trainer.step = step
        trainer._update_lr()
        return [float(pg["lr"]) for pg in trainer.opt.param_groups]

    assert lrs_at(0) == base_lrs
    assert lrs_at(850) == base_lrs

    expected_mid_scale = 0.1 + 0.9 * (1.0 - np.sqrt((925 - 850) / 150))
    for got, base_lr in zip(lrs_at(925), base_lrs, strict=True):
        assert abs(got - (base_lr * expected_mid_scale)) < 1e-12

    expected_late_scale = 0.1 + 0.9 * (1.0 - np.sqrt((999 - 850) / 150))
    for got, base_lr in zip(lrs_at(999), base_lrs, strict=True):
        assert abs(got - (base_lr * expected_late_scale)) < 1e-12

    assert lrs_at(1000) == base_lrs


def test_cosine_lr_schedule_phase_starts_after_warmup(tmp_path: Path) -> None:
    trainer = _make_trainer(
        tmp_path,
        warmup_steps=10,
        lr_schedule="cosine",
        lr_T0=100,
        lr_eta_min=0.0,
    )
    base_lrs = trainer._base_lrs()

    trainer.step = 10
    trainer._update_lr()

    assert [float(pg["lr"]) for pg in trainer.opt.param_groups] == base_lrs


def test_sqrt_release_accepts_cosine_tail_shape(tmp_path: Path) -> None:
    trainer = _make_trainer(
        tmp_path,
        warmup_steps=0,
        lr_schedule="sqrt_release",
        lr_release_cycle_steps=1000,
        lr_release_start_frac=0.85,
        lr_release_min_scale=0.1,
        lr_release_shape="cosine",
    )
    base_lrs = trainer._base_lrs()

    trainer.step = 925
    trainer._update_lr()

    expected_scale = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * ((925 - 850) / 150)))
    for got, base_lr in zip(
        [float(pg["lr"]) for pg in trainer.opt.param_groups],
        base_lrs,
        strict=True,
    ):
        assert abs(got - (base_lr * expected_scale)) < 1e-12


def test_sqrt_release_rejects_unknown_tail_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lr_release_shape"):
        _make_trainer(
            tmp_path,
            warmup_steps=0,
            lr_schedule="sqrt_release",
            lr_release_shape="linear",
        )


def test_sqrt_release_live_updates_release_knobs(tmp_path: Path) -> None:
    trainer = _make_trainer(
        tmp_path,
        warmup_steps=0,
        lr_schedule="sqrt_release",
        lr_release_cycle_steps=1000,
        lr_release_start_frac=0.85,
        lr_release_min_scale=0.1,
    )

    trainer.set_lr_release_config(
        cycle_steps=0,
        release_start_frac=0.5,
        min_scale=0.2,
        release_shape="cosine",
    )
    scheduler: Any = trainer._scheduler

    assert trainer._lr_release_cycle_steps == 0
    assert scheduler.cycle_steps == 0
    assert scheduler.release_start_frac == 0.5
    assert scheduler.min_scale == 0.2
    assert scheduler.release_shape == "cosine"


def test_sqrt_release_scheduler_accepts_cosine_checkpoint(tmp_path: Path) -> None:
    src = _make_trainer(tmp_path / "src", lr_schedule="cosine")
    ckpt = tmp_path / "cosine.pt"
    src.save(ckpt)

    loaded = _make_trainer(
        tmp_path / "dst",
        warmup_steps=0,
        lr_schedule="sqrt_release",
        lr_release_cycle_steps=1000,
        lr_release_start_frac=0.85,
        lr_release_min_scale=0.1,
    )
    loaded.load(ckpt)

    loaded.step = 925
    loaded._update_lr()
    expected_scale = 0.1 + 0.9 * (1.0 - np.sqrt((925 - 850) / 150))
    for got, base_lr in zip(
        [float(pg["lr"]) for pg in loaded.opt.param_groups],
        loaded._base_lrs(),
        strict=True,
    ):
        assert abs(got - (base_lr * expected_scale)) < 1e-12


def test_sqrt_release_zero_cycle_uses_train_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(
        tmp_path,
        warmup_steps=0,
        lr_schedule="sqrt_release",
        lr_release_cycle_steps=0,
        lr_release_start_frac=0.85,
        lr_release_min_scale=0.1,
    )
    base_lrs = trainer._base_lrs()
    seen_lrs: list[list[float]] = []
    seen_collect: list[bool] = []

    def fake_run_optimizer_step(
        *,
        step_sums: dict[str, float],
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: Any,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Any = None,
    ) -> tuple[int, float]:
        del step_opt_stats, step_acc_sums, buf, batch_size, batch_iter
        assert update_lr is False
        seen_collect.append(bool(collect_optimizer_stats))
        seen_lrs.append([float(pg["lr"]) for pg in trainer.opt.param_groups])
        for key in (
            "loss",
            "policy_loss",
            "soft_policy_loss",
            "future_policy_loss",
            "wdl_loss",
            "sf_move_loss",
            "sf_eval_loss",
            "categorical_loss",
            "volatility_loss",
            "sf_volatility_loss",
            "moves_left_loss",
        ):
            step_sums[key] = 0.0
        return 1, 0.0

    monkeypatch.setattr(trainer, "_run_optimizer_step", fake_run_optimizer_step)

    trainer.train_steps(cast(Any, None), batch_size=1, steps=100)

    assert seen_lrs[0] == base_lrs
    assert seen_lrs[85] == base_lrs
    assert seen_collect == [False] * 99 + [True]
    for got, base_lr in zip(seen_lrs[-1], base_lrs, strict=True):
        assert abs(got - (base_lr * 0.1)) < 1e-12
    assert [float(pg["lr"]) for pg in trainer.opt.param_groups] == seen_lrs[-1]


def test_sqrt_release_window_step_hits_min_scale_on_final_step() -> None:
    param = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.SGD([param], lr=0.5)
    scheduler = _SqrtReleaseLRScheduler(
        opt,
        cycle_steps=0,
        release_start_frac=0.8,
        min_scale=0.1,
    )

    scheduler.step_window(0, cycle_steps=10)
    assert float(opt.param_groups[0]["lr"]) == 0.5
    scheduler.step_window(8, cycle_steps=10)
    assert float(opt.param_groups[0]["lr"]) == 0.5
    scheduler.step_window(9, cycle_steps=10)
    assert float(opt.param_groups[0]["lr"]) == pytest.approx(0.05)
    scheduler.step_window(4, cycle_steps=5)
    assert float(opt.param_groups[0]["lr"]) == pytest.approx(0.05)
    scheduler.step_window(0, cycle_steps=1)
    assert float(opt.param_groups[0]["lr"]) == 0.5


def test_sqrt_release_zero_cycle_switches_after_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(
        tmp_path,
        warmup_steps=10,
        lr_schedule="sqrt_release",
        lr_release_cycle_steps=0,
        lr_release_start_frac=0.5,
        lr_release_min_scale=0.1,
    )
    base_lrs = trainer._base_lrs()
    seen: list[tuple[int, bool, list[float]]] = []

    def fake_run_optimizer_step(
        *,
        step_sums: dict[str, float],
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: Any,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Any = None,
    ) -> tuple[int, float]:
        del step_opt_stats, step_acc_sums, buf, batch_size, collect_optimizer_stats, batch_iter
        seen.append((int(trainer.step), bool(update_lr), [float(pg["lr"]) for pg in trainer.opt.param_groups]))
        for key in (
            "loss",
            "policy_loss",
            "soft_policy_loss",
            "future_policy_loss",
            "wdl_loss",
            "sf_move_loss",
            "sf_eval_loss",
            "categorical_loss",
            "volatility_loss",
            "sf_volatility_loss",
            "moves_left_loss",
        ):
            step_sums[key] = 0.0
        if update_lr:
            trainer._update_lr()
        return 1, 0.0

    monkeypatch.setattr(trainer, "_run_optimizer_step", fake_run_optimizer_step)

    trainer.train_steps(cast(Any, None), batch_size=1, steps=20)

    assert [update_lr for _, update_lr, _ in seen[:10]] == [True] * 10
    assert [update_lr for _, update_lr, _ in seen[10:]] == [False] * 10
    assert seen[10][2] == base_lrs

    for got, base_lr in zip(seen[-1][2], base_lrs, strict=True):
        assert abs(got - (base_lr * 0.1)) < 1e-12


def test_aurora_uses_matrix_lr_and_adam_fallback_lr(tmp_path: Path) -> None:
    trainer = _make_aurora_trainer(tmp_path)
    assert isinstance(trainer.opt, AuroraWithAuxAdam)
    assert trainer.opt._use_update_graphs is True
    aurora_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_aurora", False)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_aurora", False)]
    named_params = dict(trainer.model.named_parameters())
    aurora_param_ids = {id(param) for pg in aurora_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert len(aurora_groups) == 1
    assert all(param.ndim == 2 for param in aurora_groups[0]["params"])
    assert float(aurora_groups[0]["lr"]) == float(fallback_groups[0]["lr"]) * 20.0
    assert id(named_params["blocks.0.weight"]) in aurora_param_ids
    assert id(named_params["embed.weight"]) in fallback_param_ids


def test_aurora_cuda_graphs_can_be_disabled(tmp_path: Path) -> None:
    kwargs = trainer_kwargs_from_config(
        {"optimizer": "aurora", "aurora_cuda_graphs": False},
        log_dir=tmp_path,
    )
    assert kwargs["aurora_cuda_graphs"] is False
    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        aurora_cuda_graphs=kwargs["aurora_cuda_graphs"],
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    assert isinstance(trainer.opt, AuroraWithAuxAdam)
    assert trainer.opt._use_update_graphs is False


def test_aurora_accepts_matrix_lr_multiplier_and_weight_decay(tmp_path: Path) -> None:
    trainer = _make_custom_aurora_trainer(tmp_path)
    aurora_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_aurora", False)]
    fallback_decay_groups = [
        pg for pg in trainer.opt.param_groups
        if not pg.get("use_aurora", False) and float(pg.get("weight_decay", 0.0)) > 0.0
    ]

    assert float(aurora_groups[0]["lr"]) == float(fallback_decay_groups[0]["lr"]) * 12.0
    assert float(aurora_groups[0]["weight_decay"]) == 3e-5
    assert float(fallback_decay_groups[0]["weight_decay"]) == 7e-5


def test_load_resets_scheduler_when_optimizer_state_is_incompatible(tmp_path: Path) -> None:
    src = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="adamw",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path / "src",
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    ckpt = tmp_path / "adamw.pt"
    src.save(ckpt)

    loaded = _make_aurora_trainer(tmp_path / "dst")
    loaded.load(ckpt)

    assert len(loaded._scheduler.base_lrs) == len(loaded.opt.param_groups)
    assert loaded._base_lrs()[0] == loaded._base_lrs()[2] * 20.0


def test_soda_weight_decay_mode_replaces_only_decay_groups(tmp_path: Path) -> None:
    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        matrix_lr_multiplier=12.0,
        matrix_weight_decay=3e-5,
        aux_weight_decay=7e-5,
        weight_decay_mode="soda",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )

    assert isinstance(trainer.opt, SODAWeightDecayWrapper)
    soda_groups = [pg for pg in trainer.opt.param_groups if pg.get("soda_regularize", False)]
    no_soda_groups = [pg for pg in trainer.opt.param_groups if not pg.get("soda_regularize", False)]

    assert len(soda_groups) == 2
    assert all(float(pg.get("weight_decay", 0.0)) == 0.0 for pg in soda_groups)
    assert sorted(float(pg["soda_replaced_weight_decay"]) for pg in soda_groups) == [3e-5, 7e-5]
    assert all(float(pg.get("weight_decay", 0.0)) == 0.0 for pg in no_soda_groups)


def test_soda_wrapper_applies_anchor_average_after_base_step() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    base = torch.optim.SGD([
        {"params": [param], "lr": 0.1, "weight_decay": 0.0, "soda_regularize": True}
    ])
    opt = SODAWeightDecayWrapper(base)

    param.grad = torch.tensor([1.0])
    opt.step()
    assert torch.allclose(param, torch.tensor([0.9]))

    param.grad = torch.tensor([1.0])
    opt.step()
    assert torch.allclose(param, torch.tensor([0.8333333]), atol=1e-6)


def test_chained_optimizer_rejects_unroutable_param_group() -> None:
    p0 = torch.nn.Parameter(torch.tensor([1.0]))
    p1 = torch.nn.Parameter(torch.tensor([2.0]))
    opt = _ChainedOptimizer([
        torch.optim.SGD([p0], lr=0.1),
        torch.optim.SGD([p1], lr=0.1),
    ])

    with pytest.raises(NotImplementedError, match="cannot route"):
        opt.add_param_group({"params": [torch.nn.Parameter(torch.tensor([3.0]))]})


def test_muon_matrix_scope_can_target_mlp_without_embed(tmp_path: Path) -> None:
    trainer = _make_muon_scope_trainer(tmp_path, "mlp_only")
    muon_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_muon", False)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_muon", False)]
    named_params = dict(trainer.model.named_parameters())
    muon_param_ids = {id(param) for pg in muon_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert id(named_params["blocks.0.weight"]) not in muon_param_ids
    assert id(named_params["embed.weight"]) in fallback_param_ids


def test_aurora_matrix_scope_can_target_mlp_out_v_without_qk(tmp_path: Path) -> None:
    trainer = _make_scoped_trainer(tmp_path, "aurora", "mlp_out_v")
    aurora_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_aurora", False)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_aurora", False)]
    named_params = dict(trainer.model.named_parameters())
    aurora_param_ids = {id(param) for pg in aurora_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert id(named_params["blocks.0.ffn.weight"]) in aurora_param_ids
    assert id(named_params["blocks.0.out_proj.weight"]) in aurora_param_ids
    assert id(named_params["blocks.0.v_proj.weight"]) in aurora_param_ids
    assert id(named_params["blocks.0.q_proj.weight"]) in fallback_param_ids
    assert id(named_params["blocks.0.k_proj.weight"]) in fallback_param_ids


def test_soap_matrix_scope_can_target_mlp_with_adam_fallback(tmp_path: Path) -> None:
    trainer = _make_scoped_trainer(tmp_path, "soap", "mlp_only")
    named_params = dict(trainer.model.named_parameters())
    soap_param_ids = {id(param) for param in trainer.opt.param_groups[0]["params"]}
    fallback_param_ids = {
        id(param)
        for pg in trainer.opt.param_groups[1:]
        for param in pg["params"]
    }

    assert id(named_params["blocks.0.ffn.weight"]) in soap_param_ids
    assert id(named_params["blocks.0.out_proj.weight"]) in fallback_param_ids
    assert id(named_params["head.weight"]) in fallback_param_ids
    assert float(trainer.opt.param_groups[0]["weight_decay"]) == 3e-5
    assert any(float(pg["weight_decay"]) == 7e-5 for pg in trainer.opt.param_groups[1:])


def test_soap_soda_marks_matrix_and_adam_fallback_decay_groups(tmp_path: Path) -> None:
    trainer = Trainer(
        _TinyScopedModel(),
        device="cpu",
        lr=1e-3,
        optimizer="soap",
        matrix_optimizer_scope="mlp_only",
        matrix_weight_decay=3e-5,
        aux_weight_decay=7e-5,
        weight_decay_mode="soda",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )

    assert isinstance(trainer.opt, SODAWeightDecayWrapper)
    soda_groups = [pg for pg in trainer.opt.param_groups if pg.get("soda_regularize", False)]

    assert len(soda_groups) == 2
    assert sorted(float(pg["soda_replaced_weight_decay"]) for pg in soda_groups) == [3e-5, 7e-5]
    assert all(float(pg.get("weight_decay", 0.0)) == 0.0 for pg in trainer.opt.param_groups)


def test_zclip_max_norm_can_be_disabled_from_config(tmp_path: Path) -> None:
    kwargs = trainer_kwargs_from_config(
        {"zclip_max_norm": None, "zclip_clip_factor": 0.75},
        log_dir=tmp_path,
    )
    assert kwargs["zclip_max_norm"] is None
    assert kwargs["zclip_clip_factor"] == 0.75

    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        zclip_max_norm=kwargs["zclip_max_norm"],
        zclip_clip_factor=kwargs["zclip_clip_factor"],
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )

    assert trainer.zclip.max_grad_norm is None
    assert trainer.zclip.clip_factor == 0.75


def test_sqrt_release_config_defaults_to_train_window_cycle(tmp_path: Path) -> None:
    kwargs = trainer_kwargs_from_config(
        {"lr_schedule": "sqrt_release"},
        log_dir=tmp_path,
    )

    assert kwargs["lr_release_cycle_steps"] == 0
    assert kwargs["lr_release_shape"] == "sqrt"


def test_zclip_step_reports_hard_and_adaptive_clipping(tmp_path: Path) -> None:
    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        zclip_z_thresh=2.0,
        zclip_max_norm=1.0,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)

    _grad_norm, stats = trainer._zclip_step(collect_stats=True)

    assert stats is not None
    assert stats["hard_clip"] == 1.0
    assert stats["clipped"] == 1.0

    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        zclip_z_thresh=2.0,
        zclip_max_norm=None,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    trainer.zclip.initialized = True
    trainer.zclip.mean = 1.0
    trainer.zclip.var = 0.01
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)

    _grad_norm, stats = trainer._zclip_step(collect_stats=True)

    assert stats is not None
    assert stats["adaptive_clip"] == 1.0
    assert stats["hard_clip"] == 0.0
    assert stats["clipped"] == 1.0


# --- rl_loop_audit I13: configured param-group hyperparameters survive a load ---


def test_configured_matrix_weight_decay_survives_checkpoint_round_trip(tmp_path: Path) -> None:
    # The exact defect: Optimizer.load_state_dict replaces every group's
    # hyperparameters with the checkpoint's, so `matrix_weight_decay: 0` in the
    # yaml stayed a no-op against a checkpoint stamped with 1e-4, forever.
    donor = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        matrix_weight_decay=1e-4,
        aux_weight_decay=1e-4,
        use_amp=False,
        log_dir=tmp_path / "donor",
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)
    assert float(donor.opt.param_groups[0]["weight_decay"]) == 1e-4

    loaded = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        matrix_weight_decay=0.0,
        aux_weight_decay=3e-5,
        use_amp=False,
        log_dir=tmp_path / "loaded",
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    loaded.load(ckpt)

    aurora_groups = [pg for pg in loaded.opt.param_groups if pg.get("use_aurora", False)]
    aux_decay_groups = [
        pg for pg in loaded.opt.param_groups
        if not pg.get("use_aurora", False) and float(pg.get("weight_decay", 0.0)) > 0.0
    ]
    assert float(aurora_groups[0]["weight_decay"]) == 0.0
    assert [float(pg["weight_decay"]) for pg in aux_decay_groups] == [3e-5]


def test_configured_aurora_uw_floor_survives_checkpoint_round_trip(tmp_path: Path) -> None:
    # weight_decay was not the only group hyperparameter with no re-application
    # path — aurora_uw_floor is read per step straight off the group dict.
    donor = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        aurora_uw_floor=0.25,
        use_amp=False,
        log_dir=tmp_path / "donor",
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)

    loaded = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        aurora_uw_floor=0.0,
        use_amp=False,
        log_dir=tmp_path / "loaded",
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    loaded.load(ckpt)

    assert float(loaded.opt.param_groups[0]["aurora_uw_floor"]) == 0.0


def test_load_keeps_checkpoint_owned_group_keys(tmp_path: Path) -> None:
    # `lr` stays the checkpoint's — set_peak_lr re-applies it afterwards and the
    # scheduler owns the phase, so the re-application must not fight that.
    donor = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        warmup_steps=0,
        use_amp=False,
        log_dir=tmp_path / "donor",
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    donor_lrs = [float(pg["lr"]) for pg in donor.opt.param_groups]
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)

    loaded = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=5e-3,
        optimizer="aurora",
        warmup_steps=0,
        use_amp=False,
        log_dir=tmp_path / "loaded",
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    loaded.load(ckpt)

    assert [float(pg["lr"]) for pg in loaded.opt.param_groups] == donor_lrs


def test_load_keeps_soda_step_counter_but_reapplies_soda_config(tmp_path: Path) -> None:
    def _make_soda_trainer(label: str, *, matrix_weight_decay: float) -> Trainer:
        return Trainer(
            _TinyMuonModel(),
            device="cpu",
            lr=1e-3,
            optimizer="aurora",
            weight_decay_mode="soda",
            matrix_weight_decay=matrix_weight_decay,
            aux_weight_decay=1e-4,
            use_amp=False,
            log_dir=tmp_path / label,
            tb_log_interval=1000,
            prefetch_batches=False,
        )

    donor = _make_soda_trainer("donor", matrix_weight_decay=1e-4)
    donor_group = next(pg for pg in donor.opt.param_groups if pg.get("soda_regularize", False))
    donor_group["soda_k"] = 7
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)

    loaded = _make_soda_trainer("loaded", matrix_weight_decay=3e-5)
    loaded.load(ckpt)

    loaded_group = next(pg for pg in loaded.opt.param_groups if pg.get("soda_regularize", False))
    # soda_k is a per-group COUNTER: the checkpoint owns it.
    assert int(loaded_group["soda_k"]) == 7
    # soda_replaced_weight_decay is config-derived: this run's value wins.
    assert float(loaded_group["soda_replaced_weight_decay"]) == 3e-5


# --- rl_loop_audit I9/I11: grad-norm + clip rate reach the metric stream ---


_REQUIRED_LOSS_METRIC_KEYS = (
    "loss", "policy_loss", "soft_policy_loss", "future_policy_loss", "wdl_loss",
    "sf_move_loss", "sf_eval_loss", "categorical_loss", "volatility_loss",
    "sf_volatility_loss", "moves_left_loss",
)


def _zeroed_metrics(**overrides: Any) -> TrainMetrics:
    base: dict[str, Any] = dict.fromkeys(_REQUIRED_LOSS_METRIC_KEYS, 0.0)
    base["sf_move_acc"] = 0.0
    base.update(overrides)
    return TrainMetrics(**base)


def test_grad_clip_metric_kwargs_aggregates_the_whole_iteration() -> None:
    kwargs = trainer_mod._grad_clip_metric_kwargs(
        [1.0, 5.0, 3.0, 4.0],
        {"clipped": 2, "adaptive_clip": 1, "adaptive_bound": 1, "hard_clip": 1},
    )

    assert kwargs["grad_norm_mean"] == pytest.approx(3.25)
    assert kwargs["grad_norm_median"] == pytest.approx(3.5)
    assert kwargs["grad_norm_p95"] == pytest.approx(5.0)
    assert kwargs["grad_norm_max"] == pytest.approx(5.0)
    assert kwargs["grad_clip_rate"] == pytest.approx(0.5)
    assert kwargs["grad_adaptive_clip_rate"] == pytest.approx(0.25)
    assert kwargs["grad_adaptive_bound_rate"] == pytest.approx(0.25)
    assert kwargs["grad_hard_clip_rate"] == pytest.approx(0.25)
    assert kwargs["grad_norm_samples"] == 4
    assert trainer_mod._grad_clip_metric_kwargs([], {}) == {}


def test_run_optimizer_step_collects_clip_stats_off_the_tb_log_stride(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stats used to be gathered only when a step landed on tb_log_interval;
    # a 1-in-10 subsample cannot be aggregated into an honest clip rate.
    trainer = _make_trainer(tmp_path, optimizer="adamw", warmup_steps=0)
    trainer.step = 5
    assert trainer._should_log_step_scalars() is False

    param = next(trainer.model.parameters())

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del out, batch, kwargs
        return {"total": (param * param).sum()}

    monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})

    step_opt_stats: dict[str, float] = {}
    trainer._run_optimizer_step(
        step_sums={},
        step_acc_sums={},
        step_opt_stats=step_opt_stats,
        buf=cast(Any, None),
        batch_size=1,
        batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
    )

    assert set(step_opt_stats) >= {"grad_norm", "total_norm", "clipped", "hard_clip", "adaptive_clip", "lr"}
    assert step_opt_stats["grad_norm"] > 0.0
    assert step_opt_stats["total_norm"] == pytest.approx(step_opt_stats["grad_norm"])
    assert step_opt_stats["lr"] == pytest.approx(float(trainer._base_lrs()[0]))


def test_train_steps_reports_clip_rate_without_double_counting_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path, warmup_steps=0)
    attempts = {"n": 0}

    def fake_run_optimizer_step(
        *,
        step_sums: dict[str, float],
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: Any,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Any = None,
    ) -> tuple[int, float]:
        del step_acc_sums, buf, batch_size, update_lr, collect_optimizer_stats, batch_iter
        attempts["n"] += 1
        # The first attempt records a huge clipped norm and then dies: the retry
        # must replace it, not add to it.
        first = attempts["n"] == 1
        step_opt_stats["grad_norm"] = 100.0 if first else float(attempts["n"])
        step_opt_stats["clipped"] = 1.0 if first else 0.0
        step_opt_stats["hard_clip"] = 1.0 if first else 0.0
        step_opt_stats["adaptive_clip"] = 0.0
        step_opt_stats["lr"] = 1e-3
        if first:
            raise RuntimeError("CUDA transient test failure")
        step_sums.update(dict.fromkeys(_REQUIRED_LOSS_METRIC_KEYS, 0.0))
        return 1, 0.0

    monkeypatch.setattr(trainer, "_run_optimizer_step", fake_run_optimizer_step)
    monkeypatch.setattr(trainer_mod.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(trainer_mod.time, "sleep", lambda _seconds: None)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=3)

    assert metrics.grad_norm_samples == 3
    assert metrics.grad_norm_max == pytest.approx(4.0)
    assert metrics.grad_norm_mean == pytest.approx(3.0)
    assert metrics.grad_clip_rate == 0.0
    assert metrics.grad_hard_clip_rate == 0.0


def test_grad_norm_median_past_watch_threshold_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The median half of I11's condition, with the hard-clip half satisfied.

    UPDATED 2026-08-24: I11's condition is a conjunction (median past
    GRAD_NORM_MEDIAN_WATCH *and* hard-clip rate past
    GRAD_HARD_CLIP_RATE_WATCH), and the gate now says so. This case keeps its
    original intent — a median past the watch must still fire, and a median
    below it must still stay silent — and supplies the hard-clip rate that the
    original `_zeroed_metrics` left at 0.0. The 0.0 case is now a test of its
    own, `test_watch_is_silent_on_a_high_median_with_zero_hard_clips` in
    tests/test_zclip_watch_gate_and_optimizer_scalars.py, because it is the
    production false positive rather than an incidental default.
    """
    trainer = _make_trainer(tmp_path, zclip_max_norm=5.0)
    metrics = _zeroed_metrics(
        grad_norm_samples=200,
        grad_norm_median=trainer_mod.GRAD_NORM_MEDIAN_WATCH + 0.1,
        grad_clip_rate=0.40,
        grad_hard_clip_rate=0.40,
    )

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)
    assert "watch threshold" in caplog.text

    caplog.clear()
    metrics.grad_norm_median = trainer_mod.GRAD_NORM_MEDIAN_WATCH - 0.1
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)
    assert caplog.text == ""


# --- rl_loop_audit I19: the reported LR is not the sqrt_release trough ---


def test_train_steps_reports_iteration_mean_lr_not_the_release_trough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(
        tmp_path,
        warmup_steps=0,
        lr_schedule="sqrt_release",
        lr_release_cycle_steps=0,
        lr_release_start_frac=0.8,
        lr_release_min_scale=0.1,
    )
    base_lr = float(trainer._base_lrs()[0])

    def fake_run_optimizer_step(
        *,
        step_sums: dict[str, float],
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: Any,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Any = None,
    ) -> tuple[int, float]:
        del step_acc_sums, buf, batch_size, update_lr, collect_optimizer_stats, batch_iter
        step_opt_stats["lr"] = float(trainer.opt.param_groups[0]["lr"])
        step_sums.update(dict.fromkeys(_REQUIRED_LOSS_METRIC_KEYS, 0.0))
        return 1, 0.0

    monkeypatch.setattr(trainer, "_run_optimizer_step", fake_run_optimizer_step)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=100)

    trough = float(trainer.opt.param_groups[0]["lr"])
    assert trough == pytest.approx(base_lr * 0.1)
    assert metrics.opt_lr_max == pytest.approx(base_lr)
    # The duty cycle predicts ~0.88 of the plateau; the trough is 10x below it.
    assert metrics.opt_lr_mean == pytest.approx(base_lr * 0.88, rel=0.05)
    assert metrics.opt_lr_mean > 5.0 * trough
