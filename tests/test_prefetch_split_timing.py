"""CPU-only parity and attribution for the opt-in training batch timer."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.train.trainer import _BatchPrefetchSplitTiming, Trainer


def _trainer(*, prefetch: bool) -> Trainer:
    # Only the batch-producing methods run; no model, writer or optimizer is
    # needed to test their order, RNG consumption, and returned tensor bytes.
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer._prefetch_batches = prefetch
    return trainer


def _install_synthetic_sampler(
    trainer: Trainer, monkeypatch: pytest.MonkeyPatch, *, seed: int,
    sample_delay_s: float = 0.0, tensor_delay_s: float = 0.0,
) -> list[int]:
    rng = np.random.default_rng(seed)
    sampled: list[int] = []
    convert = trainer._host_batch_to_tensors

    def sample(_buf: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        if sample_delay_s:
            time.sleep(sample_delay_s)
        number = int(rng.integers(0, 10_000))
        sampled.append(number)
        return {
            "x": np.full((2, 1, 8, 8), number % 256, dtype=np.uint8),
            "policy_target": np.full((2, 3), number, dtype=np.float32),
            "wdl_target": np.full((2,), number % 3, dtype=np.int8),
        }

    def tensor(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        if tensor_delay_s:
            time.sleep(tensor_delay_s)
        return convert(batch)

    monkeypatch.setattr(trainer, "_sample_batch_host", sample)
    monkeypatch.setattr(trainer, "_host_batch_to_tensors", tensor)
    return sampled


def _equal_batches(
    left: list[dict[str, torch.Tensor]], right: list[dict[str, torch.Tensor]],
) -> None:
    assert len(left) == len(right)
    for original, measured in zip(left, right, strict=True):
        assert original.keys() == measured.keys()
        assert all(
            torch.equal(original[key], measured[key])
            for key in original
        )


def test_opt_in_prefetch_preserves_rng_order_and_tensor_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_sync() -> None:
        raise AssertionError("opt-in CPU timing must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbid_sync)
    legacy, timed = _trainer(prefetch=True), _trainer(prefetch=True)
    old_draws = _install_synthetic_sampler(legacy, monkeypatch, seed=71)
    new_draws = _install_synthetic_sampler(
        timed, monkeypatch, seed=71, sample_delay_s=0.012,
        tensor_delay_s=0.008,
    )
    split = _BatchPrefetchSplitTiming()
    buf = cast(Any, SimpleNamespace(exact_without_replacement=False))

    old_batches = list(legacy._iter_training_batches(
        buf, batch_size=2, mirror_prob=0.0, count=4,
    ))
    new_batches = list(timed._iter_training_batches(
        buf, batch_size=2, mirror_prob=0.0, count=4,
        split_timing=split,
    ))

    assert old_draws == new_draws
    _equal_batches(old_batches, new_batches)
    assert split.future_calls == split.tensor_calls == 4
    assert split.host_sample_calls == 0
    assert split.host_sample_s == 0.0
    assert split.future_wait_s >= 0.008
    assert split.tensor_issue_s >= 4 * 0.006


@pytest.mark.parametrize("exact", [False, True])
def test_serial_sampling_has_no_future_claim_and_preserves_bytes(
    monkeypatch: pytest.MonkeyPatch, exact: bool,
) -> None:
    # exact=True/prefetch=True is the run12 route: exact serial ownership
    # overrides the trainer's ordinary prefetch preference.
    legacy, timed = _trainer(prefetch=exact), _trainer(prefetch=exact)
    old_draws = _install_synthetic_sampler(legacy, monkeypatch, seed=19)
    new_draws = _install_synthetic_sampler(
        timed, monkeypatch, seed=19, sample_delay_s=0.008,
        tensor_delay_s=0.006,
    )
    buf = cast(Any, SimpleNamespace(exact_without_replacement=exact,
                                    host_batch_overlap=False))
    split = _BatchPrefetchSplitTiming()
    old_batches = list(legacy._iter_training_batches(
        buf, batch_size=2, mirror_prob=0.0, count=3,
    ))
    new_batches = list(timed._iter_training_batches(
        buf, batch_size=2, mirror_prob=0.0, count=3,
        split_timing=split,
    ))

    assert old_draws == new_draws
    _equal_batches(old_batches, new_batches)
    assert split.future_calls == 0
    assert split.future_wait_s == 0.0
    assert split.host_sample_calls == 3
    assert split.host_sample_s >= 3 * 0.006
    assert split.tensor_calls == 3
    assert split.tensor_issue_s >= 3 * 0.004


def test_exact_overlap_future_and_tensor_issue_keep_the_same_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy, timed = _trainer(prefetch=True), _trainer(prefetch=True)
    old_draws = _install_synthetic_sampler(legacy, monkeypatch, seed=43)
    new_draws = _install_synthetic_sampler(timed, monkeypatch, seed=43)
    buf = cast(Any, SimpleNamespace(
        plan=SimpleNamespace(host_overlap_reserve_bytes=100_000),
    ))
    split = _BatchPrefetchSplitTiming()

    old_batches = list(legacy._iter_exact_overlapped_batches(
        buf, batch_size=2, mirror_prob=0.0, count=3,
    ))
    new_batches = list(timed._iter_exact_overlapped_batches(
        buf, batch_size=2, mirror_prob=0.0, count=3,
        split_timing=split,
    ))

    assert old_draws == new_draws
    _equal_batches(old_batches, new_batches)
    assert split.future_calls == split.tensor_calls == 3
    assert split.host_sample_calls == 0
    assert split.host_sample_s == 0.0
    assert split.future_wait_s > 0.0
    assert split.tensor_issue_s > 0.0
