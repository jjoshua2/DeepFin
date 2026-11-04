"""Optional scalar metadata only: no checkpoint IO, model or training restore."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from chess_anti_engine.tune import trainable_init as init


@pytest.mark.parametrize("raw", [None, "bad", [], {}, [1], float("nan"), float("inf"), -float("inf"), 0, -1, 10**400])
def test_unusable_optional_peak_lr_is_ignored(monkeypatch, raw: Any) -> None:
    monkeypatch.setattr(init.torch, "load", lambda *_a, **_k: {"peak_lr": raw})
    assert init.peek_checkpoint_peak_lr(Path("unused")) is None
    init.guard_warm_start_lr(Path("unused"), {"lr": 0.003}, source="fixture")


@pytest.mark.parametrize("raw", [0.001, "0.001", torch.tensor(0.001), torch.tensor([0.001])])
def test_supported_positive_scalar_and_guard_contract(monkeypatch, raw: Any) -> None:
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return {"peak_lr": raw}

    monkeypatch.setattr(init.torch, "load", load)
    assert init.peek_checkpoint_peak_lr(Path("unused")) == pytest.approx(0.001)
    init.guard_warm_start_lr(Path("unused"), {"lr": 0.001}, source="fixture")
    with pytest.raises(ValueError, match="warm start from fixture"):
        init.guard_warm_start_lr(Path("unused"), {"lr": 0.003}, source="fixture")
    assert calls == [("unused", {"map_location": "cpu", "mmap": True})] * 3


def test_multiple_tensor_values_are_not_scalar_metadata(monkeypatch) -> None:
    monkeypatch.setattr(init.torch, "load", lambda *_a, **_k: {"peak_lr": torch.tensor([1., 2.])})
    assert init.peek_checkpoint_peak_lr(Path("unused")) is None


def test_optional_scalar_is_converted_once(monkeypatch) -> None:
    class Scalar:
        calls = 0

        def __float__(self) -> float:
            self.calls += 1
            return 0.001

    raw = Scalar()
    monkeypatch.setattr(init.torch, "load", lambda *_a, **_k: {"peak_lr": raw})
    assert init.peek_checkpoint_peak_lr(Path("unused")) == 0.001
    assert raw.calls == 1


@pytest.mark.parametrize("payload", [{}, None, []])
def test_missing_optional_metadata_remains_unavailable(monkeypatch, payload) -> None:
    monkeypatch.setattr(init.torch, "load", lambda *_a, **_k: payload)
    assert init.peek_checkpoint_peak_lr(Path("unused")) is None


def test_conversion_does_not_hide_unrelated_runtime_error(monkeypatch) -> None:
    class Scalar:
        def __float__(self) -> float:
            raise RuntimeError("unrelated conversion error")

    monkeypatch.setattr(init.torch, "load", lambda *_a, **_k: {"peak_lr": Scalar()})
    with pytest.raises(RuntimeError, match="unrelated conversion error"):
        init.peek_checkpoint_peak_lr(Path("unused"))


def test_mandatory_lr_configuration_errors_are_not_hidden(monkeypatch) -> None:
    monkeypatch.setattr(init.torch, "load", lambda *_a, **_k: {"peak_lr": 0.001})
    with pytest.raises(ValueError, match="could not convert string to float"):
        init.guard_warm_start_lr(Path("unused"), {"lr": "invalid"}, source="fixture")
    with pytest.raises(ValueError, match="could not convert string to float"):
        init.guard_warm_start_lr(Path("unused"), {"lr": 0.001, "warm_start_lr_max_ratio": "invalid"}, source="fixture")


def test_disabled_guard_does_not_read_metadata(monkeypatch) -> None:
    def unexpected(*_args, **_kwargs):
        pytest.fail("disabled guard must not read checkpoint metadata")

    monkeypatch.setattr(init.torch, "load", unexpected)
    init.guard_warm_start_lr(Path("unused"), {"lr": 1, "warm_start_lr_max_ratio": 0}, source="fixture")
