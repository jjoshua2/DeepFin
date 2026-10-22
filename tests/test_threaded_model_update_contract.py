"""CPU-only update contract checks; no compiler, dispatcher thread, or GPU."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from chess_anti_engine.inference_threaded import ThreadedDispatcher, _ModelUpdateRequest
from chess_anti_engine.model import ModelConfig


class _HeadModel(torch.nn.Module):
    def __init__(self, heads: int = 2, weight: float = 1.0) -> None:
        super().__init__()
        self.cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=heads)
        self._inference_only = True
        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight * (self.cfg.embed_dim // self.cfg.num_heads)


class _Compiled(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._orig_mod(x)


def _dispatcher(model: torch.nn.Module, monkeypatch):
    dispatcher = object.__new__(ThreadedDispatcher)
    dispatcher._compile_mode = "reduce-overhead"
    monkeypatch.setattr(dispatcher, "_evaluator", SimpleNamespace(model=_Compiled(model)), raising=False)
    compiled: list[torch.nn.Module] = []

    def compile_model(model: torch.nn.Module) -> torch.nn.Module:
        compiled.append(model)
        return _Compiled(model)

    monkeypatch.setattr(dispatcher, "_compile_model_on_dispatcher", compile_model)
    return dispatcher, compiled


def _update(dispatcher: ThreadedDispatcher, model: torch.nn.Module) -> _ModelUpdateRequest:
    request = _ModelUpdateRequest(model, threading.Event(), {})
    dispatcher._apply_model_update(request)
    assert request.done.is_set()
    assert request.result == {}
    return request


def test_same_architecture_reuses_compiled_wrapper_and_updates_output(monkeypatch):
    old = _HeadModel(weight=1.0)
    dispatcher, compiled = _dispatcher(old, monkeypatch)
    wrapper = dispatcher._evaluator.model
    new = _HeadModel(weight=3.0)
    _update(dispatcher, new)
    assert dispatcher._evaluator.model is wrapper
    assert compiled == []
    torch.testing.assert_close(wrapper(torch.ones(1)), new(torch.ones(1)))


def test_same_shaped_changed_heads_recompiles_and_uses_new_forward(monkeypatch):
    old = _HeadModel(heads=2)
    new = _HeadModel(heads=4)
    assert {k: v.shape for k, v in old.state_dict().items()} == {
        k: v.shape for k, v in new.state_dict().items()
    }
    dispatcher, compiled = _dispatcher(old, monkeypatch)
    wrapper = dispatcher._evaluator.model
    _update(dispatcher, new)
    assert compiled == [new]
    assert dispatcher._evaluator.model is not wrapper
    actual = dispatcher._evaluator.model(torch.ones(1))
    torch.testing.assert_close(actual, new(torch.ones(1)))
    assert not torch.equal(actual, old(torch.ones(1)))


@pytest.mark.parametrize("change", ["inference_mode", "unknown_config"])
def test_changed_or_unknown_forward_contract_replaces_model(monkeypatch, change):
    old = _HeadModel()
    new = _HeadModel()
    if change == "inference_mode":
        new._inference_only = False
    else:
        del new.cfg
    dispatcher, compiled = _dispatcher(old, monkeypatch)
    _update(dispatcher, new)
    assert compiled == [new]
    assert dispatcher._evaluator.model._orig_mod is new


def test_recompile_failure_signals_update_and_escapes_for_poison(monkeypatch):
    dispatcher, _ = _dispatcher(_HeadModel(), monkeypatch)
    failure = RuntimeError("compile failed")

    def fail_compile(_model: torch.nn.Module) -> torch.nn.Module:
        raise failure

    monkeypatch.setattr(dispatcher, "_compile_model_on_dispatcher", fail_compile)
    request = _ModelUpdateRequest(_HeadModel(heads=4), threading.Event(), {})
    with pytest.raises(RuntimeError, match="compile failed") as caught:
        dispatcher._apply_model_update(request)
    assert caught.value is failure
    assert request.done.is_set()
    assert request.result["error"] is failure


@pytest.mark.parametrize("fallback_fails", [False, True])
def test_strict_load_failure_keeps_recompile_or_poison_contract(monkeypatch, fallback_fails):
    old = _HeadModel()
    new = _HeadModel(weight=3.0)
    dispatcher, compiled = _dispatcher(old, monkeypatch)

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("strict load failed")

    monkeypatch.setattr(old, "load_state_dict", fail_load)
    if fallback_fails:
        def fail_compile(_model: torch.nn.Module) -> torch.nn.Module:
            raise RuntimeError("fallback failed")

        monkeypatch.setattr(dispatcher, "_compile_model_on_dispatcher", fail_compile)
        request = _ModelUpdateRequest(new, threading.Event(), {})
        with pytest.raises(RuntimeError, match="fallback failed"):
            dispatcher._apply_model_update(request)
        assert request.done.is_set()
        assert str(request.result["error"]) == "fallback failed"
    else:
        _update(dispatcher, new)
        assert compiled == [new]
        torch.testing.assert_close(
            dispatcher._evaluator.model(torch.ones(1)), new(torch.ones(1)),
        )
