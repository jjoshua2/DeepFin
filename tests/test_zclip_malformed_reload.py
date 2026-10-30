"""Malformed live caps use the actual setter on an inert trainer surface."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from chess_anti_engine.train.trainer import Trainer, resolve_zclip_max_norm
from chess_anti_engine.tune import trainable
from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config


class _Shim:
    set_grad_clip_max_norm = Trainer.set_grad_clip_max_norm
    grad_clip_max_norm = Trainer.grad_clip_max_norm
    # The predecessor has no config-aware method; its extracted block uses
    # the original scalar setter instead. Assertions are identical.
    set_grad_clip_from_config = getattr(Trainer, "set_grad_clip_from_config", None)

    def __init__(self) -> None:
        # No Trainer/model/optimizer initialization or training execution.
        self.zclip = SimpleNamespace(max_grad_norm=5.0)


def _live_push(trainer: _Shim, config: dict) -> None:
    tree = ast.parse(inspect.getsource(trainable.train_trial))
    # Exact production cap-push block only; no trial setup or training loop.
    statements = [node for node in ast.walk(tree) if (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_old_cap" for t in node.targets)
    ) or (
        isinstance(node, ast.If) and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr in {"set_grad_clip_max_norm", "set_grad_clip_from_config"}
    )]
    assert len(statements) == 2
    namespace = dict(vars(trainable))
    namespace.update(trainer=trainer, config=config, iteration_idx=1)
    exec(compile(ast.Module(body=sorted(statements, key=lambda n: n.lineno), type_ignores=[]), "<live-cap-push>", "exec"), namespace)


@pytest.mark.parametrize("raw", ["invalid", [], {}, [1, 2], torch.tensor([1., 2.]), 10**400])
@pytest.mark.parametrize("route", ["direct", "live"])
def test_malformed_cap_preserves_last_valid_value_and_control_flow(raw: Any, route: str, capsys) -> None:
    trainer = _Shim()
    if route == "direct":
        assert trainer.set_grad_clip_max_norm(raw) is False
    else:
        _live_push(trainer, {"zclip_max_norm": raw})
    assert trainer.grad_clip_max_norm == 5.0
    output = capsys.readouterr().out
    assert "REFUSING" in output
    assert "keeping 5.0" in output
    assert "live reload" not in output
    assert trainer.set_grad_clip_max_norm(7.0) is True
    assert trainer.grad_clip_max_norm == 7.0


@pytest.mark.parametrize("raw", ["invalid", [], {}, [1, 2], 10**400])
def test_startup_keeps_strict_conversion_failure(raw: Any) -> None:
    with pytest.raises((TypeError, ValueError, OverflowError), match=r"float|convert"):
        resolve_zclip_max_norm({"zclip_max_norm": raw})


@pytest.mark.parametrize(("config", "expected"), [
    ({}, 1.0), ({"grad_clip": "3"}, 3.0),
    ({"zclip_max_norm": None, "grad_clip": 2}, None),
    ({"zclip_max_norm": "4", "grad_clip": 2}, 4.0),
    ({"zclip_max_norm": 5}, 5.0),
])
def test_live_and_startup_precedence_and_disable_match(config: dict, expected: float | None, capsys) -> None:
    assert resolve_zclip_max_norm(config) == expected
    trainer = _Shim()
    _live_push(trainer, config)
    assert trainer.grad_clip_max_norm == expected
    output = capsys.readouterr().out
    assert "REFUSING" not in output
    assert ("live reload" in output) is (expected != 5.0)


def test_real_yaml_reload_malformed_value_is_refused(tmp_path, capsys) -> None:
    path = tmp_path / "fixture.yaml"
    path.write_text('train:\n  zclip_max_norm: invalid\n')
    config: dict = {"zclip_max_norm": 5.0}
    _reload_yaml_into_config(config, str(path), live_reload=True)
    assert config["zclip_max_norm"] == "invalid"
    trainer = _Shim()
    _live_push(trainer, config)
    assert trainer.grad_clip_max_norm == 5.0
    assert "REFUSING" in capsys.readouterr().out


def test_unrelated_conversion_runtime_error_is_not_hidden() -> None:
    class Bad:
        def __float__(self) -> float:
            raise RuntimeError("unrelated conversion failure")

    trainer = _Shim()
    with pytest.raises(RuntimeError, match="unrelated conversion failure"):
        _live_push(trainer, {"zclip_max_norm": Bad()})
    assert trainer.grad_clip_max_norm == 5.0
