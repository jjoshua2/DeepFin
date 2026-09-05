"""YAML failure modes across launch, live reload and trial validation.

These structural checks protect the operational distinctions described in
``docs/operations.md`` without requiring literal values in project guidance.
They inspect exception boundaries without starting or terminating a live trial.
Policy-temperature runtime boundaries are covered by
``test_selfplay_policy_temp_plumbing.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _module(rel: str) -> ast.Module:
    return ast.parse((_REPO / rel).read_text(encoding="utf-8"))


def _function(rel: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module(rel)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{rel} no longer defines {name}()")


def _enclosing_tries(fn: ast.FunctionDef, line: int) -> list[ast.Try]:
    return [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Try) and node.lineno <= line <= (node.end_lineno or -1)
    ]


def _call_lines(fn: ast.FunctionDef, name: str) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            lines.append(node.lineno)
    return lines


def test_mid_run_reload_swallows_the_unknown_key_error() -> None:
    """Category (a), MID-RUN: the reload is rejected and the trial survives.

    The whole body of ``_reload_yaml_into_config`` sits under one ``try`` with a
    catch-all handler, which is what turns an unknown key into a warning plus
    "keep the old config" instead of an exception the caller has to survive.
    """
    fn = _function("chess_anti_engine/tune/trainable_config_ops.py", "_reload_yaml_into_config")
    tries = [node for node in fn.body if isinstance(node, ast.Try)]
    assert len(tries) == 1, "the overlay is meant to be wrapped in exactly one try"
    handlers = tries[0].handlers
    assert handlers, "no except => an unknown key would propagate and kill the trial"
    assert any(
        h.type is None or (isinstance(h.type, ast.Name) and h.type.id in {"Exception", "BaseException"})
        for h in handlers
    ), "the handler must be broad enough to catch the loader's ValueError"


def test_launch_path_flattens_outside_any_try() -> None:
    """An unknown launch key raises before a trial or fallback config exists."""
    fn = _function("chess_anti_engine/run.py", "main")
    lines = _call_lines(fn, "flatten_run_config_defaults")
    assert lines, "run.py:main no longer flattens the yaml"
    for line in lines:
        assert not _enclosing_tries(fn, line), (
            "flatten_run_config_defaults is inside a try in run.py:main; "
            "an unknown launch key must propagate instead of silently using defaults"
        )


def test_trial_iteration_loop_has_no_except() -> None:
    """Per-iteration validation errors run cleanup and propagate out of the trial."""
    fn = _function("chess_anti_engine/tune/trainable.py", "train_trial")
    lines = _call_lines(fn, "from_dict")
    assert lines, "train_trial no longer rebuilds TrialConfig per iteration"
    enclosing = [t for line in lines for t in _enclosing_tries(fn, line)]
    assert enclosing, "the from_dict call is no longer inside the iteration-loop try"
    assert all(t.handlers == [] for t in enclosing), (
        "the iteration loop grew an except; a bad TrialConfig value no longer "
        "propagates out of the trial; reassess its failure and cleanup behavior"
    )
    assert any(t.finalbody for t in enclosing), "the cleanup finally is gone"
